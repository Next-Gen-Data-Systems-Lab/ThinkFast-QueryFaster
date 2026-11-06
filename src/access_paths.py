"""Offline evaluation of hybrid plans: fixed k, adaptive k and the scan.

Every plan here walks the embedding ranking, classifies the first k documents,
and keeps those the classifier accepts. The scan is the same plan with k = N, so
all plans sit on one cost axis: documents classified per query. The scan already
scored every document and `src.embed_ranks` saved the ranking, so this replays
each plan without a GPU.

Rules that choose k for each query
  mass     documents with similarity above mean + z * std. No model calls.
  gap      the largest drop in similarity, plus a buffer of 5 (Taguchi et al.).
  sample   classify m random documents, estimate the answer size, set k from it.
  yield    classify in chunks and stop when the recent acceptance rate falls
           below y. This chooses k during execution rather than predicting it.

`oracle_curve` is the true upper bound: it sees the answer and spends a budget
across queries for the most recall, so no rule of this kind can beat it.

`transfer` tests what deployment needs. A setting is tuned on one corpus to reach
a share of the scan's recall, then applied unchanged to another corpus.

Usage: python -m src.access_paths
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed_ranks import path_for

CHUNK = 32
WINDOW = 64
SAMPLE = 200
SEEDS = 20
BOOTSTRAP = 2000
THRESHOLDS = [0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,
              1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
YIELDS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.015]
TRANSFER_TARGET = 0.8

RUNS = [
    ("20newsgroups", 1000, "results/20newsgroups_sweep_bs32.scores.jsonl"),
    ("20newsgroups", 5000, "results/20newsgroups_sweep_bs32.scores.jsonl"),
    ("20newsgroups", 10000, "results/20newsgroups_sweep_bs32.scores.jsonl"),
    ("nfcorpus", 3633, "results/nfcorpus_nf.scores.jsonl"),
]


@dataclass
class Query:
    id: str
    sims: np.ndarray     # similarity, best first
    scores: np.ndarray   # classifier P(yes), in the same order
    gold: np.ndarray     # bool, in the same order

    @property
    def n(self) -> int:
        return len(self.sims)


def load(dataset: str, n: int, scores_file: str, results: Path = Path("results")) -> list[Query]:
    z = np.load(path_for(results, dataset, n))
    rows = {}
    with open(scores_file) as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] == "classify" and r["corpus_size"] == n:
                rows[r["query_id"]] = r      # a resumed job repeats a query; keep the last
    doc_ids = z["doc_ids"].tolist()
    row_of = {q: i for i, q in enumerate(z["query_ids"].tolist())}
    out = []
    for qid, r in rows.items():
        sims = z["sims"][row_of[qid]]
        order = np.argsort(-sims, kind="stable")
        scores = np.array([r["scores"][d] for d in doc_ids])
        gold = np.isin(np.array(doc_ids), np.array(r["gold"]))
        out.append(Query(qid, sims[order], scores[order], gold[order]))
    return out


def quality(q: Query, k: int, t: float, extra: np.ndarray | None = None) -> tuple[float, float, float]:
    """Recall, precision and F1 for classifying the top k, plus any `extra` positions."""
    mask = np.zeros(q.n, dtype=bool)
    mask[:k] = True
    if extra is not None:
        mask[extra] = True
    pred = mask & (q.scores >= t)
    tp = int((pred & q.gold).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / q.gold.sum()
    return r, p, 2 * p * r / (p + r) if p + r else 0.0


def k_mass(q: Query, z: float) -> int:
    return int(np.clip((q.sims >= q.sims.mean() + z * q.sims.std()).sum(), 1, q.n))


def k_gap(q: Query) -> int:
    head = q.sims[: int(0.9 * q.n)]
    return int(min(np.argmax(head[:-1] - head[1:]) + 1 + 5, q.n))


def k_yield(q: Query, y: float, t: float, chunk: int = CHUNK, window: int = WINDOW) -> int:
    """Depth at which the acceptance rate over the last `window` falls below y."""
    accepted = np.concatenate([[0], np.cumsum(q.scores >= t)])
    k = chunk
    while k < q.n:
        if k >= window and (accepted[k] - accepted[k - window]) / window < y:
            return k
        k += chunk
    return q.n


def rows_for(queries: list[Query], ks: list[int], t: float) -> dict:
    m = np.array([quality(q, k, t) for q, k in zip(queries, ks)])
    return {"cost": np.array(ks, dtype=float), "recall": m[:, 0],
            "precision": m[:, 1], "f1": m[:, 2]}


def plan_sample(queries: list[Query], alpha: float, t: float) -> dict:
    """Averaged over seeds. Sampled documents are paid for and may be kept."""
    acc = {"cost": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        out = {k: [] for k in acc}
        for q in queries:
            pos = rng.choice(q.n, size=min(SAMPLE, q.n), replace=False)
            k = int(np.clip(round(alpha * (q.scores[pos] >= t).mean() * q.n), 1, q.n))
            r, p, f = quality(q, k, t, pos)
            for key, v in zip(acc, (k + (pos >= k).sum(), r, p, f)):
                out[key].append(v)
        for key in acc:
            acc[key] = acc[key] + np.array(out[key], dtype=float) / SEEDS
    return acc


def matched_fixed(queries: list[Query], cost: float, t: float) -> dict:
    return rows_for(queries, [min(max(int(round(cost)), 1), q.n) for q in queries], t)


def paired(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b, with a bootstrap 95% interval over queries."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def compare(queries: list[Query], plan: dict, t: float) -> dict:
    """A plan against fixed k at the same mean cost."""
    fixed = matched_fixed(queries, plan["cost"].mean(), t)
    out = {"cost": float(plan["cost"].mean()),
           "cost_sd": float(plan["cost"].std())}
    for m in ("recall", "f1"):
        d, lo, hi = paired(plan[m], fixed[m])
        out.update({m: float(plan[m].mean()), f"fixed_{m}": float(fixed[m].mean()),
                    f"d_{m}": d, f"d_{m}_lo": lo, f"d_{m}_hi": hi})
    return out


def sweep(queries: list[Query], t: float) -> dict[str, list[dict]]:
    plans: dict[str, list[dict]] = {"mass": [], "gap": [], "sample": [], "yield": []}
    for z in [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]:
        plans["mass"].append({"param": z, **compare(
            queries, rows_for(queries, [k_mass(q, z) for q in queries], t), t)})
    plans["gap"].append({"param": 0, **compare(
        queries, rows_for(queries, [k_gap(q) for q in queries], t), t)})
    for a in [1, 2, 4, 8, 16, 32]:
        plans["sample"].append({"param": a, **compare(queries, plan_sample(queries, a, t), t)})
    for y in YIELDS:
        plans["yield"].append({"param": y, **compare(
            queries, rows_for(queries, [k_yield(q, y, t) for q in queries], t), t)})
    return plans


def hull(q: Query, t: float) -> list[tuple[int, float]]:
    """Segments (documents, recall gained) of the concave hull of recall against k."""
    hits = np.flatnonzero((q.scores >= t) & q.gold)
    points = [(0, 0.0)] + [(int(h) + 1, (i + 1) / q.gold.sum()) for i, h in enumerate(hits)]
    top: list[tuple[int, float]] = []
    for pt in points:
        while len(top) >= 2 and ((top[-1][1] - top[-2][1]) * (pt[0] - top[-2][0])
                                 <= (pt[1] - top[-2][1]) * (top[-1][0] - top[-2][0])):
            top.pop()
        top.append(pt)
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(top, top[1:])]


def oracle_curve(queries: list[Query], t: float) -> list[dict]:
    """Most mean recall at each mean cost, from the steepest segments of any query.

    Each query's segments get steeper-first order from the hull, so taking all
    segments by slope keeps every query's depth valid.
    """
    segments = sorted((seg for q in queries for seg in hull(q, t)),
                      key=lambda s: -s[1] / s[0])
    cost = np.cumsum([s[0] for s in segments]) / len(queries)
    recall = np.cumsum([s[1] for s in segments]) / len(queries)
    return [{"cost": float(c), "recall": float(r)} for c, r in zip(cost, recall)]


def oracle_at(curve: list[dict], cost: float) -> float:
    return float(np.interp(cost, [0] + [p["cost"] for p in curve],
                           [0] + [p["recall"] for p in curve]))


def setting_for(queries: list[Query], t: float, rule: str, target: float):
    """The cheapest setting of a rule that reaches `target` mean recall."""
    n = queries[0].n
    if rule == "k":
        grid = [int(round(x)) for x in np.geomspace(10, n, 80)]
    elif rule == "fraction":
        grid = [k / n for k in [int(round(x)) for x in np.geomspace(10, n, 80)]]
    else:
        grid = sorted(YIELDS, reverse=True)
    for v in grid:
        if apply_rule(queries, t, rule, v)["recall"].mean() >= target:
            return v
    return grid[-1]


def apply_rule(queries: list[Query], t: float, rule: str, value) -> dict:
    if rule == "k":
        ks = [min(int(value), q.n) for q in queries]
    elif rule == "fraction":
        ks = [max(int(round(value * q.n)), 1) for q in queries]
    else:
        ks = [k_yield(q, value, t) for q in queries]
    return rows_for(queries, ks, t)


def transfer(sets: dict[str, list[Query]], t_of: dict[str, float], pairs: list) -> list[dict]:
    out = []
    for src, dst in pairs:
        scan_src = rows_for(sets[src], [q.n for q in sets[src]], t_of[src])["recall"].mean()
        scan_dst = rows_for(sets[dst], [q.n for q in sets[dst]], t_of[dst])["recall"].mean()
        for rule in ("k", "fraction", "yield"):
            value = setting_for(sets[src], t_of[src], rule, TRANSFER_TARGET * scan_src)
            r = apply_rule(sets[dst], t_of[dst], rule, value)
            out.append({"source": src, "target": dst, "rule": rule, "value": value,
                        "cost": float(r["cost"].mean()),
                        "cost_share": float(r["cost"].mean() / sets[dst][0].n),
                        "recall": float(r["recall"].mean()),
                        "share_of_scan": float(r["recall"].mean() / scan_dst)})
    return out


def sensitivity(queries: list[Query], t: float) -> list[dict]:
    """The yield rule at other chunk and window sizes, y fixed at 0.1."""
    out = []
    for chunk, window in [(16, 32), (32, 64), (32, 128), (64, 128), (128, 256)]:
        r = rows_for(queries, [k_yield(q, 0.1, t, chunk, window) for q in queries], t)
        out.append({"chunk": chunk, "window": window, "cost": float(r["cost"].mean()),
                    "recall": float(r["recall"].mean())})
    return out


def fixed_curve(queries: list[Query], t: float, points: int = 40) -> list[dict]:
    n = queries[0].n
    ks = sorted({int(round(x)) for x in np.geomspace(10, n, points)})
    out = []
    for k in ks:
        r = rows_for(queries, [k] * len(queries), t)
        out.append({"k": k, "recall": float(r["recall"].mean()),
                    "precision": float(r["precision"].mean()), "f1": float(r["f1"].mean())})
    return out


def calibrate(queries: list[Query]) -> dict:
    """Leave one query out: choose t on the others, score it on the held-out query."""
    table = np.array([[quality(q, q.n, t) for t in THRESHOLDS] for q in queries])
    f1 = table[:, :, 2]
    held, chosen = [], []
    for i in range(len(queries)):
        best = int(np.argmax(np.delete(f1, i, axis=0).mean(axis=0)))
        chosen.append(THRESHOLDS[best])
        held.append(table[i, best])
    held = np.array(held)
    half = THRESHOLDS.index(0.5)
    d, lo, hi = paired(held[:, 2], table[:, half, 2])
    return {"grid": [{"t": t, "recall": float(table[:, j, 0].mean()),
                      "precision": float(table[:, j, 1].mean()),
                      "f1": float(table[:, j, 2].mean())} for j, t in enumerate(THRESHOLDS)],
            "chosen": sorted(set(chosen)), "median_t": float(np.median(chosen)),
            "recall": float(held[:, 0].mean()), "precision": float(held[:, 1].mean()),
            "f1": float(held[:, 2].mean()), "f1_at_half": float(table[:, half, 2].mean()),
            "d_f1": d, "d_f1_lo": lo, "d_f1_hi": hi}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/access_paths.json")
    args = p.parse_args(argv)

    report: dict = {}
    sets, calibrated = {}, {}
    for dataset, n, scores in RUNS:
        name = f"{dataset}_{n}"
        queries = sets[name] = load(dataset, n, scores)
        cal = calibrate(queries)
        calibrated[name] = cal["median_t"]
        entry = {"n_queries": len(queries),
                 "gold": [int(q.gold.sum()) for q in queries],
                 "calibration": cal, "thresholds": {}}
        for t in sorted({0.5, cal["median_t"]}, reverse=True):
            plans = sweep(queries, t)
            curve = oracle_curve(queries, t)
            for rows in plans.values():
                for row in rows:
                    row["oracle_recall"] = oracle_at(curve, row["cost"])
            entry["thresholds"][str(t)] = {"fixed": fixed_curve(queries, t), "plans": plans,
                                           "oracle": curve[:: max(len(curve) // 200, 1)],
                                           "sensitivity": sensitivity(queries, t)}
        report[name] = entry
        print(f"[access_paths] {name} done, calibrated t={cal['median_t']}", flush=True)

    pairs = [("20newsgroups_1000", "20newsgroups_5000"),
             ("20newsgroups_1000", "20newsgroups_10000"),
             ("20newsgroups_10000", "nfcorpus_3633"),
             ("nfcorpus_3633", "20newsgroups_10000")]
    report["transfer"] = {
        "half": transfer(sets, {k: 0.5 for k in sets}, pairs),
        "calibrated": transfer(sets, calibrated, pairs),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[access_paths] wrote {args.out}")


if __name__ == "__main__":
    main()
