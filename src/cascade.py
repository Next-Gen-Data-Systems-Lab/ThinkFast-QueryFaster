"""A LOTUS-style cascade, replayed from saved scores: a cheap proxy ranks every
document and the scan's model (the oracle) judges only the uncertain middle.

This is the optimisation LOTUS (Patel et al.) applies to sem_filter, with our
scans as the oracle. Per query:

  1. The proxy ranks all N documents: embedding similarity (free, since the
     index already exists) or the MiniLM cross-encoder (src.rerank_scan).
  2. The oracle labels a sample of SAMPLE documents (at most a quarter of N),
     drawn with replacement, half uniformly and half in proportion to 1 / rank,
     so that the rare positives near the top are well covered.
  3. From the sample, with inverse-probability weights, two rank cuts are set:
       b  the shallowest cut whose estimated recall of the oracle's positives
          reaches the target
       a  the deepest cut (at most b) whose estimated precision against the
          oracle reaches the target
  4. Ranks up to a are accepted without the oracle, ranks below b are rejected,
     and the band between goes to the oracle. Sampled documents keep the
     oracle's label.

The oracle's own decision is its P(yes) at a threshold chosen on the other
queries (leave one query out), as for the scan elsewhere in the report.

Two ways to set the cuts:
  point     the sample's point estimates; targets are met only on average, and
            not even that when positives are rare (few land in the sample)
  guarded   like LOTUS, which corrects for sampling error so the targets hold
            with probability 1 - DELTA: here a cut must meet its target at the
            DELTA quantile of BOOT bootstrap resamples of the sample. If no cut
            does, the oracle judges everything below a.
Results are averaged over SEEDS samples.

Cost is GPU seconds per query: N over the proxy's documents per second (zero
for embeddings) plus the oracle's calls over its documents per second, both
measured in our runs. Rates are per query where the rows allow it.

Reported per (proxy, oracle, target): the share of documents the oracle sees,
seconds, speed-up over the oracle scanning everything, recall, precision and
F1 against gold, recall and precision against the oracle's full scan, and the
F1 difference from the full scan with a bootstrap 95% interval over queries.

Usage: python -m src.cascade
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import access_paths as AP
from . import data as D
from .baselines import embed, saved
from .crossover import RATE

# (label, dataset names, size, {scorer: (scores file, rows file for the rate)});
# "{d}" is the dataset name. A rows file of None means the 20NG plateau RATE.
SETTINGS = [
    ("20NG 10K", "20newsgroups", 10000, {
        "qwen": ("results/20newsgroups_sweep_bs32.scores.jsonl", None),
        "kev": ("results/20newsgroups_kev9bfull_packed.scores.jsonl",
                "results/20newsgroups_kev9bfull_separate.jsonl"),
        "minilm": ("results/20newsgroups_minilm_separate.scores.jsonl",
                   "results/20newsgroups_minilm_separate.jsonl")}),
    ("SYNERGY", "synergy", 100000, {
        "qwen": ("results/synergy_qwen.scores.jsonl", "results/synergy_qwen.jsonl"),
        "kev": ("results/{d}_kev9b_packed.scores.jsonl", "results/{d}_kev9b_packed.jsonl"),
        "minilm": ("results/{d}_minilm_separate.scores.jsonl",
                   "results/{d}_minilm_separate.jsonl")}),
]
PAIRS = [("embed", "qwen"), ("embed", "kev"), ("minilm", "qwen"), ("minilm", "kev")]
TARGETS = (0.8, 0.9, 0.95)
SAMPLE = 500
SEEDS = 10
DELTA = 0.05
BOOT = 200
GUARDS = ("point", "guarded")


def rates(path: str | None, corpus: D.Corpus, queries: list[D.Query]) -> list[float]:
    """Documents per second for each query, from that scan's rows."""
    n = len(corpus.docs)
    if path is None:
        return [RATE] * len(queries)
    t = {}
    with open(path.format(d=corpus.name)) as fh:
        for line in fh:
            r = json.loads(line)
            if (r["arm"] == "classify" and r["corpus_size"] == n
                    and r.get("dataset", corpus.name) == corpus.name):
                t[r["query_id"]] = r["t_total"]
    mean = n * len(t) / sum(t.values())
    return [n / t[q.id] if q.id in t else mean for q in queries]


def load(names: str, n: int, files: dict, results: Path) -> list[dict]:
    """One dict per query: gold, each scorer's scores and documents per second."""
    out = []
    for sub in D.subsets(names, str(n)):
        qs = sub.queries
        gold = [np.isin([d.id for d in sub.docs], list(q.gold)) for q in qs]
        mats = {"embed": embed(sub, qs, results)}
        rate = {"embed": [np.inf] * len(qs)}
        for name, (scores, rows) in files.items():
            mats[name] = saved(Path(scores.format(d=sub.name)), sub, qs)
            rate[name] = rates(rows, sub, qs)
        missing = [k for k, m in mats.items() if m is None]
        assert not missing, f"{sub.name}: no scores for {missing}"
        for i, q in enumerate(qs):
            out.append({"id": q.id, "gold": gold[i],
                        "scores": {k: m[i] for k, m in mats.items()},
                        "rate": {k: r[i] for k, r in rate.items()}})
    return out


def oracle_labels(queries: list[dict], oracle: str) -> list[np.ndarray]:
    """The oracle's full-scan decisions, at a threshold chosen on the other queries."""
    table = np.array([[AP_f1(q["scores"][oracle] >= t, q["gold"]) for t in AP.THRESHOLDS]
                      for q in queries])
    labels = []
    for i, q in enumerate(queries):
        t = AP.THRESHOLDS[int(np.argmax(np.delete(table, i, axis=0).mean(axis=0)))]
        labels.append(q["scores"][oracle] >= t)
    return labels


def AP_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int((pred & gold).sum())
    return 2 * tp / (pred.sum() + gold.sum()) if tp else 0.0


def cascade(proxy: np.ndarray, oracle: np.ndarray, target: float,
            rng: np.random.Generator, guard: str = "point") -> tuple[np.ndarray, int]:
    """Predicted set and the number of oracle calls, for one query."""
    n = len(proxy)
    order = np.argsort(-proxy, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    w = 0.5 / n + 0.5 * (1.0 / rank) / np.sum(1.0 / np.arange(1, n + 1))
    m = min(SAMPLE, n // 4)
    picked = rng.choice(n, size=m, replace=True, p=w)
    o = oracle[picked]
    weight = o / (m * w[picked])            # each sampled positive's share of the total
    total = weight.sum()
    if total == 0:
        a = b = 0                           # the sample found no positives: reject all
    else:
        ranks = rank[picked]
        by_rank = np.argsort(ranks, kind="stable")
        cuts = ranks[by_rank]
        wt = weight[by_rank]
        if guard == "guarded":
            # each row a bootstrap resample of the sample, as counts per draw
            counts = rng.multinomial(m, np.full(m, 1.0 / m), size=BOOT)
            above = np.cumsum(counts * wt, axis=1)
            totals = above[:, -1:]
            recall = np.where(totals > 0, above / np.where(totals > 0, totals, 1), 0.0)
            recall_lo = np.quantile(recall, DELTA, axis=0)
            above_lo = np.quantile(above, DELTA, axis=0)
        else:
            above_lo = np.cumsum(wt)
            recall_lo = above_lo / total
        hit = np.nonzero(recall_lo >= target - 1e-12)[0]
        b = int(cuts[hit[0]]) if len(hit) else n
        # precision of the top c against the oracle, estimated as positives / c
        ok = [int(c) for c, pos in zip(cuts, above_lo) if c <= b and pos / c >= target]
        a = max(ok) if ok else 0
    pred = rank <= a
    band = (rank > a) & (rank <= b)
    pred |= band & oracle
    seen = np.zeros(n, dtype=bool)
    seen[picked] = True
    pred[seen] = oracle[seen]
    return pred, int((band | seen).sum())


def fixed_k(queries: list[dict], labels: list[np.ndarray], proxy: str, oracle: str,
            seconds: float) -> tuple[int, np.ndarray]:
    """The hybrid at one k for every query (the oracle judges the proxy's top k),
    with k set so the mean seconds per query match `seconds`; its F1 per query."""
    def cost(k):
        return np.mean([len(l) / q["rate"][proxy] + min(k, len(l)) / q["rate"][oracle]
                        for q, l in zip(queries, labels)])
    lo, hi = 0, max(len(l) for l in labels)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if cost(mid) < seconds else (lo, mid)
    f1 = []
    for q, l in zip(queries, labels):
        top = np.zeros(len(l), dtype=bool)
        top[np.argsort(-q["scores"][proxy], kind="stable")[:lo]] = True
        f1.append(prf(top & l, q["gold"])[2])
    return lo, np.array(f1)


def prf(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    tp = int((pred & ref).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / ref.sum() if ref.sum() else 1.0
    return r, p, (2 * p * r / (p + r) if tp else 0.0)


def evaluate(label: str, queries: list[dict]) -> dict:
    out = {"setting": label, "queries": len(queries), "pairs": []}
    full = {}
    for oracle in {o for _, o in PAIRS}:
        labels = oracle_labels(queries, oracle)
        rows = [prf(l, q["gold"]) for l, q in zip(labels, queries)]
        secs = [len(q["gold"]) / q["rate"][oracle] for q in queries]
        full[oracle] = (labels, np.array([r[2] for r in rows]))
        out[f"full_{oracle}"] = {"recall": float(np.mean([r[0] for r in rows])),
                                 "precision": float(np.mean([r[1] for r in rows])),
                                 "f1": float(np.mean([r[2] for r in rows])),
                                 "seconds": float(np.mean(secs))}
    for proxy, oracle in PAIRS:
        labels, full_f1 = full[oracle]
        for guard, target in [(g, t) for g in GUARDS for t in TARGETS]:
            per = []    # per query: mean over seeds of each measure
            for q, lab in zip(queries, labels):
                runs = []
                for seed in range(SEEDS):
                    rng = np.random.default_rng(seed)
                    pred, calls = cascade(q["scores"][proxy], lab, target, rng, guard)
                    n = len(lab)
                    secs = n / q["rate"][proxy] + calls / q["rate"][oracle]
                    g, vs = prf(pred, q["gold"]), prf(pred, lab)
                    runs.append([calls / n, secs, n / q["rate"][oracle] / secs,
                                 *g, vs[0], vs[1]])
                per.append(np.mean(runs, axis=0))
            per = np.array(per)
            d, lo, hi = AP.paired(per[:, 5], full_f1)
            k, k_f1 = fixed_k(queries, labels, proxy, oracle, float(per[:, 1].mean()))
            out["pairs"].append({
                "proxy": proxy, "oracle": oracle, "guard": guard, "target": target,
                "oracle_share": float(per[:, 0].mean()),
                "seconds": float(per[:, 1].mean()),
                "speedup": float(np.median(per[:, 2])),     # median over queries
                "recall": float(per[:, 3].mean()), "precision": float(per[:, 4].mean()),
                "f1": float(per[:, 5].mean()),
                "recall_vs_oracle": float(per[:, 6].mean()),
                "precision_vs_oracle": float(per[:, 7].mean()),
                "d_f1_vs_full": [d, lo, hi],
                # the hybrid with one k for all queries, at the same mean seconds
                "fixed_k": k, "fixed_k_f1": float(k_f1.mean()),
                "d_f1_vs_fixed_k": list(AP.paired(per[:, 5], k_f1))})
    return out


def show(r: dict) -> None:
    print(f"\n== {r['setting']} ({r['queries']} queries)")
    for o in ("qwen", "kev"):
        f = r[f"full_{o}"]
        print(f"  full {o:4s} scan: R {f['recall']:.3f} P {f['precision']:.3f} "
              f"F1 {f['f1']:.3f}, {f['seconds']:.1f} s per query")
    print(f"  {'proxy':6s} {'oracle':6s} {'cuts':7s} {'tgt':>4s} {'share':>6s} {'sec':>6s} {'speedup':>7s} "
          f"{'R':>6s} {'P':>6s} {'F1':>6s} {'R/or':>6s} {'P/or':>6s}  F1 vs full scan"
          f"          vs fixed k at equal time")
    for p in r["pairs"]:
        d, f = p["d_f1_vs_full"], p["d_f1_vs_fixed_k"]
        print(f"  {p['proxy']:6s} {p['oracle']:6s} {p['guard']:7s} {p['target']:4.2f} {p['oracle_share']:6.3f} "
              f"{p['seconds']:6.1f} {p['speedup']:6.1f}x {p['recall']:6.3f} {p['precision']:6.3f} "
              f"{p['f1']:6.3f} {p['recall_vs_oracle']:6.3f} {p['precision_vs_oracle']:6.3f}  "
              f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]  "
              f"{f[0]:+.3f} [{f[1]:+.3f}, {f[2]:+.3f}] (k={p['fixed_k']})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/cascade.json")
    args = p.parse_args(argv)
    out = []
    for label, names, n, files in SETTINGS:
        r = evaluate(label, load(names, n, files, Path(args.results)))
        show(r)
        out.append(r)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
