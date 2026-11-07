"""Kev-9B against zero-shot Qwen2.5-7B as the scan's classifier, on the same documents.

For each setting, both scans scored every (query, document) pair. This replays
them offline:

  half        threshold 0.5
  held-out    threshold chosen on the other queries (leave one query out)
  recall@100  ranking quality, independent of any threshold
  calibration expected calibration error and Brier score over all pairs
  hybrid      top k by embedding, then classify, at k = 100 and 1,000
  equal time  the scan in embedding order, given the agent's measured seconds

Differences are Kev minus Qwen, with a bootstrap 95% interval over queries.
Needs the saved embedding rankings from src.embed_ranks.

Usage: python -m src.kev_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import access_paths as AP
from .crossover import AGENT, RATE

SETTINGS = [
    ("20newsgroups", 1000, "results/20newsgroups_sweep_bs32", "results/20newsgroups_kev9b"),
    ("20newsgroups", 5000, "results/20newsgroups_sweep_bs32", "results/20newsgroups_kev9bfull"),
    ("20newsgroups", 10000, "results/20newsgroups_sweep_bs32", "results/20newsgroups_kev9bfull"),
    ("nfcorpus", 3633, "results/nfcorpus_nf", "results/nfcorpus_kev9b"),
    ("20newsgroups_logic", 1000, "results/20newsgroups_logic_logic",
     "results/20newsgroups_logic_kev9b"),
    ("20newsgroups_logic", 10000, "results/20newsgroups_logic_logic",
     "results/20newsgroups_logic_kev9b"),
]
KS = (100, 1000)
BINS = 10


def rows(path: str, n: int) -> pd.DataFrame:
    with open(path) as fh:
        r = pd.DataFrame(json.loads(line) for line in fh if line.strip())
    r = r[(r.arm == "classify") & (r.corpus_size == n)]
    return r.drop_duplicates("query_id", keep="last").set_index("query_id")


def held_out(queries: list[AP.Query]) -> tuple[np.ndarray, list[float]]:
    """Per-query (recall, precision, F1) at a threshold chosen without that query."""
    table = np.array([[AP.quality(q, q.n, t) for t in AP.THRESHOLDS] for q in queries])
    held, chosen = [], []
    for i in range(len(queries)):
        best = int(np.argmax(np.delete(table[:, :, 2], i, axis=0).mean(axis=0)))
        chosen.append(AP.THRESHOLDS[best])
        held.append(table[i, best])
    return np.array(held), chosen


def calibration(queries: list[AP.Query]) -> dict:
    p = np.concatenate([q.scores for q in queries])
    y = np.concatenate([q.gold for q in queries]).astype(float)
    bins = np.minimum((p * BINS).astype(int), BINS - 1)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
              for b in range(BINS) if (bins == b).any())
    top = p >= 0.5
    return {"ece": float(ece), "brier": float(((p - y) ** 2).mean()),
            "mean_p_above_half": float(p[top].mean()) if top.any() else None,
            "precision_above_half": float(y[top].mean()) if top.any() else None}


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    d, lo, hi = AP.paired(a, b)
    return {"d": d, "lo": lo, "hi": hi, "wins": int((a > b).sum()), "ties": int((a == b).sum())}


def rate(stem: str, mode: str, n: int) -> float | None:
    """Documents per second for one query, as Kev ran in that mode."""
    path = Path(f"{stem}_{mode}.jsonl")
    if not path.exists():
        return None
    r = rows(str(path), n)
    return float(n * len(r) / r.t_total.sum())


def setting(dataset: str, n: int, qwen: str, kev: str) -> dict:
    qs = {q.id: q for q in AP.load(dataset, n, f"{qwen}.scores.jsonl")}
    ks = {q.id: q for q in AP.load(dataset, n, f"{kev}_packed.scores.jsonl")}
    ids = sorted(set(qs) & set(ks))
    Q, K = [qs[i] for i in ids], [ks[i] for i in ids]
    assert all((a.gold == b.gold).all() for a, b in zip(Q, K)), "different documents"
    qr, kr = rows(f"{qwen}.jsonl", n).loc[ids], rows(f"{kev}_packed.jsonl", n).loc[ids]

    out = {"dataset": dataset, "n": n, "queries": len(ids),
           "kev_rate_separate": rate(kev, "separate", n),
           "kev_rate_packed": rate(kev, "packed", n)}
    for name, (A, r) in (("qwen", (Q, qr)), ("kev", (K, kr))):
        half = np.array([AP.quality(q, q.n, 0.5) for q in A])
        held, chosen = held_out(A)
        out[name] = {"half": half.mean(axis=0).tolist(), "held": held.mean(axis=0).tolist(),
                     "median_t": float(np.median(chosen)),
                     "best_f1": float(r.best_f1.mean()), "recall@100": float(r["recall@100"].mean()),
                     **calibration(A),
                     "hybrid": {k: np.array([AP.quality(q, k, 0.5) for q in A]).mean(axis=0).tolist()
                                for k in KS if k < n}}
        out[name]["_half"], out[name]["_held"] = half, held
        out[name]["_r100"] = r["recall@100"].values
    out["d_f1_half"] = diff(out["kev"]["_half"][:, 2], out["qwen"]["_half"][:, 2])
    out["d_f1_held"] = diff(out["kev"]["_held"][:, 2], out["qwen"]["_held"][:, 2])
    out["d_recall_held"] = diff(out["kev"]["_held"][:, 0], out["qwen"]["_held"][:, 0])
    out["d_recall@100"] = diff(out["kev"]["_r100"], out["qwen"]["_r100"])

    if dataset == "20newsgroups_logic":
        shapes = {}
        for shape in ("kw", "and", "or", "not"):
            idx = [i for i, q in enumerate(ids) if q.startswith(shape + "_")]
            shapes[shape] = {m: {"half": out[m]["_half"][idx].mean(axis=0).tolist(),
                                 "held": out[m]["_held"][idx].mean(axis=0).tolist()}
                             for m in ("qwen", "kev")}
        out["shapes"] = shapes

    if dataset == "20newsgroups" and Path(AGENT).exists():
        with open(AGENT) as fh:
            a = pd.DataFrame(json.loads(line) for line in fh if line.strip())
        a = a[(a.arm == "agent") & (a.corpus_size == n)].drop_duplicates(
            "query_id", keep="last").set_index("query_id").loc[ids]
        seconds = float(a.wall_s.mean())
        eq = {"agent_s": seconds, "agent_recall": float(a.recall.mean())}
        for name, A, docs_per_s in (("qwen", Q, RATE),
                                    ("kev_separate", K, out["kev_rate_separate"]),
                                    ("kev_packed", K, out["kev_rate_packed"])):
            k = min(n, int(seconds * docs_per_s))
            plan = AP.rows_for(A, [k] * len(A), 0.5)
            eq[name] = {"k": k, "recall": float(plan["recall"].mean()),
                        "vs_agent": diff(plan["recall"], a.recall.values)}
        out["equal_time"] = eq

    for name in ("qwen", "kev"):
        for key in ("_half", "_held", "_r100"):
            out[name].pop(key)
    return out


def show(r: dict) -> None:
    def f(x):
        return " ".join(f"{v:.3f}" for v in x)

    def ci(x):
        return (f"{x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] "
                f"wins {x['wins']}, ties {x['ties']}")

    print(f"\n== {r['dataset']} N={r['n']} ({r['queries']} queries)  "
          f"Kev docs/s separate {r['kev_rate_separate'] or 0:.0f}, packed {r['kev_rate_packed']:.0f}")
    for m in ("qwen", "kev"):
        x = r[m]
        hy = "  ".join(f"k={k}: {f(v)}" for k, v in x["hybrid"].items())
        print(f"  {m:4s} t=.5 R P F1 {f(x['half'])} | held-out (t~{x['median_t']:.0e}) {f(x['held'])}"
              f" | bestF1 {x['best_f1']:.3f} R@100 {x['recall@100']:.3f}"
              f" | ECE {x['ece']:.3f} Brier {x['brier']:.4f}"
              f" P>=.5: mean p {x['mean_p_above_half'] or 0:.2f} actual {x['precision_above_half'] or 0:.2f}")
        print(f"       hybrid t=.5 {hy}")
    print(f"  Kev-Qwen  F1 t=.5 {ci(r['d_f1_half'])}  F1 held {ci(r['d_f1_held'])}"
          f"  R held {ci(r['d_recall_held'])}  R@100 {ci(r['d_recall@100'])}")
    for shape, v in r.get("shapes", {}).items():
        print(f"    {shape:3s} qwen half {f(v['qwen']['half'])} held {f(v['qwen']['held'])}"
              f" | kev half {f(v['kev']['half'])} held {f(v['kev']['held'])}")
    if "equal_time" in r:
        e = r["equal_time"]
        print(f"  equal time: agent {e['agent_s']:.1f}s R {e['agent_recall']:.3f}")
        for name in ("qwen", "kev_separate", "kev_packed"):
            x = e[name]
            print(f"    {name:13s} k={x['k']:5d} R {x['recall']:.3f}  vs agent {ci(x['vs_agent'])}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/kev_compare.json")
    args = p.parse_args(argv)
    out = []
    for s in SETTINGS:
        if not Path(f"{s[3]}_packed.scores.jsonl").exists():
            print("skip, no Kev scores:", s)
            continue
        out.append(setting(*s))
        show(out[-1])
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
