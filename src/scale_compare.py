"""Scan against agent as the corpus grows: arXiv at 10K, 100K and 1M documents.

A full LLM scan of 1M documents takes about 100 minutes per query, so the job
(TAG scale, ARMS=hybrid,agent, K=20000) classifies only each query's top 20K
by embedding. The scan in embedding order is replayed from those scores: given
exactly the agent's seconds on a query, it reads k = seconds x rate documents,
where rate is the hybrid rows' documents per second. Replay is exact while
k <= 20000; a query whose budget runs past that is capped and counted.

The ranking comes from src.embed_ranks on this machine, with the same embedding
model the job used. Its fingerprint must match the job's rows, and the share of
the local top k that the job scored is reported, since CPU and GPU embeddings
can order near-ties differently.

Zero-shot Qwen is badly calibrated on these queries, so the scan is reported
both at t = 0.5 and at a threshold tuned on the 10K corpus with the query left
out (the same leave-one-out rule as elsewhere), applied at every size.

Usage: python -m src.scale_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import access_paths as AP

ROWS, SCORES = "results/arxiv_scale.jsonl", "results/arxiv_scale.scores.jsonl"
SIZES = (10_000, 100_000, 1_000_000)


def frame(path: str) -> pd.DataFrame:
    with open(path) as fh:
        return pd.DataFrame(json.loads(line) for line in fh if line.strip())


def held_out_thresholds(queries: list[AP.Query]) -> dict:
    """For each query, the threshold with the best mean F1 over the other queries."""
    f1 = {q.id: np.array([AP.quality(q, q.n, t)[2] for t in AP.THRESHOLDS]) for q in queries}
    return {i: float(AP.THRESHOLDS[int(np.argmax(np.mean([v for j, v in f1.items() if j != i],
                                                          axis=0)))]) for i in f1}


def analyse(rows: pd.DataFrame, n: int, thresholds: dict, dataset: str = "arxiv",
            scores: str = SCORES) -> dict | None:
    """thresholds: setting name -> {query id: threshold}."""
    h = rows[(rows.arm == "hybrid") & (rows.corpus_size == n)].drop_duplicates("query_id", keep="last")
    a = rows[(rows.arm == "agent") & (rows.corpus_size == n)].drop_duplicates("query_id", keep="last")
    if h.empty:
        return None
    z = np.load(AP.path_for(Path("results"), dataset, n))
    fp_rows, fp_local = set(h.fingerprint), str(z["fingerprint"])
    assert fp_rows == {fp_local}, f"N={n}: rows {fp_rows} but local ranking {fp_local}"
    k_max = int(h.k.iloc[0])
    rate = float(h.n_docs_scored.sum() / h.t_classify.sum())
    qs = {q.id: q for q in AP.load(dataset, n, scores, arm="hybrid")}
    covered = float(np.mean([(q.scores[:k_max] >= 0).mean() for q in qs.values()]))
    out = {"dataset": dataset, "n": n, "queries": len(qs), "k": k_max, "docs_per_s": rate,
           "full_scan_s": n / rate, "top_k_s": k_max / rate, "local_order_covered": covered,
           "gold_median": float(np.median([q.gold.sum() for q in qs.values()])),
           "top_k_ceiling": float(h.ann_recall_ceiling.mean()), "top_k": {}}
    for name, ts in thresholds.items():
        top = np.array([AP.quality(q, k_max, ts[i]) for i, q in qs.items()])
        out["top_k"][name] = dict(zip(("recall", "precision", "f1"), top.mean(axis=0).tolist()))
    if a.empty:
        return out
    a = a.set_index("query_id")
    ids = [i for i in a.index if i in qs]
    secs = a.loc[ids, "wall_s"].values
    want = (secs * rate).astype(int)
    ks = [min(qs[i].n, k_max, w) for i, w in zip(ids, want)]
    rec = a.loc[ids, "recall"].values
    gold = np.array([qs[i].gold.sum() for i in ids])
    out["agent"] = {
        "recall": float(rec.mean()), "precision": float(a.loc[ids, "precision"].mean()),
        "f1": float(a.loc[ids, "f1"].mean()),
        "found": float((rec * gold).mean()), "seconds": float(secs.mean()),
        "steps": float(a.loc[ids, "n_steps"].mean()),
        "docs_seen": float(a.loc[ids, "n_docs_scored"].mean()),
        "s_per_call": float(a.loc[ids, "t_llm"].sum() / a.loc[ids, "n_llm_calls"].sum())}
    out["equal_time"] = {"k_mean": float(np.mean(ks)), "capped": int((want > k_max).sum())}
    for name, ts in thresholds.items():
        m = np.array([AP.quality(qs[i], k, ts[i]) for i, k in zip(ids, ks)])
        d, lo, hi = AP.paired(m[:, 0], rec)
        out["equal_time"][name] = {
            "recall": float(m[:, 0].mean()), "precision": float(m[:, 1].mean()),
            "f1": float(m[:, 2].mean()), "found": float((m[:, 0] * gold).mean()),
            "vs_agent": {"d": d, "lo": lo, "hi": hi, "wins": int((m[:, 0] > rec).sum()),
                         "n": len(ids)}}
    out["exchange_rate"] = out["agent"]["s_per_call"] * rate
    return out


def show(r: dict) -> None:
    print(f"\n{r['dataset']} N={r['n']:,}: {r['queries']} queries, median gold {r['gold_median']:.0f}, "
          f"{r['docs_per_s']:.0f} docs/s (full scan {r['full_scan_s'] / 60:.0f} min/query)")
    print(f"  top {r['k']:,} by embedding in {r['top_k_s']:.0f}s: ceiling {r['top_k_ceiling']:.3f}, "
          f"local order covered {r['local_order_covered']:.3f}")
    for name, x in r["top_k"].items():
        print(f"    {name:9s} R/P/F1 " + " ".join(f"{v:.3f}" for v in x.values()))
    if "agent" not in r:
        print("  no agent rows yet")
        return
    a, e = r["agent"], r["equal_time"]
    print(f"  agent: R {a['recall']:.3f} P {a['precision']:.3f} F1 {a['f1']:.3f}, found {a['found']:.0f}, "
          f"{a['seconds']:.1f}s, {a['steps']:.1f} steps, {a['docs_seen']:.0f} docs seen, "
          f"{a['s_per_call']:.3f} s/call, exchange rate {r['exchange_rate']:.0f}")
    print(f"  scan at equal time: k {e['k_mean']:.0f} ({e['capped']} capped)")
    for name, x in e.items():
        if not isinstance(x, dict):
            continue
        v = x["vs_agent"]
        print(f"    {name:9s} R {x['recall']:.3f} P {x['precision']:.3f} F1 {x['f1']:.3f}, "
              f"found {x['found']:.0f}; vs agent {v['d']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}], "
              f"better on {v['wins']}/{v['n']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/scale_compare.json")
    args = p.parse_args(argv)
    rows = frame(ROWS)
    # Thresholds are tuned once, at the smallest size where the whole corpus
    # is scored, leaving each query out, and then used at every size.
    dev = AP.load("arxiv", SIZES[0], SCORES, arm="hybrid")
    thresholds = {"t=0.5": {q.id: 0.5 for q in dev}, "held-out": held_out_thresholds(dev)}
    print("held-out thresholds:", sorted(set(thresholds["held-out"].values())))
    out = []
    for n in SIZES:
        if not AP.path_for(Path("results"), "arxiv", n).exists():
            print(f"skip N={n}: no local ranking")
            continue
        r = analyse(rows, n, thresholds)
        if r:
            show(r)
            out.append(r)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
