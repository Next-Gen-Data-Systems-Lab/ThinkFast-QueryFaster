"""Every agent variant against the scan at equal time, on 20 Newsgroups.

The agent variants are a 2x2 of model (Qwen2.5-7B or 32B) and tools (embedding
search alone, or with BM25 keyword search), all one query at a time with up to
64 steps. For each, the scan in embedding order gets exactly that agent's
seconds on each query, at each scan model's measured documents per second
(Kev packed only when all 20 queries arrive together). Differences in recall
are paired over queries with a bootstrap 95% interval.

SYNERGY has the same comparison in src.synergy_compare.

Usage: python -m src.agents_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import access_paths as AP
from .crossover import RATE

AGENTS = {
    "7B, embeddings": "results/20newsgroups_agent_s64.jsonl",
    "7B, embeddings + BM25": "results/20newsgroups_agent7b2t.jsonl",
    "32B, embeddings": "results/20newsgroups_agent32b1t.jsonl",
    "32B, embeddings + BM25": "results/20newsgroups_agent32b2t.jsonl",
    # Told answer sets can be large, and held to stopping only after two
    # searches in a row find nothing (--agent-stop dry), up to 256 steps.
    "7B, persistent": "results/20newsgroups_agentdry.jsonl",
    # Sees 2,000 characters of each result, as the scan does, instead of 300.
    "7B, full text": "results/20newsgroups_agentfull.jsonl",
    # Chooses how many results each search returns (5 to 100).
    "7B, chooses k": "results/20newsgroups_agentk.jsonl",
    "7B, chooses k, persistent": "results/20newsgroups_agentkdry.jsonl",
}
SCANS = {   # scores file, and how to get documents per second at each N
    "qwen": ("results/20newsgroups_sweep_bs32.scores.jsonl", None),
    "kev": ("results/20newsgroups_{kev}_packed.scores.jsonl", "results/20newsgroups_{kev}_separate.jsonl"),
    "kev packed": ("results/20newsgroups_{kev}_packed.scores.jsonl", "results/20newsgroups_{kev}_packed.jsonl"),
    "jev": ("results/20newsgroups_{jev}_separate.scores.jsonl", "results/20newsgroups_{jev}_separate.jsonl"),
}
SIZES = (1000, 5000, 10000)


def frame(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    with open(path) as fh:
        return pd.DataFrame(json.loads(line) for line in fh if line.strip())


def tags(n: int) -> dict:
    return {"kev": "kev9b" if n == 1000 else "kev9bfull", "jev": "jev9b" if n == 1000 else "jev9bfull"}


def rate(rows_path: str | None, n: int) -> float | None:
    if rows_path is None:
        return RATE
    r = frame(rows_path.format(**tags(n)))
    if r is None:
        return None
    r = r[(r.arm == "classify") & (r.corpus_size == n)].drop_duplicates("query_id", keep="last")
    return float(n * len(r) / r.t_total.sum()) if len(r) else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/agents_compare.json")
    args = p.parse_args(argv)
    out = []
    for name, path in AGENTS.items():
        a = frame(path)
        if a is None:
            print("skip, no rows:", name)
            continue
        a = a[a.arm == "agent"]
        for n in SIZES:
            g = a[a.corpus_size == n].drop_duplicates("query_id", keep="last").set_index("query_id")
            if g.empty:
                continue
            entry = {"agent": name, "n": n, "recall": float(g.recall.mean()),
                     "precision": float(g.precision.mean()), "seconds": float(g.wall_s.mean()),
                     "steps": float(g.n_steps.mean()),
                     "s_per_call": float(g.t_llm.sum() / g.n_llm_calls.sum()), "scans": {}}
            if "n_gold_seen" in g:
                # Of the relevant documents its searches returned, how many it kept.
                entry["gold_seen"] = float(g.n_gold_seen.mean())
                entry["kept_of_seen"] = float(g.tp.sum() / max(g.n_gold_seen.sum(), 1))
                entry["mean_k"] = float(g.n_mean_k.mean())
            if "n_nudges" in g:
                entry["nudges"] = float(g.n_nudges.mean())
                entry["hit_step_limit"] = int((~g.stopped_early.astype(bool)).sum())
            if "n_keyword_searches" in g:
                entry["keyword_share"] = float(g.n_keyword_searches.sum() / max(g.n_steps.sum(), 1))
            line = (f"{name:26s} N={n:6d} agent R {entry['recall']:.3f} "
                    f"{entry['seconds']:5.1f}s {entry['s_per_call']:.2f} s/call")
            if "kept_of_seen" in entry:
                line += (f" k {entry['mean_k']:.0f}, kept {entry['kept_of_seen']:.0%} "
                         f"of {entry['gold_seen']:.0f} relevant seen")
            for scan, (scores, rows_path) in SCANS.items():
                docs_per_s = rate(rows_path, n)
                score_file = scores.format(**tags(n))
                if docs_per_s is None or not Path(score_file).exists():
                    continue
                qs = AP.load("20newsgroups", n, score_file)
                qs = [q for q in qs if q.id in g.index]
                ks = [min(q.n, int(g.loc[q.id, "wall_s"] * docs_per_s)) for q in qs]
                plan = AP.rows_for(qs, ks, 0.5)
                agent_r = g.loc[[q.id for q in qs], "recall"].values
                d, lo, hi = AP.paired(plan["recall"], agent_r)
                entry["scans"][scan] = {"docs_per_s": docs_per_s, "recall": float(plan["recall"].mean()),
                                        "d": d, "lo": lo, "hi": hi,
                                        "wins": int((plan["recall"] > agent_r).sum()),
                                        "exchange_rate": entry["s_per_call"] * docs_per_s}
                line += f" | {scan} {plan['recall'].mean():.3f} ({d:+.3f} [{lo:+.3f}, {hi:+.3f}])"
            print(line)
            out.append(entry)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
