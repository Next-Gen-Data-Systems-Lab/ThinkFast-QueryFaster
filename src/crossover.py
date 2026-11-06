"""Performance model: when does the agent beat a scan in embedding order?

Both arms run on one GPU, so cost is GPU seconds per query.

    T_scan(k)  = k / r           r: documents decided per second
    T_agent    = c * tau         c: model calls per query, tau: GPU seconds per call

The exchange rate E = tau * r says how many decisions one call costs. Let k* be
the depth at which the ordered scan first matches the agent's recall. The agent
is cheaper exactly when c * E < k*, so its break-even rate is E* = k* / c. A
measured E above E* means the scan gets the agent's recall for less.

Usage: python -m src.crossover
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import access_paths as AP

SCORES = "results/20newsgroups_sweep_bs32.scores.jsonl"
AGENT = "results/20newsgroups_agent_s64.jsonl"
BATCHED = "results/20newsgroups_agentbatch.jsonl"
BATCHED_NF = "results/nfcorpus_agentbatch.jsonl"
SIZES = (1000, 5000, 10000)
RATE = 170.0     # documents per second at the batch-size plateau, N = 5,000-10,000


def rows(path: str) -> pd.DataFrame:
    with open(path) as fh:
        return pd.DataFrame(json.loads(line) for line in fh if line.strip())


def depth_for(queries: list[AP.Query], recall: float, t: float) -> int:
    """Smallest k whose mean recall reaches `recall`."""
    gained = np.zeros(queries[0].n)
    for q in queries:
        gained += np.cumsum((q.scores >= t) & q.gold) / q.gold.sum()
    gained /= len(queries)
    reached = np.flatnonzero(gained >= recall)
    return int(reached[0] + 1) if len(reached) else queries[0].n


def equal_time(thresholds: dict[int, float]) -> list[dict]:
    """The ordered scan given exactly the agent's measured seconds."""
    agent = rows(AGENT)
    out = []
    for n in SIZES:
        queries = AP.load("20newsgroups", n, SCORES)
        a = agent[agent.corpus_size == n].set_index("query_id").loc[[q.id for q in queries]]
        seconds, calls = a.wall_s.mean(), a.n_llm_calls.mean()
        k = int(seconds * RATE)
        row = {"n": n, "agent_s": seconds, "agent_calls": calls,
               "agent_recall": a.recall.mean(), "agent_precision": a.precision.mean(),
               "agent_f1": a.f1.mean(), "agent_docs": a.n_docs_scored.mean(), "k": k}
        for name, t in (("half", 0.5), ("cal", thresholds[n])):
            plan = AP.rows_for(queries, [k] * len(queries), t)
            d, lo, hi = AP.paired(plan["recall"], a.recall.values)
            depth = depth_for(queries, a.recall.mean(), t)
            row.update({f"{name}_t": t, f"{name}_recall": plan["recall"].mean(),
                        f"{name}_precision": plan["precision"].mean(),
                        f"{name}_f1": plan["f1"].mean(),
                        f"{name}_d": d, f"{name}_lo": lo, f"{name}_hi": hi,
                        f"{name}_wins": int((plan["recall"] > a.recall.values).sum()),
                        f"{name}_depth": depth, f"{name}_breakeven": depth / calls})
        out.append(row)
    return out


def exchange_rates(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    r = rows(path)
    out = []
    for (n, c), g in r.groupby(["corpus_size", "agent_concurrency"]):
        tau = g.batch_wall_s.iloc[0] / g.n_llm_calls.sum()
        out.append({"n": int(n), "concurrency": int(c), "tau": float(tau),
                    "exchange": float(tau * RATE), "recall": float(g.recall.mean()),
                    "in_flight": float(g.batch_mean_in_flight.iloc[0]),
                    "latency_s": float(g.wall_s.mean())})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", default="results/access_paths.json")
    p.add_argument("--out", default="results/crossover.json")
    args = p.parse_args(argv)

    report = json.loads(Path(args.paths).read_text())
    thresholds = {n: report[f"20newsgroups_{n}"]["calibration"]["median_t"] for n in SIZES}
    out = {"rate": RATE, "equal_time": equal_time(thresholds),
           "exchange": exchange_rates(BATCHED), "exchange_nf": exchange_rates(BATCHED_NF)}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    for r in out["equal_time"]:
        print(f"N={r['n']}: agent R {r['agent_recall']:.3f} in {r['agent_s']:.1f}s | scan to k={r['k']}: "
              f"R {r['half_recall']:.3f} ({r['half_d']:+.3f} [{r['half_lo']:+.3f},{r['half_hi']:+.3f}], "
              f"{r['half_wins']}/20) | break-even {r['half_breakeven']:.1f}, calibrated {r['cal_breakeven']:.1f}")
    for key in ("exchange", "exchange_nf"):
        for r in out[key]:
            print(f"{key} N={r['n']} c={r['concurrency']}: {r['tau']:.3f} s/call, "
                  f"E={r['exchange']:.1f}, R {r['recall']:.3f}, in flight {r['in_flight']:.1f}")


if __name__ == "__main__":
    main()
