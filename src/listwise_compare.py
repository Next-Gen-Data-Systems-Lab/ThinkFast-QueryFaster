"""List-wise scan against the pointwise scan and the agents.

The list-wise scan (src/arms/listwise.py) shows the model 20 documents per
prompt and has it name the matches; the pointwise scan asks one yes/no question
per document. Both read documents in embedding order, so both replay the same
way: after T seconds a scan has read T x rate documents.

For 20 Newsgroups (N = 10,000) and SYNERGY (26 reviews):
  whole corpus   recall, precision and F1 of each scan, pointwise at t = 0.5,
                 and documents per second
  equal time     each scan given exactly an agent's seconds on each query,
                 against the 7B agent and the persistent 7B agent

Usage: python -m src.listwise_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import access_paths as AP
from . import synergy_compare as SC
from .crossover import RATE

N = 10_000
AGENTS_20NG = {"7B": "results/20newsgroups_agent_s64.jsonl",
               "7B, persistent": "results/20newsgroups_agentdry.jsonl"}
AGENTS_SYNERGY = {"7B": "results/synergy_qwen.jsonl",
                  "7B, persistent": "results/synergy_agentdry.jsonl"}


def frame(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    with open(path) as fh:
        return pd.DataFrame(json.loads(line) for line in fh if line.strip())


def diff(a, b) -> dict:
    d, lo, hi = AP.paired(np.asarray(a, float), np.asarray(b, float))
    return {"d": d, "lo": lo, "hi": hi, "wins": int((np.asarray(a) > np.asarray(b)).sum()),
            "n": len(a)}


def compare(scans: dict, agents: dict) -> dict:
    """scans: name -> (queries, docs per second per query). agents: name -> rows
    indexed by query id. Queries are matched by id."""
    out = {"whole": {}, "equal_time": {}}
    for name, (qs, rates) in scans.items():
        m = np.array([AP.quality(q, q.n, 0.5) for q in qs])
        out["whole"][name] = {"recall": float(m[:, 0].mean()), "precision": float(m[:, 1].mean()),
                              "f1": float(m[:, 2].mean()), "docs_per_s": float(np.mean(rates)),
                              "_r": m[:, 0], "_f1": m[:, 2]}
    names = list(scans)
    if len(names) == 2:
        a, b = names
        out["whole"]["listwise_minus_pointwise"] = {
            "recall": diff(out["whole"][a]["_r"], out["whole"][b]["_r"]),
            "f1": diff(out["whole"][a]["_f1"], out["whole"][b]["_f1"])}
    for agent, rows in agents.items():
        e = {}
        for name, (qs, rates) in scans.items():
            keep = [(q, r) for q, r in zip(qs, rates) if q.id in rows.index]
            secs = np.array([rows.loc[q.id, "wall_s"] for q, _ in keep])
            rec = np.array([rows.loc[q.id, "recall"] for q, _ in keep])
            ks = [min(q.n, int(s * r)) for (q, r), s in zip(keep, secs)]
            plan = AP.rows_for([q for q, _ in keep], ks, 0.5)
            e[name] = {"recall": float(plan["recall"].mean()), "agent_recall": float(rec.mean()),
                       "agent_seconds": float(secs.mean()), "vs_agent": diff(plan["recall"], rec)}
        out["equal_time"][agent] = e
    for v in out["whole"].values():
        v.pop("_r", None), v.pop("_f1", None)
    return out


def newsgroups() -> dict | None:
    rows = frame("results/20newsgroups_listwise.jsonl")
    if rows is None:
        return None
    rows = rows[(rows.arm == "listwise") & (rows.corpus_size == N)]
    lw = AP.load("20newsgroups", N, "results/20newsgroups_listwise.scores.jsonl", arm="listwise")
    lw_rate = float(N * len(rows) / rows.t_total.sum())
    pw = AP.load("20newsgroups", N, "results/20newsgroups_sweep_bs32.scores.jsonl")
    ids = {q.id for q in lw}
    pw = [q for q in pw if q.id in ids]
    lw = sorted(lw, key=lambda q: q.id)
    pw = sorted(pw, key=lambda q: q.id)
    agents = {}
    for name, path in AGENTS_20NG.items():
        a = frame(path)
        if a is not None:
            a = a[(a.arm == "agent") & (a.corpus_size == N)].drop_duplicates("query_id", keep="last")
            agents[name] = a.set_index("query_id")
    return compare({"listwise": (lw, [lw_rate] * len(lw)), "pointwise": (pw, [RATE] * len(pw))},
                   agents)


def synergy() -> dict | None:
    if not Path("results/synergy_listwise.jsonl").exists():
        return None
    c = SC.collect()
    lw, lw_rates, pw, pw_rates, ids = [], [], [], [], []
    for q, r, name in zip(c["queries"]["qwen"], c["rate"]["qwen"], c["per"]["review"]):
        d = "synergy_" + name
        rows = SC.rows("results/synergy_listwise.jsonl", "listwise", d)
        if rows is None:
            continue
        lq = AP.load(d, q.n, "results/synergy_listwise.scores.jsonl", arm="listwise")
        if len(lq) != 1:
            continue
        lw.append(lq[0])
        lw_rates.append(q.n / float(rows.t_total.iloc[0]))
        pw.append(q)
        pw_rates.append(r)
        ids.append(d)
    agents = {}
    for name, path in AGENTS_SYNERGY.items():
        got = [SC.rows(path, "agent", d) for d in ids]
        if all(g is not None for g in got):
            # One query per review, so key the rows by the scan's query id.
            agents[name] = pd.DataFrame(
                [g.iloc[0] for g in got], index=[q.id for q in lw])
    return compare({"listwise": (lw, lw_rates), "pointwise": (pw, pw_rates)}, agents)


def show(title: str, r: dict) -> None:
    def ci(x):
        return f"{x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] better on {x['wins']}/{x['n']}"

    print(f"\n{title}")
    for name, v in r["whole"].items():
        if name == "listwise_minus_pointwise":
            print(f"  listwise - pointwise: recall {ci(v['recall'])}; F1 {ci(v['f1'])}")
        else:
            print(f"  {name:9s} whole corpus R {v['recall']:.3f} P {v['precision']:.3f} "
                  f"F1 {v['f1']:.3f}, {v['docs_per_s']:.0f} docs/s")
    for agent, e in r["equal_time"].items():
        for name, v in e.items():
            print(f"  vs agent {agent:15s} ({v['agent_seconds']:.1f}s, R {v['agent_recall']:.3f}): "
                  f"{name} R {v['recall']:.3f}, {ci(v['vs_agent'])}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/listwise_compare.json")
    args = p.parse_args(argv)
    out = {}
    for title, fn in (("20 Newsgroups, N = 10,000", newsgroups), ("SYNERGY, 26 reviews", synergy)):
        r = fn()
        if r is None:
            print(f"skip {title}: no listwise rows yet")
            continue
        show(title, r)
        out[title] = r
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
