"""Scan against agent on SYNERGY, one systematic review at a time.

Each review is its own corpus with one find-all query, so the 26 reviews are
the unit of analysis: means are over reviews, and differences are paired over
reviews with a bootstrap 95% interval.

For each scan model (Qwen zero-shot, Kev, JEV):
  scan        the whole pool at t = 0.5, and at a threshold chosen on the other
              reviews (leave one review out)
  equal time  the scan in embedding order, given exactly the agent's seconds on
              that review, at that model's measured documents per second there
The agent is Qwen2.5-7B with embedding search (src.runner, TAG qwen). Agents with
other tools or models (TAG agent7b2t, agent32b1t, agent32b2t) are added when
their rows exist.

Usage: python -m src.synergy_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import access_paths as AP
from . import data as D

SCANS = {
    "qwen": ("results/synergy_qwen.jsonl", "results/synergy_qwen.scores.jsonl"),
    "kev": ("results/{d}_kev9b_packed.jsonl", "results/{d}_kev9b_packed.scores.jsonl"),
    "jev": ("results/{d}_jev9b_separate.jsonl", "results/{d}_jev9b_separate.scores.jsonl"),
}
AGENTS = {
    "7B, embeddings": "results/synergy_qwen.jsonl",
    "7B, embeddings + BM25": "results/synergy_agent7b2t.jsonl",
    "32B, embeddings": "results/synergy_agent32b1t.jsonl",
    "32B, embeddings + BM25": "results/synergy_agent32b2t.jsonl",
    "7B, persistent": "results/synergy_agentdry.jsonl",
    "7B, full text": "results/synergy_agentfull.jsonl",
    "7B, chooses k": "results/synergy_agentk.jsonl",
    "7B, chooses k, persistent": "results/synergy_agentkdry.jsonl",
    "7B, chooses k, persistent, full text": "results/synergy_agentkdryfull.jsonl",
}


def rows(path: str, arm: str, dataset: str) -> pd.DataFrame | None:
    p = Path(path.format(d=dataset))
    if not p.exists():
        return None
    with open(p) as fh:
        r = pd.DataFrame(json.loads(line) for line in fh if line.strip())
    r = r[(r.arm == arm) & (r.dataset == dataset)]
    return r.drop_duplicates("query_id", keep="last") if len(r) else None


def held_out(queries: list[AP.Query]) -> np.ndarray:
    table = np.array([[AP.quality(q, q.n, t) for t in AP.THRESHOLDS] for q in queries])
    out = []
    for i in range(len(queries)):
        best = int(np.argmax(np.delete(table[:, :, 2], i, axis=0).mean(axis=0)))
        out.append(table[i, best])
    return np.array(out)


def collect() -> dict:
    per: dict = {"review": [], "n": [], "gold": []}
    queries = {m: [] for m in SCANS}
    rate = {m: [] for m in SCANS}
    agents = {a: [] for a in AGENTS}
    for name in D.expand("synergy"):
        n = len(D.load(name).docs)
        found = {}
        for m, (row_path, score_path) in SCANS.items():
            r = rows(row_path, "classify", name)
            if r is None:
                break
            q = AP.load(name, n, score_path.format(d=name))
            if len(q) != 1:
                break
            found[m] = (q[0], n / float(r.t_total.iloc[0]))
        if len(found) < len(SCANS):
            print("skip, scans missing:", name)
            continue
        per["review"].append(name.removeprefix("synergy_"))
        per["n"].append(n)
        per["gold"].append(int(found["qwen"][0].gold.sum()))
        for m, (q, docs_per_s) in found.items():
            queries[m].append(q)
            rate[m].append(docs_per_s)
        for a, path in AGENTS.items():
            r = rows(path, "agent", name)
            agents[a].append(None if r is None else r.iloc[0])
    return {"per": per, "queries": queries, "rate": rate, "agents": agents}


def diff(a, b) -> dict:
    d, lo, hi = AP.paired(np.asarray(a, float), np.asarray(b, float))
    return {"d": d, "lo": lo, "hi": hi, "wins": int((np.asarray(a) > np.asarray(b)).sum()),
            "n": len(a)}


def analyse(c: dict) -> dict:
    out = {"reviews": len(c["per"]["review"]), "per_review": c["per"], "scans": {}, "agents": {}}
    for m, qs in c["queries"].items():
        half = np.array([AP.quality(q, q.n, 0.5) for q in qs])
        held = held_out(qs)
        out["scans"][m] = {
            "docs_per_s": float(np.mean(c["rate"][m])),
            "half": half.mean(axis=0).tolist(), "held": held.mean(axis=0).tolist(),
            "_half": half, "_held": held}
    for a, rs in c["agents"].items():
        if any(r is None for r in rs):
            continue
        rec = np.array([r.recall for r in rs])
        secs = np.array([r.wall_s for r in rs])
        calls = np.array([r.n_llm_calls for r in rs])
        tau = float(sum(r.t_llm for r in rs) / calls.sum())
        entry = {"recall": float(rec.mean()), "precision": float(np.mean([r.precision for r in rs])),
                 "f1": float(np.mean([r.f1 for r in rs])), "seconds": float(secs.mean()),
                 "steps": float(np.mean([r.n_steps for r in rs])),
                 "docs_seen": float(np.mean([r.n_docs_scored for r in rs])),
                 "s_per_call": tau, "equal_time": {}}
        if "n_gold_seen" in rs[0]:
            seen = np.array([r.n_gold_seen for r in rs])
            entry["gold_seen"] = float(seen.mean())
            entry["kept_of_seen"] = float(sum(r.tp for r in rs) / max(seen.sum(), 1))
            entry["mean_k"] = float(np.mean([r.n_mean_k for r in rs]))
        if "n_nudges" in rs[0]:
            entry["nudges"] = float(np.mean([r.n_nudges for r in rs]))
            entry["hit_step_limit"] = int(sum(not bool(r.stopped_early) for r in rs))
        if "n_keyword_searches" in rs[0]:
            entry["keyword_share"] = float(np.sum([r.n_keyword_searches for r in rs])
                                           / max(np.sum([r.n_steps for r in rs]), 1))
        for m, qs in c["queries"].items():
            ks = [min(q.n, int(s * r)) for q, s, r in zip(qs, secs, c["rate"][m])]
            for t_name, t in (("half", 0.5),):
                plan = AP.rows_for(qs, ks, t)
                entry["equal_time"][m] = {
                    "share_of_pool": float(np.mean([k / q.n for k, q in zip(ks, qs)])),
                    "recall": float(plan["recall"].mean()),
                    "vs_agent": diff(plan["recall"], rec),
                    "full_scan_vs_agent": diff(out["scans"][m]["_half"][:, 0], rec)}
            entry["exchange_rate"] = {m: tau * float(np.mean(c["rate"][m])) for m in c["queries"]}
        out["agents"][a] = entry
    q, k, j = (out["scans"][m]["_held"][:, 2] for m in ("qwen", "kev", "jev"))
    out["f1_held"] = {"kev_vs_qwen": diff(k, q), "jev_vs_qwen": diff(j, q)}
    for m in out["scans"]:
        out["scans"][m].pop("_half"), out["scans"][m].pop("_held")
    return out


def show(r: dict) -> None:
    def ci(x):
        return f"{x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] better on {x['wins']}/{x['n']}"

    print(f"SYNERGY, {r['reviews']} reviews")
    for m, s in r["scans"].items():
        print(f"  scan {m:4s} {s['docs_per_s']:6.1f} docs/s | t=0.5 R P F1 "
              + " ".join(f"{v:.3f}" for v in s["half"]) + " | held-out "
              + " ".join(f"{v:.3f}" for v in s["held"]))
    print(f"  F1 held-out: Kev-Qwen {ci(r['f1_held']['kev_vs_qwen'])}; "
          f"JEV-Qwen {ci(r['f1_held']['jev_vs_qwen'])}")
    for a, e in r["agents"].items():
        print(f"\n  agent {a}: R {e['recall']:.3f} P {e['precision']:.3f} F1 {e['f1']:.3f} "
              f"{e['seconds']:.1f}s {e['steps']:.1f} steps, {e['docs_seen']:.0f} docs seen, "
              f"{e['s_per_call']:.3f} s/call"
              + (f", keyword share {e['keyword_share']:.2f}" if "keyword_share" in e else "")
              + (f", k {e['mean_k']:.0f}, kept {e['kept_of_seen']:.0%} of "
                 f"{e['gold_seen']:.1f} relevant seen" if "kept_of_seen" in e else ""))
        for m, x in e["equal_time"].items():
            print(f"    {m:4s} same time: {x['share_of_pool']:.0%} of pool, R {x['recall']:.3f}, "
                  f"vs agent {ci(x['vs_agent'])}; exchange rate {e['exchange_rate'][m]:.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/synergy_compare.json")
    args = p.parse_args(argv)
    r = analyse(collect())
    show(r)
    Path(args.out).write_text(json.dumps(r, indent=1, default=float))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
