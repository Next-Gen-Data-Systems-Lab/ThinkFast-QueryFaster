"""Print every number the deck and report use, from the files in results/.

Usage: python -m src.summarize_runs results

Sections with no results yet are skipped, so it runs at any stage.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCAN_THRESHOLDS = (0.5, 0.1, 0.01, 0.001, 1e-4)
AGENT_BUDGETS = ((8, "agent"), (32, "agent_s32"), (64, "agent_s64"))


def read_rows(path: str | Path) -> pd.DataFrame:
    with open(path) as fh:
        return pd.DataFrame(json.loads(line) for line in fh if line.strip())


def seconds_per_call(agent: pd.DataFrame) -> float:
    return agent.t_llm.sum() / agent.n_llm_calls.sum()


def print_sweep(results: str) -> None:
    files = [f for f in sorted(glob.glob(f"{results}/20newsgroups_sweep_bs*.jsonl"))
             if "scores" not in f]
    if not files:
        return
    sweep = pd.concat(read_rows(f) for f in files)
    scan = sweep[sweep.arm == "classify"]
    print("gpus", sweep.gpu.unique(), "batch sizes", sorted(sweep.batch_size.unique()))
    print("\n== Scan documents per second, by batch size and N")
    print(scan.pivot_table(index="batch_size", columns="corpus_size",
                           values="n_docs_per_sec").round(1))
    print("tokens per second", scan.groupby("batch_size").n_prefill_tokens_per_sec.mean().round(0).to_dict())
    print("peak GPU MB", scan.groupby("batch_size").peak_gpu_mb.mean().round(0).to_dict())
    print("\n== Arms at batch size 32")
    columns = ["recall", "precision", "f1", "best_f1", "recall@100", "recall@1000",
               "wall_s", "n_docs_scored", "ann_recall_ceiling"]
    print(sweep[sweep.batch_size == 32].groupby(["corpus_size", "arm"])[columns].mean().round(3))


def print_agent_budgets(results: str) -> None:
    print("\n== Agent by step limit")
    for limit, name in AGENT_BUDGETS:
        path = Path(results) / f"20newsgroups_{name}.jsonl"
        if not path.exists():
            continue
        agent = read_rows(path)
        table = agent.groupby("corpus_size").agg(
            recall=("recall", "mean"), precision=("precision", "mean"),
            s_per_query=("wall_s", "mean"), steps=("n_steps", "mean"),
            calls=("n_llm_calls", "mean"), docs=("n_docs_scored", "mean"),
            found=("tp", "mean"), gold=("n_gold", "mean"),
            chose_to_stop=("stopped_early", "mean"))
        print(f"limit {limit}\n{table.round(3).to_string()}")
        print(f"seconds per call {seconds_per_call(agent):.3f}, "
              f"generated {agent.n_gen_tokens.sum() / agent.n_llm_calls.sum():.1f}, "
              f"prompt {agent.n_prompt_tokens.sum() / agent.n_llm_calls.sum():.0f} tokens per call")


def print_thresholds(path: Path, label: str) -> None:
    if not path.exists():
        return
    with path.open() as fh:
        rows = [r for r in map(json.loads, fh) if r["arm"] == "classify"]
    for n in sorted({r["corpus_size"] for r in rows}):
        group = [r for r in rows if r["corpus_size"] == n]
        points = []
        for t in SCAN_THRESHOLDS:
            recall, precision, f1 = [], [], []
            for r in group:
                gold = set(r["gold"])
                kept = {d for d, s in r["scores"].items() if s >= t}
                tp = len(kept & gold)
                p, c = (tp / len(kept) if kept else 0.0), tp / len(gold)
                recall.append(c)
                precision.append(p)
                f1.append(2 * p * c / (p + c) if p + c else 0.0)
            points.append((t, round(float(np.mean(recall)), 3), round(float(np.mean(precision)), 3),
                           round(float(np.mean(f1)), 3)))
        scores = np.concatenate([np.fromiter(r["scores"].values(), float) for r in group])
        print(f"{label} N={n} (t, R, P, F1): {points}")
        print(f"   share of scores below 0.001: {(scores < 1e-3).mean():.3f}, "
              f"above 0.5: {(scores > 0.5).mean():.3f}")


def print_file(path: Path, title: str, columns: list[str], by: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    rows = read_rows(path)
    print(f"\n== {title}")
    print(rows.groupby(by)[[c for c in columns if c in rows]].mean().round(3))
    return rows


def print_batched_agent(results: str) -> None:
    for dataset in ("20newsgroups", "nfcorpus"):
        path = Path(results) / f"{dataset}_agentbatch.jsonl"
        if not path.exists():
            continue
        rows = read_rows(path)
        table = rows.groupby(["corpus_size", "agent_concurrency"]).agg(
            wall=("batch_wall_s", "first"), calls=("n_llm_calls", "sum"),
            in_flight=("batch_mean_in_flight", "first"), recall=("recall", "mean"),
            precision=("precision", "mean"), latency=("wall_s", "mean"),
            steps=("n_steps", "mean"))
        table["s_per_call"] = table.wall / table.calls
        print(f"\n== Batched agent, {dataset}")
        print(table.round(3).to_string())


def print_doc_lengths(results: str) -> None:
    files = sorted(glob.glob(f"{results}/20newsgroups_len*.jsonl"))
    files = [f for f in files if "scores" not in f]
    if not files:
        return
    table = []
    for f in files:
        rows = read_rows(f)
        table.append({"max_chars": int(Path(f).stem.split("len")[1]),
                      "docs_per_s": rows.n_docs_per_sec.mean(),
                      "tokens_per_s": rows.n_prefill_tokens_per_sec.mean(),
                      "recall": rows.recall.mean(), "precision": rows.precision.mean()})
    print("\n== Document length, N = 1,000")
    print(pd.DataFrame(table).sort_values("max_chars").round(3).to_string(index=False))


def main(results: str) -> None:
    pd.set_option("display.width", 190, "display.max_columns", 40, "display.max_rows", 80)
    root = Path(results)
    print_sweep(results)
    print_agent_budgets(results)
    print("\n== Scan thresholds from saved scores")
    print_thresholds(root / "20newsgroups_sweep_bs32.scores.jsonl", "7B")
    print_thresholds(root / "20newsgroups_small.scores.jsonl", "1.5B")
    print_thresholds(root / "nfcorpus_nf.scores.jsonl", "NFCorpus")
    quality = ["recall", "precision", "f1", "best_f1", "recall@10", "recall@100",
               "ndcg@10", "wall_s", "n_docs_scored"]
    print_file(root / "nfcorpus_nf.jsonl", "NFCorpus", quality, ["arm"])
    small = print_file(root / "20newsgroups_small.jsonl", "Qwen 1.5B", quality,
                       ["corpus_size", "arm"])
    if small is not None:
        print(f"seconds per call {seconds_per_call(small[small.arm == 'agent']):.3f}")
    print_batched_agent(results)
    print_doc_lengths(results)
    print_file(root / "20newsgroups_logic.jsonl", "Keyword and multi-condition queries",
               quality, ["corpus_size", "arm"])
    phi = print_file(root / "20newsgroups_phi.jsonl", "Phi-3.5-mini", quality,
                     ["corpus_size", "arm"])
    if phi is not None:
        print(f"seconds per call {seconds_per_call(phi[phi.arm == 'agent']):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
