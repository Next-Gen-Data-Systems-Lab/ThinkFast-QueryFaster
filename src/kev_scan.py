"""Scan with Kev, an open Jev-like decision model, instead of zero-shot Qwen.

Kev reads the document first and the question after it, so one pass over a
document can answer every query. Two modes measure what that is worth:

  separate   one question per request, one query at a time, like the Qwen scan
  packed     one request per document, carrying every query

Runs in-process, with no HTTP, and reads raw probabilities: the API rounds to
four decimals, which would erase the low-threshold analysis. Rows and scores use
the same format as src.runner, so src.access_paths works on them unchanged.

Needs Kev's own environment (Python 3.12+), so it runs through chtc/run_kev_scan.sh.

Usage: python -m src.kev_scan --run jaredpalmer/kev-9b@v1.0 --corpus-sizes 1000 --tag kev9b
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from . import data as D
from . import metrics as M

WARMUP_DOCS = 256
BATCH = 64     # kev.serve.MAX_BATCH, the most requests its model thread takes at once
QUESTION = "Does this document match this search request: {query}?"


def build(run: str):
    """Kev's own serving setup, as in kev.serve.main, without the web server."""
    import torch
    from kev.checkpoint import LoadOptions, fused_available
    from kev.serve import Server, _serving_device

    opts = LoadOptions.from_env()
    if opts.backend is None:
        opts = replace(opts, backend="auto")
    device, checkpoint = _serving_device("auto", opts, run)
    if device == "mps" and opts.attn is None:
        opts = replace(opts, attn="sdpa")
    if device != "cpu" and opts.dtype is None:
        opts = replace(opts, dtype=torch.bfloat16)
    if device == "cuda" and torch.version.hip is None:
        if opts.cuda_graphs is None:
            opts = replace(opts, cuda_graphs=True)
        if opts.fused is None:
            opts = replace(opts, fused=fused_available())
    tok, model = checkpoint.load(device, opts)
    print(f"[kev] {run} on {device} via {model.backend} ({model.dtype}), "
          f"graphs={opts.cuda_graphs} fused={opts.fused}", flush=True)
    return Server(checkpoint, tok, model, device), opts


def record(doc: D.Doc, queries: list[D.Query]) -> dict:
    from kev.api import Noul, SystemOneRequest, to_record

    questions = {q.id: Noul(type="noul", instructions=QUESTION.format(query=q.text))
                 for q in queries}
    request = SystemOneRequest(state=doc.text, model="kev-latest", questions=questions)
    return to_record(request)[0]


def ask(server, docs: list[D.Doc], queries: list[D.Query]) -> tuple[dict, float]:
    """P(yes) per query per document, and the seconds it took."""
    t0 = time.perf_counter()
    pending = [(d, server.submit(record(d, queries))) for d in docs]
    scores: dict[str, dict[str, float]] = {q.id: {} for q in queries}
    for doc, future in pending:
        probabilities, _ = future.result()
        for q, p in zip(queries, probabilities):
            scores[q.id][doc.id] = float(p[1])
    return scores, time.perf_counter() - t0


def write(stem: Path, base: dict, mode: str, timed: list, sub: D.Corpus, threshold: float) -> list:
    results = []
    with open(f"{stem}.jsonl", "a") as rows, open(f"{stem}.scores.jsonl", "a") as saved:
        for q, by_doc, seconds in timed:
            result = M.ArmResult(
                arm="classify", query_id=q.id,
                predicted={d for d, s in by_doc.items() if s >= threshold},
                scores=by_doc, timing={"total": seconds},
                counts={"docs_scored": float(len(by_doc)),
                        "docs_per_sec": len(by_doc) / seconds})
            rows.write(json.dumps({**base, "kev_mode": mode,
                                   **M.flatten(result, q.gold)}) + "\n")
            saved.write(json.dumps({"arm": "classify", "dataset": sub.name, "query_id": q.id,
                                    "corpus_size": len(sub.docs), "batch_size": BATCH,
                                    "k": None, "gold": sorted(q.gold),
                                    "scores": by_doc}) + "\n")
            results.append(result)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="jaredpalmer/kev-9b@v1.0")
    p.add_argument("--dataset", default="20newsgroups",
                   help="comma-separated; 'synergy' means all 26 reviews")
    p.add_argument("--corpus-sizes", default="1000")
    p.add_argument("--modes", default="separate,packed")
    p.add_argument("--tag", default="kev")
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)

    import torch

    server, opts = build(args.run)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    results = Path(args.results)
    results.mkdir(exist_ok=True)

    for sub in D.subsets(args.dataset, args.corpus_sizes, args.max_chars, args.seed):
        queries = sub.queries[: args.max_queries] if args.max_queries else sub.queries
        base = {"dataset": sub.name, "corpus_size": len(sub.docs), "batch_size": BATCH,
                "k": None, "model": args.run, "gpu": gpu, "kev_graphs": opts.cuda_graphs,
                "kev_fused": opts.fused, "fingerprint": D.fingerprint(sub)}
        ask(server, sub.docs[:WARMUP_DOCS], queries[:1])
        ask(server, sub.docs[:WARMUP_DOCS], queries)
        server.wait_idle()
        print(f"[kev] warm, n={len(sub.docs)} queries={len(queries)}", flush=True)

        for mode in args.modes.split(","):
            if mode == "separate":
                timed = []
                for q in queries:
                    scores, seconds = ask(server, sub.docs, [q])
                    timed.append((q, scores[q.id], seconds))
            else:
                scores, seconds = ask(server, sub.docs, queries)
                timed = [(q, scores[q.id], seconds / len(queries)) for q in queries]

            done = write(results / f"{sub.name}_{args.tag}_{mode}", base, mode, timed,
                         sub, args.threshold)
            total = sum(t[2] for t in timed)
            summary = M.summarize(done, {q.id: q.gold for q in queries})
            print(f"[kev] {mode:8s} n={len(sub.docs)} "
                  f"{len(sub.docs) * len(queries) / total:8.1f} decisions/s ({total:.1f}s) "
                  f"R={summary['mean_recall']:.3f} P={summary['mean_precision']:.3f} "
                  f"F1={summary['mean_f1']:.3f} bestF1={summary.get('mean_best_f1', 0):.3f}",
                  flush=True)
    server.close()


if __name__ == "__main__":
    main()
