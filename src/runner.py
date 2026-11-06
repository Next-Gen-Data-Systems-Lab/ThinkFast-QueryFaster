"""Experiment driver.

Writes one JSONL row per (arm, config, query) and skips rows already present,
so a run killed by a CHTC preemption resumes where it stopped and plots are
regenerated from disk without touching the GPU.

Loop order matters: the vLLM engine is expensive to construct and max_num_seqs
is fixed at construction, so batch size is the OUTER loop and the engine is
built once per batch size.

Examples
--------
# offline smoke test, no GPU, no download
python -m src.runner --dataset synthetic --arms classify,ann,hybrid \\
    --model mock --embed-backend tfidf --corpus-sizes 500,1000 \\
    --batch-sizes 32,128 --out results/smoke.jsonl

# the midterm run
python -m src.runner --dataset 20newsgroups --arms classify,ann,hybrid \\
    --model Qwen/Qwen2.5-7B-Instruct --corpus-sizes 1000,5000,10000 \\
    --batch-sizes 32,64,128,256 --out results/midterm.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

from . import data as D
from . import metrics as M
from .arms import ann as ANN
from .arms import classify as CLS

# Every request is prefill only (max_tokens=1), so vLLM's per-step token budget
# caps the real batch, not max_num_seqs. Its 2048-token default fits about 7 of
# our ~270-token prompts, which would make the batch-size sweep measure nothing.
# 512 per slot lets max_num_seqs be the limit. Sizing this from max_model_len
# instead would ask vLLM to profile a 524K-token step at batch size 256.
TOKENS_PER_SEQ = 512


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def env_stamp(args) -> dict:
    """Recorded on every row so results are reproducible and comparable."""
    stamp = {
        "host": socket.gethostname(),
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "model": args.model,
        "embed_backend": args.embed_backend,
        "embed_model": args.embed_model,
    }
    for mod in ("torch", "vllm", "sentence_transformers"):
        try:
            stamp[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            pass
    return stamp


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r.get("arm"), r.get("dataset"), r.get("corpus_size"),
                      r.get("batch_size"), r.get("k"), r.get("query_id")))
    return keys


class Writer:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a")

    def write(self, row: dict):
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="20newsgroups",
                   choices=sorted(D.LOADERS))
    p.add_argument("--arms", default="classify,ann,hybrid")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="decision model, or 'mock' for an offline smoke test")
    p.add_argument("--corpus-sizes", default="1000,5000,10000")
    p.add_argument("--batch-sizes", default="256",
                   help="vLLM max_num_seqs values; the throughput sweep")
    p.add_argument("--k", type=int, default=100,
                   help="candidates for the ann and hybrid arms")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-queries", type=int, default=0,
                   help="0 = all queries in the dataset")
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=0,
                   help=f"vLLM per-step token budget; 0 = batch size x {TOKENS_PER_SEQ}")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--embed-backend", default="sentence-transformers",
                   choices=["sentence-transformers", "tfidf"])
    p.add_argument("--embed-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="results/run.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    sizes = [int(x) for x in args.corpus_sizes.split(",") if x.strip()]
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    out = Path(args.out)
    seen = done_keys(out)
    stamp = env_stamp(args)

    print(f"[runner] dataset={args.dataset} arms={arms}", flush=True)
    print(f"[runner] corpus_sizes={sizes} batch_sizes={batches} k={args.k}", flush=True)
    print(f"[runner] gpu={stamp['gpu']} resume={len(seen)} rows already done", flush=True)

    kw = {} if args.dataset == "synthetic" else {"max_chars": args.max_chars}
    full = D.load(args.dataset, **kw)
    print(f"[runner] loaded {len(full.docs)} docs, {len(full.queries)} queries "
          f"[{D.fingerprint(full)}]", flush=True)

    writer = Writer(out)
    needs_model = {"classify", "hybrid"} & set(arms)
    needs_index = {"ann", "hybrid"} & set(arms)

    try:
        for bs in batches:
            model = None
            budget = args.max_num_batched_tokens or max(
                bs * TOKENS_PER_SEQ, args.max_model_len)
            if needs_model:
                print(f"[runner] building decision model (max_num_seqs={bs}, "
                      f"max_num_batched_tokens={budget})", flush=True)
                t0 = time.perf_counter()
                model = CLS.build(
                    args.model, max_num_seqs=bs,
                    max_num_batched_tokens=budget,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    seed=args.seed,
                )
                print(f"[runner]   ready in {time.perf_counter()-t0:.1f}s", flush=True)

            for n in sizes:
                sub = full.subsample(n, seed=args.seed)
                qs = sub.queries[: args.max_queries] if args.max_queries else sub.queries
                docs_by_id = {d.id: d for d in sub.docs}
                golds = {q.id: q.gold for q in qs}
                base = {
                    **stamp,
                    "dataset": sub.name,
                    "corpus_size": len(sub.docs),
                    "batch_size": bs,
                    "max_num_batched_tokens": budget,
                    "fingerprint": D.fingerprint(sub),
                }

                index = None
                if needs_index:
                    index = ANN.Index(backend=args.embed_backend,
                                      model=args.embed_model)
                    bs_s = index.build(sub.docs)
                    print(f"[runner] n={len(sub.docs)} index built in {bs_s:.1f}s",
                          flush=True)

                for arm in arms:
                    results = []
                    t_arm = time.perf_counter()
                    for q in qs:
                        # classify has no k; the key must match what is written
                        # to the row or resume never skips it.
                        kk = None if arm == "classify" else args.k
                        key = (arm, sub.name, len(sub.docs), bs, kk, q.id)
                        if key in seen:
                            continue
                        if arm == "classify":
                            r = CLS.run(model, q, sub.docs, args.threshold)
                        elif arm == "ann":
                            r = ANN.run_ann(index, q, args.k)
                        elif arm == "hybrid":
                            r = ANN.run_hybrid(index, model, q, docs_by_id,
                                               args.k, args.threshold)
                        else:
                            raise ValueError(f"unknown arm {arm!r}")
                        row = {**base, "k": kk, **M.flatten(r, q.gold)}
                        writer.write(row)
                        seen.add(key)
                        results.append(r)

                    if results:
                        s = M.summarize(results, golds)
                        print(f"[runner] {arm:9s} n={len(sub.docs):6d} bs={bs:4d} "
                              f"R={s.get('mean_recall', 0):.3f} "
                              f"P={s.get('mean_precision', 0):.3f} "
                              f"F1={s.get('mean_f1', 0):.3f} "
                              f"bestF1={s.get('mean_best_f1', float('nan')):.3f} "
                              f"{s.get('docs_per_sec', 0):8.1f} doc/s "
                              f"({time.perf_counter()-t_arm:.1f}s)", flush=True)

            del model
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
    finally:
        writer.close()

    print(f"[runner] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
