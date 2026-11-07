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
from .arms import agent as AG
from .arms import ann as ANN
from .arms import classify as CLS
from .arms import keyword as KW

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
                      r.get("batch_size"), r.get("k"), r.get("query_id"),
                      r.get("agent_concurrency")))
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
                   help="one or more comma-separated names; 'synergy' means all 26 "
                        "reviews. Corpus sizes above a pool's size use the whole pool.")
    p.add_argument("--arms", default="classify,ann,hybrid")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="decision model, or 'mock' for an offline smoke test")
    p.add_argument("--corpus-sizes", default="1000,5000,10000")
    p.add_argument("--batch-sizes", default="256",
                   help="vLLM max_num_seqs values; the throughput sweep")
    p.add_argument("--k", type=int, default=100,
                   help="candidates for the ann and hybrid arms")
    p.add_argument("--agent-tools", default="semantic",
                   help="semantic, or semantic,keyword to add BM25 search")
    p.add_argument("--agent-stop", default="agent", choices=AG.STOP_RULES,
                   help="agent: stop when the agent says so; dry: only after two "
                        "searches in a row find nothing, with a prompt that says "
                        "answer sets can be large")
    p.add_argument("--agent-k", type=int, default=20,
                   help="documents the agent sees per search")
    p.add_argument("--agent-max-steps", type=int, default=8,
                   help="hard cap on agent search iterations")
    p.add_argument("--agent-concurrency", default="",
                   help="queries in flight, as a sweep such as 1,4,16; empty "
                        "runs them strictly one at a time")
    p.add_argument("--agent-replicas", type=int, default=1,
                   help="run each query this many times, to reach concurrency "
                        "beyond the number of queries")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-queries", type=int, default=0,
                   help="0 = all queries in the dataset")
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--max-model-len", type=int, default=0,
                   help="0 = 2048, 4096 with the agent arm, 8192 past 2000 characters")
    p.add_argument("--max-num-batched-tokens", type=int, default=0,
                   help=f"vLLM per-step token budget; 0 = batch size x {TOKENS_PER_SEQ}")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--embed-backend", default="sentence-transformers",
                   choices=["sentence-transformers", "tfidf"])
    p.add_argument("--embed-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="results/run.jsonl")
    p.add_argument("--scores-out", default="",
                   help="also write per-document scores here, so thresholds can "
                        "be re-evaluated without another GPU run")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not args.max_model_len:
        # Noisy text can run close to one token per character.
        # SYNERGY prompts also carry eligibility criteria of up to 2,754 characters.
        long_query = any(n.startswith("synergy") for n in D.expand(args.dataset))
        args.max_model_len = (8192 if args.max_chars > 2000 or long_query
                              else 4096 if "agent" in arms else 2048)
    sizes = [int(x) for x in args.corpus_sizes.split(",") if x.strip()]
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    concurrency = [int(x) for x in args.agent_concurrency.split(",") if x.strip()]
    if concurrency and max(concurrency) > min(batches):
        p.error("--agent-concurrency cannot exceed the batch size")
    out = Path(args.out)
    seen = done_keys(out)
    stamp = env_stamp(args)

    print(f"[runner] dataset={args.dataset} arms={arms}", flush=True)
    print(f"[runner] corpus_sizes={sizes} batch_sizes={batches} k={args.k}", flush=True)
    print(f"[runner] gpu={stamp['gpu']} resume={len(seen)} rows already done", flush=True)

    names = D.expand(args.dataset)
    unknown = [n for n in names if n not in D.LOADERS]
    if unknown:
        p.error(f"unknown dataset(s) {unknown}")
    fulls = []
    for name in names:
        full = D.load(name, **({} if name == "synthetic" else {"max_chars": args.max_chars}))
        print(f"[runner] loaded {name}: {len(full.docs)} docs, {len(full.queries)} queries "
              f"[{D.fingerprint(full)}]", flush=True)
        fulls.append(full)

    writer = Writer(out)
    score_writer = Writer(Path(args.scores_out)) if args.scores_out else None
    needs_model = {"classify", "hybrid", "agent"} & set(arms)
    needs_index = {"ann", "hybrid", "agent"} & set(arms)

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

            for full, n in [(f, n) for f in fulls for n in sizes]:
                sub = full.subsample(n, seed=args.seed)
                if any(len(full.docs) <= m < n for m in sizes):
                    continue    # a smaller size already used the whole pool
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

                agent = tool = None
                if "agent" in arms:
                    tools = tuple(x.strip() for x in args.agent_tools.split(",") if x.strip())
                    agent = AG.SearchAgent(AG.build_chat(model), args.model,
                                           max_steps=args.agent_max_steps, tools=tools,
                                           stop=args.agent_stop)
                    keyword = None
                    if "keyword" in tools:
                        keyword = KW.KeywordIndex()
                        kw_s = keyword.build(sub.docs)
                        print(f"[runner] n={len(sub.docs)} keyword index built in "
                              f"{kw_s:.1f}s", flush=True)
                    tool = AG.RetrievalTool(index, docs_by_id, k=args.agent_k,
                                            keyword=keyword)

                for arm in arms:
                    if arm == "agent" and concurrency:
                        for c in concurrency:
                            copies = qs if args.agent_replicas == 1 else [
                                D.Query(f"{q.id}#{r}", q.text, q.gold)
                                for r in range(args.agent_replicas) for q in qs]
                            golds.update({q.id: q.gold for q in copies})
                            todo = [q for q in copies if ("agent", sub.name, len(sub.docs),
                                                          bs, None, q.id, c) not in seen]
                            if not todo:
                                continue
                            rs, batch = AG.run_many(
                                agent, lambda: AG.RetrievalTool(
                                    index, docs_by_id, k=args.agent_k,
                                    keyword=keyword), todo, c)
                            calls = sum(r.counts["llm_calls"] for r in rs)
                            for r in rs:
                                row = {**base, "k": None, "agent_concurrency": c,
                                       **batch, **M.flatten(r, golds[r.query_id])}
                                writer.write(row)
                                seen.add(("agent", sub.name, len(sub.docs), bs, None,
                                          r.query_id, c))
                            s = M.summarize(rs, golds)
                            print(f"[runner] agent     n={len(sub.docs):6d} c={c:4d} "
                                  f"R={s.get('mean_recall', 0):.3f} "
                                  f"P={s.get('mean_precision', 0):.3f} "
                                  f"{batch['batch_wall_s'] / calls:.3f} s/step "
                                  f"in flight {batch['batch_mean_in_flight']:.1f} "
                                  f"({batch['batch_wall_s']:.1f}s)", flush=True)
                        continue

                    results = []
                    t_arm = time.perf_counter()
                    for q in qs:
                        # Only ann and hybrid use k; the key must match what is
                        # written to the row or resume never skips it.
                        kk = args.k if arm in ("ann", "hybrid") else None
                        key = (arm, sub.name, len(sub.docs), bs, kk, q.id, None)
                        if key in seen:
                            continue
                        if arm == "classify":
                            r = CLS.run(model, q, sub.docs, args.threshold)
                        elif arm == "ann":
                            r = ANN.run_ann(index, q, args.k)
                        elif arm == "hybrid":
                            r = ANN.run_hybrid(index, model, q, docs_by_id,
                                               args.k, args.threshold)
                        elif arm == "agent":
                            r = AG.run(agent, tool, q)
                        else:
                            raise ValueError(f"unknown arm {arm!r}")
                        row = {**base, "k": kk, **M.flatten(r, q.gold)}
                        # Scores first: if the job dies between the two writes,
                        # resume repeats this query and the reader keeps the
                        # last score row, rather than the row having no scores.
                        if score_writer and r.scores:
                            score_writer.write({
                                "arm": arm, "dataset": sub.name, "query_id": q.id,
                                "corpus_size": len(sub.docs), "batch_size": bs,
                                "k": kk, "gold": sorted(q.gold),
                                "scores": r.scores,
                            })
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
        if score_writer:
            score_writer.close()

    print(f"[runner] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
