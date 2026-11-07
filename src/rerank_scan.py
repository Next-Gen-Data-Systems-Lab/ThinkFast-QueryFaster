"""Scan with a cross-encoder reranker: the cheap neural baseline for the LLM scans.

A cross-encoder reads the query and the document together, as the scan does,
but it is a small encoder trained for relevance ranking rather than an LLM. If
it separates the answer set as well as an LLM scan at a fraction of the cost,
the LLM is not needed. Scores are the sigmoid of the relevance logit.

  bge     BAAI/bge-reranker-v2-m3 (568M, Apache-2.0), max length 1024
  minilm  cross-encoder/ms-marco-MiniLM-L6-v2 (22M, Apache-2.0), max length 512

Weights are pinned; the architectures are stock transformers classes, so no
code from the model repositories runs. Long pairs are truncated from the
longer side. Rows and scores use the runner's format, so src.baselines and
src.access_paths read them unchanged.

Usage: python -m src.rerank_scan --model bge --dataset 20newsgroups --corpus-sizes 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import data as D
from . import metrics as M
from .jev_scan import length_batches
from .kev_scan import write

MODELS = {
    "bge": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", 1024),
    "minilm": ("cross-encoder/ms-marco-MiniLM-L6-v2",
               "233902d25c440f23af6f7d6e94d2946bac0bee0a", 512),
}
WARMUP_DOCS = 256


class Reranker:
    def __init__(self, name: str, batch_tokens: int = 131072, max_batch: int = 256,
                 device: str = "cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.repo, revision, self.max_length = MODELS[name]
        self.tok = AutoTokenizer.from_pretrained(self.repo, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.repo, revision=revision,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        self.device = device
        self.revision = revision
        self.batch_tokens, self.max_batch = batch_tokens, max_batch

    def score(self, query: str, texts: list[str]) -> list[float]:
        torch = self.torch
        enc = self.tok([query] * len(texts), texts, truncation="longest_first",
                       max_length=self.max_length)
        ids, types = enc["input_ids"], enc.get("token_type_ids")
        out = [0.0] * len(texts)
        for batch in length_batches([len(x) for x in ids], self.max_batch, self.batch_tokens):
            feats = self.tok.pad({"input_ids": [ids[i] for i in batch],
                                  **({"token_type_ids": [types[i] for i in batch]}
                                     if types is not None else {})},
                                 return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**{k: v.to(self.device) for k, v in feats.items()}).logits
            p = torch.sigmoid(logits[:, 0].float()).tolist()
            for i, s in zip(batch, p):
                out[i] = s
        return out


def ask(model: Reranker, docs: list[D.Doc], query: D.Query) -> tuple[dict, float]:
    sync = model.torch.cuda.synchronize if model.device == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    p = model.score(query.text, [d.text for d in docs])
    sync()
    return {d.id: s for d, s in zip(docs, p)}, time.perf_counter() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bge", choices=sorted(MODELS))
    p.add_argument("--dataset", default="20newsgroups",
                   help="comma-separated; 'synergy' means all 26 reviews")
    p.add_argument("--corpus-sizes", default="1000")
    p.add_argument("--tag", default=None, help="default: the model name")
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    p.add_argument("--device", default="cuda", help="cpu for a local smoke test")
    p.add_argument("--warmup-docs", type=int, default=WARMUP_DOCS)
    args = p.parse_args(argv)
    tag = args.tag or args.model

    model = Reranker(args.model, device=args.device)
    gpu = model.torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"[rerank] {model.repo}@{model.revision[:8]} on {gpu}", flush=True)
    results = Path(args.results)
    results.mkdir(exist_ok=True)

    for sub in D.subsets(args.dataset, args.corpus_sizes, args.max_chars, args.seed):
        queries = sub.queries[: args.max_queries] if args.max_queries else sub.queries
        base = {"dataset": sub.name, "corpus_size": len(sub.docs),
                "batch_size": model.max_batch, "k": None,
                "model": f"{model.repo}@{model.revision[:8]}", "gpu": gpu,
                "fingerprint": D.fingerprint(sub)}
        ask(model, sub.docs[:args.warmup_docs], queries[0])
        timed = []
        for q in queries:
            scores, seconds = ask(model, sub.docs, q)
            timed.append((q, scores, seconds))
        done = write(results / f"{sub.name}_{tag}_separate", base, "separate", timed, sub,
                     args.threshold)
        total = sum(t[2] for t in timed)
        summary = M.summarize(done, {q.id: q.gold for q in queries})
        print(f"[rerank] {sub.name} n={len(sub.docs)} "
              f"{len(sub.docs) * len(queries) / total:8.1f} decisions/s ({total:.1f}s) "
              f"R={summary['mean_recall']:.3f} P={summary['mean_precision']:.3f} "
              f"F1={summary['mean_f1']:.3f} bestF1={summary.get('mean_best_f1', 0):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
