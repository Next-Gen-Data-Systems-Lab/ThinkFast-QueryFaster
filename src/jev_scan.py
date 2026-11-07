"""Scan with JEV-9B, an open Jev-like decision model, instead of zero-shot Qwen.

JEV-9B (autotrust/JEV-9B, Apache-2.0) is Qwen3.5-9B with a rank-16 LoRA and a
24-slot linear decision head. A yes/no question is a `noul` decision: one prefill
of a plain-text template, then the head on the last token's final hidden state,
a per-kind temperature, and a softmax over (false, true). No decoding.

Unlike Kev, the question is part of the same prompt as the document, so there is
no packed mode: each (query, document) pair is one sequence. This uses the
model card's transformers + peft path, which it reports as fast as vLLM for
decisions, rather than the development vLLM build its server needs. Sequences
are right-padded and read at their last real token, so a document's score
cannot depend on what it was batched with; `check` confirms this on each run.

Weights are pinned to one Hugging Face revision. The architecture is the stock
transformers Qwen3.5 model, so no code from the model repository runs.

Rows and scores use the same format as src.runner and src.kev_scan.

Usage: python -m src.jev_scan --corpus-sizes 1000 --tag jev9b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import data as D
from . import metrics as M
from .kev_scan import write

REPO = "autotrust/JEV-9B"
REVISION = "7ac669b8fb476ba926491e3d1c39f9330d68dbed"
QUESTION = "Does this document match this search request: {query}?"
WARMUP_DOCS = 256


def prompt(doc: D.Doc, query: D.Query) -> str:
    return (f"[kind] noul\n[state] {doc.text}\n"
            f"[question] {QUESTION.format(query=query.text)}\n"
            "[options]\nfalse\ntrue\n[decision]:")


def length_batches(lengths: list[int], max_batch: int, batch_tokens: int) -> list[list[int]]:
    """Indices grouped by length, each group within a padded-token budget."""
    batches, current = [], []
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        # Sorted by length, so the newest sequence sets the padded width.
        if current and (len(current) == max_batch
                        or (len(current) + 1) * lengths[i] > batch_tokens):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


class Jev:
    def __init__(self, repo: str = REPO, revision: str = REVISION,
                 batch_tokens: int = 65536, max_batch: int = 128):
        import torch
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=torch.bfloat16, device_map="cuda")
        self.model = PeftModel.from_pretrained(
            base, repo, subfolder="adapter", revision=revision).merge_and_unload().eval()
        head = load_file(hf_hub_download(repo, "head.safetensors", revision=revision))
        cfg = json.load(open(hf_hub_download(repo, "judge_config.json", revision=revision)))
        temps = json.load(open(hf_hub_download(repo, "calibration.json", revision=revision)))
        lo, hi = cfg["slots"]["ranges"]["noul"]
        assert cfg["slots"]["verbalizers"][lo:hi] == ["false", "true"]
        self.W = head["proj.weight"][lo:hi].float().cuda()
        self.b = head["proj.bias"][lo:hi].float().cuda()
        self.T = float(temps["per_kind"]["noul"])
        self.batch_tokens, self.max_batch = batch_tokens, max_batch

    def _forward(self, ids: list[list[int]]) -> list[float]:
        torch = self.torch
        width = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device="cuda")
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device="cuda")
        last = mask.sum(dim=1) - 1
        with torch.inference_mode():
            h = self.model.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            h = h[torch.arange(len(ids), device="cuda"), last].float()
            p = torch.softmax((h @ self.W.T + self.b) / self.T, dim=-1)[:, 1]
        return p.tolist()

    def score(self, texts: list[str]) -> list[float]:
        """P(true) for each prompt, batched by length under a token budget."""
        ids = self.tok(texts, add_special_tokens=False)["input_ids"]
        out = [0.0] * len(ids)
        for batch in length_batches([len(x) for x in ids], self.max_batch, self.batch_tokens):
            for j, p in zip(batch, self._forward([ids[j] for j in batch])):
                out[j] = p
        return out


def kernels() -> str:
    """Which optional fast kernels import: fla (gated delta rule) and causal_conv1d."""
    out = []
    for name in ("fla", "causal_conv1d"):
        try:
            __import__(name)
            out.append(f"{name}=yes")
        except Exception:
            out.append(f"{name}=no")
    return " ".join(out)


def check(jev: Jev, docs: list[D.Doc], query: D.Query, n: int = 16) -> float:
    """Largest gap between batched and one-at-a-time scores."""
    texts = [prompt(d, query) for d in docs[:n]]
    batched = jev.score(texts)
    single = [jev._forward([jev.tok(t, add_special_tokens=False)["input_ids"]])[0] for t in texts]
    return max(abs(a - b) for a, b in zip(batched, single))


def ask(jev: Jev, docs: list[D.Doc], query: D.Query) -> tuple[dict, float]:
    jev.torch.cuda.synchronize()
    t0 = time.perf_counter()
    p = jev.score([prompt(d, query) for d in docs])
    jev.torch.cuda.synchronize()
    return {d.id: s for d, s in zip(docs, p)}, time.perf_counter() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="20newsgroups",
                   help="comma-separated; 'synergy' means all 26 reviews")
    p.add_argument("--corpus-sizes", default="1000")
    p.add_argument("--tag", default="jev9b")
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-tokens", type=int, default=65536)
    p.add_argument("--max-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)

    jev = Jev(batch_tokens=args.batch_tokens, max_batch=args.max_batch)
    torch = jev.torch
    gpu = torch.cuda.get_device_name(0)
    fast = kernels()
    print(f"[jev] {REPO}@{REVISION[:8]} on {gpu}, kernels {fast}", flush=True)

    results = Path(args.results)
    results.mkdir(exist_ok=True)

    for sub in D.subsets(args.dataset, args.corpus_sizes, args.max_chars, args.seed):
        queries = sub.queries[: args.max_queries] if args.max_queries else sub.queries
        base = {"dataset": sub.name, "corpus_size": len(sub.docs), "batch_size": args.max_batch,
                "k": None, "model": f"{REPO}@{REVISION[:8]}", "gpu": gpu,
                "jev_fast_kernels": fast, "fingerprint": D.fingerprint(sub)}
        ask(jev, sub.docs[:WARMUP_DOCS], queries[0])
        gap = check(jev, sub.docs, queries[0])
        print(f"[jev] warm, n={len(sub.docs)} queries={len(queries)} "
              f"batched vs single max |dp|={gap:.2e}", flush=True)

        timed = []
        for q in queries:
            scores, seconds = ask(jev, sub.docs, q)
            timed.append((q, scores, seconds))
        done = write(results / f"{sub.name}_{args.tag}_separate", {**base, "batch_gap": gap},
                     "separate", timed, sub, args.threshold)
        total = sum(t[2] for t in timed)
        summary = M.summarize(done, {q.id: q.gold for q in queries})
        print(f"[jev] n={len(sub.docs)} {len(sub.docs) * len(queries) / total:8.1f} decisions/s "
              f"({total:.1f}s) R={summary['mean_recall']:.3f} P={summary['mean_precision']:.3f} "
              f"F1={summary['mean_f1']:.3f} bestF1={summary.get('mean_best_f1', 0):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
