"""Approach B: batched (query, document) classification.

Design note, worth stating in the report
----------------------------------------
Rather than depending on one vendor checkpoint, we implement the System One
decision pattern directly on top of an open-weights model: the prompt ends
where the answer begins, we generate exactly ONE token, and we read the
probability mass on "yes" versus "no" from the LM head. No autoregressive
generation, and one forward pass per (query, document) pair. The result is a
probability rather than a parsed string, but it is not calibrated: 97% of scores
fall below 0.001, and the best cutoff runs from 0.5 to 1e-8 by query shape.

This is simpler than the open Jev-like models, which train an adapter or a small
block on a frozen base. Here nothing is trained. It buys us three things a
purpose-built checkpoint would not:

  1. The same served model backs the agent arm, so the comparison isolates
     execution architecture rather than confounding it with model quality.
  2. Model size becomes a free experimental variable, which matters because
     the thesis is about batch throughput.
  3. No gated downloads or custom serving stacks in a five-day window.

A purpose-built decision model (Kev, JEV-9B) needs its own environment, so it
runs through src/kev_scan.py and writes the same score files.
"""

from __future__ import annotations

import os
import time

from ..metrics import ArmResult, stopwatch

YES = ("yes", "Yes", "YES", " yes", " Yes", "▁yes", "Ġyes", "ĠYes")
NO = ("no", "No", "NO", " no", " No", "▁no", "Ġno", "ĠNo")

PROMPT = (
    "You are a document filter. Decide whether the document matches the "
    "search request.\n\n"
    "Search request: {query}\n\n"
    "Document:\n{doc}\n\n"
    "Does this document match the search request? Answer with exactly one "
    "word: yes or no."
)


def peak_gpu_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return 0.0


class DecisionModel:
    """One forward pass per pair, probability of 'yes' from the LM head.

    max_num_seqs is the knob the throughput sweep turns: it caps how many
    sequences vLLM runs concurrently, which is the effective batch size.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_num_seqs: int = 256,
        max_num_batched_tokens: int | None = None,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "auto",
        seed: int = 0,
        prefix_caching: bool = True,
        cuda_graph: str = "full",
    ):
        from vllm import LLM, SamplingParams

        self.model_name = model
        self.max_num_seqs = max_num_seqs
        t0 = time.perf_counter()
        self.llm = LLM(
            model=model,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            enable_prefix_caching=prefix_caching,
            # vLLM 0.6.3 decodes in CUDA graphs up to 8,192 tokens and eagerly
            # past that, and past that its output came out as garbage. "full"
            # covers the whole context, "eager" uses no graphs; both fix it.
            enforce_eager=cuda_graph == "eager",
            **({"max_seq_len_to_capture": max_model_len}
               if cuda_graph == "full" else {}),
            # vLLM refuses prefix caching if a model declares a sliding window,
            # as Phi-3.5 does. Our context is far shorter than any such window.
            disable_sliding_window=True,
            disable_log_stats=True,
        )
        self.load_s = time.perf_counter() - t0
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(
            max_tokens=1, temperature=0.0, logprobs=20,
            allowed_token_ids=self._label_ids() or None,
        )

    def _label_ids(self) -> list[int]:
        ids = set()
        for w in YES + NO:
            for enc in (w, w.strip()):
                try:
                    t = self.tok.encode(enc, add_special_tokens=False)
                except Exception:
                    continue
                if len(t) == 1:
                    ids.add(t[0])
        return sorted(ids)

    def _render(self, query: str, doc: str) -> str:
        body = PROMPT.format(query=query, doc=doc)
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return body + "\nAnswer:"

    @staticmethod
    def _prob_yes(logprobs) -> float:
        """P(yes) / (P(yes) + P(no)) from the first generated token."""
        import math
        if not logprobs:
            return 0.0
        y = n = 0.0
        for lp in logprobs.values():
            tok = (getattr(lp, "decoded_token", None) or "")
            s = tok.strip().lower()
            p = math.exp(lp.logprob)
            if s == "yes":
                y += p
            elif s == "no":
                n += p
        tot = y + n
        return (y / tot) if tot > 0 else 0.0

    def score(self, query: str, docs: list) -> tuple[dict[str, float], dict]:
        """Score every document against one query. Returns (probs, stats)."""
        prompts = [self._render(query, d.text) for d in docs]
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        elapsed = time.perf_counter() - t0

        probs, prompt_tokens = {}, 0
        for doc, out in zip(docs, outs):
            prompt_tokens += len(out.prompt_token_ids)
            lps = out.outputs[0].logprobs
            probs[doc.id] = self._prob_yes(lps[0] if lps else None)
        return probs, {
            "generate_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "docs_scored": len(docs),
            "docs_per_sec": len(docs) / elapsed if elapsed else 0.0,
            "prefill_tokens_per_sec": prompt_tokens / elapsed if elapsed else 0.0,
        }


def run(model: DecisionModel, query, docs, threshold: float = 0.5) -> ArmResult:
    timing: dict[str, float] = {}
    with stopwatch(timing, "total"):
        probs, stats = model.score(query.text, docs)
    timing["generate"] = stats.pop("generate_s")
    return ArmResult(
        arm="classify",
        query_id=query.id,
        predicted={d for d, p in probs.items() if p >= threshold},
        scores=probs,
        timing=timing,
        counts=stats,
        extra={
            "model": model.model_name,
            "max_num_seqs": model.max_num_seqs,
            "threshold": threshold,
            "peak_gpu_mb": peak_gpu_mb(),
        },
    )


# --------------------------------------------------------------------------
# Offline stand-in: lets the runner, metrics and plots be exercised with no GPU
# --------------------------------------------------------------------------

class MockDecisionModel:
    """Lexical overlap in place of a model. Smoke tests only, never reported."""

    def __init__(self, max_num_seqs: int = 256, per_doc_s: float = 0.0, **_):
        self.model_name = "mock-lexical"
        self.max_num_seqs = max_num_seqs
        self.per_doc_s = per_doc_s
        self.load_s = 0.0

    def score(self, query: str, docs: list) -> tuple[dict[str, float], dict]:
        qt = {w for w in query.lower().split() if len(w) > 3}
        t0 = time.perf_counter()
        probs, toks = {}, 0
        for d in docs:
            dt = {w for w in d.text.lower().split() if len(w) > 3}
            probs[d.id] = len(qt & dt) / len(qt) if qt else 0.0
            toks += len(d.text) // 4
        if self.per_doc_s:
            time.sleep(self.per_doc_s * len(docs) / max(1, self.max_num_seqs))
        elapsed = max(time.perf_counter() - t0, 1e-9)
        return probs, {
            "generate_s": elapsed,
            "prompt_tokens": toks,
            "docs_scored": len(docs),
            "docs_per_sec": len(docs) / elapsed,
            "prefill_tokens_per_sec": toks / elapsed,
        }


def build(model: str, **kw):
    if model == "mock" or os.environ.get("CS839_MOCK"):
        return MockDecisionModel(**{k: v for k, v in kw.items()
                                    if k in ("max_num_seqs", "per_doc_s")})
    return DecisionModel(model=model, **kw)
