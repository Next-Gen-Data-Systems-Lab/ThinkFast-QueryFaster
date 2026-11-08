"""List-wise scan: the model reads documents in groups and names the matches.

The pointwise scan (classify.py) asks one yes/no question per document and
reads P(yes). This arm is what an LLM "without tools" amounts to once the corpus
is larger than its context window: the corpus is cut into groups of `group`
documents, each group is one prompt, and the model replies with the numbers of
the documents that match. It reads the same text per document as the pointwise
scan, but the request once per group instead of once per document, and it
writes its answer instead of reading one token's probability.

Groups follow the embedding ranking, so the arm can be replayed like the
pointwise scan: its scores are 1.0 for documents it named and 0.0 for the rest,
and reading the first k documents in embedding order costs k / rate.
"""

from __future__ import annotations

import time

from ..metrics import ArmResult
from .agent import parse_reply

SYSTEM_PROMPT = """You screen documents for a search request. You see a numbered list of documents. Reply with one JSON object and nothing else:
{"keep": [<numbers of the documents that match the request>]}

Use an empty list if none of them match."""


def render(query_text: str, docs: list) -> str:
    lines = [f"Search request: {query_text}", "", "Documents:"]
    # One line per document, so the numbering stays unambiguous.
    lines += [f"[{i}] {' '.join(d.text.split())}" for i, d in enumerate(docs, 1)]
    return "\n".join(lines)


def run(chat, index, query, docs: list, group: int = 20) -> ArmResult:
    """Screen every document in `docs` for one query, `group` per prompt."""
    by_id = {d.id: d for d in docs}
    t0 = time.perf_counter()
    order = [d for d, _ in index.search(query.text, len(docs)) if d in by_id]
    ranked = set(order)
    order += [d.id for d in docs if d.id not in ranked]
    t_retrieve = time.perf_counter() - t0

    chunks = [order[i:i + group] for i in range(0, len(order), group)]
    conversations = [[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": render(query.text, [by_id[d] for d in c])}]
                     for c in chunks]
    cut = getattr(chat, "truncated", 0)
    t1 = time.perf_counter()
    replies = chat.chat_many(conversations, max_tokens=16 + 5 * group)
    t_generate = time.perf_counter() - t1

    scores = {d: 0.0 for d in order}
    failures = 0
    for chunk, reply in zip(chunks, replies):
        parsed = parse_reply(reply.text)
        keep = parsed.get("keep") if isinstance(parsed, dict) else None
        if not isinstance(keep, list):
            failures += 1
            continue
        for i in keep:
            if isinstance(i, int) and 1 <= i <= len(chunk):
                scores[chunk[i - 1]] = 1.0

    return ArmResult(
        arm="listwise",
        query_id=query.id,
        predicted={d for d, s in scores.items() if s >= 0.5},
        scores=scores,
        timing={"total": t_retrieve + t_generate, "retrieve": t_retrieve,
                "classify": t_generate},
        counts={"docs_scored": float(len(order)), "prompts": float(len(chunks)),
                "prompt_tokens": float(sum(r.prompt_tokens for r in replies)),
                "gen_tokens": float(sum(r.gen_tokens for r in replies)),
                "parse_failures": float(failures),
                # prompts cut to fit the context window, dropping their last documents
                "truncated_prompts": float(getattr(chat, "truncated", 0) - cut)},
        extra={"group": group, "order": "embedding",
               "model": getattr(chat, "model_name", None)},
    )
