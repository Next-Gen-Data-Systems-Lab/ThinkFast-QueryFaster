"""Approach A: iterative LLM search agent.

Why this arm decides the project
--------------------------------
H1 and H2 both compare against the agent. If the agent side is an analytic
estimate while the classifier side is measured, we have not compared two
systems -- we have compared a measurement to an assumption, and that is the
first thing a reader will push on. This arm is MEASURED.

What the agent does
-------------------
Loop until the agent says stop or max_steps is reached:
  1. reason about the query and what it has found so far
  2. emit a search call (the retrieval tool below)
  3. inspect the returned snippets, keep the ids it judges relevant
  4. decide whether to search again

The termination decision in step 4 is the whole point. A full scan is
exhaustive by construction; an agent has to guess when it is done. `steps` and
`stopped_early` are recorded on every result -- those columns are the evidence
for H1.

Choices that favour the agent
-----------------------------
So that a classifier win cannot be put down to a weak baseline:
  * The tool never shows a document twice, so repeating a search pages deeper.
  * Each step sees the latest results plus a short summary of the search so
    far, not the whole transcript, which keeps prompts small and steps fast.
  * Queries run one at a time, so each one has the engine to itself.

Serving
-------
The agent calls the SAME in-process vLLM engine that backs the classifier arm,
so the comparison isolates execution architecture rather than model quality,
and one job needs no separate server.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from ..metrics import ArmResult, stopwatch

SYSTEM_PROMPT = """You are a search agent. Your job is to find ALL documents in a corpus that match a search request. You cannot read the corpus directly. You can only search it, and each search shows documents you have not seen yet.

Each turn you see the latest search results. Reply with one JSON object and nothing else:
{"thought": "<one short sentence>", "keep": [<numbers of the results that match the request>], "next_search": "<your next search query, or null to stop>"}

Stop only when you believe you have found every matching document."""

MAX_REPLY_TOKENS = 256


@dataclass
class ChatReply:
    text: str
    prompt_tokens: int
    gen_tokens: int
    seconds: float


class LocalChat:
    """Chat on the engine already loaded for the classifier arm."""

    def __init__(self, decision_model):
        self.llm = decision_model.llm
        self.tok = decision_model.tok

    def chat(self, messages: list[dict], max_tokens: int,
             temperature: float = 0.0) -> ChatReply:
        from vllm import SamplingParams

        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        t0 = time.perf_counter()
        out = self.llm.generate([prompt], params, use_tqdm=False)[0]
        return ChatReply(out.outputs[0].text, len(out.prompt_token_ids),
                         len(out.outputs[0].token_ids),
                         time.perf_counter() - t0)


class MockChat:
    """Lexical stand-in for offline smoke tests. Never reported."""

    def __init__(self, searches: int = 3):
        self.searches = searches

    def chat(self, messages: list[dict], max_tokens: int,
             temperature: float = 0.0) -> ChatReply:
        text = messages[-1]["content"]
        request = text.split("\n", 1)[0].removeprefix("Search request: ")
        words = {w for w in request.lower().split() if len(w) > 3}
        keep = [int(n) for n, body in re.findall(r"^\[(\d+)\] (.*)$", text, re.M)
                if words & set(body.lower().split())]
        done = text.count(" | ") + 1 if "Searches so far: (none)" not in text else 0
        reply = {"thought": "mock", "keep": keep,
                 "next_search": request if done < self.searches else None}
        return ChatReply(json.dumps(reply), len(text) // 4, 20, 0.0)


def build_chat(decision_model):
    from .classify import MockDecisionModel

    if isinstance(decision_model, MockDecisionModel):
        return MockChat()
    return LocalChat(decision_model)


def parse_reply(text: str) -> dict | None:
    """The first JSON object in a reply, or None if there isn't a usable one."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def render_turn(query_text: str, kept: int, searches: list[str],
                results: list[dict]) -> str:
    lines = [f"Search request: {query_text}",
             f"Documents kept so far: {kept}",
             "Searches so far: " + (" | ".join(searches) or "(none)"), ""]
    if results:
        lines.append("Latest results:")
        lines += [f"[{i}] {r['snippet']}" for i, r in enumerate(results, 1)]
    else:
        lines.append("No results yet. Write your first search query.")
    return "\n".join(lines)


class RetrievalTool:
    """The search tool the agent calls. Wraps the same index the ANN arm uses.

    Counting matters: every call is a round trip the agent pays serially, so
    `calls` and `docs_returned` go straight into the latency model.
    """

    def __init__(self, index, docs_by_id: dict, k: int = 10,
                 snippet_chars: int = 300):
        self.index = index
        self.docs_by_id = docs_by_id
        self.k = k
        self.snippet_chars = snippet_chars
        self.calls = 0
        self.docs_returned = 0
        self.seconds = 0.0

    def search(self, query: str, k: int | None = None,
               exclude: frozenset | set = frozenset()) -> list[dict]:
        """Top k documents for the query, skipping any id in `exclude`."""
        k = k or self.k
        t0 = time.perf_counter()
        hits = self.index.search(query, k + len(exclude))
        hits = [h for h in hits if h[0] not in exclude][:k]
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.docs_returned += len(hits)
        return [
            {"id": did,
             "score": score,
             "snippet": self.docs_by_id[did].text[: self.snippet_chars]}
            for did, score in hits if did in self.docs_by_id
        ]

    def reset(self):
        self.calls = self.docs_returned = 0
        self.seconds = 0.0


class SearchAgent:
    """Iterative search agent.

    Parameters
    ----------
    client     : has .chat(messages, max_tokens, temperature) -> ChatReply
    model      : model id, the same one the classifier arm uses
    max_steps  : hard cap on search iterations; record whether it was hit
    """

    def __init__(self, client, model: str, max_steps: int = 8,
                 temperature: float = 0.0):
        self.client = client
        self.model_name = model
        self.max_steps = max_steps
        self.temperature = temperature

    def find(self, query_text: str, tool: RetrievalTool) -> tuple[set[str], dict]:
        """Return (matching document ids, stats).

        stats always includes:
            steps           int    search iterations actually taken
            stopped_early   bool   True if the agent chose to stop, False if
                                   max_steps cut it off
            gen_tokens      int    tokens generated across all steps
            prompt_tokens   int    tokens prefilled across all steps
            llm_seconds     float  time inside model calls (excludes tool time)
            llm_calls       int    model calls, one more than steps when the
                                   agent judges its last results
            parse_failures  int    replies that were not usable JSON
        """
        kept: set[str] = set()
        seen: set[str] = set()
        searches: list[str] = []
        results: list[dict] = []
        stats = {"steps": 0, "stopped_early": False, "gen_tokens": 0,
                 "prompt_tokens": 0, "llm_seconds": 0.0, "llm_calls": 0,
                 "parse_failures": 0}

        while True:
            reply = self.client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": render_turn(
                     query_text, len(kept), searches, results)}],
                MAX_REPLY_TOKENS, self.temperature)
            stats["llm_calls"] += 1
            stats["gen_tokens"] += reply.gen_tokens
            stats["prompt_tokens"] += reply.prompt_tokens
            stats["llm_seconds"] += reply.seconds

            parsed = parse_reply(reply.text)
            if parsed is None:
                stats["parse_failures"] += 1
                parsed = {}
            keep = parsed.get("keep")
            for i in keep if isinstance(keep, list) else []:
                if isinstance(i, int) and 1 <= i <= len(results):
                    kept.add(results[i - 1]["id"])

            next_search = parsed.get("next_search")
            if not isinstance(next_search, str) or not next_search.strip():
                # A bad first reply must not end the run with nothing searched.
                if stats["steps"] == 0:
                    next_search = query_text
                else:
                    stats["stopped_early"] = True
                    break
            if stats["steps"] >= self.max_steps:
                break

            results = tool.search(next_search, exclude=seen)
            seen.update(r["id"] for r in results)
            searches.append(next_search)
            stats["steps"] += 1
            if not results:
                stats["stopped_early"] = True
                break

        return kept, stats


def run(agent: SearchAgent, tool: RetrievalTool, query) -> ArmResult:
    """Run the agent on one query and package the result."""
    tool.reset()
    timing: dict[str, float] = {}
    with stopwatch(timing, "total"):
        matches, stats = agent.find(query.text, tool)

    timing["tool"] = tool.seconds
    timing["llm"] = stats.get("llm_seconds", 0.0)
    return ArmResult(
        arm="agent",
        query_id=query.id,
        predicted=set(matches),
        scores={},                       # an agent returns a set, not a ranking
        timing=timing,
        counts={
            "docs_scored": float(tool.docs_returned),
            "tool_calls": float(tool.calls),
            "steps": float(stats.get("steps", 0)),
            "gen_tokens": float(stats.get("gen_tokens", 0)),
            "prompt_tokens": float(stats.get("prompt_tokens", 0)),
            "llm_calls": float(stats.get("llm_calls", 0)),
            "parse_failures": float(stats.get("parse_failures", 0)),
        },
        extra={
            "model": agent.model_name,
            "max_steps": agent.max_steps,
            "stopped_early": bool(stats.get("stopped_early", False)),
        },
    )
