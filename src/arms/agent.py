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
  * By default queries run one at a time, so each one has the engine to itself.

Batching across queries
-----------------------
A reasoning step is decode-bound, so one query leaves most of the GPU idle.
`run_many` keeps several queries in flight and sends their next steps to the
engine together. That is what a server with many users would do, and it is the
fair comparison for the exchange rate: GPU seconds per step, not latency.
Rounds are synchronous, so a round lasts as long as its longest reply. That
makes the measured gain a lower bound on continuous batching.

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
from collections import deque
from dataclasses import dataclass

from ..metrics import ArmResult, stopwatch

SYSTEM_PROMPT = """You are a search agent. Your job is to find ALL documents in a corpus that match a search request. You cannot read the corpus directly. You can only search it, and each search shows documents you have not seen yet.

Each turn you see the latest search results. Reply with one JSON object and nothing else:
{"thought": "<one short sentence>", "keep": [<numbers of the results that match the request>], "next_search": "<your next search query, or null to stop>"}

Stop only when you believe you have found every matching document."""

# With a second tool, the reply also names the kind of search to run.
SYSTEM_PROMPT_TOOLS = """You are a search agent. Your job is to find ALL documents in a corpus that match a search request. You cannot read the corpus directly. You can only search it, and each search shows documents you have not seen yet.

There are two kinds of search:
- "semantic" finds documents whose meaning is close to your query.
- "keyword" finds documents that contain your query's words, ranked by BM25. Use it for exact terms, names, drug or gene names, and phrases.

Each turn you see the latest search results. Reply with one JSON object and nothing else:
{"thought": "<one short sentence>", "keep": [<numbers of the results that match the request>], "next_search": "<your next search query, or null to stop>", "search_type": "semantic" or "keyword"}

Stop only when you believe you have found every matching document."""

TOOLS = ("semantic", "keyword")
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
        return self.chat_many([messages], max_tokens, temperature)[0]

    def chat_many(self, conversations: list[list[dict]], max_tokens: int,
                  temperature: float = 0.0) -> list[ChatReply]:
        """One engine call for all conversations. `seconds` is the shared wait."""
        from vllm import SamplingParams

        prompts = [self.tok.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in conversations]
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - t0
        return [ChatReply(o.outputs[0].text, len(o.prompt_token_ids),
                          len(o.outputs[0].token_ids), seconds) for o in outs]


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
        if '"keyword"' in messages[0]["content"]:
            reply["search_type"] = TOOLS[done % 2]
        return ChatReply(json.dumps(reply), len(text) // 4, 20, 0.0)

    def chat_many(self, conversations: list[list[dict]], max_tokens: int,
                  temperature: float = 0.0) -> list[ChatReply]:
        return [self.chat(m, max_tokens, temperature) for m in conversations]


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
                 snippet_chars: int = 300, keyword=None):
        self.index = index
        self.keyword = keyword
        self.docs_by_id = docs_by_id
        self.k = k
        self.snippet_chars = snippet_chars
        self.calls = 0
        self.docs_returned = 0
        self.seconds = 0.0

    def search(self, query: str, k: int | None = None,
               exclude: frozenset | set = frozenset(),
               mode: str = "semantic") -> list[dict]:
        """Top k documents for the query, skipping any id in `exclude`.
        mode "keyword" uses the BM25 index when the tool has one."""
        k = k or self.k
        t0 = time.perf_counter()
        index = self.keyword if mode == "keyword" and self.keyword is not None else self.index
        hits = index.search(query, k + len(exclude))
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
                 temperature: float = 0.0, tools: tuple = ("semantic",)):
        unknown = set(tools) - set(TOOLS)
        if unknown or not tools:
            raise ValueError(f"tools must be a subset of {TOOLS}, not {tools}")
        self.client = client
        self.model_name = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.tools = tuple(tools)

    def find(self, query_text: str, tool: RetrievalTool) -> tuple[set[str], dict]:
        """Run one query to completion. See `_steps` for what comes back."""
        steps = self._steps(query_text, tool)
        try:
            messages = next(steps)
            while True:
                messages = steps.send(self.client.chat(
                    messages, MAX_REPLY_TOKENS, self.temperature))
        except StopIteration as done:
            return done.value

    def _steps(self, query_text: str, tool: RetrievalTool, session: str = ""):
        """Generator: yields messages, receives a ChatReply, returns the answer.

        Keeping the loop free of model calls lets `find` and `run_many` share it.
        `session` marks a replica of a query, so its prompts differ from the
        original's and prefix caching cannot make the repeat look cheap.

        Returns (matching document ids, stats).

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
        prompt = SYSTEM_PROMPT_TOOLS if len(self.tools) > 1 else SYSTEM_PROMPT
        system = prompt + (f"\n\nSession {session}." if session else "")
        kept: set[str] = set()
        seen: set[str] = set()
        searches: list[str] = []
        results: list[dict] = []
        stats = {"steps": 0, "stopped_early": False, "gen_tokens": 0,
                 "prompt_tokens": 0, "llm_seconds": 0.0, "llm_calls": 0,
                 "parse_failures": 0, "keyword_searches": 0}

        while True:
            reply = yield [
                {"role": "system", "content": system},
                {"role": "user", "content": render_turn(
                    query_text, len(kept), searches, results)}]
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

            mode = parsed.get("search_type")
            if mode not in self.tools:
                mode = self.tools[0]
            results = tool.search(next_search, exclude=seen, mode=mode)
            seen.update(r["id"] for r in results)
            searches.append(f"[{mode}] {next_search}" if len(self.tools) > 1 else next_search)
            stats["keyword_searches"] += mode == "keyword"
            stats["steps"] += 1
            if not results:
                stats["stopped_early"] = True
                break

        return kept, stats


def package(agent: SearchAgent, tool: RetrievalTool, query, matches: set[str],
            stats: dict, seconds: float) -> ArmResult:
    timing = {"total": seconds, "tool": tool.seconds,
              "llm": stats.get("llm_seconds", 0.0)}
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
            "keyword_searches": float(stats.get("keyword_searches", 0)),
        },
        extra={
            "model": agent.model_name,
            "max_steps": agent.max_steps,
            "tools": ",".join(agent.tools),
            "stopped_early": bool(stats.get("stopped_early", False)),
        },
    )


def run(agent: SearchAgent, tool: RetrievalTool, query) -> ArmResult:
    """Run the agent on one query and package the result."""
    tool.reset()
    t0 = time.perf_counter()
    matches, stats = agent.find(query.text, tool)
    return package(agent, tool, query, matches, stats, time.perf_counter() - t0)


def run_many(agent: SearchAgent, new_tool, queries: list,
             concurrency: int) -> tuple[list[ArmResult], dict]:
    """Run queries with up to `concurrency` in flight on one engine.

    `new_tool` makes a RetrievalTool, because each query keeps its own counters.
    A result's wall time is that query's latency, which overlaps with others, so
    cost comes from the returned batch stats: wall_s / llm_calls is GPU seconds
    per step, and that is the number the exchange rate needs.
    """
    waiting = deque(queries)
    active: list[dict] = []
    results: list[ArmResult] = []
    rounds = in_flight = 0
    t0 = time.perf_counter()

    def admit():
        while waiting and len(active) < concurrency:
            query, tool = waiting.popleft(), new_tool()
            steps = agent._steps(query.text, tool, query.id.partition("#")[2])
            active.append({"query": query, "tool": tool, "steps": steps,
                           "messages": next(steps),
                           "start": time.perf_counter()})

    admit()
    while active:
        replies = agent.client.chat_many(
            [a["messages"] for a in active], MAX_REPLY_TOKENS, agent.temperature)
        rounds += 1
        in_flight += len(active)
        still_running = []
        for a, reply in zip(active, replies):
            try:
                a["messages"] = a["steps"].send(reply)
                still_running.append(a)
            except StopIteration as done:
                matches, stats = done.value
                results.append(package(agent, a["tool"], a["query"], matches, stats,
                                       time.perf_counter() - a["start"]))
        active[:] = still_running
        admit()

    return results, {"batch_wall_s": time.perf_counter() - t0,
                     "batch_rounds": rounds,
                     "batch_mean_in_flight": in_flight / rounds if rounds else 0.0}
