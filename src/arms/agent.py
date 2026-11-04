"""Approach A: iterative LLM search agent. INTERFACE ONLY -- owner: Nida.

This file fixes the contract so the agent arm can be built in parallel with
everything else and dropped into the runner without renegotiating anything.
Fill in `SearchAgent.find` and delete the NotImplementedError.

Why this arm decides the project
--------------------------------
H1 and H2 both compare against the agent. If the agent side is an analytic
estimate while the classifier side is measured, we have not compared two
systems -- we have compared a measurement to an assumption, and that is the
first thing a reader will push on. This arm must be MEASURED.

What the agent does
-------------------
Loop until the agent says stop or max_steps is reached:
  1. reason about the query and what it has found so far
  2. emit a search call (the retrieval tool below)
  3. inspect the returned snippets, keep the ids it judges relevant
  4. decide whether to search again

The termination decision in step 4 is the whole point. A full scan is
exhaustive by construction; an agent has to guess when it is done. Record
`steps` and `stopped_early` on every result -- those columns are the evidence
for H1.

Serving
-------
Use the SAME vLLM engine that backs the classifier arm, so the comparison
isolates execution architecture rather than model quality. In practice that
means running `vllm serve <model>` once and pointing both arms at it, rather
than constructing two in-process engines (they would contend for GPU memory).
Record engine and model version on every row.
"""

from __future__ import annotations

import time

from ..metrics import ArmResult, stopwatch


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

    def search(self, query: str, k: int | None = None) -> list[dict]:
        t0 = time.perf_counter()
        hits = self.index.search(query, k or self.k)
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
    """Iterative search agent. IMPLEMENT ME.

    Parameters
    ----------
    client     : an OpenAI-compatible client pointed at the local vLLM server
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

        stats MUST include:
            steps          int    search iterations actually taken
            stopped_early  bool   True if the agent chose to stop, False if
                                  max_steps cut it off
            gen_tokens     int    tokens generated across all steps
            prompt_tokens  int    tokens prefilled across all steps
            llm_seconds    float  time inside model calls (excludes tool time)
        """
        raise NotImplementedError(
            "Agent arm not implemented. Owner: Nida. See module docstring."
        )


def run(agent: SearchAgent, tool: RetrievalTool, query) -> ArmResult:
    """Driver. Works unchanged once SearchAgent.find is implemented."""
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
        },
        extra={
            "model": agent.model_name,
            "max_steps": agent.max_steps,
            "stopped_early": bool(stats.get("stopped_early", False)),
        },
    )
