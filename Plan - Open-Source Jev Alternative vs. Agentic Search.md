# CS 839 Project Plan

## Open-source Jev-style classification vs. agentic search

# Project goal

Study when **parallel document classification can outperform an LLM search agent on semantic retrieval**. Before the Nov 10 midway check, we need a clear experiment design and initial feasibility results.

# 1\. Experiment design

## Research question

For a query that asks for all matching documents, when is it faster to classify every document in parallel than to use an LLM agent that searches iteratively? 

What are the tradeoffs in **recall, latency, throughput, and GPU resource use**?

## Approach A: Conventional LLM search agent

An open-weight LLM reasons about the query, calls a keyword or vector search tool, examines results, and may search again. We will choose an LLM and inspect an open-source inference engine to understand prefill, token decoding, batching, and GPU behavior. Estimate tool-call and repeated-step costs. 

A working search agent is implemented in `src/arms/agent.py`, so agent latency is measured rather than estimated. 

An agent also needs a way to check whether it found all matching documents. For this, we would need to look for a dataset with these properties.

## Approach B: Open-source Jev-like classifier

For each document, evaluate the pair (query, document) and return match or no match. 

Independent decisions can be batched on a GPU. Aggregate the matching document IDs. We will test an open-source alternative such as Laya or [Kev](https://github.com/jaredpalmer/kev). 

These models are candidates, not confirmed drop-in replacements for Jev. Feel free to suggest others.

## Dataset and queries

Select a small, publicly available document corpus that is easy to download and check. Possible options include research abstracts, public reports, or course descriptions. 

Use **simple keyword queries, multi-condition queries, and semantic queries**. 

For example, search for documents about GPU scheduling that also discuss memory limits. 

Avoid criteria that cannot be verified from the document text.

## What we will vary and measure

Vary **corpus size, document length, query complexity, and classifier batch size**. For the initial pilot, we aim for about 30 to 50 documents, a few queries, and small batch sizes. 

Measure **classifier latency, documents per second, and GPU memory use**. If profiling tools are available, record **utilization** as well. 

For a small labeled subset, measure **precision and recall**. Keep model quality and speed separate.

Use the same document and query set across comparisons. 

Record hardware, model version, prompts, and input lengths so results are reproducible.

# 2\. Simple performance model

Agent latency: T\_agent ≈ sum over steps of (prefill time \+ generated tokens × decode time \+ tool-call time).

Classifier latency: T\_parallel ≈ ceil(number of documents / effective batch size) × time per batch \+ aggregation time.

These are starting approximations. Real batching, memory bandwidth, context lengths, and retrieval completeness will affect the comparison. The central output later will be a crossover plot showing which approach is preferable under different assumptions.

# 3\. Work plan

Nabanita owns all of this work. Status as of Nov 6:

## LLM and inference engine

- [x] Serve Qwen2.5-7B-Instruct with vLLM 0.6.3 on CHTC
- [x] Build the search agent on the same engine as the classifier
- [ ] Measure agent cost per step (job submitted)

## Classifier

- [x] Single-token yes/no classifier on Qwen2.5-7B-Instruct, as a Jev-style stand-in
- [x] Batch-size test on an A100 40GB: throughput levels off around batch size 16-32
- [ ] Full sweep across corpus sizes (job submitted)
- [ ] Test Laya or Kev behind the same interface

## Dataset, evaluation, and midway summary

- [x] 20 Newsgroups as the primary corpus: every document is labelled, so the answer set for each query is complete
- [x] Metrics: recall, precision, F1, ranking quality, latency, documents per second, GPU memory
- [ ] Add simple keyword and multi-condition queries
- [ ] Fit the performance model and plot the crossover
- [ ] Write the midway summary
