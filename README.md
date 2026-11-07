# Batched classification vs. agentic search

Proposal document: https://www.overleaf.com/project/6ac8329001f35b6ac41611bb

**Question:** For a query that asks for *all* matching documents, when is it
better to classify every document in parallel than to let an LLM agent search
iteratively?

The arms, all behind one interface and scored against the same complete answer
sets:

| Arm | What it does | Reads every document? | Cost shape |
|---|---|---|---|
| `classify` (the scan) | one yes/no forward pass per (query, document) pair, batched | yes | `O(N)` calls, parallel |
| `classify` with Kev-9B (`src/kev_scan.py`) | the same scan with an open Jev-like decision model | yes | `O(N)`, one pass per document can answer every query |
| `agent` | LLM search loop: write a query, get top 10, keep matches, search again or stop | no | serial in steps |
| `agent`, batched (`--agent-concurrency`) | the same agent, many queries in flight on one GPU | no | cheaper per step, slower per query |
| `ann` | top-k by embedding similarity, no LLM | no | `O(log N)`, no model calls |
| `hybrid` | top-k by embedding, then classify those k | no (yes at k = N) | `O(k)` calls |

Only the scan is exhaustive. The other arms are not meant to be: they are
cheaper, and they pay for it in recall, because recall is always measured
against every matching document. Comparisons between arms are made at equal
cost (time or decisions), never at equal effort. `hybrid` with k = N is the
scan in embedding order, so cutting it at k is "stop after k documents". The
adaptive k work tested rules for choosing that stopping point per query.

## Status

*Last updated Nov 7, 2026. Measured on one H200 with Qwen2.5-7B, plus Kev-9B as a second scan.*

All four arms run on CHTC, including a measured search agent. Results are in
`results/`, and `python -m src.summarize_runs results` prints the numbers.

| N | Scan recall | Scan s/query | Agent recall | Agent s/query |
|---:|---:|---:|---:|---:|
| 1,000 | 0.29 | 5.8 | 0.23 | 2.1 |
| 5,000 | 0.31 | 29 | 0.14 | 4.6 |
| 10,000 | 0.31 | 60 | 0.09 | 5.0 |

Scan is at batch size 32 and threshold 0.5. The agent is allowed 64 steps.
Precision is similar for both, 0.68-0.76.

- **Throughput:** levels off at batch size 32, about 170 documents per second.
- **Agent stopping:** extra budget doesn't help. At N = 10,000 it stops by choice
  after about 10 steps, with 44 of 500 relevant documents found.
- **Exchange rate:** one agent step costs about 78 scan decisions (0.46 s of wall
  time at 170 documents per second), falling to about 11 when calls are batched.
- **Limits:** the scan's lead mostly disappears on NFCorpus, where hybrid has the
  best F1. Qwen 1.5B scans about 2.4x faster but ranks much worse.
- **A purpose-built classifier fixes most of the scan's recall problem.** Kev-9B
  as the scan reaches recall 0.78 at t = 0.5 on 20 Newsgroups (Qwen 0.31) and
  F1 0.64 (Qwen 0.42). Packing all 20 queries into one pass per document scores
  540 documents per second per query. See [Kev-9B as the scan](#kev-9b-as-the-scan).

**Next:** rehearse the midterm talk (around Nov 10), then work through the plan
under [Next steps](#next-steps) for the final report (due Dec 7).

### Offline analysis of hybrid plans

Every plan walks the embedding ranking, classifies the first k documents and
keeps those accepted, so the scan is the plan with k = N. Replaying from saved
scores reproduces the CHTC rows: same fingerprints, identical top-k, and the same
classifier scores. `python -m src.access_paths` and `python -m src.crossover` run
it. All numbers are 20 Newsgroups with Qwen2.5-7B unless stated.

- **Adaptive k does not beat fixed k at equal cost.** Score mass and the largest
  gap tie it, and estimating answer size from a random sample loses. The yield
  rule (classify in chunks, stop when the acceptance rate falls below y) ties it.
- **There is little to gain.** A budget-optimal oracle that sees the answers adds
  only 0.01-0.04 recall over fixed k on 20 Newsgroups. Answer sizes barely vary
  there (329-555 at N = 10K), so k has to follow N, not the query. NFCorpus has
  more headroom at small budgets (0.14 to 0.18 at about 65 documents), but no
  rule reaches it. This agrees with Meng et al. (SIGIR 2024) on fixed depth.
- **Settings transfer differently.** Tuned for 80% of the scan's recall on one
  corpus, then applied unchanged:

  | Tuned on, applied to | Fixed k | k as a share of N | Yield rule |
  |---|---:|---:|---:|
  | 20NG 1K, 20NG 10K | 0.21 | **0.83** | 0.63 |
  | 20NG 10K, NFCorpus | 0.95 | 0.92 | 0.80 |
  | NFCorpus, 20NG 10K | 0.28 | 0.56 | **0.82** |

  The table shows the share of scan recall reached, at threshold 0.5. A share of
  N works within a corpus, but only the yield rule reached 80% in both directions
  across datasets. It undershoots as N grows. That is four pairs, so it is a lead
  rather than a result.
- **The threshold matters more than k.** Chosen by leave-one-query-out, t is about
  1e-8 on 20 Newsgroups. At N = 10K that takes recall from 0.31 to 0.63, precision
  from 0.76 to 0.59, and F1 from 0.42 to 0.58 (+0.16, 95% CI 0.11 to 0.21). Only
  0.09% of decisions change between batch sizes 1 and 32. It does not carry over:
  NFCorpus prefers 0.9, and 1e-8 there gives F1 0.06.
- **At equal time, the ordered scan beats the agent.** Given the agent's own 5.0 s
  at N = 10K, it reaches k = 849 and recall 0.256 against 0.087 at threshold 0.5
  (+0.168, 95% CI 0.112 to 0.232, better on 19 of 20 queries), at precision 0.79
  against 0.73. N = 5K gives +0.142. At N = 1K the gap is not significant (+0.048,
  CI -0.003 to 0.099).
- **Break-even exchange rate:** the agent would match the ordered scan only if a
  call cost about 11 decisions at N = 10K (6 with the calibrated threshold).
- **Batching lowers the exchange rate to break-even, and no further.** Recall does
  not change. At N = 10K, on one H200, with each query run 16 times to reach high
  concurrency (session tags stop prefix caching from helping):

  | Mean in flight | s per call | Decisions per call | Scan R, same GPU time | Agent R | Latency |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 0.46 | 78 | 0.25 | 0.08 | 4.5 s |
  | 9 | 0.09 | 15 | 0.10 | 0.09 | 9.5 s |
  | 22 | 0.084 | 14 | 0.12 | 0.09 | 29 s |
  | 53 | 0.063 | 11 | 0.09 | 0.09 | 106 s |

  So at t = 0.5 a fully batched agent ties the scan (+0.003, 95% CI -0.019 to 0.026),
  at recall 0.09, and with about 100 s of latency. With t = 1e-8 the scan keeps its
  lead (+0.060, CI 0.038 to 0.083, better on 17 of 20). Batching makes the agent
  cheaper, not more complete: recall stays at 0.09 whatever the concurrency.
- **The crossover:** one at a time, the scan wins at equal GPU time. Agents reach
  parity only with dozens of queries in flight, and only up to the agent's own
  recall. Above that, the scan is the only arm that gets there.
- **NFCorpus crosses, narrowly.** At its own 95 documents per second, a call falls
  from 33 decisions to 3.7. On the 50 queries with scan scores, the agent passes the
  scan on recall only at 84 in flight (0.15 against 0.11, CI -0.095 to -0.010), and
  not on F1. That is one run, and the agent's recall there was 0.12-0.13 in the
  other four runs, so treat it as a sign of noise as much as of a crossover.
- **The exchange rate is the same order across models, not the same number:** 78
  for Qwen 7B, 80-90 for Qwen 1.5B, 122 for Phi-3.5-mini (210 documents per second,
  0.57 s per call, 100 generated tokens). But the 1.5B and Phi agents stop after 1
  to 1.5 searches, so those two rates come from almost trivial runs. Only the 7B
  agent really searches, so the agent comparison rests on one model. Phi's scan
  ranks well (recall@100 of 0.55 at N = 1K, against 0.75 for 7B and 0.37 for 1.5B).
- **Adaptive k also fails where answer sizes vary.** The keyword and multi-condition
  queries span 18 to 1,046 answers at N = 10K, and the oracle has up to 0.07 more
  recall than fixed k. Score mass and yield still tie fixed k, and sampling loses.
- **Query shape decides the scan's quality.** At N = 10K and t = 0.5, with the same
  arms on 25 queries:

  | Shape (queries) | Scan R / P | Agent R / P |
  |---|---:|---:|
  | Keyword (10) | 0.82 / 0.69 | 0.26 / 0.67 |
  | AND (6) | 0.72 / 0.59 | 0.24 / 0.62 |
  | OR (5) | 0.32 / 0.88 | 0.06 / 0.88 |
  | NOT (4) | 0.10 / 0.74 | 0.06 / 0.63 |

  The best threshold follows the shape. Narrow, exact queries (keyword, AND) peak
  near 0.5, while broad topic queries (OR, NOT, categories) want about 1e-8, where
  NOT reaches recall 0.51. So one global cutoff is wrong, and the NOT failure looks
  like the broad-topic problem rather than a failure of negation, though four
  queries is too few to be sure. Keyword gold sets come from exact word matches, so
  a plain text search scores 1.0 on them by construction. They measure how reliable
  the model is, and are not a case for using one.
- **Cost follows tokens.** Cutting documents to 250, 500, 1,000, 2,000 and 4,000
  characters gives 369, 276, 212, 174 and 149 documents per second, at 44-53K tokens
  per second throughout. Recall is 0.21, 0.25, 0.27, 0.29 and 0.29, so 1,000
  characters is 22% faster than 2,000 for 0.02 recall. The exchange rate scales the
  same way, because it is a rate per document.
- **Index cost:** on a laptop, the HNSW graph for 10K documents builds in under a
  second and inserts take 0.11 ms per document. Search keeps at least 99.9% of the
  exact top k up to k = 3,000 when `ef` is at least k. Embedding is the real cost
  (6.3 s on the H200), so HNSW changes nothing at this scale.

### Kev-9B as the scan

Kev-9B is an open, Jev-like decision model (a LoRA and pointer head on
Qwen3.5-9B-Base, Apache-2.0). It reads the document first and the questions after
it, so one pass over a document can answer every query ("packed"). It ran on the
same documents as Qwen (fingerprints match) on one H200, through
`src/kev_scan.py`. `python -m src.kev_compare` replays both from saved scores and
writes `results/kev_compare.json`. Differences are Kev minus Qwen, with a
bootstrap 95% interval over queries.

| Setting | Qwen F1 at 0.5 | Kev F1 at 0.5 | Qwen F1, held-out t | Kev F1, held-out t | Held-out difference |
|---|---:|---:|---:|---:|---:|
| 20NG N = 1K | 0.40 | 0.63 | 0.56 | 0.63 | +0.07 [0.04, 0.10] |
| 20NG N = 5K | 0.41 | 0.64 | 0.58 | 0.65 | +0.07 [0.04, 0.10] |
| 20NG N = 10K | 0.42 | 0.64 | 0.58 | 0.65 | +0.07 [0.04, 0.11] |
| NFCorpus | 0.12 | 0.15 | 0.12 | 0.18 | +0.06 [0.02, 0.10] |
| Logic N = 1K | 0.59 | 0.73 | 0.57 | 0.76 | +0.19 [0.12, 0.25] |
| Logic N = 10K | 0.56 | 0.70 | 0.58 | 0.77 | +0.19 [0.15, 0.23] |

- **Recall without threshold tuning.** At t = 0.5, Kev's recall is 0.78 at N = 10K
  against Qwen's 0.31. Qwen needs t = 1e-8 to reach 0.63. Kev's held-out threshold
  is 0.5 to 0.9 in every setting, so the threshold problem is mostly a Qwen problem.
- **Query shape stops mattering much.** At N = 10K and t = 0.5, Kev's F1 is 0.74
  (keyword), 0.54 (AND), 0.79 (OR), 0.70 (NOT). Qwen's is 0.73, 0.63, 0.45, 0.18.
  NOT goes from recall 0.10 to 0.78. Kev is weaker than Qwen only on AND at 0.5,
  where it accepts too much (precision 0.40); with a held-out threshold it wins
  there too (0.78 against 0.58).
- **Equal time against the agent (20NG, t = 0.5).** Given the agent's own seconds
  per query, the scan in embedding order reaches:

  | N | Agent | Qwen scan | Kev, one query at a time | Kev, packed |
  |---:|---:|---:|---:|---:|
  | 1,000 | 0.23 | 0.28 | 0.68 | 0.77 (whole corpus) |
  | 5,000 | 0.14 | 0.28 | 0.59 | 0.75 |
  | 10,000 | 0.09 | 0.26 | 0.48 | 0.71 |

  Kev beats the agent on all 20 queries at every size. One query at a time Kev
  scores 121 documents per second, slower than Qwen's 170, and wins on accuracy.
  Packed is 540 documents per second per query, but only when the queries arrive
  together, so it fits batch workloads, not one-off questions.
- **Exchange rate with Kev:** one Qwen agent step (0.46 s) buys about 56 Kev
  decisions one query at a time, or about 250 packed.
- **Hybrid improves too.** Classifying the top 1,000 by embedding at N = 10K gives
  recall 0.57 with Kev against 0.27 with Qwen.
- **NFCorpus is still hard.** Kev is the best arm there (held-out F1 0.18, against
  0.15 for Qwen's hybrid and 0.12 for the agent), but recall does not improve
  (0.19 for both at the held-out threshold). With 50 queries it packs 724
  documents per second per query.
- **Kev is not better calibrated.** Its expected calibration error is 0.06,
  against 0.04 for Qwen. Above 0.5 it predicts 0.81 on average and is right
  about half the time. Qwen is extreme in the other direction: almost everything
  is near 0 or 1.
- **Caveats:** Kev is a different, larger base model trained for decisions, so
  this is not the same-model comparison the agent arm has. Neither 20 Newsgroups
  nor NFCorpus is in its training data. Packed throughput depends on how many
  queries share a pass: 20 here, 50 on NFCorpus, 25 on the logic set.

Related work to cite: ranked list truncation (Choppy, AttnCut, and Meng et al.
2024 for LLM re-ranking), adaptive-k for RAG (Taguchi et al. 2025), stopping rules
for technology-assisted review (Cormack and Grossman 2016; Yang et al. 2021), and
semantic filter cascades (LOTUS; Kim et al. 2026). Sources are at the end of
[Next steps](#next-steps). Meng et al. and LOTUS were read in full; the others
from their abstracts.

---

## Quick start

Offline, no GPU, no downloads — exercises the whole pipeline in about ten
seconds:

```bash
pip install numpy scikit-learn pandas matplotlib

python -m src.runner --dataset synthetic --arms classify,ann,hybrid \
    --model mock --embed-backend tfidf \
    --corpus-sizes 500,1000,2000 --batch-sizes 32,128,512 \
    --out results/smoke.jsonl

python -m src.analyze --results results/smoke.jsonl --outdir figs
```

Run this before every CHTC submit. It catches typos in seconds rather than
after a model has spent ten minutes loading onto a GPU.

## The midterm run (CHTC)

Copy the project to the access point, then submit from there:

```bash
bash chtc/upload.sh

ssh chtc
cd ThinkFast-QueryFaster
condor_submit chtc/midterm.sub
```

`upload.sh` uses the `chtc` SSH alias; set `CHTC_HOST` to use a different host.
It does not copy `chtc/.env`, because the job does not need a token.

Run a short test job first. It writes to its own output file and compares two
batch sizes, so it also shows whether the sweep changes throughput:

```bash
condor_submit SIZES=1000 BATCHES=32,256 TAG=test chtc/midterm.sub
```

To run the search agent, which handles queries one at a time and so needs only
one batch size:

```bash
condor_submit ARMS=agent BATCHES=32 TAG=agent chtc/midterm.sub
```

For the full sweep, submit one job per batch size so they run in parallel on
identical GPUs. Each one needs its own `TAG`, or their results overwrite each
other. `H200$` leaves out the H200 NVL.

```bash
for bs in 1 4 8 16 32 64 256; do
  condor_submit BATCHES=$bs TAG=sweep_bs$bs GPU='H200$' LENGTH=short chtc/midterm.sub
done
condor_submit ARMS=agent BATCHES=32 TAG=agent GPU='H200$' LENGTH=short chtc/midterm.sub
```

HTCondor reads `src/` when a job starts, not when it is submitted, so uploading
to a folder with a queued job changes the code that job will run. To work on
the code while a job waits, upload somewhere else with
`REMOTE_DIR=another-folder bash chtc/upload.sh`.

Results come back to `results/` and `figs/` in the project folder on the access
point. The last batch size in a run also writes per-document scores to
`results/<dataset>_<tag>.scores.jsonl`, so thresholds can be re-evaluated
without another GPU run. The job only runs on A100 and H200 GPUs; pass `GPU=H200` to pin one type
so throughput numbers stay comparable between runs.

Equivalent by hand on an interactive GPU session:

```bash
python -m src.runner --dataset 20newsgroups --arms classify,ann,hybrid \
    --model Qwen/Qwen2.5-7B-Instruct \
    --corpus-sizes 1000,5000,10000 --batch-sizes 32,64,128,256 \
    --k 100 --out results/midterm.jsonl

python -m src.analyze --results results/midterm.jsonl --outdir figs/midterm
```

Runs are **resumable**: the runner skips any (arm, corpus size, batch size, k,
query) already present in the output file, so a preempted job picks up where
it stopped.

---

## Datasets

| Name | Docs | Queries | Relevant/query | Notes |
|---|---:|---:|---:|---|
| `20newsgroups` | ~18.8K | 20 | hundreds | Every doc carries a category label, so "find all documents about cryptography" has an **exact** answer set. Ships with scikit-learn. **Primary.** |
| `nfcorpus` | ~3.6K | ~320 | ~38 | Real IR relevance judgments. Needs `datasets`. Second dataset, for generality. |
| `20newsgroups_logic` | ~18.8K | 25 | 18-1,046 | Keyword, AND, OR and NOT queries on the same documents; gold from labels and exact word matches. |
| `synergy_<review>` | 257-48K | 1 each | 9-762 | **Main benchmark for the short paper.** 26 systematic reviews ([SYNERGY](https://github.com/asreview/synergy-dataset), CC0). Each is its own corpus: the review's screened candidate pool, with its inclusions as the complete answer set and its eligibility criteria as the query (`_title` variants use only the review title). `synergy` means all 26. |
| `synthetic` | any | 5 | N/5 | Offline smoke tests only. Never reported. |

SYNERGY is built locally, because it needs network access:
`python -m src.synergy_build` reads the labels at a pinned commit, fetches titles
and abstracts from OpenAlex, fills missing abstracts from PubMed where a record
has a PMID, and writes `data/synergy/` (not in git; `chtc/upload.sh` copies it).
About 30% of records stay title-only, close to the official build; screeners see
those records too. Pools are never merged, since a record in one review's pool
was never screened against another review's criteria. Records are shuffled with
a fixed seed, because the source files list inclusions first.

RCV1 would also fit, but its text needs a signed NIST agreement; the public
scikit-learn version is TF-IDF vectors only.

20 Newsgroups is loaded with headers, footers and quoted replies stripped. The
headers contain the newsgroup name; leaving them in makes the task trivial and
the measurement meaningless.

---

## How the classifier works

Rather than depending on one vendor checkpoint, we implement the System One
decision pattern directly on an open-weights model: the prompt ends where the
answer begins, we generate **exactly one token**, and we read the probability
mass on "yes" versus "no" from the LM head. One forward pass per pair and no
autoregressive generation. Nothing is trained, and the output is a probability
rather than a parsed string, though it is not calibrated.

Three reasons this beats chasing a checkpoint in a five-day window:

1. The **same served model** backs the agent arm, so the comparison isolates
   execution architecture rather than confounding it with model quality.
2. Model size becomes a free experimental variable — and the thesis is about
   batch throughput.
3. No gated downloads, no custom serving stack.

A purpose-built decision model (Kev, JEV-9B) needs its own environment, so it runs
through `src/kev_scan.py` and `chtc/run_kev.sh`, and writes the same score files.

`--batch-sizes` sets vLLM's `max_num_seqs`, which caps concurrent sequences and
is therefore the effective batch size. It is the **outer** loop in the runner,
because the engine is expensive to build and `max_num_seqs` is fixed at
construction.

---

## What the smoke run already shows

From `figs/summary.md` on synthetic data, with k=100:

| Arm | N | Recall | Precision |
|---|---:|---:|---:|
| Classify | 2,000 | **1.000** | 1.000 |
| ANN | 2,000 | 0.250 | 1.000 |
| Hybrid | 2,000 | 0.250 | 1.000 |

Hybrid recall equals its retrieval ceiling exactly. When a query has more
relevant documents than `k`, retrieve-then-classify **cannot** reach them no
matter how good the classifier is — 80% of synthetic queries hit this. Every
hybrid row records `ann_recall_ceiling` and `k_vs_gold_ratio` so the effect is
measured rather than inferred.

This is the caveat on "hybrid dominates everywhere", and it is a real result
for the report: the hybrid wins on throughput but inherits a hard recall cap,
while the full scan is exhaustive by construction. That is H1's argument,
visible before any GPU work.

**Caveat:** synthetic data only. The numbers above are a pipeline check, not
evidence. Real numbers come from the 20 Newsgroups run.

---

## Layout

```
src/
  data.py          corpora + query sets + subsampling + fingerprinting
  metrics.py       ArmResult, set metrics, nDCG, threshold sweep
  runner.py        experiment driver (resumable, JSONL)
  analyze.py       figures + markdown tables for the report
  summarize_runs.py  every number the deck and report use
  embed_ranks.py   saves all query-document similarities, for offline replay
  access_paths.py  fixed k, adaptive k, oracle, threshold calibration, transfer
  crossover.py     performance model, equal-time comparison, exchange rates
  report_figs.py   final report figures
  kev_scan.py      scan with Kev, in its own environment on CHTC
  jev_scan.py      scan with JEV-9B (transformers + peft), on CHTC
  kev_compare.py   Kev or JEV against Qwen from saved scores, and against the agent at equal time
  index_costs.py   HNSW build, insert and deep-k recall (needs hnswlib)
  arms/
    classify.py    Approach B  (+ MockDecisionModel for offline tests)
    ann.py         Approaches C and D
    agent.py       Approach A (+ MockChat for offline tests)
chtc/
  midterm.sub      HTCondor submit file
  run.sh           job executable (runs smoke test first)
  run_kev.sh, kev.sub   the Kev job, pinned to one Kev commit
  upload.sh        copies the project to the access point
```

### The interface (fixed — do not renegotiate)

Every arm returns an `ArmResult`:

```python
ArmResult(
    arm="classify",
    query_id="q_sci.crypt",
    predicted={"ng00123", ...},     # doc ids the arm says match
    scores={"ng00123": 0.91, ...},  # optional ranking
    timing={"total": 4.2, ...},     # seconds per stage
    counts={"docs_scored": 10000},  # calls, tokens, docs
    extra={"model": "...", ...},    # anything to record per row
)
```

`metrics.flatten()` turns one of these into a JSONL row. Add an arm by writing
a function that returns `ArmResult` and adding a branch in `runner.py`.

### Agent arm

`src/arms/agent.py` runs the search loop on the same in-process vLLM engine as
the classifier, so both arms use one model and one job needs no separate server.
Each step shows the agent up to `--agent-k` unseen documents as 300-character
snippets, and `--agent-max-steps` caps the search.

Every result records `steps` and `stopped_early`. An agent has to guess when it
is finished; a full scan is exhaustive by construction. Those two columns are
the evidence for H1, which is the strongest claim in the project.

---

## Next steps

Done so far:

- [x] Confirm the container tag and GPU flag in `chtc/midterm.sub`
- [x] Agent arm — the arm H1 and H2 are measured against
- [x] Second dataset (NFCorpus) and second model (Qwen2.5-1.5B)
- [x] Measure the exchange rate (about 70 scan decisions per agent step)
- [x] Check the ranked list truncation, TAR stopping and semantic filter literature
- [x] Sweep `k` for the hybrid arm from the saved scores
- [x] Build and test adaptive k: it does not beat fixed k at equal cost
- [x] Calibrate the classifier threshold on held-out queries
- [x] Clean up `src/summarize_runs.py`
- [x] Swap exact search for HNSW and report index build and insert cost
- [x] Batch agent calls across queries (`agentbatch`)
- [x] Repeat the adaptive k test on keyword and multi-condition queries
- [x] Document-length sweep
- [x] Fit the performance model and measure the crossover (`agentrep`)
- [x] Second decision model family (Phi-3.5-mini)

### Before the final report (Dec 7)

The two contributions are adaptive k and the exchange rate.

- [x] Kev-9B scan on 20 Newsgroups (1K, 5K, 10K), NFCorpus and the logic queries,
      one query at a time and packed (`chtc/kev.sub`, `src.kev_compare`)
- [ ] JEV-9B scan on the same settings (Apache-2.0, pinned revision 7ac669b).
      Pilot done: N = 1K F1 0.61 at t = 0.5 (Qwen 0.40, Kev 0.63), 81 decisions/s,
      which is close to compute-bound for a 9B prefill (Kev reaches 540 only by
      packing queries). Sweep running as jobs 12437151-53.
      Compare with `python -m src.kev_compare --model jev`.

### Short paper (decided Nov 7)

Target a 4-page short paper. Reviewers will ask why an LLM is needed, whether
the agent is a strawman, and whether 10K documents is enough.

- [ ] Cheap baselines (`src/baselines.py`, `src/rerank_scan.py`). Offline so far,
      with tuned cutoffs: BM25 F1 0.37 and embeddings 0.48 against Qwen 0.58 and
      Kev 0.65 at 20NG 10K; WSS@95 0.04 / 0.41 / 0.55 / 0.74. On the logic queries
      BM25 and embeddings reach 0.31-0.32 against 0.58 and 0.77. On NFCorpus BM25
      ties Qwen (0.16 against 0.12, CI crosses 0) and Kev is best (0.18).
      Cross-encoders (bge-reranker-v2-m3, ms-marco-MiniLM) running as jobs
      12437161-66 (`chtc/scan.sub`).
- [ ] SYNERGY as the main benchmark: loader done, data build running. Then
      embeddings, Qwen scan and agent, Kev, JEV and rerankers on all 26 reviews.
- [ ] A stronger agent: keyword search as a second tool, and a larger model.
- [ ] arXiv metadata (CC0) at 100K and 1M documents, for the crossover at scale.
- [ ] Rewrite `report/` as the 4-page paper.
- [ ] Final report: draft in `report/report.tex` (LaTeX, compile on Overleaf),
      complete except the JEV-9B results. The story changes: low recall was
      mostly a Qwen problem, so the scan-against-agent result gets stronger with
      a real decision model, and the threshold finding becomes a zero-shot caveat.

### Future work

Not planned before Dec 7:

- Query rewriting: an expensive model decomposes the predicate once and a cheap
  model executes it N times
- Laya, skipped: 512-token limit, reportedly near random zero-shot
- Hosted Jev: closed, so accuracy only, and billed per call
- Update rate

We are skipping the confidence cascade because it is too close to JEVDB.

### Sources

- [Choppy](https://arxiv.org/abs/2004.13012), [AttnCut](https://arxiv.org/abs/2102.12793)
- [Ranked List Truncation for LLM-based Re-Ranking](https://arxiv.org/abs/2404.18185)
- [Adaptive-k](https://arxiv.org/abs/2506.08479)
- [Heuristic Stopping Rules for TAR](https://arxiv.org/abs/2106.09871) (Yang, Lewis and Frieder 2021)
- [Engineering Quality and Reliability in TAR](https://cormack.uwaterloo.ca/reliability) (Cormack and Grossman, SIGIR 2016)
- [Semantic Operators (LOTUS)](https://arxiv.org/abs/2407.11418)
- [Fast LLM-Based Semantic Filtering](https://arxiv.org/abs/2606.08090)
