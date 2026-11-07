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
| `agent` | LLM search loop: write a query, get the top 20 unseen documents, keep matches, search again or stop. `--agent-tools semantic,keyword` adds BM25 search as a second tool | no | serial in steps |
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
- **Scale (arXiv, 10K to 1M papers):** the agent's effort barely grows with the
  corpus (7 to 12 steps, 143 to 239 papers read, always stopping by choice)
  while answer sets grow a hundredfold, so its recall falls from 0.19 to 0.008.
  At equal time and with a held-out threshold, the scan finds 89 / 384 / 646
  relevant papers per query against the agent's 22 / 44 / 68, better on 20 of
  20 queries at every size. At t = 0.5 the two tie at 10K. See the arXiv item
  under [Before the final report](#before-the-final-report-dec-7).

**Next:** rehearse the midterm talk (Nov 10; the deck was updated Nov 7 with
all results), and finish the two experiments under
[Before the final report](#before-the-final-report-dec-7): a persistent agent,
and Kev scanning arXiv at 100K and 1M.

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
| `arxiv` | 1M | 20 | 15-581 at 10K, 1.4K-59K at 1M | **Scale.** A fixed 1M-paper sample (first 1M unique ids by MD5) of the CC0 arXiv metadata snapshot `librarian-bots/arxiv-metadata-snapshot` at a pinned revision; title and abstract. 20 category queries; a paper is relevant if the category is among its own (cross-lists included), so answer sets are complete. Built on first use into `data/arxiv/` (32 s locally, from 2.7 GB of parquet); jobs build their own copy. |
| `synthetic` | any | 5 | N/5 | Offline smoke tests only. Never reported. |

SYNERGY is built locally, because it needs network access:
`python -m src.synergy_build` reads the labels at a pinned commit, takes title
and abstract from PubMed where a record has a PMID and from OpenAlex otherwise
(and OpenAlex abstracts where PubMed has none), and writes `data/synergy/` (not
in git; `chtc/upload.sh` copies it). OpenAlex needs a free API key in
`OPENALEX_API_KEY`, read from the environment or the project's `.env`. 8.7% of
records stay title-only; screeners see those records too. Pools are never merged, since a record in one review's pool
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

## Layout

`results/README.md` says which results file backs which table and which script
reads it.

```
src/
  data.py            corpora, query sets, subsampling, fingerprints
  metrics.py         ArmResult, set metrics, threshold sweep
  runner.py          GPU experiment driver for Qwen (resumable, JSONL)
  arms/
    classify.py      the scan (+ MockDecisionModel for offline tests)
    ann.py           embedding search, and hybrid (search, then classify the top k)
    agent.py         the search agent (+ MockChat); stop rules "agent" and "dry"
    keyword.py       BM25, the agent's optional second tool
  kev_scan.py        scan with Kev-9B, in Kev's own environment on CHTC
  jev_scan.py        scan with JEV-9B (transformers + peft), on CHTC
  rerank_scan.py     scan with a cross-encoder reranker (bge, MiniLM)
  synergy_build.py   builds the SYNERGY texts from PubMed and OpenAlex
  embed_ranks.py     saves query-document similarities for offline replay
  # analysis: replays saved scores, no GPU needed
  summarize_runs.py  the deck's and the first report tables' numbers
  access_paths.py    fixed k, adaptive k, oracle, thresholds
  crossover.py       exchange rates, equal time on 20 Newsgroups
  kev_compare.py     Kev or JEV against Qwen
  baselines.py       all seven scorers (tab:scorers)
  agents_compare.py  every agent against every scan, 20 Newsgroups
  synergy_compare.py the same per SYNERGY review
  scale_compare.py   arXiv at 10K, 100K and 1M
  index_costs.py     HNSW against exact search (needs hnswlib)
  analyze.py         quick plots of one run (output to figs/, not committed)
  report_figs.py     report/figs/exchange.png and frontier.png
  paper_figs.py      recall against time (paper, report and, with --slide, the deck)
chtc/
  midterm.sub, run.sh      Qwen jobs (scan, hybrid, agent)
  kev.sub, run_kev.sh      Kev jobs, pinned to one Kev commit
  scan.sub, run_scan.sh    JEV and reranker jobs (Python 3.12 environment)
  jev.sub, run_jev.sh      the older JEV-only job
  upload.sh                copies the project to the access point
report/                    the course report (LaTeX, compile on Overleaf)
paper/                     the short paper draft (paused)
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
- [x] JEV-9B scan on the same settings (Apache-2.0, pinned revision 7ac669b),
      `python -m src.kev_compare --model jev`. F1 at t = 0.5 / tuned:
      20NG 10K 0.64 / 0.63 (Qwen 0.42 / 0.58, Kev 0.64 / 0.65); logic 10K
      0.79 / 0.79 (Qwen 0.56 / 0.58, Kev 0.70 / 0.77); NFCorpus 0.20 / 0.20 (Qwen
      0.12 / 0.12, Kev 0.15 / 0.18). NOT queries F1 0.63 against Qwen's 0.18. Tuned
      threshold 0.3-0.5 everywhere. Recall in the agent's time at 20NG 10K: 0.43
      (Qwen 0.26, Kev 0.48 separate). 73-127 decisions/s, close to compute-bound
      for a 9B prefill. ECE 0.045-0.056; well calibrated on logic (0.83 predicted,
      0.82 actual).
- [ ] Cheap baselines (`src/baselines.py`, `src/rerank_scan.py`), tuned F1 with
      the cutoff chosen on other queries, and WSS@95:

      | Setting | BM25 | Embed | bge | Qwen | Kev | JEV |
      |---|---:|---:|---:|---:|---:|---:|
      | 20NG 10K | 0.37 | 0.48 | 0.35 | 0.58 | 0.65 | 0.63 |
      | Logic 10K | 0.32 | 0.31 | 0.23 | 0.58 | 0.77 | 0.79 |
      | NFCorpus | 0.16 | 0.12 | 0.13 | 0.12 | 0.18 | 0.20 |
      | SYNERGY | 0.11 | 0.15 | | | | |

      WSS@95 at 20NG 10K: 0.04 / 0.41 / 0.25 / 0.55 / 0.74 / 0.60. The bge
      cross-encoder (1,100 decisions/s, 9x Qwen) never beats embeddings; zero-shot
      Qwen only ties BM25 on NFCorpus, but both decision models beat it. MiniLM
      jobs running; SYNERGY LLM and reranker scans queued.
- [x] SYNERGY as the main benchmark (`python -m src.synergy_compare`). Built:
      168,946 records, 2,831 inclusions (official: 169,288 and 2,834); 8.7%
      title-only. Over the 26 reviews:

      | Scorer | AP | F1 tuned | WSS@95 | docs/s |
      |---|---:|---:|---:|---:|
      | BM25 | 0.10 | 0.11 | 0.23 | |
      | Embeddings | 0.16 | 0.15 | 0.45 | |
      | MiniLM / bge | 0.14 / 0.10 | 0.13 / 0.10 | 0.34 / 0.26 | |
      | Qwen 7B scan | 0.36 | **0.32** | **0.58** | 109 |
      | Kev-9B | **0.38** | 0.22 | 0.55 | 58 |
      | JEV-9B | 0.34 | 0.25 | 0.38 | 46 |

      On long, multi-condition criteria zero-shot Qwen ranks as well as the
      decision models and has the best tuned F1 (Kev - Qwen -0.10 [-0.17, -0.05],
      JEV - Qwen -0.08 [-0.16, -0.01]). Kev at t = 0.5 keeps recall 0.69 at
      precision 0.17. The agent (7B, embeddings) reaches recall 0.16 in 5.3 s, 11
      steps. In the agent's own time per review, the scan in embedding order reaches
      0.26 with Qwen (+0.10 [0.05, 0.16], 17/26 reviews), 0.37 with Kev (+0.21,
      22/26) and 0.21 with JEV (+0.05 [0.004, 0.09], 16/26). Exchange rate 48 (Qwen),
      lower than 20NG's 78 because the criteria make prompts longer.
- [x] A stronger agent (`python -m src.agents_compare`, `python -m
      src.synergy_compare`): BM25 keyword search as a second tool
      (`--agent-tools semantic,keyword`, `src/arms/keyword.py`, shared with the
      BM25 baseline) and Qwen2.5-32B, agent only, up to 64 steps. Agent recall,
      and scan recall in each agent's own seconds per query (t = 0.5):

      | Agent | 20NG 10K agent | Qwen | Kev | JEV | SYNERGY agent | Qwen | Kev | JEV |
      |---|---:|---:|---:|---:|---:|---:|---:|---:|
      | 7B, embeddings | 0.09 | 0.22 | 0.43 | 0.38 | 0.16 | 0.26 | 0.37 | 0.21 |
      | 7B, + BM25 | 0.10 | 0.24 | 0.49 | 0.41 | 0.13 | 0.29 | 0.46 | 0.28 |
      | 32B, embeddings | 0.17 | 0.29 | 0.68 | 0.57 | 0.21 | 0.32 | 0.62 | 0.40 |
      | 32B, + BM25 | 0.13 | 0.29 | 0.63 | 0.54 | 0.16 | 0.31 | 0.55 | 0.34 |

      Every scan beats every agent on both datasets; all intervals are above
      zero. 32B with embeddings is the best agent, but its steps cost about 1.4 s
      (exchange rate about 160 against the Qwen scan), so the scan's margin grows.
      Adding BM25 lowers recall for both models: the agents switch to keyword
      search (55-90% of searches on SYNERGY) and find less. This table gives
      each query its own agent time; the older crossover used the mean, which
      gives slightly higher scan recall (0.26 against 0.22 for Qwen) because
      recall is concave in k. Use the per-query numbers in the paper.
- [x] arXiv at 10K, 100K and 1M documents (job 12437254, TAG scale, 2.4 h on
      an H200): hybrid with k = 20,000 (each query's top 20K by embedding is
      classified) and the 7B agent. `python -m src.scale_compare` replays it from
      the local rankings (`src.embed_ranks`; the 1M ranking took 1 h 55 min on the
      Mac). Fingerprints match the job's rows, the job scored all of the local
      top 20K, and no equal-time budget went past 20K, so the replay is exact.
      Held-out threshold: tuned on 10K leaving each query out (10^-8, as on 20
      Newsgroups), then used at every size.

      | | 10K | 100K | 1M |
      |---|---:|---:|---:|
      | Median answer set | 148 | 1,436 | 14,222 |
      | Agent steps / papers read / seconds | 7.2 / 143 / 3.9 | 10.6 / 212 / 6.2 | 11.9 / 239 / 8.2 |
      | Agent recall / found | 0.19 / 22 | 0.05 / 44 | 0.008 / 68 |
      | Scan depth in agent's time | 540 | 889 | 1,172 |
      | Scan recall t=0.5 | 0.16 | 0.09 | 0.024 |
      | Scan recall held-out / found | 0.57 / 89 | 0.29 / 384 | 0.069 / 646 |
      | Scan - agent, held-out [95% CI] | +0.38 [0.31, 0.44] | +0.24 [0.20, 0.29] | +0.061 [0.036, 0.091] |
      | Scan - agent, t=0.5 | -0.04 [-0.10, 0.03] | +0.04 [0.02, 0.06] | +0.015 [0.011, 0.020] |
      | Top 20K, held-out: recall | 0.63 | 0.60 | 0.44 |

      Qwen scans arXiv abstracts at about 140 documents per second, so a full
      1M scan would take 117 minutes per query; the top 20K takes 140 s. The
      exchange rate is 66-81 at every size. The embedding's top 20K holds 59% of
      each answer set at 1M, which caps the hybrid. Only the 7B agent and the
      Qwen scan were run here.
- [ ] Short paper: draft in `paper/paper.tex` (ACM sigconf, about 2,000 words,
      `refs.bib`, figure from `python -m src.paper_figs`). Paused (Nov 7) to
      focus on experiments and the report; the arXiv results still need to go
      in. `paper_figs` draws a third panel for arXiv at 100K (`--arxiv-n`; chosen
      Nov 7 because at 1M every recall is under 0.12 on the shared axis and the
      agent's point sits on the curve). The arXiv curve ends where the classified
      top 20K runs out. The report uses the same figure. `report/` stays
      the longer course report and still needs the new results.
- [ ] Final report: draft in `report/report.tex` (LaTeX, compile on Overleaf).
      All results are in, including JEV-9B, SYNERGY and arXiv scale (Table
      `tab:scale`); no TODOs left. Not yet compiled. The story changes: low recall was
      mostly a Qwen problem, so the scan-against-agent result gets stronger with
      a real decision model, and the threshold finding becomes a zero-shot caveat.
- [ ] Is the agent's early stopping a prompt problem? (submitted Nov 7, jobs
      12437628-30, TAG agentdry). `--agent-stop dry` (submit option `AGENT_STOP=dry`)
      tells the 7B agent that answer sets can hold thousands of documents and
      holds it to a stopping rule from technology-assisted review: it may stop
      only after two searches in a row find nothing it keeps. An earlier stop is
      refused once with a note; a second refusal in a row ends the run. Up to 256
      steps; only the latest 40 searches are listed in its prompt. Runs on 20
      Newsgroups 10K, all 26 SYNERGY reviews and arXiv 100K.
      `src.agents_compare`, `src.synergy_compare` and `src.scale_compare` pick it
      up as "7B, persistent", and `paper_figs` draws it as a square.
- [ ] Kev at scale (submitted Nov 7): packed scans of arXiv at 100K (job
      12437631, TAG kev9b, about 1 h) and 1M (12437632, TAG kev9b1m, about 10 h at
      the 27 documents per second packed rate seen on 20 Newsgroups). A full 1M
      scan answers all 20 queries in one pass, so it shows how complete an
      exhaustive scan gets where the agent reaches 0.008. The job builds the
      arXiv sample itself; `run_kev.sh` adds pyarrow to Kev's environment for it.

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
