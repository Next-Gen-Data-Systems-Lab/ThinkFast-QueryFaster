# Batched classification vs. agentic search

Proposal document: https://www.overleaf.com/project/6ac8329001f35b6ac41611bb

**Question:** For a query that asks for *all* matching documents, when is it
better to classify every document in parallel than to let an LLM agent search
iteratively?

Four execution arms behind one interface:

| Arm | What it does | Cost shape | 
|---|---|---|
| `agent` | iterative LLM search, reason → search → inspect → repeat | serial in steps |
| `classify` | one forward pass per (query, document) pair, batched | `O(N)` calls, parallel |
| `ann` | embed the corpus, retrieve top-k by similarity | `O(log N)`, no model calls |
| `hybrid` | ANN for candidates, then classify those k | `O(k)` calls |

## Status

*Last updated Nov 6, 2026. Measured on one H200 with Qwen2.5-7B.*

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
- **Exchange rate:** one agent step costs about 70 scan decisions (0.41 s at 170
  documents per second), or 80-90 on Qwen 1.5B.
- **Limits:** the scan's lead mostly disappears on NFCorpus, where hybrid has the
  best F1. Qwen 1.5B scans about 2.4x faster but ranks much worse. Agent calls
  were not batched across queries.

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
  call cost about 11 decisions at N = 10K (6 with the calibrated threshold), not 70.
  Agent calls were not batched in this number.
- **Index cost:** on a laptop, the HNSW graph for 10K documents builds in under a
  second and inserts take 0.11 ms per document. Search keeps at least 99.9% of the
  exact top k up to k = 3,000 when `ef` is at least k. Embedding is the real cost
  (6.3 s on the H200), so HNSW changes nothing at this scale.

Related work to cite: ranked list truncation (Choppy, AttnCut, and Meng et al.
2024 for LLM re-ranking), adaptive-k for RAG (Taguchi et al. 2025), stopping rules
for technology-assisted review (Cormack and Grossman), and semantic filter
cascades (LOTUS; Kim et al. 2026). Sources are at the end of
[Next steps](#next-steps). Only abstracts have been read so far.

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
| `synthetic` | any | 5 | N/5 | Offline smoke tests only. Never reported. |

20 Newsgroups is loaded with headers, footers and quoted replies stripped. The
headers contain the newsgroup name; leaving them in makes the task trivial and
the measurement meaningless.

---

## How the classifier works

Rather than depending on one vendor checkpoint, we implement the System One
decision pattern directly on an open-weights model: the prompt ends where the
answer begins, we generate **exactly one token**, and we read the probability
mass on "yes" versus "no" from the LM head. One forward pass per pair, no
autoregressive generation, and a calibrated probability rather than a parsed
string.

Three reasons this beats chasing a checkpoint in a five-day window:

1. The **same served model** backs the agent arm, so the comparison isolates
   execution architecture rather than confounding it with model quality.
2. Model size becomes a free experimental variable — and the thesis is about
   batch throughput.
3. No gated downloads, no custom serving stack.

A purpose-built decision model (Laya, JEV-9B, NeoHorse-Jev-4B) swaps in behind
the same interface later, to confirm the finding is architectural rather than
specific to one model.

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
  index_costs.py   HNSW build, insert and deep-k recall (needs hnswlib)
  arms/
    classify.py    Approach B  (+ MockDecisionModel for offline tests)
    ann.py         Approaches C and D
    agent.py       Approach A (+ MockChat for offline tests)
chtc/
  midterm.sub      HTCondor submit file
  run.sh           job executable (runs smoke test first)
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

### Before the final report (Dec 7)

The two contributions are adaptive k and the exchange rate.

- [ ] Batch agent calls across queries, which would lower the exchange rate
      (`agentbatch` jobs queued on CHTC)
- [ ] Fit the performance model and plot the crossover between scan and agent
      (`src/crossover.py` is ready for the batched numbers)
- [ ] Repeat the adaptive k test on the logic queries, where answer sizes vary
- [ ] Write the final report

### If time allows

- [ ] Keyword and multi-condition queries (`20newsgroups_logic`, job queued)
- [ ] Document-length sweep (`len*` jobs queued)
- [ ] Second decision model to show the finding is architectural (Phi-3.5-mini,
      job queued)

### Future work

Not planned before Dec 7:

- Query rewriting: an expensive model decomposes the predicate once and a cheap
  model executes it N times
- Laya and Kev behind the same interface
- Update rate

We are skipping the confidence cascade because it is too close to JEVDB.

### Sources

- [Choppy](https://arxiv.org/abs/2004.13012), [AttnCut](https://arxiv.org/abs/2102.12793)
- [Ranked List Truncation for LLM-based Re-Ranking](https://arxiv.org/abs/2404.18185)
- [Adaptive-k](https://arxiv.org/abs/2506.08479)
- [Heuristic Stopping Rules for TAR](https://arxiv.org/abs/2106.09871)
- [Semantic Operators (LOTUS)](https://arxiv.org/abs/2407.11418)
- [Fast LLM-Based Semantic Filtering](https://arxiv.org/abs/2606.08090)
