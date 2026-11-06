# Batched classification vs. agentic search

Proposal document: https://www.overleaf.com/project/6ac8329001f35b6ac41611bb

**Question.** For a query that asks for *all* matching documents, when is it
better to classify every document in parallel than to let an LLM agent search
iteratively?

Four execution arms behind one interface:

| Arm | What it does | Cost shape |
|---|---|---|
| `agent` | iterative LLM search, reason → search → inspect → repeat | serial in steps |
| `classify` | one forward pass per (query, document) pair, batched | `O(N)` calls, parallel |
| `ann` | embed the corpus, retrieve top-k by similarity | `O(log N)`, no model calls |
| `hybrid` | ANN for candidates, then classify those k | `O(k)` calls |

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

## Open items

- [x] Confirm the container tag and GPU flag in `chtc/midterm.sub`
- [x] Agent arm — the arm H1 and H2 are measured against
- [ ] Simple keyword and multi-condition queries; all current queries are
      semantic category queries
- [ ] Document-length sweep
- [ ] Calibrate the classifier threshold from the saved scores
- [ ] Sweep `k` for the hybrid arm; the ceiling finding says `k` should scale
      with expected gold-set size, which is a result worth plotting
- [ ] Swap exact search for HNSW and report index build and incremental
      insert cost (the maintenance axis)
- [ ] Second decision model to show the finding is architectural
