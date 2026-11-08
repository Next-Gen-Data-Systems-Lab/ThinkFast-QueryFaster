# Scan or search?

**The question.** Some requests need *every* matching document: all studies for
a medical review, all emails about a deal. An LLM can answer them two ways:

- **Scan:** ask "does this document match?" of every document. It is slow on a
  big collection, but it never skips anything.
- **Search:** let an LLM agent type searches, read the results, keep the good
  ones and decide when it is done. It is fast, but it can miss things.

Which one finds more of the answers for the same cost?

## What we did

We ran both on collections where every right answer is known:

| Collection | What it is | Size |
|---|---|---|
| 20 Newsgroups | forum posts with topic labels; our development set | 10K posts, 20 queries |
| SYNERGY | 26 real medical systematic reviews; the query is the review's eligibility criteria | 257 to 48K papers each |
| arXiv | paper abstracts with category labels, to test scale | 10K to 1M papers, 20 queries |

The scan and the agent use the same model (Qwen2.5-7B) on the same GPU (one
H200), so only the strategy differs. We also scanned with two open models
trained for yes/no decisions (Kev-9B and JEV-9B), and made the agent as strong
as we could: a bigger model (32B), keyword search, a rule against stopping
early, full documents instead of snippets, and control over how many results
each search returns.

To compare fairly, we gave the scan exactly the agent's time on each query and
let it read documents in order of embedding similarity, then counted how many
right answers each found (recall).

## What we found

1. **A scan with a decision model beats every agent.** In the agent's own
   time, Kev finds far more of the answers on all three collections.
2. **Agents quit early.** Even with thousands of answers, the agent reads a few
   hundred documents and stops by choice, never at its step limit.
3. **A stronger agent helps, but not enough.** Telling it to keep going and
   letting it choose how many results to read raises its recall 2 to 3x. That
   ties a plain Qwen scan, but Kev still wins by a wide margin.
4. **The agent is only as good as its judgment.** Of the relevant documents it
   sees, it keeps about 1 in 3, the same share as the Qwen scan. More
   searching cannot fix that.
5. **One agent step costs as much as 48 to 230 scan decisions.** Running many
   agents together brings that down to break-even, but at about 100 s per query.
6. **Smaller results:** decision models win on short queries but not on long
   medical criteria; cheap scorers (BM25, embeddings, rerankers) trail every LLM
   scan; choosing scan depth per query does not beat a fixed depth; putting 20
   documents in one prompt is no faster and much less precise.

Recall, with the scan given the agent's time on each query:

| | Plain 7B agent | Strongest agent | Qwen scan | Kev scan |
|---|---:|---:|---:|---:|
| 20 Newsgroups, 10K | 0.09 | 0.28 | 0.30 (a tie) | 0.71 |
| SYNERGY, 26 reviews | 0.16 | 0.29 | 0.31 (a tie) | 0.55 |
| arXiv, 100K | 0.05 | 0.18 (persistent only) | 0.15 (a tie) | about 0.69 |

The Qwen and Kev columns use the strongest agent's time (the persistent agent's
on arXiv). With a threshold tuned on other queries, the Qwen scan also beats
every agent.

In one line: when you need every answer, check every document with a model
built for yes/no decisions, because an agent cannot tell what it missed.

## Where things are

- **Report:** `report/report.tex`, compiled on Overleaf. It has every number,
  with 95% intervals.
- **Midterm deck:** https://claude.ai/artifact/E1R5FHuvJHniGchsHttgeE
- **Results:** `results/`. `results/README.md` says which file backs which
  table and which script reads it.
- **Detailed notes:** `docs/experiment-log.md` has every experiment's numbers,
  settings and job numbers, as of Nov 8.
- **Short paper:** `paper/` (paused).

## Status

- **Running:** the list-wise scan on SYNERGY reviews 8 to 26 (one job per review,
  after a vLLM crash on long prompts), and Kev scanning all of arXiv at 1M.
- **To do:** rehearse the talk; when the list-wise jobs finish, fill in the one
  sentence marked TODO in the report; compile the report on Overleaf.
- **Not planned before Dec 7:** query rewriting, hosted Jev, corpora that change
  over time.

## Code

```
src/
  data.py, metrics.py   collections, queries, scoring
  runner.py             runs the scan, agent and other arms on a GPU
  arms/                 classify.py (scan), agent.py, listwise.py, ann.py, keyword.py
  kev_scan.py, jev_scan.py, rerank_scan.py   scans with other models
  *_compare.py, access_paths.py, crossover.py, baselines.py
                        analysis: replay saved scores, no GPU needed
  paper_figs.py         the recall-against-time figure (report and deck)
chtc/                   job files for the CHTC cluster (each .sub has a run_*.sh)
report/  paper/  docs/  results/
```

Every arm returns an `ArmResult` (the documents it picked, scores, timings,
counts), and `metrics.flatten()` turns it into one JSONL row.

### Test offline (no GPU, about ten seconds)

```bash
pip install numpy scikit-learn pandas matplotlib
python -m src.runner --dataset synthetic --arms classify,ann,hybrid,listwise \
    --model mock --embed-backend tfidf --corpus-sizes 500,1000 --batch-sizes 32 \
    --out /tmp/smoke.jsonl
```

Run this before every cluster job; it catches typos before a model loads.

### Run on CHTC

```bash
bash chtc/upload.sh        # copy the project to the access point
ssh chtc
cd ThinkFast-QueryFaster
condor_submit DATASET=20newsgroups ARMS=agent BATCHES=32 TAG=myrun GPU='H200$' LENGTH=short chtc/scan_vs_agent.sub
```

- Give each job its own `TAG`, or results overwrite each other.
- A queued job reads `src/` when it starts, so don't upload over a queued job.
- `GPU='H200$'` keeps timings comparable with the other runs.
- All submit options, and the problems we have hit, are in `CLAUDE.md`.

### Data

20 Newsgroups and NFCorpus download on first use. SYNERGY is built locally with
`python -m src.synergy_build` (needs network and a free `OPENALEX_API_KEY` in
`.env`); arXiv builds itself on first use into `data/arxiv/`. 20 Newsgroups is
loaded without headers, which would give the topic away.
