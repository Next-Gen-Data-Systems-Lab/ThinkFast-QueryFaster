# ThinkFast-QueryFaster

CS 839 project: for a query that asks for *all* matching documents, when is it
better to classify every document in parallel than to let an LLM agent search?

- **Dates:** midterm presentation (8 minutes) on Nov 10, 2026. Final report
  due Dec 9, 2026.
- **Source of truth:** the plan (`Plan - Open-Source Jev Alternative vs. Agentic
  Search.md`) over the older proposal. Update rate is future work.

Keep this file and `README.md` updated with progress as work happens.

## Decisions

- Main dataset: 20 Newsgroups, because every document is labelled and so each
  answer set is complete. Second dataset: NFCorpus, 50 queries.
- Main model: Qwen2.5-7B-Instruct for both classifier and agent, so the
  comparison isolates architecture. Second model: Qwen2.5-1.5B.
- Kev-9B runs as a second scan arm alongside Qwen, not instead of it. Done on
  all datasets: it fixes most of the scan's recall and threshold problems, so the
  report treats low recall as a zero-shot Qwen limit. JEV-9B only if it adds
  something. Laya is skipped.
- Aim for a 4-page short paper (decided Nov 7). Main benchmark: SYNERGY (26
  systematic reviews, one corpus and one query each, never pooled). arXiv
  metadata for scale. 20 Newsgroups stays as the development set. RCV1 needs a
  signed NIST agreement, so it is not planned.
- Contribution before Dec 7: the exchange rate (one agent step buys N
  classification decisions) and adaptive k. Query rewriting is future work. Skip
  the confidence cascade, which is too close to JEVDB.
- Position adaptive k as cost-based access-path selection, and check the ranked
  list truncation literature first. Done: the test came back negative, so report
  it as a negative result with the oracle bound. The strongest finding is the
  scan in embedding order beating the agent at equal time, one query at a time.
  Batched agents only tie it, at the agent's low recall. On arXiv up to 1M the
  agent reads about 200 papers whatever N is, so the scan's lead in papers found
  grows with N (done Nov 7; report Table `tab:scale`).
- Ayesha's agent variants (done Nov 8, report `sec:freer`): longer snippets barely
  help; choosing k is the strongest agent, and with the persistent stop rule it
  ties the zero-shot Qwen scan, while Kev and a tuned threshold still win. The
  agent keeps only about a third of the relevant documents it sees, the same rate
  as the Qwen scan at t = 0.5. The list-wise scan ("no tools") is no faster than
  pointwise and loses precision.

## Running on CHTC

`ssh chtc` needs an interactive login. Claude reuses that connection, so it has to be open.

```bash
bash chtc/upload.sh                 # copies to ~/ThinkFast-QueryFaster on ap2001
ssh chtc
cd ThinkFast-QueryFaster
condor_submit BATCHES=32 TAG=name GPU='H200$' LENGTH=short chtc/scan_vs_agent.sub
```

Submit-time options: `DATASET MODEL SIZES BATCHES K TAG ARMS MAX_QUERIES
AGENT_STEPS AGENT_CONCURRENCY AGENT_REPLICAS AGENT_TOOLS AGENT_STOP AGENT_SNIPPET AGENT_KMAX LISTWISE_GROUP PREFIX_CACHE MAX_CHARS GPU AVOID LENGTH
MEMORY DISK`. `AVOID` is a regular expression on machine names to keep a job off. Qwen2.5-32B needs `DISK=200GB MEMORY=96GB`. The default `GPU` is
`A100|H200`, so pass `GPU='H200$'` for timings comparable with other runs.

## Things that have broken before

- **Comments:** `condor_submit` keeps a trailing `#` comment as part of the
  value. Comments go on their own lines.
- **`transfer_input_files`:** `src/` copies only the contents. Use `src`.
- **vLLM image:** its entrypoint is the API server, so
  `docker_override_entrypoint = true` is required.
- **`CUDA_VISIBLE_DEVICES`:** HTCondor sets a UUID and vLLM 0.6.3 needs an
  integer. `run_scan_vs_agent.sh` converts it.
- **Uploading:** HTCondor reads `src/` when a job starts, not when it is
  submitted. Uploading over a queued job changes what it runs.
- **`TAG`:** parallel jobs need different tags or their results overwrite.
- **GPUs:** `H200$` leaves out the H200 NVL. Blackwell GPUs are too new for vLLM
  0.6.3, and `scan_vs_agent.sub` keeps to compute capability 8.0-9.0.
- **SSH watchers:** a dropped connection also empties `condor_q` output. Check the
  ssh exit code, or a dead connection looks like finished jobs.
- **Sliding window:** vLLM refuses prefix caching for models that declare one, such
  as Phi-3.5. `classify.py` passes `disable_sliding_window=True`.
- **`getpwuid(): uid not found`:** the container has no user for HTCondor's uid,
  which kills torch inductor. `run_kev_scan.sh` sets `USER`, `LOGNAME` and the
  inductor and Triton cache dirs.
- **SYNERGY data:** build it locally with `python -m src.synergy_build` (needs
  network), then `chtc/upload.sh` copies `data/`. Datasets accept comma lists and
  `synergy` for all 26; a size above the pool size means the whole pool.
- **arXiv at 1M:** a full LLM scan is about 117 minutes per query, so jobs
  classify each query's top 20K by embedding (`ARMS=hybrid K=20000`) and replay
  equal-time plans from those scores (`python -m src.scale_compare`). `upload.sh`
  leaves `data/arxiv` out.
- **vLLM crash on list-wise SYNERGY:** with prefix caching on, vLLM 0.6.3 hit a CUDA
  illegal memory access on review 8 (Jeyaraman_2020) three times, and with
  `PREFIX_CACHE=off` on review 9. All four ran on host gpu5000 (aliases
  dbrundage-/zliu-chtcgpu5000); other work there is fine. Long list-wise jobs now
  run one review per job with `AVOID=gpu5000`.
- **Token budget:** `max_num_batched_tokens` defaults to 2048, which limits
  prefill-only work. The runner sets batch size x 512. Don't size it from
  `max_model_len`: at batch size 256 that means profiling a 524K-token step.

## Code notes

- All arms return `ArmResult`. `metrics.flatten()` makes a JSONL row.
- The resume key must match the row: only `ann` and `hybrid` store `k`.
- `--scores-out` saves per-document scores, so thresholds and hybrid at other
  values of k can be evaluated without another GPU run.
- Hybrid plans replay offline: `src.embed_ranks` saves the ranking (needs
  sentence-transformers), then `src.access_paths` and `src.crossover` use the
  saved scores. Compare rules with fixed k at equal mean cost, never by eye.
- Kev job: `condor_submit GPU='H200$' TAG=kev9b DATASET=... SIZES=... chtc/kev_scan.sub`
  (also `MODES MAX_QUERIES LENGTH`). `python -m src.kev_compare` compares offline.
- Scan jobs in the Python 3.12 environment: `condor_submit MODULE=jev_scan|rerank_scan
  MODEL=bge|minilm|- TAG=... DATASET=... SIZES=... chtc/jev_rerank_scan.sub`. It transfers
  `data/`. The 20NG, logic and NFCorpus JEV results came from the removed `jev.sub`
  (jobs 12437150-53), which ran the same environment and module. Pinned HF revision; stock transformers Qwen3.5, so no
  repo code runs. Compare with `python -m src.kev_compare --model jev`.
- `src/kev_scan.py` runs only in Kev's environment through `chtc/run_kev_scan.sh`, pinned
  to a commit whose code was read. Kev's code has not been run on this Mac.
- `SearchAgent._steps` is a generator, so `find` and `run_many` share one loop.
  A batched row's `wall_s` is latency; cost is `batch_wall_s` over calls.
- The local test command is under "Test offline" in the README. There is no system
  `timeout` command on this Mac.

## Progress

The short paper is `paper/paper.tex` (ACM sigconf; figure from
`python -m src.paper_figs`). The final report is `report/report.tex` with `refs.bib` and `figs/`, compiled on
Overleaf (no LaTeX on this Mac or on CHTC). Keep its numbers in step with README.

`README.md` is a short plain-language overview (what we did, what we found,
status); keep it short. Detailed numbers go in the report, and the old lab notes
are archived in `docs/experiment-log.md` (not updated). `results/README.md` maps
each results file to its experiment, script and report table: add a row when a new
TAG lands. Teammates read the repo too, so keep stray test outputs out of `results/`. The midterm deck is
https://claude.ai/artifact/E1R5FHuvJHniGchsHttgeE (10 slides plus 2 backups, with
timed speaker notes, about 7 minutes; updated Nov 8 with the corrected headline,
a decision-model scan beats every agent and zero-shot Qwen ties the strongest,
and a slide on the freer agents). Its figure comes from
`python -m src.paper_figs --slide`, which shows four agents (`SLIDE_AGENTS`).
