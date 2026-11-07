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
  Batched agents only tie it, at the agent's low recall.

## Running on CHTC

`ssh chtc` needs an interactive login. Claude reuses that connection, so it has to be open.

```bash
bash chtc/upload.sh                 # copies to ~/ThinkFast-QueryFaster on ap2001
ssh chtc
cd ThinkFast-QueryFaster
condor_submit BATCHES=32 TAG=name GPU='H200$' LENGTH=short chtc/midterm.sub
```

Submit-time options: `DATASET MODEL SIZES BATCHES K TAG ARMS MAX_QUERIES
AGENT_STEPS AGENT_CONCURRENCY AGENT_REPLICAS MAX_CHARS GPU LENGTH`.

## Things that have broken before

- **Comments:** `condor_submit` keeps a trailing `#` comment as part of the
  value. Comments go on their own lines.
- **`transfer_input_files`:** `src/` copies only the contents. Use `src`.
- **vLLM image:** its entrypoint is the API server, so
  `docker_override_entrypoint = true` is required.
- **`CUDA_VISIBLE_DEVICES`:** HTCondor sets a UUID and vLLM 0.6.3 needs an
  integer. `run.sh` converts it.
- **Uploading:** HTCondor reads `src/` when a job starts, not when it is
  submitted. Uploading over a queued job changes what it runs.
- **`TAG`:** parallel jobs need different tags or their results overwrite.
- **GPUs:** `H200$` leaves out the H200 NVL. Blackwell GPUs are too new for vLLM
  0.6.3, and `midterm.sub` keeps to compute capability 8.0-9.0.
- **SSH watchers:** a dropped connection also empties `condor_q` output. Check the
  ssh exit code, or a dead connection looks like finished jobs.
- **Sliding window:** vLLM refuses prefix caching for models that declare one, such
  as Phi-3.5. `classify.py` passes `disable_sliding_window=True`.
- **`getpwuid(): uid not found`:** the container has no user for HTCondor's uid,
  which kills torch inductor. `run_kev.sh` sets `USER`, `LOGNAME` and the
  inductor and Triton cache dirs.
- **SYNERGY data:** build it locally with `python -m src.synergy_build` (needs
  network), then `chtc/upload.sh` copies `data/`. Datasets accept comma lists and
  `synergy` for all 26; a size above the pool size means the whole pool.
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
- Kev job: `condor_submit GPU='H200$' TAG=kev9b DATASET=... SIZES=... chtc/kev.sub`
  (also `MODES MAX_QUERIES LENGTH`). `python -m src.kev_compare` compares offline.
- Scan jobs in the Python 3.12 environment: `condor_submit MODULE=jev_scan|rerank_scan
  MODEL=bge|minilm|- TAG=... DATASET=... SIZES=... chtc/scan.sub`. It transfers
  `data/`. `chtc/jev.sub` is the older JEV-only version. Pinned HF revision; stock transformers Qwen3.5, so no
  repo code runs. Compare with `python -m src.kev_compare --model jev`.
- `src/kev_scan.py` runs only in Kev's environment through `chtc/run_kev.sh`, pinned
  to a commit whose code was read. Kev's code has not been run on this Mac.
- `SearchAgent._steps` is a generator, so `find` and `run_many` share one loop.
  A batched row's `wall_s` is latency; cost is `batch_wall_s` over calls.
- The local test command is in the README quick start. There is no system
  `timeout` command on this Mac.

## Progress

The final report is `report/report.tex` with `refs.bib` and `figs/`, compiled on
Overleaf (no LaTeX on this Mac or on CHTC). Keep its numbers in step with README.

Results and status are in the Status section of `README.md`. The midterm deck is
https://claude.ai/artifact/E1R5FHuvJHniGchsHttgeE (10 slides plus a backup, with
timed speaker notes).
