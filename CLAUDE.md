# ThinkFast-QueryFaster

CS 839 project: for a query that asks for *all* matching documents, when is it
better to classify every document in parallel than to let an LLM agent search?

- **People:** Ayesha, Nabanita and Nida present together. Nabanita does most of
  the work, so nothing should wait on the others.
- **Dates:** midterm presentation (8 minutes) around Nov 10, 2026. Final report
  due Dec 7, 2026.
- **Source of truth:** the plan (`Plan - Open-Source Jev Alternative vs. Agentic
  Search.md`) over the older proposal. Update rate is future work.

Keep this file and `README.md` updated with progress as work happens.

## Decisions

- Main dataset: 20 Newsgroups, because every document is labelled and so each
  answer set is complete. Second dataset: NFCorpus, 50 queries.
- Main model: Qwen2.5-7B-Instruct for both classifier and agent, so the
  comparison isolates architecture. Second model: Qwen2.5-1.5B.
- Laya and Kev are future work.
- Contribution before Dec 7: the exchange rate (one agent step buys N
  classification decisions) and adaptive k. Query rewriting is future work. Skip
  the confidence cascade, which is too close to JEVDB.
- Position adaptive k as cost-based access-path selection, and check the ranked
  list truncation literature first.

## Running on CHTC

`ssh chtc` needs an interactive login, and port 22 may need the campus network
or VPN. Claude reuses that connection, so it has to be open.

```bash
bash chtc/upload.sh                 # copies to ~/ThinkFast-QueryFaster on ap2001
ssh chtc
cd ThinkFast-QueryFaster
condor_submit BATCHES=32 TAG=name GPU='H200$' LENGTH=short chtc/midterm.sub
```

Submit-time options: `DATASET MODEL SIZES BATCHES K TAG ARMS MAX_QUERIES
AGENT_STEPS GPU LENGTH`.

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
- **Token budget:** `max_num_batched_tokens` defaults to 2048, which limits
  prefill-only work. The runner sets batch size x 512. Don't size it from
  `max_model_len`: at batch size 256 that means profiling a 524K-token step.

## Code notes

- All arms return `ArmResult`. `metrics.flatten()` makes a JSONL row.
- The resume key must match the row: only `ann` and `hybrid` store `k`.
- `--scores-out` saves per-document scores, so thresholds and hybrid at other
  values of k can be evaluated without another GPU run.
- The local test command is in the README quick start. There is no system
  `timeout` command on this Mac.

## Progress

Results and status are in the Status section of `README.md`. The midterm deck is
https://claude.ai/artifact/E1R5FHuvJHniGchsHttgeE (10 slides plus a backup, with
timed speaker notes). The original Google Slides deck can't be edited from here
because no Slides editor connector is available.
