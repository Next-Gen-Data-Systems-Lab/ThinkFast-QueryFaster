# Results index

Every number in the report and the deck comes from a file in this folder. Most
of them are replayed by a script without a GPU. This page says which file holds
what, and which script reads it.

## How the files are named

`<dataset>_<tag>.jsonl` holds one row per (arm, corpus size, query) from one job.
`<tag>` is the `TAG` it was submitted with. Rows carry `recall`, `precision`,
`f1`, timings (`t_total`, `wall_s` for agents) and the run's settings (model,
GPU, `fingerprint` of the corpus sample).

`<dataset>_<tag>.scores.jsonl` holds the scan's score for every document, per
query (`scores`: document id to P(yes), and `gold`). Thresholds, scan depths and
equal-time comparisons are replayed from these.

`<dataset>_embed_<N>.npz` is the embedding ranking from `python -m
src.embed_ranks`: query-by-document similarities, document and query ids, and the
corpus fingerprint, which must match the rows it is replayed with.

`*.json` files are summaries written by the analysis scripts. Rerun the script
to refresh one.

## The files

All runs are on one H200 unless the file's `gpu` field says otherwise.

| Files | What they hold | Read by | Report |
|---|---|---|---|
| `20newsgroups_sweep_bs{1,4,8,16,32,64,256}` | Qwen2.5-7B scan, ANN and hybrid at 1K, 5K and 10K, one batch size each. **bs32 is the reference scan** used everywhere. The other batch sizes give throughput; their scores differ from bs32 only by GPU numerics (at most 104 of 332,000 decisions flip). | `summarize_runs`, `access_paths`, `crossover`, `agents_compare` | `tab:headline`, `tab:batch`, `sec:adaptive` |
| `20newsgroups_agent`, `_agent_s32`, `_agent_s64` | 7B agent with 8, 32 and 64 steps allowed. **s64 is the reference agent.** | `summarize_runs`, `crossover`, `agents_compare` | `tab:headline`, deck "stops on its own" |
| `20newsgroups_agent7b2t`, `_agent32b1t`, `_agent32b2t` | Stronger agents: 7B with BM25 added, 32B, 32B with BM25. 64 steps. | `agents_compare` | `tab:agents`, `fig:time` |
| `{20newsgroups,synergy,arxiv}_agentdry` | The persistent 7B agent (`AGENT_STOP=dry`, up to 256 steps): told answer sets can be large, and allowed to stop only after two searches in a row find nothing. | `agents_compare`, `synergy_compare`, `scale_compare` | `tab:agents`, `sec:freer`, `sec:scale` |
| `20newsgroups_agentbatch`, `_agentrep`, `nfcorpus_agentbatch` | Many agents sharing the GPU (concurrency 1 to 256; `agentrep` repeats queries to fill large batches). | `crossover` | `tab:batch`, `fig:exchange` |
| `20newsgroups_small`, `_phi` | Qwen2.5-1.5B and Phi-3.5 as scan and agent. | `summarize_runs` | discussion |
| `20newsgroups_len{250,500,1000,2000,4000}` | Scan speed and quality when documents are cut to N characters. | `summarize_runs` | methods |
| `20newsgroups_logic_logic` | Qwen on the logic set (AND, OR, NOT of topics). | `access_paths`, `kev_compare`, `baselines` | `tab:shape` |
| `20newsgroups_kev9b_*`, `_kev9bfull_*` | Kev-9B scan at 1K (`kev9b`) and 5K/10K (`kev9bfull`). `packed` answers all 20 queries in one pass; `separate` is one query at a time. | `kev_compare`, `agents_compare` | `tab:decision`, `tab:kevtime` |
| `20newsgroups_jev9b_separate`, `_jev9bfull_separate` | JEV-9B scan at 1K and 10K (jobs 12437150-53, from the since-removed `jev.sub`; `chtc/jev_rerank_scan.sub` with `MODULE=jev_scan` reproduces them). | `kev_compare --model jev`, `agents_compare` | `tab:decision`, `tab:kevtime` |
| `20newsgroups_jev9bslow_separate` | JEV-9B without its fast kernels: a check that they change only 2 of 20,000 decisions. | none (kept as evidence) | none |
| `20newsgroups_logic_{kev9b_packed,jev9b_separate}` | Decision models on the logic set. | `kev_compare`, `baselines` | `tab:decision` |
| `*_bge_separate`, `*_minilm_separate` | Cross-encoder rerankers (bge-reranker-v2-m3, MiniLM) as the scan, on 20NG, logic, NFCorpus and every SYNERGY review. | `baselines` | `tab:scorers` |
| `nfcorpus_nf`, `nfcorpus_{kev9b,jev9b}_*` | NFCorpus, 50 queries: all four arms with Qwen, plus the decision models. | `access_paths`, `crossover`, `baselines` | `tab:transfer`, `tab:scorers` |
| `synergy_qwen` | SYNERGY, all 26 reviews in one file: Qwen scan, ANN, hybrid and the 7B agent. | `synergy_compare`, `baselines` | `sec:synergy`, `tab:agents` |
| `synergy_agent7b2t`, `_agent32b1t`, `_agent32b2t` | Stronger agents on all 26 reviews. | `synergy_compare` | `tab:agents`, `fig:time` |
| `synergy_<review>_{kev9b_packed,jev9b_separate,bge_separate,minilm_separate}` | Decision models and rerankers, one file per review. | `synergy_compare`, `baselines` | `sec:synergy`, `tab:scorers` |
| `arxiv_scale` | arXiv at 10K, 100K and 1M: Qwen classifies each query's top 20K by embedding (`hybrid`, k = 20,000), plus the 7B agent. | `scale_compare`, `paper_figs` | `tab:scale`, `fig:time` |
| `{20newsgroups,synergy}_agentfull` | 7B agent seeing 2,000 characters per result instead of 300 (`AGENT_SNIPPET=2000`). | `agents_compare`, `synergy_compare` | `tab:agents`, `sec:freer` |
| `{20newsgroups,synergy}_agentk`, `_agentkdry` | 7B agent choosing k per search, 5 to 100 (`AGENT_KMAX=100`); `agentkdry` is also persistent, and is the strongest agent. Rows carry `n_gold_seen` and `n_mean_k`. | `agents_compare`, `synergy_compare` | `tab:agents`, `sec:freer` |
| `{20newsgroups,synergy}_listwise` | List-wise scan: 20 documents per prompt, the model names the matches; scores are 0/1. SYNERGY has 7 reviews until jobs 12439055-73 finish. | `listwise_compare` | `sec:freer` |
| `arxiv_kev9b_packed` | Kev scanning all of arXiv 100K, every query in one pass (job 12437631). | `paper_figs` (arXiv panel) | `sec:scale`, `fig:time` |
| `*_embed_<N>.npz` | Embedding rankings for 20NG, logic, NFCorpus, each SYNERGY review and arXiv. The arXiv 1M file took 2 hours to compute on a laptop. | everything that replays | |

### Pending

| Files | What they will hold |
|---|---|
| `synergy_listwise` | The rest of the 26 reviews (jobs 12439055-73, one per review, `synergy_<review>_listwise`, after four vLLM crashes on one host; merge into `synergy_listwise`). |
| `arxiv_kev9b1m_packed` | Kev scanning all of arXiv 1M. Its full scores file is about 1 GB, over GitHub's 100 MB limit, so it is not committed; a compact copy will be. |

## Summaries

| File | Written by |
|---|---|
| `access_paths.json` | `python -m src.access_paths` (fixed and adaptive k, oracle, thresholds) |
| `crossover.json` | `python -m src.crossover` (exchange rates, equal time on 20NG) |
| `kev_compare.json`, `jev_compare.json` | `python -m src.kev_compare [--model jev]` |
| `baselines.json` | `python -m src.baselines` (all scorers, `tab:scorers`) |
| `agents_compare.json` | `python -m src.agents_compare` (every agent against every scan, 20NG) |
| `synergy_compare.json` | `python -m src.synergy_compare` |
| `listwise_compare.json` | `python -m src.listwise_compare` (list-wise against pointwise scan, and both against the agents) |
| `scale_compare.json` | `python -m src.scale_compare` (arXiv) |
| `index_costs.json` | `python -m src.index_costs` (HNSW against exact search) |

`python -m src.summarize_runs results` prints the numbers in the deck and the
report's first tables.
