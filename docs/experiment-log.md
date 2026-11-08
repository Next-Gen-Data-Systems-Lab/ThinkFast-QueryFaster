# Experiment log

The detailed notes behind the README, as of Nov 8, 2026: every experiment's
numbers, intervals, job numbers and settings, and the checklist they were run
from. The README has the summary and the report (`report/report.tex`) has the
final numbers; this file is history and is not kept up to date.

## Status

*Last updated Nov 8, 2026. One H200; Qwen2.5-7B is both scan and agent unless
noted. Every number below has a 95% interval and a script in the "Before the
final report" checklist; `results/README.md` says which file backs which result.*

### What we found

**1. At equal time, a scan with an open decision model beats every agent,
including the strongest; a zero-shot scan only ties an agent told to keep
searching.** Each agent's seconds on each query are given to a scan reading
documents in embedding order. Recall:

| | 7B agent | Persistent 7B agent | Strongest 7B agent (chooses k, persistent) | Qwen scan, its time | Kev scan, its time |
|---|---:|---:|---:|---:|---:|
| 20 Newsgroups, 10K | 0.087 | 0.224 | 0.281 | 0.303 (+0.02, a tie) | 0.708 (+0.43) |
| SYNERGY, 26 reviews | 0.159 | 0.257 | 0.285 | 0.312 (+0.03, a tie) | 0.554 (+0.27) |
| arXiv, 100K | 0.046 | 0.176 | not run | 0.145 against the persistent agent (-0.03, a tie) | 0.69 (one query at a time, estimated) |

The "its time" columns use the strongest agent's seconds where it ran, the
persistent agent's on arXiv. Against the plain 7B agent every scan wins. Zero-shot
Qwen is the weak scan because it is badly calibrated: on topic queries the best
threshold is about 1e-8, not 0.5. With a threshold tuned on the other queries
(1e-8 on 20 Newsgroups and arXiv, 0.03 on SYNERGY) it beats the persistent agent
everywhere (+0.35, +0.15 and +0.36 recall) and the strongest agent on both
datasets it ran on (+0.32 and +0.13), all intervals above zero.

**2. Agents stop by choice, and their effort does not grow with the answer
set.** With 64 steps allowed, the 7B agent stops after about 10. On arXiv, from
10K to 1M papers, answer sets grow 100x and the agent reads 143 to 239 papers.
Telling it answer sets can be large, and refusing a stop while its searches
still find matches, raises its recall 1.6 to 3.8x, at 3 to 5 times the time.

**3. The exchange rate.** One agent step costs as much GPU time as 48 to 230
scan decisions (78 for the 7B agent on 20 Newsgroups; 11 when about 50 agents
share the GPU, which is break-even, at the agent's low recall and about 100 s
latency). A 32B agent costs more than it gains; BM25 as a second tool hurts.

**4. Which model should scan.** Kev-9B and JEV-9B beat zero-shot Qwen on short
queries (F1 on 20 Newsgroups 0.65 and 0.63 against 0.58; on logical queries 0.77
and 0.79 against 0.58) but not on long eligibility criteria (SYNERGY 0.22 and
0.25 against 0.32). BM25, embeddings and cross-encoder rerankers trail the LLM
scans almost everywhere.

**5. Choosing scan depth per query does not beat a fixed depth;** an oracle adds
at most 0.07 recall (a negative result, kept in the report).

**6. Giving the agent more freedom helps it, but not past the scan (Ayesha's
questions).** On 20 Newsgroups 10K and SYNERGY, recall and seconds per query:

| Agent | 20NG R | 20NG s | SYNERGY R | SYNERGY s |
|---|---:|---:|---:|---:|
| 7B, 300-character results, k = 20 (reference) | 0.087 | 5.0 | 0.159 | 5.3 |
| Sees 2,000 characters per result, as the scan does | 0.100 | 7.5 | 0.191 | 7.4 |
| Chooses k per search (5 to 100; it picks about 47) | 0.230 | 18.2 | 0.230 | 14.4 |
| Chooses k and persistent | 0.281 | 33.2 | 0.285 | 20.8 |

- **Snippet length was not the handicap.** Full text adds 0.01-0.03 recall and
  costs 1.5x the time.
- **Choosing k is the big lever.** The agent asks for more per search, reads far
  more (1,065 documents per SYNERGY review against 219), and recall rises 1.4 to
  2.6x. It also spends 3 to 4 times the seconds, and the equal-time scan keeps up.
- **The agent's judgment, not its search, caps it.** Counting the relevant
  documents its searches return, it keeps only 36-38% of them on 20 Newsgroups
  and 20-41% on SYNERGY. The same model keeps 31% and 32% of them when it
  scans at t = 0.5, so the agent cannot get past the zero-shot judge it is
  built on.
- **An LLM "without tools" is the list-wise scan:** 20 documents per prompt, the
  model names the matches. On 20 Newsgroups it is no faster than the pointwise
  scan (170 documents per second each, since both read every document once), has
  the same recall (0.339 against 0.310, a tie) and much lower precision (0.25
  against 0.76; F1 -0.14, worse on 19 of 20 queries). It names 733 documents per
  query for 500 relevant ones, and its 0/1 answers cannot be re-thresholded.
  SYNERGY has 7 of 26 reviews so far, with the same pattern (precision 0.035).

### Running

| Jobs | What | Why |
|---|---|---|
| 12439055-73 | The list-wise scan on SYNERGY reviews 8 to 26, one job per review (`DATASET=synergy_<review>`), prefix caching off, kept off host gpu5000 (`AVOID=gpu5000`). Four runs of all 26 in one job crashed with a CUDA illegal memory access in vLLM, all on that host: three on review 8 with caching on, one on review 9 with it off (review 8 then finished, but with recall 0.06 and 22 unparsable replies, so it is rerun too). Other workloads on that host match other hosts, so this looks like a vLLM bug with long prompts | Finishes finding 6 |
| 12437632 | Kev scanning all of arXiv 1M | How complete an exhaustive scan gets at scale |

**Next:** rehearse for Nov 10. The deck (main-result slide, a new slide on the
agent variants, the summary) and the report (abstract, `tab:agents`, new
`sec:freer`, arXiv in `sec:scale`, discussion, conclusion) now carry findings 1
and 6. When 12439055-73 land, replace the report's 7-review SYNERGY list-wise
sentence (marked TODO in a comment). The midterm-era numbers (one query at a time, without equal time) are in
the deck's backup slides and `python -m src.summarize_runs results`.

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


## Next steps

Done so far:

- [x] Confirm the container tag and GPU flag in `chtc/scan_vs_agent.sub`
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
      one query at a time and packed (`chtc/kev_scan.sub`, `src.kev_compare`)
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
      All results are in, including JEV-9B, SYNERGY, arXiv scale (Table
      `tab:scale`), the persistent and freer agents and the list-wise scan
      (`sec:freer`, Nov 8). One TODO comment: the SYNERGY list-wise sentence.
      Not yet compiled. The story changes: low recall was
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

      **20 Newsgroups 10K (job 12437628, done):** stopping was a large part of
      the agent's problem. Recall rises from 0.087 to 0.224 (best of all agents;
      32B reached 0.17), with 46 steps instead of 10, 922 documents read instead
      of 198, and 20.6 s instead of 5.0 s. It still stopped by choice on all 20
      queries (at most 132 steps), after 1.1 refused stops on average. In its
      time the scan still wins: Qwen 0.292 (+0.068, 95% interval [+0.010,
      +0.138], the narrowest margin of any agent), Kev 0.654, JEV 0.553.

      **SYNERGY and arXiv 100K (jobs 12437629-30, done):** the persistent agent
      ties the zero-shot scan at t = 0.5. SYNERGY: agent recall 0.257 (was
      0.159), Qwen in the same time 0.305, +0.048 [-0.017, +0.113]; JEV +0.074
      [-0.002, +0.154]; Kev still wins, +0.261 [+0.157, +0.350]. arXiv 100K: agent
      0.176 (was 0.046), Qwen at t = 0.5 0.145, -0.031 [-0.079, +0.020]; with the
      held-out threshold the scan wins, +0.362 [+0.302, +0.423]. So "every scan
      beats every agent" holds only for a decision model or a calibrated
      threshold once the agent is told to persist.
- [ ] Is the agent handicapped, and does a scan need one document per prompt?
      (jobs 12438047-54, TAGs agentfull, agentk, agentkdry, listwise, on 20
      Newsgroups 10K and all of SYNERGY). Raised by Ayesha. Done except the
      SYNERGY list-wise scan, which crashed after 7 reviews (CUDA illegal memory
      access in vLLM) four times on one host, and was resubmitted as one job
      per review (12439055-73). Results are finding 6 in
      Status; `python -m src.agents_compare`, `src.synergy_compare` and
      `src.listwise_compare` print them.
      - `AGENT_SNIPPET=2000` (`--agent-snippet-chars`): the agent sees 2,000
        characters of each result, the text the scan sees, instead of 300.
      - `AGENT_KMAX=100` (`--agent-k-max`): the agent picks how many results each
        search returns, 5 to 100, instead of a fixed 20; `agentkdry` adds the
        persistent stop rule.
      - `ARMS=listwise` (`src/arms/listwise.py`, `--listwise-group`): what an LLM
        without search tools amounts to once the corpus is larger than its
        context. Each prompt holds 20 documents in embedding order and the model
        names the matches. Scores are saved as 0/1, so it replays like the
        pointwise scan; `python -m src.listwise_compare` compares the two and
        both against the agents at equal time.
      - Every agent row now records `n_gold_seen` (relevant documents its
        searches returned; `tp` is how many it kept) and `n_mean_k`. Prompts that
        would overflow the context are cut from the end and counted in
        `n_truncated_prompts`. The runner sizes `max_model_len` for long prompts.
- [x] Kev at arXiv 100K (job 12437631, 66 min, packed: one pass answers all 20
      queries, 550 documents per second per query). Whole corpus at t = 0.5:
      recall 0.90, precision 0.32 (Kev errs toward yes here; best F1 0.59 at
      t = 0.9). In the agents' time it wins on 20 of 20 queries: against the 7B
      agent +0.612 [+0.535, +0.680], against the persistent agent +0.684 [+0.638,
      +0.730]. One query at a time (rate estimated at 123 documents per second
      from the 20 Newsgroups ratio of separate to packed) it still wins: +0.293
      and +0.516 at t = 0.5. The scores file is 80 MB, under GitHub's limit.
- [ ] Kev at scale (submitted Nov 7): packed scans of arXiv at 100K (job
      12437631, TAG kev9b, about 1 h) and 1M (12437632, TAG kev9b1m, about 10 h at
      the 27 documents per second packed rate seen on 20 Newsgroups). A full 1M
      scan answers all 20 queries in one pass, so it shows how complete an
      exhaustive scan gets where the agent reaches 0.008. The job builds the
      arXiv sample itself; `run_kev_scan.sh` adds pyarrow to Kev's environment for it.
- [ ] Job files were renamed on Nov 7 (`midterm.sub` to `scan_vs_agent.sub`,
      `kev.sub` to `kev_scan.sub`, `scan.sub` to `jev_rerank_scan.sub`, each with its
      `run_*.sh`). `jev.sub` was removed: `jev_rerank_scan.sub` runs the same pinned
      environment and `src.jev_scan`, so it reproduces that job's results exactly. The old copies stay on the
      access point until jobs 12437629-32 have started, because a queued job reads
      its script when it starts; delete them after that.

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
