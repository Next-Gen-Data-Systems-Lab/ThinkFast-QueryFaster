"""Cheap baselines against the LLM scans: does a find-all query need an LLM at all?

Every scorer gives each (query, document) pair a number; the question is how
well that number separates the answer set. Compared on the same documents:

  bm25        Okapi BM25 on the query text (k1 = 1.2, b = 0.75), no model
  embed       cosine similarity, all-MiniLM-L6-v2 (the ANN arm's index)
  qwen, kev, jev, ...   the LLM scans' P(yes), from their saved score files

Per query:
  ap          average precision over the whole ranking
  best_f1     the best F1 any cutoff reaches (an oracle)
  f1          F1 at a cutoff chosen on the other queries (leave one query out).
              LLM scores are probabilities, so the cutoff is a threshold on
              them; BM25 and cosine are not comparable across queries, so they
              are z-scored within the query first.
  wss95       work saved over sampling at 95% recall, the screening metric:
              the share of documents a reviewer can skip, reading in score
              order, and still find 95% of the answers, minus 0.05

Differences in F1 against the reference scorer are paired over queries with a
bootstrap 95% interval. Needs the saved embedding rankings (src.embed_ranks).

Usage: python -m src.baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import access_paths as AP
from . import data as D
from .embed_ranks import path_for

K1, B = 1.2, 0.75
Z_GRID = list(np.arange(-1.0, 8.01, 0.25))

# (label, dataset names, size, {scorer: score file}, max queries); "{d}" is the
# dataset name. NFCorpus scans covered its first 50 queries.
SETTINGS = [
    ("20NG 1K", "20newsgroups", 1000, {
        "qwen": "results/20newsgroups_sweep_bs32.scores.jsonl",
        "kev": "results/20newsgroups_kev9b_packed.scores.jsonl",
        "jev": "results/20newsgroups_jev9b_separate.scores.jsonl"}),
    ("20NG 10K", "20newsgroups", 10000, {
        "qwen": "results/20newsgroups_sweep_bs32.scores.jsonl",
        "kev": "results/20newsgroups_kev9bfull_packed.scores.jsonl",
        "jev": "results/20newsgroups_jev9bfull_separate.scores.jsonl"}),
    ("Logic 10K", "20newsgroups_logic", 10000, {
        "qwen": "results/20newsgroups_logic_logic.scores.jsonl",
        "kev": "results/20newsgroups_logic_kev9b_packed.scores.jsonl",
        "jev": "results/20newsgroups_logic_jev9b_separate.scores.jsonl"}),
    ("NFCorpus", "nfcorpus", 3633, {
        "qwen": "results/nfcorpus_nf.scores.jsonl",
        "kev": "results/nfcorpus_kev9b_packed.scores.jsonl",
        "jev": "results/nfcorpus_jev9b_separate.scores.jsonl"}, 50),
    ("SYNERGY", "synergy", 100000, {
        "qwen": "results/synergy_qwen.scores.jsonl",
        "kev": "results/{d}_kev9b_packed.scores.jsonl",
        "jev": "results/{d}_jev9b_separate.scores.jsonl"}),
]
RAW = {"bm25", "embed"}      # not probabilities: z-score within the query


def tokenizer():
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(stop_words="english", lowercase=True)


def bm25(corpus: D.Corpus, queries: list[D.Query]) -> np.ndarray:
    vec = tokenizer()
    tf = vec.fit_transform([d.text for d in corpus.docs]).tocsc().astype(np.float64)
    n = tf.shape[0]
    length = np.asarray(tf.sum(axis=1)).ravel()
    df = np.diff(tf.indptr)
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    norm = K1 * (1 - B + B * length / max(length.mean(), 1e-9))
    out = np.zeros((len(queries), n))
    analyse = vec.build_analyzer()
    vocab = vec.vocabulary_
    for i, q in enumerate(queries):
        for term in set(analyse(q.text)):
            j = vocab.get(term)
            if j is None:
                continue
            col = tf[:, j]
            rows, f = col.indices, col.data
            out[i, rows] += idf[j] * f * (K1 + 1) / (f + norm[rows])
    return out


def embed(corpus: D.Corpus, queries: list[D.Query], results: Path) -> np.ndarray | None:
    path = path_for(results, corpus.name, len(corpus.docs))
    if not path.exists():
        return None
    z = np.load(path)
    assert str(z["fingerprint"]) == D.fingerprint(corpus), f"stale {path}"
    col = {d: i for i, d in enumerate(z["doc_ids"].tolist())}
    row = {q: i for i, q in enumerate(z["query_ids"].tolist())}
    order = [col[d.id] for d in corpus.docs]
    return np.stack([z["sims"][row[q.id]][order] for q in queries])


def saved(path: Path, corpus: D.Corpus, queries: list[D.Query]) -> np.ndarray | None:
    """Saved P(yes) for these documents, or None if any query is missing."""
    if not path.exists():
        return None
    rows = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if (r["arm"] == "classify" and r["corpus_size"] == len(corpus.docs)
                    and r.get("dataset", corpus.name) == corpus.name):
                rows[r["query_id"]] = r["scores"]
    if not all(q.id in rows for q in queries):
        return None
    return np.array([[rows[q.id][d.id] for d in corpus.docs] for q in queries])


def per_query(scores: np.ndarray, gold: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    hits = gold[order].astype(float)
    tp = np.cumsum(hits)
    n_gold = gold.sum()
    ranks = np.arange(1, len(hits) + 1)
    precision, recall = tp / ranks, tp / n_gold
    f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    depth95 = int(np.searchsorted(recall, 0.95 - 1e-12) + 1)
    return {"ap": float((precision * hits).sum() / n_gold),
            "best_f1": float(f1.max()),
            "wss95": float((len(hits) - depth95) / len(hits) - 0.05),
            "recall@gold": float(tp[int(n_gold) - 1] / n_gold)}


def f1_at(scores: np.ndarray, gold: np.ndarray, t: float) -> float:
    pred = scores >= t
    tp = (pred & gold).sum()
    if tp == 0:
        return 0.0
    p, r = tp / pred.sum(), tp / gold.sum()
    return float(2 * p * r / (p + r))


def held_out_f1(scores: list[np.ndarray], golds: list[np.ndarray], raw: bool) -> np.ndarray:
    if raw:
        scores = [(s - s.mean()) / (s.std() + 1e-12) for s in scores]
    grid = Z_GRID if raw else AP.THRESHOLDS
    table = np.array([[f1_at(s, g, t) for t in grid] for s, g in zip(scores, golds)])
    out = []
    for i in range(len(scores)):
        best = int(np.argmax(np.delete(table, i, axis=0).mean(axis=0)))
        out.append(table[i, best])
    return np.array(out)


def evaluate(label: str, names: str, n: int, files: dict, results: Path,
             max_queries: int = 0) -> dict:
    per: dict[str, list] = {}       # scorer -> list of (scores, gold) per query
    for sub in D.subsets(names, str(n)):
        qs = sub.queries[:max_queries] if max_queries else sub.queries
        golds = [np.isin([d.id for d in sub.docs], list(q.gold)) for q in qs]
        found = {"bm25": bm25(sub, qs), "embed": embed(sub, qs, results)}
        for name, pattern in files.items():
            found[name] = saved(Path(pattern.format(d=sub.name)), sub, qs)
        for name, mat in found.items():
            per.setdefault(name, [])
            if mat is None:
                per[name].append(None)
            else:
                per[name] += list(zip(mat, golds))
    # Keep only scorers that cover every query, so all compare on the same set.
    complete = {k: v for k, v in per.items() if v and all(x is not None for x in v)}
    out = {"setting": label, "queries": len(next(iter(complete.values()))), "scorers": {}}
    held = {}
    for name, pairs in complete.items():
        rows = [per_query(s, g) for s, g in pairs]
        held[name] = held_out_f1([s for s, _ in pairs], [g for _, g in pairs], name in RAW)
        out["scorers"][name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out["scorers"][name]["f1"] = float(held[name].mean())
    ref = "qwen" if "qwen" in held else None
    if ref:
        for name in held:
            if name != ref:
                d, lo, hi = AP.paired(held[name], held[ref])
                out["scorers"][name]["d_f1_vs_qwen"] = [d, lo, hi]
    return out


def show(r: dict) -> None:
    print(f"\n== {r['setting']} ({r['queries']} queries)")
    print(f"  {'scorer':6s} {'AP':>6s} {'bestF1':>7s} {'F1':>6s} {'WSS@95':>7s} {'R@|gold|':>8s}  F1 vs qwen")
    for name, m in r["scorers"].items():
        d = m.get("d_f1_vs_qwen")
        ci = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""
        print(f"  {name:6s} {m['ap']:6.3f} {m['best_f1']:7.3f} {m['f1']:6.3f} "
              f"{m['wss95']:7.3f} {m['recall@gold']:8.3f}  {ci}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="comma-separated setting labels")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/baselines.json")
    args = p.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    out = []
    for label, names, n, files, *limit in SETTINGS:
        if only and label not in only:
            continue
        try:
            out.append(evaluate(label, names, n, files, Path(args.results), *limit))
        except FileNotFoundError as e:
            print(f"skip {label}: {e}")
            continue
        show(out[-1])
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
