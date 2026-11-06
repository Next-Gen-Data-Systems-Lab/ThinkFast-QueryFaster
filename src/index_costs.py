"""What the index costs, and what approximate search does to deep retrieval.

The scan needs no index. Every other arm pays for one, so this measures the
HNSW graph build, incremental inserts, and how much of the exact top k it keeps.
Hybrid plans go deep, so recall is reported at large k as well as at k = 100.

Needs hnswlib, which is not in the CHTC image. Graph times are from whatever
machine this runs on; embedding time on the GPU is `index_build_s` in the rows.

Usage: python -m src.index_costs --corpus-size 10000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from .arms import ann as ANN

KS = (100, 1000, 3000)
EF = (100, 400, 1600, 4000)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-size", type=int, default=10000)
    p.add_argument("--insert-share", type=float, default=0.1)
    p.add_argument("--out", default="results/index_costs.json")
    args = p.parse_args(argv)

    import hnswlib

    corpus = D.load("20newsgroups").subsample(args.corpus_size, seed=0)
    index = ANN.Index()
    embed_s = index.build(corpus.docs)
    mat = index.mat
    queries = index._embed([q.text for q in corpus.queries], is_query=True)
    n, dim = mat.shape
    held = int(n * args.insert_share)

    graph = hnswlib.Index(space="ip", dim=dim)
    graph.init_index(max_elements=n, ef_construction=200, M=16)
    graph.set_num_threads(4)
    t0 = time.perf_counter()
    graph.add_items(mat[: n - held], np.arange(n - held))
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    graph.add_items(mat[n - held:], np.arange(n - held, n))
    insert_s = time.perf_counter() - t0

    exact = np.argsort(-(queries @ mat.T), axis=1)
    t0 = time.perf_counter()
    for _ in range(20):
        queries @ mat.T
    exact_ms = (time.perf_counter() - t0) / 20 / len(queries) * 1000

    rows = []
    for ef in EF:
        graph.set_ef(ef)
        for k in KS:
            if k > ef:
                continue
            t0 = time.perf_counter()
            found, _ = graph.knn_query(queries, k=k)
            ms = (time.perf_counter() - t0) / len(queries) * 1000
            overlap = np.mean([len(set(found[i]) & set(exact[i, :k])) / k
                               for i in range(len(queries))])
            rows.append({"ef": ef, "k": k, "recall_vs_exact": float(overlap), "ms_per_query": ms})

    out = {"corpus_size": n, "dim": dim, "embed_s": embed_s,
           "graph_build_s": build_s, "graph_build_docs": n - held,
           "insert_s": insert_s, "insert_docs": held,
           "insert_ms_per_doc": insert_s / held * 1000,
           "exact_ms_per_query": exact_ms, "search": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
