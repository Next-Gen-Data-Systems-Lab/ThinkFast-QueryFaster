"""Save every query-document similarity, so retrieval can be replayed offline.

The classifier scored every document during the scan, so any hybrid plan can be
replayed without a GPU if the full embedding ranking is available. This builds
that ranking with the same index the arms use. It needs sentence-transformers,
and takes a few minutes on a laptop.

Usage: python -m src.embed_ranks --dataset 20newsgroups --corpus-sizes 1000,5000,10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import data as D
from .arms import ann as ANN


def path_for(results: Path, dataset: str, n: int) -> Path:
    return results / f"{dataset}_embed_{n}.npz"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="20newsgroups", choices=sorted(D.LOADERS))
    p.add_argument("--corpus-sizes", default="1000,5000,10000")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)

    full = D.load(args.dataset, max_chars=args.max_chars)
    for n in [int(x) for x in args.corpus_sizes.split(",") if x.strip()]:
        sub = full.subsample(n, seed=args.seed)
        index = ANN.Index(model=args.embed_model)
        seconds = index.build(sub.docs)
        queries = index._embed([q.text for q in sub.queries], is_query=True)
        out = path_for(Path(args.results), sub.name, len(sub.docs))
        np.savez_compressed(
            out, sims=(queries @ index.mat.T).astype(np.float32),
            doc_ids=np.array(index.doc_ids), query_ids=np.array([q.id for q in sub.queries]),
            fingerprint=D.fingerprint(sub))
        print(f"{out}: {len(sub.queries)} queries x {len(sub.docs)} docs, "
              f"embedded in {seconds:.1f}s, {D.fingerprint(sub)}", flush=True)


if __name__ == "__main__":
    main()
