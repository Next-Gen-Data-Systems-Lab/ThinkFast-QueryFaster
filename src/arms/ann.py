"""Approach C: embedding retrieval, and Approach D: retrieve-then-classify.

Index build time is measured separately from query time, because it is the
cost the scanning arm never pays. That difference is the whole argument for
why the crossover moves under corpus churn.
"""

from __future__ import annotations

import time

import numpy as np

from ..metrics import ArmResult, stopwatch


class Index:
    """Dense embeddings + exact inner product, or TF-IDF offline.

    Exact search rather than HNSW is deliberate for the midterm: it removes
    recall-vs-speed index tuning as a confound while we establish the arms.
    Swap in hnswlib for the final run and report both.
    """

    def __init__(self, backend: str = "sentence-transformers",
                 model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 256, device: str | None = None):
        self.backend = backend
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self._enc = None
        self.doc_ids: list[str] = []
        self.mat: np.ndarray | None = None
        self.build_s = 0.0
        self.encode_s = 0.0

    def _encoder(self):
        if self._enc is None and self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(self.model_name, device=self.device)
        return self._enc

    def _embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self.backend == "tfidf":
            if is_query:
                v = self._vec.transform(texts)
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._vec = TfidfVectorizer(max_features=50_000, stop_words="english")
                v = self._vec.fit_transform(texts)
            v = np.asarray(v.todense(), dtype=np.float32)
        else:
            v = self._encoder().encode(
                texts, batch_size=self.batch_size,
                convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-9)

    def build(self, docs: list) -> float:
        """Embed and index the corpus. Returns build seconds."""
        t0 = time.perf_counter()
        self.doc_ids = [d.id for d in docs]
        self.mat = self._embed([d.text for d in docs], is_query=False)
        self.build_s = time.perf_counter() - t0
        return self.build_s

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self._embed([query], is_query=True)[0]
        sims = self.mat @ q
        k = min(k, len(self.doc_ids))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]

    def add(self, docs: list) -> float:
        """Incremental insert -- the maintenance cost under corpus churn."""
        t0 = time.perf_counter()
        new = self._embed([d.text for d in docs], is_query=False)
        self.mat = np.vstack([self.mat, new]) if self.mat is not None else new
        self.doc_ids.extend(d.id for d in docs)
        return time.perf_counter() - t0


def run_ann(index: Index, query, k: int = 100) -> ArmResult:
    """Approach C: top-k by similarity, no model calls at all."""
    timing: dict[str, float] = {}
    with stopwatch(timing, "total"):
        hits = index.search(query.text, k)
    scores = {d: s for d, s in hits}
    return ArmResult(
        arm="ann",
        query_id=query.id,
        predicted=set(scores),
        scores=scores,
        timing=timing,
        counts={"docs_scored": 0.0, "docs_retrieved": float(len(hits))},
        extra={"k": k, "embed_model": index.model_name,
               "index_build_s": index.build_s},
    )


def run_hybrid(index: Index, model, query, docs_by_id: dict,
               k: int = 100, threshold: float = 0.5) -> ArmResult:
    """Approach D: retrieve k candidates, then classify only those k.

    Cost is independent of corpus size, which is why this is expected to win
    on throughput almost everywhere.

    It inherits a hard recall ceiling from retrieval: anything outside the top
    k can never be recovered no matter how good the classifier is. We record
    that ceiling on every row, because when a query has more relevant
    documents than k, the ceiling -- not classifier quality -- is what caps
    this arm. That is the central caveat on "hybrid dominates".
    """
    timing: dict[str, float] = {}
    with stopwatch(timing, "retrieve"):
        hits = index.search(query.text, k)
    retrieved = {d for d, _ in hits}
    cand = [docs_by_id[d] for d, _ in hits if d in docs_by_id]

    with stopwatch(timing, "classify"):
        probs, stats = model.score(query.text, cand)
    timing["total"] = timing["retrieve"] + timing["classify"]
    stats.pop("generate_s", None)

    gold = query.gold
    ceiling = len(retrieved & gold) / len(gold) if gold else 0.0

    return ArmResult(
        arm="hybrid",
        query_id=query.id,
        predicted={d for d, p in probs.items() if p >= threshold},
        scores=probs,
        timing=timing,
        counts={**stats, "docs_retrieved": float(len(hits))},
        extra={"k": k, "threshold": threshold,
               "model": getattr(model, "model_name", "?"),
               "max_num_seqs": getattr(model, "max_num_seqs", None),
               "embed_model": index.model_name,
               "index_build_s": index.build_s,
               # fraction of gold reachable at all given k
               "ann_recall_ceiling": ceiling,
               "k_vs_gold_ratio": k / len(gold) if gold else float("inf")},
    )
