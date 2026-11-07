"""BM25 keyword search: the agent's second tool and the BM25 baseline.

Okapi BM25 (k1 = 1.2, b = 0.75) over scikit-learn's English tokenizer with stop
words removed. `scores` gives every document's score for one query, which the
baseline uses; `search` gives the top k, which the agent's tool uses. Same
interface as the ANN index: build(docs), then search(query, k).
"""

from __future__ import annotations

import time

import numpy as np

K1, B = 1.2, 0.75


class KeywordIndex:
    def build(self, docs: list) -> float:
        from sklearn.feature_extraction.text import CountVectorizer

        t0 = time.perf_counter()
        self.doc_ids = [d.id for d in docs]
        self.vec = CountVectorizer(stop_words="english", lowercase=True)
        self.tf = self.vec.fit_transform([d.text for d in docs]).tocsc().astype(np.float64)
        n = self.tf.shape[0]
        length = np.asarray(self.tf.sum(axis=1)).ravel()
        df = np.diff(self.tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.norm = K1 * (1 - B + B * length / max(length.mean(), 1e-9))
        self.analyse = self.vec.build_analyzer()
        return time.perf_counter() - t0

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.tf.shape[0])
        vocab = self.vec.vocabulary_
        for term in set(self.analyse(query)):
            j = vocab.get(term)
            if j is None:
                continue
            start, end = self.tf.indptr[j], self.tf.indptr[j + 1]
            rows, f = self.tf.indices[start:end], self.tf.data[start:end]
            out[rows] += self.idf[j] * f * (K1 + 1) / (f + self.norm[rows])
        return out

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        top = top[np.argsort(-s[top], kind="stable")]
        # Documents sharing no word with the query are not results.
        return [(self.doc_ids[i], float(s[i])) for i in top if s[i] > 0]
