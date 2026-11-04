"""Result records and quality metrics.

Every arm returns an ArmResult. The runner turns one into a flat dict and
appends it to a JSONL file, so plots can be regenerated without re-running
anything on the GPU.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ArmResult:
    arm: str
    query_id: str
    predicted: set[str]                       # doc ids the arm says match
    scores: dict[str, float] = field(default_factory=dict)   # doc id -> score
    timing: dict[str, float] = field(default_factory=dict)   # stage -> seconds
    counts: dict[str, float] = field(default_factory=dict)   # calls, tokens, ...
    extra: dict = field(default_factory=dict)

    @property
    def wall(self) -> float:
        return self.timing.get("total", sum(self.timing.values()))


@contextmanager
def stopwatch(into: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        into[key] = into.get(key, 0.0) + time.perf_counter() - t0


# --------------------------------------------------------------------------
# Set metrics -- the "find all matching documents" task
# --------------------------------------------------------------------------

def set_metrics(predicted: set[str], gold: frozenset[str]) -> dict[str, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_predicted": len(predicted), "n_gold": len(gold),
    }


# --------------------------------------------------------------------------
# Ranked metrics -- for arms that return scores (ANN, hybrid, scored classify)
# --------------------------------------------------------------------------

def _ranked(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def ndcg_at_k(scores: dict[str, float], gold: frozenset[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    if not gold or not scores:
        return 0.0
    ranked = _ranked(scores)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in gold)
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg else 0.0


def recall_at_k(scores: dict[str, float], gold: frozenset[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(_ranked(scores)[:k]) & gold) / len(gold)


def ranked_metrics(scores: dict[str, float], gold: frozenset[str],
                   ks: tuple[int, ...] = (10, 100, 1000)) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(scores, gold, k)
        out[f"recall@{k}"] = recall_at_k(scores, gold, k)
    return out


def sweep_threshold(scores: dict[str, float], gold: frozenset[str],
                    steps: int = 41) -> list[dict[str, float]]:
    """F1 across decision thresholds.

    A decision model gives a calibrated probability, so the operating point is
    a free parameter. Reporting a single F1 at p>0.5 understates the arm;
    reporting the curve shows the precision/recall tradeoff honestly.
    """
    if not scores:
        return []
    rows = []
    for i in range(steps):
        t = i / (steps - 1)
        pred = {d for d, s in scores.items() if s >= t}
        m = set_metrics(pred, gold)
        m["threshold"] = t
        rows.append(m)
    return rows


def best_f1(scores: dict[str, float], gold: frozenset[str]) -> dict[str, float]:
    rows = sweep_threshold(scores, gold)
    return max(rows, key=lambda r: r["f1"]) if rows else {}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarize(results: list[ArmResult], golds: dict[str, frozenset[str]],
              ks: tuple[int, ...] = (10, 100, 1000)) -> dict[str, float]:
    """Macro-average over queries, plus throughput totals."""
    if not results:
        return {}
    acc: dict[str, list[float]] = {}
    total_docs = total_wall = 0.0
    for r in results:
        gold = golds[r.query_id]
        m = set_metrics(r.predicted, gold)
        if r.scores:
            m.update(ranked_metrics(r.scores, gold, ks))
            bf = best_f1(r.scores, gold)
            if bf:
                m["best_f1"] = bf["f1"]
                m["best_f1_threshold"] = bf["threshold"]
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
        total_docs += r.counts.get("docs_scored", 0.0)
        total_wall += r.wall

    out = {f"mean_{k}": sum(v) / len(v) for k, v in acc.items()}
    out["n_queries"] = len(results)
    out["total_wall_s"] = total_wall
    out["mean_wall_s"] = total_wall / len(results)
    out["total_docs_scored"] = total_docs
    if total_wall > 0:
        out["docs_per_sec"] = total_docs / total_wall
    walls = sorted(r.wall for r in results)
    out["p50_wall_s"] = walls[len(walls) // 2]
    out["p95_wall_s"] = walls[min(len(walls) - 1, int(0.95 * len(walls)))]
    return out


def flatten(r: ArmResult, gold: frozenset[str], ks=(10, 100, 1000)) -> dict:
    """One ArmResult -> one flat JSONL row."""
    row = {"arm": r.arm, "query_id": r.query_id}
    row.update(set_metrics(r.predicted, gold))
    if r.scores:
        row.update(ranked_metrics(r.scores, gold, ks))
        bf = best_f1(r.scores, gold)
        if bf:
            row["best_f1"] = bf["f1"]
            row["best_f1_threshold"] = bf["threshold"]
    row.update({f"t_{k}": v for k, v in r.timing.items()})
    row.update({f"n_{k}": v for k, v in r.counts.items()})
    row["wall_s"] = r.wall
    row.update(r.extra)
    return row
