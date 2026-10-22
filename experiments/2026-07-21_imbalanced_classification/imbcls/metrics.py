"""
metrics.py
----------
Evaluation metrics: standard classification quality plus the score-smoothness /
ranking-diagnostic family that motivates the whole experiment.

Reuses ``ml_elements`` metric objects for AP / AUC (calling ``.fn`` directly so
ranking scores are not clipped) and computes log-loss / Brier only for scores
tagged as probabilities.

The score-smoothness measures (entropy, tie rate, max gap) come from
``ml_elements.scores``.

Functions
---------
compute_classification_metrics  AP / AUC / log-loss / Brier (log-loss & Brier gated).
eval_row                Assemble one metrics.csv row.
compute_bucket_metrics  Per-bucket lift table for operational ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_elements.metrics import AUC, AVG_PRECISION, BRIER, LOGLOSS
from ml_elements.scores import max_score_gap, score_entropy, tie_rate


def compute_classification_metrics(y: np.ndarray, score: np.ndarray, is_prob: bool) -> dict:
    """AP / AUC always; log-loss / Brier only when ``score`` is a probability."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    out = {
        "average_precision": float(AVG_PRECISION.fn(y, score)),
        "roc_auc": float(AUC.fn(y, score)),
        "log_loss": np.nan,
        "brier_score": np.nan,
    }
    if is_prob:
        pc = np.clip(score, 1e-8, 1.0 - 1e-8)
        out["log_loss"] = float(LOGLOSS.fn(y, pc))
        out["brier_score"] = float(BRIER.fn(y, pc))
    return out


def eval_row(
    y: np.ndarray,
    score: np.ndarray,
    is_prob: bool,
    keys: dict,
    timings: dict,
    counts: dict,
) -> dict:
    """
    Build a full metrics row: identifying ``keys`` + classification metrics +
    score-smoothness metrics + ``timings`` + ``counts``.
    """
    entropy, norm_entropy = score_entropy(score)
    ties = tie_rate(score)
    row = dict(keys)
    row.update(compute_classification_metrics(y, score, is_prob))
    row["score_entropy"] = entropy
    row["normalized_score_entropy"] = norm_entropy
    row["tie_rate"] = ties["tie_rate"]
    row["n_unique_scores"] = ties["n_unique_scores"]
    row["occupied_bins"] = ties["occupied_bins"]
    row["max_score_gap"] = max_score_gap(score)
    row.update(timings)
    row.update(counts)
    return row


def compute_bucket_metrics(
    y: np.ndarray,
    score: np.ndarray,
    n_buckets: int,
    base_rate: float,
) -> pd.DataFrame:
    """
    Split scores into ``n_buckets`` equal-width rank buckets (bucket 0 = lowest
    scores) and report count, positives, positive rate, mean score and lift vs
    the base rate for each.
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.int64)
    ranks[order] = np.arange(len(score))
    bucket = np.minimum((ranks * n_buckets) // len(score), n_buckets - 1)

    rows = []
    for b in range(n_buckets):
        mask = bucket == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        pos = int(y[mask].sum())
        rate = pos / cnt
        rows.append(
            {
                "n_buckets": n_buckets,
                "bucket_id": b,
                "count": cnt,
                "positive_count": pos,
                "positive_rate": rate,
                "mean_score": float(score[mask].mean()),
                "lift_vs_base_rate": (rate / base_rate) if base_rate > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
