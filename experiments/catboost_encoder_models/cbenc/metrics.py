"""
metrics.py
----------
Standard classification metrics — every score here is a genuine probability,
so (unlike the ``imbalanced_classification`` sibling) there is no ranking-score
gating needed.
"""

from __future__ import annotations

import numpy as np

from ml_elements.metrics import AUC, AVG_PRECISION, BRIER, LOGLOSS


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """AP, AUC, log-loss and Brier score for a probability vector."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    pc = np.clip(score, 1e-8, 1.0 - 1e-8)
    return {
        "average_precision": float(AVG_PRECISION.fn(y, score)),
        "roc_auc": float(AUC.fn(y, score)),
        "log_loss": float(LOGLOSS.fn(y, pc)),
        "brier_score": float(BRIER.fn(y, pc)),
    }


def eval_row(y: np.ndarray, score: np.ndarray, keys: dict, timings: dict, counts: dict) -> dict:
    """Assemble one ``metrics.csv`` row: keys + classification metrics + timings + counts."""
    row = dict(keys)
    row.update(classification_metrics(y, score))
    row.update(timings)
    row.update(counts)
    return row
