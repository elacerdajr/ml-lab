"""
scores.py
---------
Working with a model's scores: how they are shaped, how to transform them,
and how to turn labels or margins into the scores a model needs.

Shape of a score vector (wrapped as ``Metric`` objects in ``metrics.py``)
    score_entropy(scores)        Shannon entropy of a 50-bin histogram on [0, 1],
                                 and its normalisation by log(50).
    tie_rate(scores)             Share of duplicated scores, unique count,
                                 occupied histogram bins.
    max_score_gap(scores)        Largest gap between adjacent sorted scores.

Post-hoc transforms of probabilities p
    shrink(p, pi, lam)           (1-λ)p + λπ. Monotonic, so ranking (AP, AUC) is kept.
    noise_rank(p, row_ids, alpha, seed)
                                 αp + (1-α)u with a stable per-row u in [0, 1).
                                 A ranking score, not a probability.
    row_noise(row_ids, seed)     That u: SplitMix64 of row id and seed.

Labels and calibration
    smooth_labels(y, pi, lam)    Soft target (1-λ)y + λπ.
    to_weighted_rows(X, y_soft)  Exact soft-label reduction for cross-entropy
                                 models: a positive copy weighted y', a negative
                                 copy weighted 1-y'.
    calibrate_sigmoid(est, cv)   Platt-sigmoid wrapper for margin models (SVMs).
    positive_scores(model, X)    predict_proba[:, 1], or clipped predict for regressors.

Moved from ``experiments/2026-07-21_imbalanced_classification/imbcls`` unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_ENTROPY_BINS = 50


# ── shape of a score vector ──────────────────────────────────────────────────

def score_entropy(scores: np.ndarray, bins: int = N_ENTROPY_BINS) -> tuple[float, float]:
    """Return ``(entropy, normalized_entropy)`` from a ``bins``-bin histogram over [0,1]."""
    hist, _ = np.histogram(scores, bins=bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0, 0.0
    p = hist / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    return entropy, entropy / np.log(bins)


def tie_rate(scores: np.ndarray, bins: int = N_ENTROPY_BINS) -> dict:
    """Tie rate, number of unique scores, and number of occupied histogram bins."""
    n = len(scores)
    n_unique = int(np.unique(scores).size)
    hist, _ = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    return {
        "tie_rate": float(1.0 - n_unique / n) if n else 0.0,
        "n_unique_scores": n_unique,
        "occupied_bins": int((hist > 0).sum()),
    }


def max_score_gap(scores: np.ndarray) -> float:
    """Maximum gap between adjacent values of the sorted score vector."""
    if len(scores) < 2:
        return 0.0
    return float(np.max(np.diff(np.sort(scores))))


# ── post-hoc transforms ──────────────────────────────────────────────────────

def shrink(p: np.ndarray, pi: float, lam: float) -> np.ndarray:
    """``(1-λ)p + λπ``: a monotonic map, so Average Precision is preserved."""
    return (1.0 - lam) * np.asarray(p, dtype=float) + lam * pi


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser on a uint64 array (pure integer math, reproducible)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z


def row_noise(row_ids: np.ndarray, seed: int) -> np.ndarray:
    """
    Stable per-row pseudo-random value in ``[0, 1)``.

    Derived from a SplitMix64 hash of ``row_id`` mixed with ``seed``, so it is
    reproducible across runs and processes (unlike ``hash()``, which is
    PYTHONHASHSEED-salted).
    """
    key = np.asarray(row_ids).astype(np.int64).astype(np.uint64)
    # Mix the seed in Python-int space (arbitrary precision) then mask to 64 bits,
    # so the scalar multiply never triggers a numpy overflow warning.
    seed_mix = np.uint64((int(seed) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    key = key + seed_mix
    z = _splitmix64(key)
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def noise_rank(p: np.ndarray, row_ids: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    """Ranking score ``αp + (1-α)u`` (not a calibrated probability)."""
    u = row_noise(row_ids, seed)
    return alpha * np.asarray(p, dtype=float) + (1.0 - alpha) * u


# ── labels and calibration ───────────────────────────────────────────────────

def smooth_labels(y: np.ndarray, pi: float, lam: float) -> np.ndarray:
    """Soft target ``(1-λ)y + λπ``."""
    y = np.asarray(y, dtype=float)
    return (1.0 - lam) * y + lam * pi


def to_weighted_rows(
    X: pd.DataFrame,
    y_soft: np.ndarray,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Exact soft-label reduction for cross-entropy models.

    Each row with soft label ``y'`` becomes a positive copy (weight ``y'``) and a
    negative copy (weight ``1-y'``); copies with weight ``≤ eps`` are dropped so
    hard labels collapse back to a single weight-1 row.

    Returns
    -------
    (X2, y2, w2)
        Duplicated feature frame, hard 0/1 labels, and sample weights.
    """
    X = X.reset_index(drop=True)
    ys = np.clip(np.asarray(y_soft, dtype=float), 0.0, 1.0)
    X2 = pd.concat([X, X], ignore_index=True)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([ys, 1.0 - ys])
    keep = w2 > eps
    return X2.loc[keep].reset_index(drop=True), y2[keep], w2[keep]


def calibrate_sigmoid(estimator, cv: int = 3):
    """
    Wrap ``estimator`` in Platt sigmoid calibration with internal ``cv``-fold CV.

    For margin models (``LinearSVC``, ``SVC(probability=False)``) that expose
    only ``decision_function``. Using ``cv`` rather than ``prefit`` avoids a
    manual holdout split.
    """
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(estimator, method="sigmoid", cv=cv)


def positive_scores(model, X) -> np.ndarray:
    """
    Positive-class score for a fitted model.

    ``predict_proba[:, 1]`` for classifiers; a clipped ``predict`` for
    regressor-based soft-label models.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.0, 1.0)
