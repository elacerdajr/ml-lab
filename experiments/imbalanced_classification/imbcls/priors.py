"""
priors.py
---------
The five prior mechanisms, all anchored on the **true** base rate
``pi = y_train_full.mean() ≈ 0.001`` (never the undersampled rate).

1. none                — train normally on the undersample.
2. label smoothing     — soft target ``y' = (1-λ)y + λπ``.
3. synthetic soft pts   — append rows drawn from the empirical train_full
                          distribution with soft label ``y* = π``.
4. post-hoc shrinkage   — ``p' = (1-λ)p + λπ`` (monotonic → AP-preserving).
5. deterministic noise  — ranking score ``r = αp + (1-α)u`` where ``u`` is a
                          stable per-row pseudo-random value in ``[0,1)``.

Priors 2 & 3 require a model that consumes soft labels. For log-loss models
(logistic, RFF+logistic) :func:`to_weighted_rows` gives the *exact* reduction:
a soft label ``y'`` equals a positive row of weight ``y'`` plus a negative row
of weight ``1-y'`` under cross-entropy. CatBoost consumes soft labels natively
via ``loss_function="CrossEntropy"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import CAT_FEATURES, FEATURES, NUM_FEATURES

# ── Prior 2: label smoothing ─────────────────────────────────────────────────

def label_smoothing(y: np.ndarray, pi: float, lam: float) -> np.ndarray:
    """Soft target ``(1-λ)y + λπ``."""
    y = np.asarray(y, dtype=float)
    return (1.0 - lam) * y + lam * pi


# ── Prior 3: synthetic soft prior points ─────────────────────────────────────

def make_synthetic_prior_points(
    train_full: pd.DataFrame,
    n_points: int,
    pi: float,
    cfg: Config,
    seed: int,
) -> pd.DataFrame:
    """
    Draw ``n_points`` synthetic rows from the empirical ``train_full`` distribution.

    Numerical features are sampled from real rows then perturbed with Gaussian
    noise scaled by ``synthetic_num_perturb × feature_std``; categoricals are
    resampled from empirical frequencies. Every synthetic row carries soft label
    ``y* = pi``.
    """
    rng = np.random.default_rng(seed)
    n_src = len(train_full)
    src_idx = rng.integers(0, n_src, size=n_points)

    data: dict[str, np.ndarray] = {}
    for col in CAT_FEATURES:
        # sample categorical values from empirical frequencies
        vals = train_full[col].to_numpy()
        freq_idx = rng.integers(0, n_src, size=n_points)
        data[col] = pd.Categorical(
            vals[freq_idx], categories=train_full[col].cat.categories
        )
    for col in NUM_FEATURES:
        base = train_full[col].to_numpy()[src_idx].astype(np.float32)
        std = float(train_full[col].to_numpy().std())
        noise = rng.normal(0.0, cfg.priors.synthetic_num_perturb * std, size=n_points)
        data[col] = (base + noise).astype(np.float32)

    out = pd.DataFrame(data)
    out["y"] = float(pi)                       # soft label
    out["row_id"] = -(np.arange(1, n_points + 1, dtype=np.int64))  # negative = synthetic
    return out[[*FEATURES, "y", "row_id"]]


# ── Soft-label reduction for log-loss models ─────────────────────────────────

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


# ── Prior 4: post-hoc shrinkage ──────────────────────────────────────────────

def posthoc_shrinkage(p: np.ndarray, pi: float, lam: float) -> np.ndarray:
    """``(1-λ)p + λπ`` — a monotonic map, so Average Precision is preserved."""
    return (1.0 - lam) * np.asarray(p, dtype=float) + lam * pi


# ── Prior 5: deterministic noise ranking score ───────────────────────────────

def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser on a uint64 array (pure integer math, reproducible)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z


def deterministic_noise_score(row_ids: np.ndarray, seed: int) -> np.ndarray:
    """
    Stable per-row pseudo-random value in ``[0, 1)``.

    Derived from a SplitMix64 hash of ``row_id`` mixed with ``seed`` — reproducible
    across runs and processes (unlike ``hash()``, which is PYTHONHASHSEED-salted).
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
    u = deterministic_noise_score(row_ids, seed)
    return alpha * np.asarray(p, dtype=float) + (1.0 - alpha) * u
