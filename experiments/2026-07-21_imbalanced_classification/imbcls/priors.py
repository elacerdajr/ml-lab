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

Only prior 3 is specific to this experiment and lives here. Label smoothing,
shrinkage and the noise ranking score are ``ml_elements.scores.smooth_labels``,
``shrink`` and ``noise_rank``.

Priors 2 & 3 require a model that consumes soft labels. For log-loss models
(logistic, RFF+logistic) ``ml_elements.scores.to_weighted_rows`` gives the *exact* reduction:
a soft label ``y'`` equals a positive row of weight ``y'`` plus a negative row
of weight ``1-y'`` under cross-entropy. CatBoost consumes soft labels natively
via ``loss_function="CrossEntropy"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import CAT_FEATURES, FEATURES, NUM_FEATURES

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
