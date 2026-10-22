"""
scoring.py
----------
Post-hoc score transforms (priors 4 & 5). These apply to *every* fitted model
with no extra training: given a model probability vector ``p`` they produce the
raw score, shrinkage variants and deterministic-noise ranking variants.

Each transform is tagged ``is_probability`` so downstream metrics know whether
log-loss / Brier are meaningful (they are not for the ranking score).
"""

from __future__ import annotations

import numpy as np

from ml_elements.scores import noise_rank, shrink

from .config import Config


def apply_transforms(
    p: np.ndarray,
    row_ids: np.ndarray,
    pi: float,
    cfg: Config,
    seed: int,
) -> dict[str, tuple[np.ndarray, bool]]:
    """
    Build the score-transform family for one prediction vector.

    Returns
    -------
    dict
        ``{transform_name: (score, is_probability)}`` with keys
        ``raw``, ``shrink@{λ}`` and ``noise@{α}``.
    """
    p = np.asarray(p, dtype=float)
    out: dict[str, tuple[np.ndarray, bool]] = {"raw": (p, True)}

    for lam in cfg.priors.shrinkage_lambdas:
        out[f"shrink@{lam:g}"] = (shrink(p, pi, lam), True)

    for alpha in cfg.priors.noise_alphas:
        out[f"noise@{alpha:g}"] = (noise_rank(p, row_ids, alpha, seed), False)

    return out
