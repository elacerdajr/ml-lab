"""
data.py
-------
Synthetic data generation, splitting and undersampling.

This is a direct port of ``experiments/imbalanced_classification/imbcls/data.py``
(same logit formula, same seed handling) so all three experiments share an
identical data-generating process — copied rather than imported, matching the
repo convention that every experiment folder is self-contained.

The generative process is a controlled nonlinear logit so the positive class is
rare (~0.1%) but *not* random: the target depends on some ``cat_1`` levels, a
subset of high-risk ``cat_2`` levels, a smooth ``sin(num_1)`` effect, a
threshold effect on ``num_2``, a categorical×numerical interaction, and noise.
The intercept is calibrated by bisection so ``mean(p) ≈ positive_rate``.

Key rule
--------
``pi`` is the **true training base rate** ``y_train_full.mean() ≈ 0.001``, never
the 10% undersampled rate. Models train on ``train_under`` but are evaluated on
``val_full`` / ``test_full`` which preserve the real base rate.

Functions
---------
generate_full            Build the full population DataFrame with a stable ``row_id``.
split_dataset            Stratified train/val/test split preserving the base rate.
make_undersampled_train  Keep all train positives, subsample negatives to 10%.
compute_pi               The true base rate from ``train_full``.
data_summary             Assemble the ``data_summary.json`` payload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

CAT_FEATURES = ["cat_1", "cat_2"]
NUM_FEATURES = ["num_1", "num_2"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _solve_intercept(base_logit: np.ndarray, target_mean: float) -> float:
    """
    Bisection-solve an additive intercept ``b`` so ``mean(sigmoid(base+b))`` equals
    ``target_mean``. ``mean`` is monotone increasing in ``b`` so bisection is safe.
    """
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _sigmoid(base_logit + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_full(cfg: Config) -> pd.DataFrame:
    """
    Generate the full population.

    Returns
    -------
    pandas.DataFrame
        Columns ``cat_1, cat_2, num_1, num_2, y, row_id`` where ``cat_*`` are
        stored as pandas ``category`` dtype and ``row_id`` is a stable integer
        carried through every split.
    """
    d = cfg.data
    rng = np.random.default_rng(cfg.seed)
    n = d.n_full

    cat_1 = rng.integers(0, d.cat_1_values, size=n)
    cat_2 = rng.integers(0, d.cat_2_values, size=n)
    num_1 = rng.normal(0.0, 1.0, size=n)
    num_2 = rng.normal(0.0, 1.0, size=n)

    # Per-level categorical effects (fixed by a dedicated rng so they are stable).
    eff_rng = np.random.default_rng(cfg.seed + 1)
    effect_cat_1 = eff_rng.normal(0.0, 0.8, size=d.cat_1_values)
    effect_cat_2 = np.zeros(d.cat_2_values)
    high_risk = eff_rng.choice(d.cat_2_values, size=d.cat_2_high_risk, replace=False)
    effect_cat_2[high_risk] = eff_rng.uniform(1.5, 3.0, size=d.cat_2_high_risk)

    # Nonlinear categorical×numerical interaction: high-risk cat_2 amplifies num_1,
    # and cat_1 level 0 flips the sign of the num_2 threshold effect.
    is_high_risk = np.isin(cat_2, high_risk).astype(float)
    interaction = 1.2 * is_high_risk * num_1 - 0.8 * (cat_1 == 0).astype(float) * (num_2 > 0)

    base_logit = (
        effect_cat_1[cat_1]
        + effect_cat_2[cat_2]
        + 1.5 * np.sin(num_1)
        + 1.0 * (num_2 > d.num_2_threshold).astype(float)
        + interaction
        + rng.normal(0.0, d.noise_sigma, size=n)
    )

    intercept = _solve_intercept(base_logit, d.positive_rate)
    p = _sigmoid(base_logit + intercept)
    y = (rng.random(n) < p).astype(np.int8)

    df = pd.DataFrame(
        {
            "cat_1": pd.Categorical(cat_1.astype(str)),
            "cat_2": pd.Categorical(cat_2.astype(str)),
            "num_1": num_1.astype(np.float32),
            "num_2": num_2.astype(np.float32),
            "y": y,
            "row_id": np.arange(n, dtype=np.int64),
        }
    )
    return df


def split_dataset(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """
    Stratified train/val/test split preserving the natural base rate in every part.

    Returns
    -------
    dict
        Keys ``train_full``, ``val_full``, ``test_full``.
    """
    from sklearn.model_selection import train_test_split

    s = cfg.splits
    idx = np.arange(len(df))
    train_idx, holdout_idx = train_test_split(
        idx,
        test_size=s.val_fraction + s.test_fraction,
        stratify=df["y"].to_numpy(),
        random_state=cfg.seed,
    )
    rel_test = s.test_fraction / (s.val_fraction + s.test_fraction)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=rel_test,
        stratify=df["y"].to_numpy()[holdout_idx],
        random_state=cfg.seed,
    )
    return {
        "train_full": df.iloc[train_idx].reset_index(drop=True),
        "val_full": df.iloc[val_idx].reset_index(drop=True),
        "test_full": df.iloc[test_idx].reset_index(drop=True),
    }


def compute_pi(train_full: pd.DataFrame) -> float:
    """Return the true training base rate ``pi = y_train_full.mean()`` (≈ 0.001)."""
    return float(train_full["y"].mean())


def make_undersampled_train(train_full: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Build ``train_under``: keep **all** positives, subsample negatives so positives
    are ``undersample.positive_rate`` (default 10%) of the result.
    """
    rng = np.random.default_rng(cfg.seed + 7)
    pos = train_full[train_full["y"] == 1]
    neg = train_full[train_full["y"] == 0]

    target_rate = cfg.undersample.positive_rate
    n_pos = len(pos)
    n_neg_keep = min(len(neg), int(round(n_pos * (1.0 - target_rate) / target_rate)))
    neg_idx = rng.choice(len(neg), size=n_neg_keep, replace=False)
    under = pd.concat([pos, neg.iloc[neg_idx]], ignore_index=True)
    return under.sample(frac=1.0, random_state=cfg.seed + 8).reset_index(drop=True)


def data_summary(
    df_full: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    train_under: pd.DataFrame,
    pi: float,
    cfg: Config,
) -> dict:
    """Assemble the JSON-serialisable data summary."""

    def _stat(frame: pd.DataFrame) -> dict:
        return {
            "n": int(len(frame)),
            "n_positive": int(frame["y"].sum()),
            "positive_rate": float(frame["y"].mean()),
        }

    return {
        "profile": cfg.profile,
        "seed": cfg.seed,
        "target_positive_rate": cfg.data.positive_rate,
        "pi_true_train_base_rate": pi,
        "features": {"categorical": CAT_FEATURES, "numerical": NUM_FEATURES},
        "cat_1_values": cfg.data.cat_1_values,
        "cat_2_values": cfg.data.cat_2_values,
        "full": _stat(df_full),
        "train_full": _stat(splits["train_full"]),
        "val_full": _stat(splits["val_full"]),
        "test_full": _stat(splits["test_full"]),
        "train_under": _stat(train_under),
    }
