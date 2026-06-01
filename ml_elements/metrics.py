"""
metrics.py
----------
Metric building blocks.

A ``Metric`` is a frozen dataclass that bundles three things:
    - a human-readable ``name``
    - a ``direction`` ("higher" or "lower") so improvement can be computed
      without guessing
    - a scoring ``fn`` that maps (y_true, y_score) -> float

Pre-built metrics
-----------------
AUC                  ROC-AUC. Direction: higher.
LOGLOSS              Binary cross-entropy. Direction: lower.
AVG_PRECISION        sklearn average_precision_score. Direction: higher.
AVG_PRECISION_SMOOTH Beta-binomial precision-prior AP. Direction: higher.
                     Use ``make_smooth_ap(prior_mean, prior_strength)``
                     to control the prior.

Custom metrics
--------------
Any callable (y_true, y_score) -> float can be wrapped:

    >>> from sklearn.metrics import f1_score
    >>> F1 = Metric("f1", "higher", lambda y, p: f1_score(y, p > 0.5))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .protocols import MetricFn


@dataclass(frozen=True)
class Metric:
    """
    A single evaluation metric with its optimization direction.

    Parameters
    ----------
    name : str
        Human-readable identifier used in DataFrames and plots.
    direction : {"higher", "lower"}
        "higher" means larger values are better (AUC, AP).
        "lower" means smaller values are better (log-loss).
    fn : MetricFn
        Callable ``(y_true: ndarray, y_score: ndarray) -> float``.

    Examples
    --------
    >>> AUC.name
    'auc'
    >>> AUC.direction
    'higher'
    >>> AUC.fn(y_true, scores)
    0.842
    """

    name: str
    direction: Literal["higher", "lower"]
    fn: MetricFn

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Compute the metric, clipping y_score to avoid numerical issues.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0 / 1).
        y_score : np.ndarray
            Model scores or probabilities.

        Returns
        -------
        float
        """
        y_score = np.clip(np.asarray(y_score, dtype=float), 1e-8, 1 - 1e-8)
        return float(self.fn(np.asarray(y_true), y_score))

    def improvement(self, score_challenger: float, score_baseline: float) -> float:
        """
        Signed improvement: positive means challenger is better.

        For "higher" metrics: challenger - baseline.
        For "lower" metrics: baseline - challenger.

        Parameters
        ----------
        score_challenger : float
        score_baseline : float

        Returns
        -------
        float
        """
        delta = score_challenger - score_baseline
        return delta if self.direction == "higher" else -delta


# ---------------------------------------------------------------------------
# Smooth average precision helpers (migrated from feature_information_studies)
# ---------------------------------------------------------------------------

def _average_precision_smooth(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prior_mean: float = 0.1,
    prior_strength: float = 20,
) -> float:
    """
    Beta-binomial precision-prior average precision.

    At each positive rank k, precision is smoothed:

        P_smooth(k) = (TP(k) + alpha) / (k + alpha + beta)

    where alpha = prior_mean * prior_strength,
          beta  = (1 - prior_mean) * prior_strength.

    As prior_strength -> 0, this recovers standard sklearn AP.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Scores. Higher = more likely positive.
    prior_mean : float
        Prior expected precision (p0).
    prior_strength : float
        Prior strength in pseudo-observations (m).

    Returns
    -------
    float
        Smoothed AP, or np.nan if no positives in y_true.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    n_pos = int(y_sorted.sum())
    if n_pos == 0:
        return np.nan

    alpha = prior_mean * prior_strength
    beta_val = (1.0 - prior_mean) * prior_strength

    tp = np.cumsum(y_sorted)
    rank = np.arange(1, len(y_sorted) + 1)
    precision_smooth = (tp + alpha) / (rank + alpha + beta_val)

    return float(precision_smooth[y_sorted == 1].mean())


def make_smooth_ap(
    prior_mean: float = 0.1,
    prior_strength: float = 20,
) -> Metric:
    """
    Build a smooth average-precision ``Metric`` with custom prior knobs.

    Parameters
    ----------
    prior_mean : float
        Prior expected precision p0.
    prior_strength : float
        Prior strength m (pseudo-observation count).

    Returns
    -------
    Metric
        ``AVG_PRECISION_SMOOTH``-style metric with the given prior.
    """
    def fn(y_true: np.ndarray, y_score: np.ndarray) -> float:
        return _average_precision_smooth(
            y_true,
            y_score,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
        )

    return Metric(
        name=f"avg_precision_smooth(p0={prior_mean},m={prior_strength})",
        direction="higher",
        fn=fn,
    )


# ---------------------------------------------------------------------------
# Pre-built metric instances
# ---------------------------------------------------------------------------

AUC: Metric = Metric(
    name="auc",
    direction="higher",
    fn=lambda y, p: roc_auc_score(y, p),
)

LOGLOSS: Metric = Metric(
    name="logloss",
    direction="lower",
    fn=lambda y, p: log_loss(y, p),
)

AVG_PRECISION: Metric = Metric(
    name="average_precision",
    direction="higher",
    fn=lambda y, p: average_precision_score(y, p),
)

AVG_PRECISION_SMOOTH: Metric = make_smooth_ap(prior_mean=0.1, prior_strength=20)


ALL_METRICS: dict[str, Metric] = {
    m.name: m
    for m in [AUC, LOGLOSS, AVG_PRECISION, AVG_PRECISION_SMOOTH]
}
