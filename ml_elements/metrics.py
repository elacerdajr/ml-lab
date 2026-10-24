"""
metrics.py
----------
``Metric``: a named, directed score, and the pre-built metrics.

A metric has a ``name``, a ``direction`` ("higher" or "lower" is better) and
one or more weighted terms. Each term scores one of:

    on="scores"   fn(y_true, y_score)          AUC, AP, log-loss, score entropy
    on="model"    fn(model)                    tree count, coefficient norm
    on="all"      fn(y_true, y_score, model)   anything else

Arithmetic on metrics returns a metric, so penalties and blends need no
separate type:

    >>> AVG_PRECISION - 0.01 * N_ITER          # penalise boosting rounds
    >>> 0.7 * AUC + 0.3 * AVG_PRECISION        # blend
    >>> (AVG_PRECISION - 0.005 * N_LEAVES).rename("ap_leaf_penalised")

The left operand sets the name and direction; multiplying by a negative number
flips the direction. ``Objective``, ``ModelMetric`` and ``make_objective`` are
kept as aliases for this one type.

Pre-built metrics
-----------------
AUC, AVG_PRECISION, AVG_PRECISION_SMOOTH (higher)   LOGLOSS, BRIER (lower)
SCORE_ENTROPY, NORMALIZED_SCORE_ENTROPY (higher)     TIE_RATE, MAX_SCORE_GAP (lower)
N_ITER, N_LEAVES, COEF_L1 (model terms, lower)

Custom metrics
--------------
    >>> from sklearn.metrics import f1_score
    >>> F1 = Metric("f1", "higher", lambda y, p: f1_score(y, p > 0.5))
    >>> TREES = Metric("n_trees", "lower", lambda m: float(m.tree_count_), on="model")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .scores import max_score_gap, score_entropy, tie_rate

Direction = Literal["higher", "lower"]
On = Literal["scores", "model", "all"]


@dataclass(frozen=True, init=False)
class Metric:
    """
    A score with a direction, built from one or more weighted terms.

    Parameters
    ----------
    name : str
        Identifier used as a column name in result tables.
    direction : {"higher", "lower"}
        Which way is better.
    fn : callable
        The scoring function; its arguments depend on ``on``.
    on : {"scores", "model", "all"}
        What ``fn`` reads: ``(y_true, y_score)``, ``(model)``, or
        ``(y_true, y_score, model)``.
    """

    name: str
    direction: Direction
    terms: tuple = field(repr=False)  # ((weight, on, fn), ...)

    def __init__(
        self,
        name: str,
        direction: Direction,
        fn: Callable | None = None,
        on: On = "scores",
        *,
        terms: tuple | None = None,
    ) -> None:
        if direction not in ("higher", "lower"):
            raise ValueError(f"direction must be 'higher' or 'lower', got {direction!r}")
        if (fn is None) == (terms is None):
            raise TypeError("Give a metric either fn or terms.")
        if terms is None:
            if on not in ("scores", "model", "all"):
                raise ValueError(f"on must be 'scores', 'model' or 'all', got {on!r}")
            terms = ((1.0, on, fn),)
        else:
            # Old Objective terms were (weight, fn(y, s, m)) pairs.
            terms = tuple(t if len(t) == 3 else (t[0], "all", t[1]) for t in terms)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "terms", terms)

    @property
    def fn(self) -> Callable:
        """The scoring function of a single-term metric, uncalled and unclipped."""
        if len(self.terms) != 1 or self.terms[0][0] != 1.0:
            raise AttributeError(f"{self.name!r} combines several terms; use .score().")
        return self.terms[0][2]

    @property
    def on(self) -> On:
        """What a single-term metric reads; "all" for combinations of different kinds."""
        kinds = {kind for _, kind, _ in self.terms}
        return kinds.pop() if len(kinds) == 1 else "all"

    def score(self, y_true: np.ndarray, y_score: np.ndarray, model: Any = None) -> float:
        """
        Evaluate the metric. ``y_score`` is clipped to [1e-8, 1 - 1e-8] first.

        ``model`` is required when any term reads the fitted model.
        """
        y_score = np.clip(np.asarray(y_score, dtype=float), 1e-8, 1 - 1e-8)
        y_true = np.asarray(y_true)
        total = 0.0
        for weight, kind, fn in self.terms:
            if kind == "scores":
                value = fn(y_true, y_score)
            elif kind == "model":
                if model is None:
                    raise ValueError(f"{self.name!r} scores the fitted model; pass model=.")
                value = fn(model)
            else:
                value = fn(y_true, y_score, model)
            total += weight * value
        return float(total)

    def improvement(self, score_challenger: float, score_baseline: float) -> float:
        """Signed improvement: positive means the challenger is better."""
        delta = score_challenger - score_baseline
        return delta if self.direction == "higher" else -delta

    def as_scorer(self) -> Callable[[Any, Any, Any], float]:
        """
        An sklearn scorer ``(estimator, X, y) -> float`` for this metric.

        Pass it as ``scoring=`` to estimators with early stopping, e.g.
        ``make_hgb(early_stopping=True, scoring=(AVG_PRECISION - 0.001 * N_ITER).as_scorer())``.
        sklearn maximises scorers, so a "lower" metric is negated.
        """
        sign = 1.0 if self.direction == "higher" else -1.0

        def scorer(estimator, X, y) -> float:
            p = estimator.predict_proba(X)[:, 1]
            return sign * self.score(np.asarray(y), p, model=estimator)

        scorer.__name__ = f"scorer({self.name})"
        return scorer

    def rename(self, name: str) -> Metric:
        """A copy with a new name."""
        return Metric(name, self.direction, terms=self.terms)

    def with_direction(self, direction: Direction) -> Metric:
        """A copy with a different direction."""
        return Metric(self.name, direction, terms=self.terms)

    # ── arithmetic ────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce(other: Any) -> Metric:
        if isinstance(other, Metric):
            return other
        if isinstance(other, (int, float)):
            c = float(other)
            return Metric(str(c), "higher", lambda y, s, _c=c: _c)
        raise TypeError(f"Cannot combine Metric with {type(other).__name__!r}")

    def _merge(self, other: Any, sign: float) -> Metric:
        rhs = Metric._coerce(other)
        name = f"{self.name} + {rhs.name}" if sign > 0 else f"{self.name} - {rhs.name}"
        terms = self.terms + tuple((sign * w, kind, fn) for w, kind, fn in rhs.terms)
        return Metric(name, self.direction, terms=terms)

    def __add__(self, other):
        return self._merge(other, +1.0)

    def __radd__(self, other):
        return Metric._coerce(other).__add__(self)

    def __sub__(self, other):
        return self._merge(other, -1.0)

    def __rsub__(self, other):
        return Metric._coerce(other).__sub__(self)

    def __mul__(self, scalar: float) -> Metric:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Can only multiply a Metric by a number, got {type(scalar).__name__!r}")
        s = float(scalar)
        name = f"-{self.name}" if s == -1.0 else f"{s} * {self.name}"
        direction = self.direction if s >= 0 else ("lower" if self.direction == "higher" else "higher")
        return Metric(name, direction, terms=tuple((w * s, kind, fn) for w, kind, fn in self.terms))

    def __rmul__(self, scalar: float) -> Metric:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Metric:
        if not isinstance(scalar, (int, float)) or scalar == 0:
            raise TypeError("Divisor must be a non-zero number")
        return self.__mul__(1.0 / float(scalar))

    def __neg__(self) -> Metric:
        return self.__mul__(-1.0)


# ── aliases for the former Objective / ModelMetric types ─────────────────────

Objective = Metric


def ModelMetric(name: str, direction: Direction, fn: Callable[[Any], float]) -> Metric:  # noqa: N802
    """A metric of the fitted model, e.g. its tree count. Same as ``Metric(..., on="model")``."""
    return Metric(name, direction, lambda m: float(fn(m)), on="model")


def make_objective(
    fn: Callable[[np.ndarray, np.ndarray, Any], float],
    name: str,
    direction: Direction = "higher",
) -> Metric:
    """A metric from ``fn(y_true, y_score, model)``, for non-linear combinations."""
    return Metric(name, direction, fn, on="all")


# ── smooth average precision ─────────────────────────────────────────────────

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

    As prior_strength -> 0, this recovers standard sklearn AP. Returns NaN
    when there are no positives.
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


def make_smooth_ap(prior_mean: float = 0.1, prior_strength: float = 20) -> Metric:
    """Smooth average precision with prior expected precision p0 and strength m."""
    def fn(y_true: np.ndarray, y_score: np.ndarray) -> float:
        return _average_precision_smooth(
            y_true, y_score, prior_mean=prior_mean, prior_strength=prior_strength
        )

    return Metric(f"avg_precision_smooth(p0={prior_mean},m={prior_strength})", "higher", fn)


# ── pre-built metrics ────────────────────────────────────────────────────────

AUC = Metric("auc", "higher", lambda y, p: roc_auc_score(y, p))
LOGLOSS = Metric("logloss", "lower", lambda y, p: log_loss(y, p))
AVG_PRECISION = Metric("average_precision", "higher", lambda y, p: average_precision_score(y, p))
BRIER = Metric("brier_score", "lower", lambda y, p: brier_score_loss(y, p))
AVG_PRECISION_SMOOTH = make_smooth_ap(prior_mean=0.1, prior_strength=20)

# Shape of the score vector (labels unused): smooth, untied scores rank better
# under heavy imbalance. See scores.py.
SCORE_ENTROPY = Metric("score_entropy", "higher", lambda y, p: score_entropy(p)[0])
NORMALIZED_SCORE_ENTROPY = Metric(
    "normalized_score_entropy", "higher", lambda y, p: score_entropy(p)[1]
)
TIE_RATE = Metric("tie_rate", "lower", lambda y, p: tie_rate(p)["tie_rate"])
MAX_SCORE_GAP = Metric("max_score_gap", "lower", lambda y, p: max_score_gap(p))

# Fitted-model terms, usually as penalties: AVG_PRECISION - 0.01 * N_ITER.
N_ITER = Metric(
    "n_iter", "lower",
    # LogisticRegression stores n_iter_ as a one-element array; boosters as an int.
    lambda m: float(np.max(np.atleast_1d(getattr(m, "n_iter_", 0)))),
    on="model",
)
N_LEAVES = Metric(
    "n_leaves", "lower", lambda m: float(np.sum(getattr(m, "n_leaves_", [0]))), on="model"
)
COEF_L1 = Metric(
    "coef_l1", "lower",
    lambda m: float(np.abs(np.asarray(getattr(m, "coef_", [[0]]))).sum()),
    on="model",
)

ALL_METRICS: dict[str, Metric] = {
    m.name: m for m in [AUC, LOGLOSS, AVG_PRECISION, BRIER, AVG_PRECISION_SMOOTH]
}
