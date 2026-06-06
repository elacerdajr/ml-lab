"""
objectives.py
-------------
Composite scoring objectives.

An ``Objective`` is a weighted linear combination of scoring *terms*.
Each term can be:

- A **prediction metric** — depends on (y_true, y_score), e.g. AP, AUC.
- A **model metric**      — depends on the fitted model, e.g. tree count,
  coefficient L1 norm, inference time.

Arithmetic operators let you compose objectives without boilerplate::

    from ml_elements import AVG_PRECISION, AUC
    from ml_elements.objectives import ModelMetric, N_ITER, N_LEAVES

    # penalise model complexity
    obj = AVG_PRECISION - 0.01 * N_ITER

    # blend two prediction metrics
    obj = 0.7 * AUC + 0.3 * AVG_PRECISION

    # arbitrary expression via make_objective()
    obj = make_objective(
        fn=lambda y, s, m: average_precision_score(y, s) / (1 + m.n_iter_),
        name="ap_per_iter",
    )

``Objective`` satisfies the same interface as ``Metric``
(``name``, ``direction``, ``score``, ``improvement``) and can be passed
wherever a ``Metric`` is expected — ``TrialRunner``, ``Study``,
``plot_study``, etc.

Pre-built model metrics
-----------------------
N_ITER      Number of boosting iterations (HGB / CatBoost).
N_LEAVES    Total leaf count across all trees (HGB).
COEF_L1     L1 norm of linear coefficients (LogisticRegression).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

# Lazy import of Metric to avoid circular dependency at module load time.
# (metrics.py also lazily imports from objectives.py for its operators.)


# ── unified term signature ────────────────────────────────────────────────────
# Every term is (weight: float, fn: (y_true, y_score, model) -> float).
# Prediction-only terms ignore `model`; model-only terms ignore y_true/y_score.

_TermFn = Callable[[np.ndarray, np.ndarray, Any], float]


def _lift_pred(fn: Callable) -> _TermFn:
    """Wrap a (y_true, y_score) callable into the unified signature."""
    return lambda y_true, y_score, model: fn(y_true, y_score)


def _lift_model(fn: Callable) -> _TermFn:
    """Wrap a (model,) callable into the unified signature."""
    return lambda y_true, y_score, model: fn(model)


# ── ModelMetric ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelMetric:
    """
    A metric that depends on the fitted model, not its predictions.

    Typical uses: number of trees, parameter count, L1 weight norm,
    inference latency — any quantity you want to trade off against AP/AUC.

    Parameters
    ----------
    name : str
    direction : {"higher", "lower"}
        Almost always ``"lower"`` for complexity / size penalties.
    fn : Callable[[model], float]
        Receives the fitted estimator and returns a scalar.

    Examples
    --------
    >>> N_ITER = ModelMetric("n_iter", "lower", fn=lambda m: float(m.n_iter_))
    >>> obj = AVG_PRECISION - 0.01 * N_ITER
    """

    name: str
    direction: Literal["higher", "lower"]
    fn: Callable[[Any], float]

    def score(self, model: Any) -> float:
        return float(self.fn(model))

    # ── promote to Objective for arithmetic ───────────────────────────────────

    def _as_obj(self) -> "Objective":
        return Objective(
            name=self.name,
            direction=self.direction,
            terms=((1.0, _lift_model(self.fn)),),
        )

    def __add__(self, other):       return self._as_obj().__add__(other)
    def __radd__(self, other):      return self._as_obj().__radd__(other)
    def __sub__(self, other):       return self._as_obj().__sub__(other)
    def __rsub__(self, other):      return self._as_obj().__rsub__(other)
    def __mul__(self, s: float):    return self._as_obj().__mul__(s)
    def __rmul__(self, s: float):   return self.__mul__(s)
    def __truediv__(self, s: float):return self._as_obj().__truediv__(s)
    def __neg__(self):              return self.__mul__(-1.0)


# ── Objective ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Objective:
    """
    Composite scoring objective: a weighted linear combination of terms.

    Build via arithmetic operators on ``Metric`` / ``ModelMetric`` objects,
    or via :func:`make_objective` for non-linear expressions.

    Parameters
    ----------
    name : str
    direction : {"higher", "lower"}
    terms : tuple of (weight, fn) pairs
        Internal representation. Prefer arithmetic operators.

    Interface
    ---------
    Same as ``Metric``: ``name``, ``direction``, ``score``, ``improvement``.
    The extra ``model`` argument to ``score`` is optional — if all terms
    are prediction-based, ``model`` is unused.

    Examples
    --------
    >>> from ml_elements import AVG_PRECISION, AUC
    >>> from ml_elements.objectives import N_ITER, N_LEAVES

    Penalise model complexity:

    >>> obj = AVG_PRECISION - 0.01 * N_ITER
    >>> obj.score(y_true, y_score, model)
    0.298

    Blend two metrics:

    >>> obj = 0.6 * AVG_PRECISION + 0.4 * AUC

    Rename after construction:

    >>> obj = (AVG_PRECISION - 0.005 * N_LEAVES).rename("ap_leaf_penalised")

    Use as primary metric in a Study (works identically to Metric):

    >>> study = Study(runner, primary_metric=obj)
    """

    name: str
    direction: Literal["higher", "lower"]
    terms: tuple = field(repr=False)   # tuple[tuple[float, _TermFn], ...]

    # ── scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        model: Any = None,
    ) -> float:
        """
        Evaluate the composite objective.

        Parameters
        ----------
        y_true : array-like of 0/1
        y_score : array-like of scores / probabilities
        model : fitted estimator
            Required when any term is model-intrinsic. Safe to omit when
            the objective contains only prediction-based terms.
        """
        y_score = np.clip(np.asarray(y_score, dtype=float), 1e-8, 1 - 1e-8)
        y_true  = np.asarray(y_true)
        return float(sum(w * fn(y_true, y_score, model) for w, fn in self.terms))

    def improvement(self, score_challenger: float, score_baseline: float) -> float:
        """Signed improvement (positive = challenger is better)."""
        delta = score_challenger - score_baseline
        return delta if self.direction == "higher" else -delta

    # ── convenience ───────────────────────────────────────────────────────────

    def rename(self, name: str) -> "Objective":
        """Return a copy with a new name (does not mutate)."""
        return Objective(name=name, direction=self.direction, terms=self.terms)

    def with_direction(self, direction: Literal["higher", "lower"]) -> "Objective":
        """Return a copy with a different optimisation direction."""
        return Objective(name=self.name, direction=direction, terms=self.terms)

    # ── internal coercion ─────────────────────────────────────────────────────

    @staticmethod
    def _coerce(other: Any) -> "Objective":
        """Convert Metric / ModelMetric / scalar to Objective."""
        if isinstance(other, Objective):
            return other
        if isinstance(other, ModelMetric):
            return other._as_obj()
        # Lazy import to avoid circular dependency
        from .metrics import Metric
        if isinstance(other, Metric):
            return Objective(
                name=other.name,
                direction=other.direction,
                terms=((1.0, _lift_pred(other.fn)),),
            )
        if isinstance(other, (int, float)):
            c = float(other)
            return Objective(
                name=str(c),
                direction="higher",
                terms=((1.0, lambda y, s, m, _c=c: _c),),
            )
        raise TypeError(
            f"Cannot combine Objective with {type(other).__name__!r}"
        )

    def _merge(self, other: Any, sign: float) -> "Objective":
        rhs  = Objective._coerce(other)
        name = (
            f"{self.name} + {rhs.name}"
            if sign > 0
            else f"{self.name} - {rhs.name}"
        )
        combined = self.terms + tuple((sign * w, fn) for w, fn in rhs.terms)
        return Objective(name=name, direction=self.direction, terms=combined)

    # ── arithmetic operators ──────────────────────────────────────────────────

    def __add__(self, other):
        return self._merge(other, +1.0)

    def __radd__(self, other):
        return Objective._coerce(other).__add__(self)

    def __sub__(self, other):
        return self._merge(other, -1.0)

    def __rsub__(self, other):
        return Objective._coerce(other).__sub__(self)

    def __mul__(self, scalar: float) -> "Objective":
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Can only multiply Objective by a scalar, got {type(scalar).__name__!r}")
        s    = float(scalar)
        name = f"-{self.name}" if s == -1.0 else f"{s} * {self.name}"
        dir_ = self.direction if s >= 0 else (
            "lower" if self.direction == "higher" else "higher"
        )
        return Objective(
            name=name,
            direction=dir_,
            terms=tuple((w * s, fn) for w, fn in self.terms),
        )

    def __rmul__(self, scalar: float) -> "Objective":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Objective":
        if not isinstance(scalar, (int, float)) or scalar == 0:
            raise TypeError("Divisor must be a non-zero scalar")
        return self.__mul__(1.0 / float(scalar))

    def __neg__(self) -> "Objective":
        return self.__mul__(-1.0)


# ── factory ───────────────────────────────────────────────────────────────────

def make_objective(
    fn: Callable[[np.ndarray, np.ndarray, Any], float],
    name: str,
    direction: Literal["higher", "lower"] = "higher",
) -> Objective:
    """
    Build an Objective from an arbitrary callable.

    Use when the objective is **not** a weighted linear combination
    (e.g. ``AP / (1 + n_iter)``).

    Parameters
    ----------
    fn : callable (y_true, y_score, model) -> float
        ``model`` may be ``None`` if not needed.
    name : str
    direction : {"higher", "lower"}

    Examples
    --------
    >>> from sklearn.metrics import average_precision_score
    >>> ratio = make_objective(
    ...     fn=lambda y, s, m: average_precision_score(y, s) / (1 + m.n_iter_),
    ...     name="ap_per_iter",
    ... )
    """
    return Objective(name=name, direction=direction, terms=((1.0, fn),))


# ── pre-built model metrics ───────────────────────────────────────────────────

N_ITER = ModelMetric(
    name="n_iter",
    direction="lower",
    fn=lambda m: float(getattr(m, "n_iter_", 0)),
)
"""Number of boosting iterations. Works with HGB and CatBoost."""

N_LEAVES = ModelMetric(
    name="n_leaves",
    direction="lower",
    fn=lambda m: float(np.sum(getattr(m, "n_leaves_", [0]))),
)
"""Total leaf count across all trees (HistGradientBoosting)."""

COEF_L1 = ModelMetric(
    name="coef_l1",
    direction="lower",
    fn=lambda m: float(np.abs(np.asarray(getattr(m, "coef_", [[0]]))).sum()),
)
"""L1 norm of linear model coefficients (LogisticRegression, LinearSVC, etc.)."""
