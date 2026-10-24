"""
runner.py
---------
``TrialRunner``: the former way to evaluate one trial, kept as a thin wrapper.

    TrialRunner(setups, model_factory, metrics, budget).run(trial)

is ``evaluate([trial], setups, metrics, budget, model=model_factory)``'s only
``TrialResult``, with identical seeds and scores. New code should call
``evaluate`` directly; see ``evaluate.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from .evaluate import TrialResult, _as_setups, evaluate
from .metrics import Metric
from .protocols import ModelBackend
from .trial import DataBudget, Trial

__all__ = ["TrialResult", "TrialRunner"]


class TrialRunner:
    """
    Evaluates setups on one trial at a time. Prefer ``evaluate``.

    Parameters
    ----------
    setups : dict[str, list[str] | tuple[list[str], Callable]]
        Setup name to feature columns, or to ``(features, model_factory)``.
    model_factory : Callable[[], ModelBackend] | None
        Default model for setups without their own.
    metrics : list[Metric]
    budget : DataBudget
    target_col : str
    """

    def __init__(
        self,
        setups: dict[str, list[str] | tuple[list[str], Callable]],
        model_factory: Callable[[], ModelBackend] | None,
        metrics: list[Metric],
        budget: DataBudget,
        target_col: str = "y",
    ) -> None:
        if not metrics:
            raise ValueError("metrics must not be empty.")
        self._setups = _as_setups(setups)
        self.setups: dict[str, list[str]] = {s.name: s.features for s in self._setups}
        self.model_factory = model_factory
        self.metrics = metrics
        self.budget = budget
        self.target_col = target_col

    def run(self, trial: Trial) -> TrialResult:
        """Execute one trial: sample data, fit models, score on repeated test draws."""
        result = evaluate(
            [trial], self._setups, self.metrics, self.budget,
            model=self.model_factory, target=self.target_col,
        )
        return result.trial_results[0]
