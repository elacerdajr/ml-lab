"""
study.py
--------
``Study``: the former way to run and compare a list of trials, kept as a thin
wrapper over ``evaluate`` and ``compare``.

    study = Study(runner, primary_metric=AUC)
    result = study.run(trials)                       # evaluate(trials, ...)
    study.improvements(result, "base", "chal")       # compare(..., per_repeat=True)
    study.summarize(improvements)                    # compare(...) without metric means
    study.full_summary(result, "base", "chal")       # compare(...)

``StudyResult`` is ``Result``. New code should call ``evaluate`` and
``compare`` directly; see ``evaluate.py``.
"""

from __future__ import annotations

import pandas as pd

from .evaluate import Result, compare, evaluate
from .metrics import Metric
from .runner import TrialRunner
from .trial import Trial

StudyResult = Result

__all__ = ["Study", "StudyResult"]


class Study:
    """
    Runs a list of trials through a ``TrialRunner``. Prefer ``evaluate``.

    Parameters
    ----------
    runner : TrialRunner
    primary_metric : Metric
        The metric ``improvements`` and ``summarize`` compare on.
    n_jobs : int
        Parallel trials via joblib (``-1`` = all CPUs).
    """

    def __init__(self, runner: TrialRunner, primary_metric: Metric, n_jobs: int = 1) -> None:
        self.runner = runner
        self.primary_metric = primary_metric
        self.n_jobs = n_jobs

    def run(self, trials: list[Trial]) -> Result:
        """Run all trials and return their ``Result``."""
        r = self.runner
        return evaluate(
            trials, r._setups, r.metrics, r.budget,
            model=r.model_factory, target=r.target_col, n_jobs=self.n_jobs, progress=True,
        )

    def improvements(self, result: Result, baseline: str, challenger: str) -> pd.DataFrame:
        """Per-repeat improvement of challenger over baseline on the primary metric."""
        return compare(result, baseline, challenger, self.primary_metric, per_repeat=True)

    def summarize(self, improvements: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the output of ``improvements`` to one row per trial."""
        baseline = improvements["baseline_setup"].iloc[0]
        challenger = improvements["challenger_setup"].iloc[0]
        return (
            improvements
            .groupby(["trial_name", "trial_value"])
            .agg(
                mean_improvement=("improvement", "mean"),
                std_improvement=("improvement", "std"),
                p_improvement_gt_0=("improvement", lambda x: float((x > 0).mean())),
                mean_baseline_score=(baseline, "mean"),
                mean_challenger_score=(challenger, "mean"),
                n_repeats=("improvement", "count"),
            )
            .reset_index()
        )

    def full_summary(self, result: Result, baseline: str, challenger: str) -> pd.DataFrame:
        """``summarize`` plus ``{setup}_{metric}_mean`` for every tracked metric."""
        return compare(result, baseline, challenger, self.primary_metric)
