"""
study.py
--------
``StudyResult`` and ``Study``: run a list of trials and aggregate results.

A ``Study`` is the top-level experiment runner. Give it a list of ``Trial``
objects — from a manual list, a ``SobolSearch``, or anything else — and call
``.run(trials)``. It returns a ``StudyResult`` with every ``TrialResult``
plus a combined scores DataFrame, ready for improvement analysis.

Parallelism
-----------
``Study(n_jobs=N)`` runs trials in parallel using ``joblib.Parallel``.
Trials are fully independent so this is trivially safe:

    study = Study(runner, primary_metric=AUC, n_jobs=4)

Progress
--------
Install ``rich`` for a live progress bar. Falls back to plain ``print`` if
not available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from .metrics import Metric
from .runner import TrialResult, TrialRunner
from .trial import Trial


@dataclass
class StudyResult:
    """
    Complete output of a ``Study.run()`` call.

    Attributes
    ----------
    trial_results : list[TrialResult]
        One ``TrialResult`` per trial, in the same order as the input list.
        Access models, train data, and per-trial scores through these.
    scores : pd.DataFrame
        All trial scores concatenated. Columns: ``trial_name``,
        ``trial_value``, ``repeat``, ``test_seed``, ``setup``,
        ``features``, + one column per metric.

    Examples
    --------
    Get the challenger model from the 3rd trial:

        result.trial_results[2].models["challenger"].predict_proba(X)

    Filter scores to one setup:

        result.scores[result.scores["setup"] == "challenger"]
    """

    trial_results: list[TrialResult]
    scores: pd.DataFrame


class Study:
    """
    Runs a list of trials and aggregates the results.

    Parameters
    ----------
    runner : TrialRunner
        Configured trial runner (setups, model factory, metrics, budget).
    primary_metric : Metric
        The metric used for improvement computation and summarisation.
        All metrics in the runner are still scored; this one determines
        what "better" means.
    n_jobs : int
        Number of parallel workers. 1 = sequential. -1 = all CPUs.
        Requires ``joblib`` (``pip install joblib``).

    Examples
    --------
    >>> study = Study(runner, primary_metric=AUC, n_jobs=4)
    >>> result = study.run(trials)
    >>> improv = study.improvements(result, baseline="baseline", challenger="challenger")
    >>> summary = study.summarize(improv)
    """

    def __init__(
        self,
        runner: TrialRunner,
        primary_metric: Metric,
        n_jobs: int = 1,
    ) -> None:
        self.runner = runner
        self.primary_metric = primary_metric
        self.n_jobs = n_jobs

    def run(self, trials: list[Trial]) -> StudyResult:
        """
        Run all trials and return a ``StudyResult``.

        Parameters
        ----------
        trials : list[Trial]
            Trials to run. Order is preserved in ``StudyResult.trial_results``.

        Returns
        -------
        StudyResult
        """
        if not trials:
            raise ValueError("trials must not be empty.")

        trial_results = self._run_trials(trials)
        scores = pd.concat(
            [tr.scores for tr in trial_results], ignore_index=True
        )

        return StudyResult(trial_results=trial_results, scores=scores)

    def improvements(
        self,
        result: StudyResult,
        baseline: str,
        challenger: str,
    ) -> pd.DataFrame:
        """
        Compute per-repeat improvement: challenger metric minus baseline metric
        (sign-corrected for lower-is-better metrics).

        Parameters
        ----------
        result : StudyResult
        baseline : str
            Setup name that acts as the reference, e.g. ``"baseline"``.
        challenger : str
            Setup name being evaluated, e.g. ``"challenger"``.

        Returns
        -------
        pd.DataFrame
            Wide format. One row per (trial_name, trial_value, repeat).
            Columns include both raw setup scores, ``raw_delta``,
            and ``improvement`` (always positive-is-better).
        """
        metric = self.primary_metric
        key_cols = ["trial_name", "trial_value", "repeat"]

        wide = (
            result.scores
            .pivot_table(index=key_cols, columns="setup", values=metric.name)
            .reset_index()
        )

        for col in (baseline, challenger):
            if col not in wide.columns:
                raise ValueError(
                    f"Setup {col!r} not found. Available: {list(wide.columns)}"
                )

        raw_delta = wide[challenger] - wide[baseline]
        improvement = raw_delta if metric.direction == "higher" else -raw_delta

        wide["baseline_setup"] = baseline
        wide["challenger_setup"] = challenger
        wide["metric"] = metric.name
        wide["raw_delta"] = raw_delta
        wide["improvement"] = improvement

        return wide

    def summarize(self, improvements: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate improvement statistics across repeats.

        Parameters
        ----------
        improvements : pd.DataFrame
            Output of ``improvements()``.

        Returns
        -------
        pd.DataFrame
            One row per (trial_name, trial_value). Columns:
            ``mean_improvement``, ``std_improvement``,
            ``p_improvement_gt_0``, ``mean_baseline_score``,
            ``mean_challenger_score``, ``n_repeats``.
        """
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

    def full_summary(
        self,
        result: StudyResult,
        baseline: str,
        challenger: str,
    ) -> pd.DataFrame:
        """
        Like ``summarize()``, but adds mean scores for **every** tracked metric
        and both setups as extra columns.

        Parameters
        ----------
        result : StudyResult
        baseline : str
        challenger : str

        Returns
        -------
        pd.DataFrame
            One row per (trial_name, trial_value). All columns from
            ``summarize()`` plus ``{setup}_{metric}_mean`` for every
            (setup, metric) combination.

        Examples
        --------
        >>> summary = study.full_summary(result, baseline="without_x3", challenger="with_x3")
        >>> summary[["trial_value", "without_x3_auc_mean", "with_x3_auc_mean",
        ...           "without_x3_average_precision_mean", "with_x3_average_precision_mean"]]
        """
        improv = self.improvements(result, baseline, challenger)
        summary = self.summarize(improv)

        key_cols = ["trial_name", "trial_value"]
        scores = result.scores
        metric_names = [m.name for m in self.runner.metrics]

        for setup in (baseline, challenger):
            subset = scores[scores["setup"] == setup]
            means = (
                subset
                .groupby(key_cols)[metric_names]
                .mean()
                .rename(columns={m: f"{setup}_{m}_mean" for m in metric_names})
                .reset_index()
            )
            summary = summary.merge(means, on=key_cols, how="left")

        return summary

    def _run_trials(self, trials: list[Trial]) -> list[TrialResult]:
        if self.n_jobs == 1:
            return self._run_sequential(trials)
        return self._run_parallel(trials)

    def _run_sequential(self, trials: list[Trial]) -> list[TrialResult]:
        results = []
        use_rich, console, progress_cls = _try_import_rich()

        if use_rich:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Running trials…", total=len(trials))
                for trial in trials:
                    progress.update(
                        task,
                        description=f"[cyan]{trial.name}={trial.value:.4g}[/cyan]",
                    )
                    results.append(self.runner.run(trial))
                    progress.advance(task)
        else:
            for i, trial in enumerate(trials, 1):
                print(f"Trial {i}/{len(trials)}: {trial.name}={trial.value:.4g}")
                results.append(self.runner.run(trial))

        return results

    def _run_parallel(self, trials: list[Trial]) -> list[TrialResult]:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "Parallel execution requires joblib. Install with: pip install joblib"
            ) from exc

        results: list[TrialResult] = Parallel(n_jobs=self.n_jobs)(
            delayed(self.runner.run)(trial) for trial in trials
        )
        return results


def _try_import_rich() -> tuple[bool, object | None, object | None]:
    try:
        from rich.console import Console
        return True, Console(), None
    except ImportError:
        return False, None, None
