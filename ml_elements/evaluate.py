"""
evaluate.py
-----------
``Setup``, ``Result``, ``evaluate`` and ``compare``: the core of an experiment.

    trials = sweep(...)                                  # list[Trial]
    result = evaluate(trials, setups, metrics, budget)   # Result
    table  = compare(result, "baseline", "challenger", AVG_PRECISION)

``evaluate`` does, for each trial:
  1. Sample train / valid data from the trial's population.
  2. Fit one model per setup.
  3. Score every setup on ``budget.n_repeats`` fresh test draws.

Seeds: train ``seed_train + seed_offset``, valid ``seed_valid + seed_offset``,
test repeat r ``seed_test_base + seed_offset * 1000 + r``. These match the
former ``TrialRunner``, so results are identical.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .protocols import ModelBackend
from .trial import DataBudget, Trial

ModelFactory = Callable[[], ModelBackend]
ScoreTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Setup:
    """
    One thing being evaluated: a name, its feature columns, its model, and an
    optional transform of the model's positive-class scores.

    Parameters
    ----------
    name : str
        Label used in result tables, e.g. ``"baseline"``.
    features : list[str]
        Columns the model sees.
    model : zero-argument factory, optional
        Overrides the trial's and ``evaluate``'s default model.
    scores : Callable[[ndarray], ndarray], optional
        Applied to ``predict_proba(X)[:, 1]`` before scoring, e.g.
        ``lambda p: shrink(p, pi=0.001, lam=0.1)``.

    Examples
    --------
    >>> Setup("baseline", ["x1", "x2"])
    >>> Setup("hgb", ["x1", "x2", "x3"], model=make_hgb(iterations=100))
    """

    name: str
    features: list[str]
    model: ModelFactory | None = None
    scores: ScoreTransform | None = None


@dataclass
class TrialResult:
    """
    One trial's part of a ``Result``: the fitted models, the data they saw,
    and the per-repeat scores.

    Attributes
    ----------
    trial : Trial
    models : dict[str, ModelBackend]
        Fitted model per setup name. A setup whose fit failed is absent.
    df_train, df_valid : pd.DataFrame
        Training and validation samples (validation is never used for fitting).
    scores : pd.DataFrame
        One row per (setup, repeat): ``trial_name``, ``trial_value``,
        ``repeat``, ``test_seed``, ``setup``, ``features``, then one column
        per metric.
    """

    trial: Trial
    models: dict[str, ModelBackend]
    df_train: pd.DataFrame
    df_valid: pd.DataFrame
    scores: pd.DataFrame


@dataclass
class Result:
    """
    Output of ``evaluate``.

    Attributes
    ----------
    trial_results : list[TrialResult]
        One per trial, in input order. Use it to reach fitted models:
        ``result.trial_results[2].models["challenger"]``.
    scores : pd.DataFrame
        All trials' score rows concatenated.
    metrics : list[str]
        Metric column names, in the order given to ``evaluate``.
    failures : pd.DataFrame
        One row per setup that failed when ``on_error="record"``:
        ``trial_name``, ``trial_value``, ``setup``, ``stage``, ``error``.
    """

    trial_results: list[TrialResult]
    scores: pd.DataFrame
    metrics: list[str] = field(default_factory=list)
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)


SetupSpec = Setup | list[str] | tuple


def evaluate(
    trials: Trial | Sequence[Trial],
    setups: Sequence[Setup] | dict[str, SetupSpec],
    metrics: Sequence,
    budget: DataBudget,
    *,
    model: ModelFactory | None = None,
    target: str = "y",
    on_error: Literal["raise", "record"] = "raise",
    n_jobs: int = 1,
    progress: bool = False,
) -> Result:
    """
    Fit every setup on every trial and score it on repeated test draws.

    Parameters
    ----------
    trials : Trial or list[Trial]
    setups : list[Setup], or a dict ``name -> features`` / ``name -> (features, factory)``
    metrics : list[Metric]
        Each is scored on every (setup, repeat).
    budget : DataBudget
        Default sizes and seeds; a trial's ``budget_override`` wins.
    model : zero-argument factory, optional
        Default model for setups and trials that name none.
    target : str
        Label column in population samples.
    on_error : {"raise", "record"}
        ``"record"`` keeps going when one setup's fit or prediction raises:
        its metric values become NaN and the error lands in ``Result.failures``.
    n_jobs : int
        Parallel trials via joblib (``-1`` = all CPUs).
    progress : bool
        Show a progress bar (rich if installed, else print).

    Returns
    -------
    Result
    """
    trial_list = [trials] if isinstance(trials, Trial) else list(trials)
    if not trial_list:
        raise ValueError("trials must not be empty.")
    setup_list = _as_setups(setups)
    if not metrics:
        raise ValueError("metrics must not be empty.")

    def one(trial: Trial) -> tuple[TrialResult, list[dict]]:
        return _evaluate_trial(trial, setup_list, list(metrics), budget, model, target, on_error)

    outcomes = _map(one, trial_list, n_jobs=n_jobs, progress=progress)
    trial_results = [tr for tr, _ in outcomes]
    failures = [f for _, fs in outcomes for f in fs]
    return Result(
        trial_results=trial_results,
        scores=pd.concat([tr.scores for tr in trial_results], ignore_index=True),
        metrics=[m.name for m in metrics],
        failures=pd.DataFrame(failures, columns=["trial_name", "trial_value", "setup", "stage", "error"]),
    )


def compare(
    result: Result,
    baseline: str,
    challenger: str,
    metric,
    *,
    per_repeat: bool = False,
) -> pd.DataFrame:
    """
    Compare two setups on ``metric``, trial by trial.

    ``improvement`` is challenger minus baseline, sign-flipped for
    lower-is-better metrics, so positive always means the challenger is better.

    Parameters
    ----------
    per_repeat : bool
        False (default): one row per trial with ``mean_improvement``,
        ``std_improvement``, ``p_improvement_gt_0``, ``mean_baseline_score``,
        ``mean_challenger_score``, ``n_repeats``, plus ``{setup}_{metric}_mean``
        for every metric in the result.
        True: one row per (trial, repeat) with both raw scores, ``raw_delta``
        and ``improvement``.
    """
    key_cols = ["trial_name", "trial_value", "repeat"]
    wide = (
        result.scores
        .pivot_table(index=key_cols, columns="setup", values=metric.name)
        .reset_index()
    )
    for col in (baseline, challenger):
        if col not in wide.columns:
            raise ValueError(f"Setup {col!r} not found. Available: {list(wide.columns)}")

    raw_delta = wide[challenger] - wide[baseline]
    wide["baseline_setup"] = baseline
    wide["challenger_setup"] = challenger
    wide["metric"] = metric.name
    wide["raw_delta"] = raw_delta
    wide["improvement"] = raw_delta if metric.direction == "higher" else -raw_delta
    if per_repeat:
        return wide

    summary = (
        wide
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
    metric_names = result.metrics or [metric.name]
    for setup in (baseline, challenger):
        means = (
            result.scores[result.scores["setup"] == setup]
            .groupby(["trial_name", "trial_value"])[metric_names]
            .mean()
            .rename(columns={m: f"{setup}_{m}_mean" for m in metric_names})
            .reset_index()
        )
        summary = summary.merge(means, on=["trial_name", "trial_value"], how="left")
    return summary


# ── internals ────────────────────────────────────────────────────────────────

def _as_setups(setups) -> list[Setup]:
    if isinstance(setups, dict):
        out = []
        for name, spec in setups.items():
            if isinstance(spec, Setup):
                out.append(spec)
            elif isinstance(spec, list):
                out.append(Setup(name, list(spec)))
            else:
                features, factory = spec
                out.append(Setup(name, list(features), model=factory))
        setups = out
    setups = list(setups)
    if len(setups) < 2:
        raise ValueError("setups must contain at least two entries (baseline and challenger).")
    names = [s.name for s in setups]
    if len(set(names)) != len(names):
        raise ValueError(f"setup names must be unique, got {names}")
    return setups


def _evaluate_trial(trial, setups, metrics, budget, default_model, target, on_error):
    budget = trial.budget_override if trial.budget_override is not None else budget
    trial_model = trial.model_factory if trial.model_factory is not None else default_model
    population = trial.population

    df_train = population.sample(budget.n_train, budget.seed_train + trial.seed_offset)
    df_valid = population.sample(budget.n_valid, budget.seed_valid + trial.seed_offset)
    y_train = df_train[target]

    failures: list[dict] = []

    def note_failure(setup: Setup, stage: str) -> None:
        failures.append({
            "trial_name": trial.name, "trial_value": trial.value, "setup": setup.name,
            "stage": stage, "error": traceback.format_exc(limit=1).strip().splitlines()[-1],
        })

    models: dict[str, ModelBackend] = {}
    for setup in setups:
        factory = setup.model or trial_model
        if factory is None:
            raise ValueError(
                f"No model for setup {setup.name!r}. Give the setup a model, "
                "the trial a model_factory, or evaluate(model=...)."
            )
        try:
            fitted = factory()
            fitted.fit(df_train[setup.features], y_train)
            models[setup.name] = fitted
        except Exception:
            if on_error == "raise":
                raise
            note_failure(setup, "fit")

    rows = []
    for repeat in range(1, budget.n_repeats + 1):
        test_seed = budget.seed_test_base + trial.seed_offset * 1_000 + repeat
        df_test = population.sample(budget.n_test, test_seed)
        y_test = df_test[target].to_numpy()
        for setup in setups:
            row = {
                "trial_name": trial.name,
                "trial_value": trial.value,
                "repeat": repeat,
                "test_seed": test_seed,
                "setup": setup.name,
                "features": ",".join(setup.features),
            }
            fitted = models.get(setup.name)
            values = {m.name: np.nan for m in metrics}
            if fitted is not None:
                try:
                    p_hat = fitted.predict_proba(df_test[setup.features])[:, 1]
                    if setup.scores is not None:
                        p_hat = setup.scores(p_hat)
                    values = {m.name: m.score(y_test, p_hat, model=fitted) for m in metrics}
                except Exception:
                    if on_error == "raise":
                        raise
                    if repeat == 1:
                        note_failure(setup, "score")
            row.update(values)
            rows.append(row)

    return TrialResult(trial, models, df_train, df_valid, pd.DataFrame(rows)), failures


def _map(fn, items, *, n_jobs: int, progress: bool):
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError("Parallel evaluation requires joblib: uv sync --extra viz") from exc
        return Parallel(n_jobs=n_jobs)(delayed(fn)(item) for item in items)
    if not progress:
        return [fn(item) for item in items]
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
    except ImportError:
        out = []
        for i, item in enumerate(items, 1):
            print(f"Trial {i}/{len(items)}: {item.name}={item.value:.4g}")
            out.append(fn(item))
        return out
    out = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as bar:
        task = bar.add_task("Running trials…", total=len(items))
        for item in items:
            bar.update(task, description=f"[cyan]{item.name}={item.value:.4g}[/cyan]")
            out.append(fn(item))
            bar.advance(task)
    return out
