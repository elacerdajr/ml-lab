"""
runner.py
---------
``TrialResult`` and ``TrialRunner``: the atomic unit of an experiment.

A ``TrialRunner`` knows how to run exactly one ``Trial``:
  1. Sample train / valid / test data from the trial's DGP.
  2. Fit one model per setup.
  3. Evaluate each model on N repeated test draws.
  4. Return a ``TrialResult`` that keeps **everything** — models, data, scores.

The models in ``TrialResult.models`` are fully accessible after the run:

    result = runner.run(trial)
    p_hat  = result.models["challenger"].predict_proba(X_new)[:, 1]
    coef   = result.models["baseline"].coef_

``TrialRunner`` is stateless — call ``run()`` as many times as needed.
``Study`` uses it internally but you can use it directly for a single trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .metrics import Metric
from .protocols import ModelBackend
from .trial import DataBudget, Trial


@dataclass
class TrialResult:
    """
    Complete output of a single trial run.

    Attributes
    ----------
    trial : Trial
        The trial that produced this result.
    models : dict[str, ModelBackend]
        Fitted model for each setup, keyed by setup name.
        Access fitted models directly:
            ``result.models["challenger"].predict_proba(X)``
    df_train : pd.DataFrame
        Training data used to fit the models.
    df_valid : pd.DataFrame
        Validation data (available for calibration, threshold tuning, etc.).
    scores : pd.DataFrame
        One row per (setup, repeat). Columns include all computed metrics,
        ``repeat``, ``test_seed``, ``setup``, ``trial_name``, ``trial_value``.
    """

    trial: Trial
    models: dict[str, ModelBackend]
    df_train: pd.DataFrame
    df_valid: pd.DataFrame
    scores: pd.DataFrame


class TrialRunner:
    """
    Fits and evaluates models for a single trial.

    The runner is configured once with shared settings (setups, model
    factory, metrics, budget) and then called with individual ``Trial``
    objects. Any trial can override the model factory or budget via
    ``Trial.model_factory`` and ``Trial.budget_override``.

    Parameters
    ----------
    setups : dict[str, list[str] | tuple[list[str], Callable]]
        Mapping from setup name to either:

        * ``list[str]`` — feature columns; the shared ``model_factory`` is used.
        * ``(list[str], factory)`` — feature columns **and** a per-setup
          model factory that overrides the shared one for this setup only.

        Examples::

            # same model, different features
            setups = {
                "baseline":   ["x1", "x2"],
                "challenger": ["x1", "x2", "x3"],
            }

            # different models, same features
            setups = {
                "auc_model": (["x1", "x2", "x3"], make_hgb(scoring="roc_auc")),
                "ap_model":  (["x1", "x2", "x3"], make_hgb(scoring="average_precision")),
            }

    model_factory : Callable[[], ModelBackend]
        Default zero-argument factory. Used for any setup that does not
        specify its own factory. May be ``None`` if every setup supplies
        its own factory.
    metrics : list[Metric]
        All metrics to score on each test repeat. All are recorded in
        ``TrialResult.scores``; the primary metric (for improvement
        direction) is selected at the ``Study`` level.
    budget : DataBudget
        Default data budget. Overridable per-trial.
    target_col : str
        Name of the label column in samples from the DGP.

    Examples
    --------
    >>> runner = TrialRunner(
    ...     setups={"baseline": ["x1", "x2"], "challenger": ["x1", "x2", "x3"]},
    ...     model_factory=make_logistic(),
    ...     metrics=[AUC, AVG_PRECISION],
    ...     budget=DataBudget(n_train=2000, n_valid=500, n_test=2000,
    ...                       seed_train=101, seed_valid=202,
    ...                       seed_test_base=10_000, n_repeats=20),
    ... )
    >>> result = runner.run(trial)
    >>> result.models["challenger"].coef_
    """

    def __init__(
        self,
        setups: dict[str, "list[str] | tuple[list[str], Callable]"],
        model_factory: "Callable[[], ModelBackend] | None",
        metrics: list[Metric],
        budget: DataBudget,
        target_col: str = "y",
    ) -> None:
        if len(setups) < 2:
            raise ValueError("setups must contain at least two entries (baseline and challenger).")
        if not metrics:
            raise ValueError("metrics must not be empty.")

        # Normalise setups: always store features and per-setup factory separately.
        # Per-setup factory takes precedence over model_factory; None = use default.
        self.setups: dict[str, list[str]] = {}
        self._setup_factories: dict[str, Callable | None] = {}
        for name, spec in setups.items():
            if isinstance(spec, list):
                self.setups[name] = spec
                self._setup_factories[name] = None
            else:
                features, fac = spec
                self.setups[name] = list(features)
                self._setup_factories[name] = fac

        self.model_factory = model_factory
        self.metrics = metrics
        self.budget = budget
        self.target_col = target_col

    def run(self, trial: Trial) -> TrialResult:
        """
        Execute one trial: sample data, fit models, score on repeated test draws.

        Parameters
        ----------
        trial : Trial
            The trial to run. Any ``model_factory`` or ``budget_override``
            set on the trial takes precedence over the runner defaults.

        Returns
        -------
        TrialResult
            All fitted models, data splits, and scores.
        """
        budget = trial.budget_override if trial.budget_override is not None else self.budget
        factory = trial.model_factory if trial.model_factory is not None else self.model_factory

        train_seed = budget.seed_train + trial.seed_offset
        valid_seed = budget.seed_valid + trial.seed_offset

        df_train = trial.dgp.sample(budget.n_train, train_seed)
        df_valid = trial.dgp.sample(budget.n_valid, valid_seed)

        models = self._fit_models(df_train, factory)

        score_rows = []
        for repeat in range(1, budget.n_repeats + 1):
            test_seed = budget.seed_test_base + trial.seed_offset * 1_000 + repeat
            df_test = trial.dgp.sample(budget.n_test, test_seed)
            score_rows.extend(
                self._score_repeat(models, df_test, trial, repeat, test_seed)
            )

        scores = pd.DataFrame(score_rows)

        return TrialResult(
            trial=trial,
            models=models,
            df_train=df_train,
            df_valid=df_valid,
            scores=scores,
        )

    def _fit_models(
        self,
        df_train: pd.DataFrame,
        factory: "Callable[[], ModelBackend] | None",
    ) -> dict[str, ModelBackend]:
        models: dict[str, ModelBackend] = {}
        y_train = df_train[self.target_col]

        for setup_name, features in self.setups.items():
            setup_fac = self._setup_factories.get(setup_name) or factory
            if setup_fac is None:
                raise ValueError(
                    f"No model factory for setup {setup_name!r}. "
                    "Provide model_factory on the runner or a per-setup factory in setups."
                )
            model = setup_fac()
            model.fit(df_train[features], y_train)
            models[setup_name] = model

        return models

    def _score_repeat(
        self,
        models: dict[str, ModelBackend],
        df_test: pd.DataFrame,
        trial: Trial,
        repeat: int,
        test_seed: int,
    ) -> list[dict]:
        y_test = df_test[self.target_col].to_numpy()
        rows = []

        for setup_name, features in self.setups.items():
            model = models[setup_name]
            p_hat = model.predict_proba(df_test[features])[:, 1]

            row: dict = {
                "trial_name": trial.name,
                "trial_value": trial.value,
                "repeat": repeat,
                "test_seed": test_seed,
                "setup": setup_name,
                "features": ",".join(features),
            }
            for metric in self.metrics:
                row[metric.name] = metric.score(y_test, p_hat, model=model)

            rows.append(row)

        return rows
