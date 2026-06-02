"""
ml_elements
-----------
Composable building blocks for ML experiments.

Quick start
-----------
Import everything in one line:

    from ml_elements import *

Or import specific blocks:

    from ml_elements import (
        GaussianBinaryDGP, RealDataDGP, ShiftedDGP,
        make_logistic, make_hgb, make_catboost, make_sklearn,
        AUC, LOGLOSS, AVG_PRECISION, AVG_PRECISION_SMOOTH, Metric, make_smooth_ap,
        DataBudget, Trial,
        TrialRunner, TrialResult,
        Study, StudyResult,
        ScenarioSweep,                       # sweep experiment design knobs
        SobolSearch, RandomSearch, ManualSearch,  # search model hyperparameters
        Comparator, APComparison,
        plot_study, plot_calibration, plot_feature_importance,
        plot_search_heatmap, plot_score_distributions,
    )

Two kinds of knob — keep them separate
---------------------------------------
ScenarioSweep    *Sweep* DGP design knobs (p_pos, info, ...). You vary the
                 experiment to map a response curve; there is no "best" value.
SobolSearch /    *Search* model hyperparameters (learning_rate, depth, ...)
RandomSearch     to find the single best model within a fixed scenario.
See ``sweep.py`` for the full distinction.

Vocabulary
----------
DataBudget       How much data to use (n_train, n_valid, n_test, seeds, n_repeats)
Trial            One experimental condition (DGP + condition label + optional overrides)
TrialResult      Output of one trial: fitted models, train/val data, per-repeat scores
Study            Runs a list of trials, aggregates results, computes improvements
StudyResult      Full output of Study.run(): all TrialResults + combined scores DataFrame
ScenarioSweep    Generates trials by sweeping DGP design knobs on a grid
SobolSearch      Generates trials from a Sobol quasi-random hyperparameter grid
RandomSearch     Generates trials from uniform random hyperparameter sampling
ManualSearch     Generates trials from an explicit list of parameter dicts
Comparator       Paired bootstrap AP comparison of setups (see APComparison)

Typical flow
------------
    budget = DataBudget(n_train=2000, n_valid=500, n_test=2000,
                        seed_train=101, seed_valid=202, seed_test_base=10_000,
                        n_repeats=20)

    setups = {"baseline": ["x1", "x2"], "challenger": ["x1", "x2", "x3"]}

    runner = TrialRunner(setups=setups, model_factory=make_logistic(),
                         metrics=[AUC, AVG_PRECISION], budget=budget)
    study  = Study(runner, primary_metric=AUC, n_jobs=1)

    # Sweep the class balance — a design choice, not something to optimize:
    trials = ScenarioSweep.over("p_pos", [0.02, 0.05, 0.10, 0.25, 0.50]).trials(
        dgp_fn=lambda s: GaussianBinaryDGP(
            p_pos=s["p_pos"],
            info={"x1": 0.85, "x2": 0.55, "x3": 0.15},
        )
    )

    result  = study.run(trials)
    improv  = study.improvements(result, baseline="baseline", challenger="challenger")
    summary = study.summarize(improv)

    fig = plot_study(summary, metric=AUC)
    fig.savefig("study_result.png")

    # Access models after the run:
    model = result.trial_results[2].models["challenger"]
    model.predict_proba(X_new)[:, 1]
"""

from .analysis import APComparison, Comparator
from .dgp import GaussianBinaryDGP, RealDataDGP, ShiftedDGP
from .metrics import (
    ALL_METRICS,
    AVG_PRECISION,
    AVG_PRECISION_SMOOTH,
    AUC,
    LOGLOSS,
    Metric,
    make_smooth_ap,
)
from .models import make_catboost, make_hgb, make_logistic, make_sklearn
from .plots import (
    plot_calibration,
    plot_feature_importance,
    plot_score_distributions,
    plot_search_heatmap,
    plot_study,
)
from .protocols import DGP, MetricFn, ModelBackend
from .runner import TrialResult, TrialRunner
from .search import ManualSearch, RandomSearch, SobolSearch
from .study import Study, StudyResult
from .sweep import ScenarioSweep
from .trial import DataBudget, Trial

__all__ = [
    # Protocols
    "DGP",
    "ModelBackend",
    "MetricFn",
    # DGPs
    "GaussianBinaryDGP",
    "RealDataDGP",
    "ShiftedDGP",
    # Model factories
    "make_logistic",
    "make_hgb",
    "make_catboost",
    "make_sklearn",
    # Metrics
    "Metric",
    "AUC",
    "LOGLOSS",
    "AVG_PRECISION",
    "AVG_PRECISION_SMOOTH",
    "ALL_METRICS",
    "make_smooth_ap",
    # Experiment primitives
    "DataBudget",
    "Trial",
    # Runner
    "TrialRunner",
    "TrialResult",
    # Study
    "Study",
    "StudyResult",
    # Scenario sweeps (DGP design knobs)
    "ScenarioSweep",
    # Hyperparameter search (model knobs)
    "SobolSearch",
    "RandomSearch",
    "ManualSearch",
    # Analysis
    "Comparator",
    "APComparison",
    # Plots
    "plot_study",
    "plot_calibration",
    "plot_feature_importance",
    "plot_search_heatmap",
    "plot_score_distributions",
]
