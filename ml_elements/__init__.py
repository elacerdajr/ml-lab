"""
ml_exp
------
Composable building blocks for ML experiments.

Quick start
-----------
Import everything in one line:

    from ml_exp import *

Or import specific blocks:

    from ml_exp import (
        GaussianBinaryDGP, RealDataDGP, ShiftedDGP,
        make_logistic, make_hgb, make_catboost, make_sklearn,
        AUC, LOGLOSS, AVG_PRECISION, AVG_PRECISION_SMOOTH, Metric, make_smooth_ap,
        DataBudget, Trial,
        TrialRunner, TrialResult,
        Study, StudyResult,
        SobolSearch, RandomSearch, ManualSearch,
        Comparator,
        plot_study, plot_calibration, plot_feature_importance,
        plot_search_heatmap, plot_score_distributions,
    )

Vocabulary
----------
DataBudget       How much data to use (n_train, n_valid, n_test, seeds, n_repeats)
Trial            One experimental condition (DGP + condition label + optional overrides)
TrialResult      Output of one trial: fitted models, train/val data, per-repeat scores
Study            Runs a list of trials, aggregates results, computes improvements
StudyResult      Full output of Study.run(): all TrialResults + combined scores DataFrame
SobolSearch      Generates trials from a Sobol quasi-random parameter grid
RandomSearch     Generates trials from uniform random sampling
ManualSearch     Generates trials from an explicit list of parameter dicts
Comparator       Statistical comparison via BayesianAPComparator

Typical flow
------------
    budget = DataBudget(n_train=2000, n_valid=500, n_test=2000,
                        seed_train=101, seed_valid=202, seed_test_base=10_000,
                        n_repeats=20)

    setups = {"baseline": ["x1", "x2"], "challenger": ["x1", "x2", "x3"]}

    runner = TrialRunner(setups=setups, model_factory=make_logistic(),
                         metrics=[AUC, AVG_PRECISION], budget=budget)
    study  = Study(runner, primary_metric=AUC, n_jobs=1)

    trials = ManualSearch(
        [{"p_pos": p} for p in [0.02, 0.05, 0.10, 0.25, 0.50]],
        trial_name="p_pos",
    ).trials(
        dgp_fn=lambda p: GaussianBinaryDGP(
            p_pos=p["p_pos"],
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

from .analysis import Comparator
from .rules import RuleCondition, Rule, RuleSet, RuleExtractor
from .rule_viz import rule_matrix_html, save_rule_report
from .embedding_pca import (
    build_embedding_texts,
    encode_with_sentence_transformer,
    fit_embedding_pca,
    load_site_descriptions,
    plot_first_three_pcs,
    plot_pca_variance,
)
from .dgp import GaussianBinaryDGP, RealDataDGP, ShiftedDGP
from .objectives import (
    COEF_L1,
    N_ITER,
    N_LEAVES,
    ModelMetric,
    Objective,
    make_objective,
)
from .metrics import (
    ALL_METRICS,
    AVG_PRECISION,
    AVG_PRECISION_SMOOTH,
    AUC,
    BRIER,
    LOGLOSS,
    Metric,
    make_smooth_ap,
)
from .models import (
    make_catboost,
    make_decision_tree,
    make_figs,
    make_greedy_rule_list,
    make_hgb,
    make_logistic,
    make_rule_fit,
    make_sklearn,
)
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
    "make_decision_tree",
    "make_figs",
    "make_greedy_rule_list",
    "make_rule_fit",
    "make_sklearn",
    # Objectives
    "Objective",
    "ModelMetric",
    "make_objective",
    "N_ITER",
    "N_LEAVES",
    "COEF_L1",
    # Metrics
    "Metric",
    "AUC",
    "LOGLOSS",
    "BRIER",
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
    # Search
    "SobolSearch",
    "RandomSearch",
    "ManualSearch",
    # Analysis
    "Comparator",
    # Rule extraction and visualization
    "RuleCondition",
    "Rule",
    "RuleSet",
    "RuleExtractor",
    "rule_matrix_html",
    "save_rule_report",
    # Embedding PCA utilities
    "build_embedding_texts",
    "encode_with_sentence_transformer",
    "fit_embedding_pca",
    "load_site_descriptions",
    "plot_first_three_pcs",
    "plot_pca_variance",
    # Plots
    "plot_study",
    "plot_calibration",
    "plot_feature_importance",
    "plot_search_heatmap",
    "plot_score_distributions",
]
