"""
ml_elements
-----------
Composable building blocks for ML experiments.

Vocabulary
----------
Population     What a trial samples labelled rows from: ``sample(n, seed)``.
               GaussianPopulation, ResampledPopulation, ShiftedPopulation.
Setup          One thing being evaluated: name, features, model, optional score transform.
Metric         A named score with a direction. Arithmetic on metrics returns a metric.
Trial          One condition to test: a population plus a budget (DataBudget).
Result         What ``evaluate`` returns: per-repeat scores, fitted models, failures.
Run            One execution of an experiment folder under one profile.
RuleSet        Rules extracted from an interpretable model.

Operators
---------
population.sample(n, seed)            -> DataFrame
sweep(space, population=...)          -> list[Trial]
evaluate(trials, setups, metrics, budget) -> Result
compare(result, baseline, challenger, metric) -> DataFrame
embed_leaves(model, X)                -> leaf-index feature matrix
extract_rules(model, X, y, features)  -> RuleSet
record(run, tables)                   -> output folder with run_meta.json

Typical flow
------------
    from ml_elements import *

    trials = sweep(
        [{"p_pos": p} for p in [0.02, 0.05, 0.10, 0.25, 0.50]],
        population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info={"x1": 0.85, "x2": 0.55, "x3": 0.15}),
        name="p_pos",
    )
    setups = [Setup("baseline", ["x1", "x2"]), Setup("challenger", ["x1", "x2", "x3"])]
    result = evaluate(trials, setups, [AUC, AVG_PRECISION], DataBudget.quick(n=2000, n_repeats=20),
                      model=make_logistic())
    table = compare(result, "baseline", "challenger", AVG_PRECISION)

    result.trial_results[2].models["challenger"].predict_proba(X_new)[:, 1]

Former names
------------
GaussianBinaryDGP, RealDataDGP, ShiftedDGP, DGP       -> the Population names
TrialRunner, Study, StudyResult                       -> evaluate, compare, Result
SobolSearch, RandomSearch, ManualSearch               -> sweep
Objective, ModelMetric, make_objective                -> Metric
RuleExtractor                                         -> extract_rules
These still work. ``Comparator`` was removed: it depended on a
``bayesian_ap_comparator`` module that is not in the repository.
"""

from .dgp import GaussianBinaryDGP, RealDataDGP, ShiftedDGP
from .embedding_pca import (
    build_embedding_texts,
    encode_with_sentence_transformer,
    fit_embedding_pca,
    load_site_descriptions,
    plot_first_three_pcs,
    plot_pca_variance,
)
from .evaluate import Result, Setup, TrialResult, compare, evaluate
from .metrics import (
    ALL_METRICS,
    AUC,
    AVG_PRECISION,
    AVG_PRECISION_SMOOTH,
    BRIER,
    COEF_L1,
    LOGLOSS,
    MAX_SCORE_GAP,
    N_ITER,
    N_LEAVES,
    NORMALIZED_SCORE_ENTROPY,
    SCORE_ENTROPY,
    TIE_RATE,
    Metric,
    ModelMetric,
    Objective,
    make_objective,
    make_smooth_ap,
)
from .models import (
    embed_leaves,
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
from .population import GaussianPopulation, ResampledPopulation, ShiftedPopulation
from .protocols import DGP, MetricFn, ModelBackend, Population
from .rich_logger import RichLogger
from .rule_viz import rule_matrix_html, save_rule_report
from .rules import Rule, RuleCondition, RuleExtractor, RuleSet, extract_rules
from .run import Run, record, report_figures
from .runner import TrialRunner
from .scores import (
    calibrate_sigmoid,
    max_score_gap,
    noise_rank,
    positive_scores,
    row_noise,
    score_entropy,
    shrink,
    smooth_labels,
    tie_rate,
    to_weighted_rows,
)
from .search import ManualSearch, RandomSearch, SobolSearch
from .study import Study, StudyResult
from .sweep import sweep, sweep_points
from .trial import DataBudget, Trial

__all__ = [
    # Nouns
    "Population",
    "GaussianPopulation",
    "ResampledPopulation",
    "ShiftedPopulation",
    "Setup",
    "Metric",
    "Trial",
    "DataBudget",
    "Result",
    "TrialResult",
    "Run",
    "RuleSet",
    # Operators
    "sweep",
    "sweep_points",
    "evaluate",
    "compare",
    "embed_leaves",
    "extract_rules",
    "record",
    "report_figures",
    # Metrics
    "AUC",
    "AVG_PRECISION",
    "AVG_PRECISION_SMOOTH",
    "LOGLOSS",
    "BRIER",
    "SCORE_ENTROPY",
    "NORMALIZED_SCORE_ENTROPY",
    "TIE_RATE",
    "MAX_SCORE_GAP",
    "N_ITER",
    "N_LEAVES",
    "COEF_L1",
    "ALL_METRICS",
    "make_smooth_ap",
    # Scores
    "score_entropy",
    "tie_rate",
    "max_score_gap",
    "shrink",
    "noise_rank",
    "row_noise",
    "smooth_labels",
    "to_weighted_rows",
    "calibrate_sigmoid",
    "positive_scores",
    # Model factories
    "make_logistic",
    "make_hgb",
    "make_catboost",
    "make_decision_tree",
    "make_figs",
    "make_greedy_rule_list",
    "make_rule_fit",
    "make_sklearn",
    # Protocols
    "ModelBackend",
    "MetricFn",
    # Rules and reports
    "Rule",
    "RuleCondition",
    "rule_matrix_html",
    "save_rule_report",
    # Embedding PCA utilities
    "build_embedding_texts",
    "encode_with_sentence_transformer",
    "fit_embedding_pca",
    "load_site_descriptions",
    "plot_first_three_pcs",
    "plot_pca_variance",
    # Plots and logging
    "plot_study",
    "plot_calibration",
    "plot_feature_importance",
    "plot_search_heatmap",
    "plot_score_distributions",
    "RichLogger",
    # Former names
    "DGP",
    "GaussianBinaryDGP",
    "RealDataDGP",
    "ShiftedDGP",
    "TrialRunner",
    "Study",
    "StudyResult",
    "SobolSearch",
    "RandomSearch",
    "ManualSearch",
    "Objective",
    "ModelMetric",
    "make_objective",
    "RuleExtractor",
]
