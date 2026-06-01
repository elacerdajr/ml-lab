# ml_exp — ML Experiment Blocks

Composable building blocks for binary classification experiments. Each block does one thing and connects to the others through clean interfaces — swap any piece without touching the rest.

## Install

```bash
pip install numpy pandas scikit-learn matplotlib scipy
# optional: pip install catboost joblib rich
```

## Vocabulary

| Name | What it is |
|---|---|
| `DataBudget` | How much data to use — train/valid/test sizes, seeds, number of test repeats |
| `Trial` | One experimental condition: a DGP, a label, and optional per-trial overrides |
| `TrialResult` | Everything from one trial — fitted models, data splits, per-repeat scores |
| `Study` | Runs a list of trials, aggregates results, computes improvements |
| `StudyResult` | Full output: all `TrialResult`s + combined scores DataFrame |
| `SobolSearch` | Generates trials from a quasi-random Sobol grid |
| `RandomSearch` | Generates trials from uniform random sampling |
| `ManualSearch` | Generates trials from an explicit list of parameter dicts |
| `Comparator` | Statistical comparison using bootstrap AP distributions |

---

## Quick start

```python
from ml_exp import (
    GaussianBinaryDGP,
    make_logistic,
    AUC, AVG_PRECISION,
    DataBudget, Trial,
    TrialRunner, Study,
    ManualSearch,
    plot_study,
)

# 1. How much data
budget = DataBudget(
    n_train=2_000, n_valid=500, n_test=2_000,
    seed_train=101, seed_valid=202, seed_test_base=10_000,
    n_repeats=20,
)

# 2. What feature sets to compare
setups = {
    "baseline":   ["x1", "x2"],
    "challenger": ["x1", "x2", "x3"],
}

# 3. Configure the runner
runner = TrialRunner(
    setups=setups,
    model_factory=make_logistic(),
    metrics=[AUC, AVG_PRECISION],
    budget=budget,
)
study = Study(runner, primary_metric=AUC)

# 4. Define what to vary
trials = ManualSearch(
    [{"p_pos": p} for p in [0.02, 0.05, 0.10, 0.25, 0.50]],
    trial_name="p_pos",
).trials(
    dgp_fn=lambda p: GaussianBinaryDGP(
        p_pos=p["p_pos"],
        info={"x1": 0.85, "x2": 0.55, "x3": 0.15},
    )
)

# 5. Run
result  = study.run(trials)
improv  = study.improvements(result, baseline="baseline", challenger="challenger")
summary = study.summarize(improv)

# 6. Plot
fig = plot_study(summary, metric=AUC)
fig.savefig("study_result.png")
```

---

## Access models after the run

```python
# Direct access — no re-fitting needed
model = result.trial_results[2].models["challenger"]
p_hat = model.predict_proba(X_new)[:, 1]

# Feature importance (trees) or coefficients (logistic)
model.feature_importances_
model.coef_
```

---

## Sobol hyperparameter search

```python
from ml_exp import SobolSearch, make_hgb

search = SobolSearch(
    param_space={"p_pos": (0.02, 0.5), "x3_info": (0.0, 1.5), "lr": (0.01, 0.3)},
    n_points=32,
    seed=0,
)

trials = search.trials(
    dgp_fn=lambda p: GaussianBinaryDGP(
        p_pos=p["p_pos"],
        info={"x1": 0.85, "x2": 0.55, "x3": p["x3_info"]},
    ),
    model_fn=lambda p: make_hgb(learning_rate=p["lr"]),
)

result = study.run(trials)
```

---

## Swap model — zero other changes

```python
from ml_exp import make_hgb, make_catboost

# HGB
runner_hgb = TrialRunner(setups=setups, model_factory=make_hgb(), metrics=[AUC], budget=budget)

# CatBoost
runner_cat = TrialRunner(setups=setups, model_factory=make_catboost(depth=6), metrics=[AUC], budget=budget)
```

---

## Swap DGP to real data — same runner

```python
from ml_exp import RealDataDGP

real_dgp   = RealDataDGP(df=df_historical, target_col="click")
real_trial = Trial(name="real", value=0.0, dgp=real_dgp, seed_offset=0)

result = study.run([real_trial])
```

---

## Add label noise / covariate shift

```python
from ml_exp import ShiftedDGP

noisy = ShiftedDGP(
    base=GaussianBinaryDGP(p_pos=0.15, info={"x1": 0.85, "x2": 0.55}),
    shift_fn=lambda df: df.assign(x1=df["x1"] + 2.0),  # shift x1 at test time
)
```

---

## Parallel execution

```python
study = Study(runner, primary_metric=AUC, n_jobs=8)  # all other code unchanged
```

Requires `pip install joblib`.

---

## Statistical comparison

```python
from ml_exp import Comparator

cmp = Comparator(n_boot=3_000)

# Full study overview (one row per trial)
report = cmp.full_report(result, baseline="baseline", challenger="challenger")
report[["trial_value", "challenger_ap_observed", "p_challenger_beats_baseline"]]

# Drill into one trial
bac = cmp.for_trial(result, trial_value=0.10, baseline="baseline", challenger="challenger")
bac.ranking_report()
bac.pairwise_report()
```

---

## Plots

```python
from ml_exp import (
    plot_study,
    plot_calibration,
    plot_feature_importance,
    plot_search_heatmap,
    plot_score_distributions,
)

# Improvement curve
plot_study(summary, metric=AUC)

# Calibration on validation set
plot_calibration(result.trial_results[0], setup="challenger")

# Feature importance / coefficients
plot_feature_importance(result.trial_results[0], setup="challenger")

# Score spread across trial values
plot_score_distributions(result, metric_name="auc")

# 2D heatmap after Sobol search (requires param_x / param_y columns in summary)
plot_search_heatmap(result, param_x="p_pos", param_y="x3_info", summary=sobol_summary)
```

---

## Custom metric

```python
from ml_exp import Metric
from sklearn.metrics import f1_score

F1 = Metric(
    name="f1",
    direction="higher",
    fn=lambda y, p: f1_score(y, p > 0.5),
)

runner = TrialRunner(..., metrics=[AUC, F1], ...)
```

---

## File layout

```
ml_exp/
├── protocols.py   DGP, ModelBackend, MetricFn — structural interfaces
├── dgp.py         GaussianBinaryDGP, RealDataDGP, ShiftedDGP
├── models.py      make_logistic, make_hgb, make_catboost, make_sklearn
├── metrics.py     Metric, AUC, LOGLOSS, AVG_PRECISION, AVG_PRECISION_SMOOTH
├── trial.py       DataBudget, Trial
├── runner.py      TrialRunner, TrialResult
├── study.py       Study, StudyResult
├── search.py      SobolSearch, RandomSearch, ManualSearch
├── analysis.py    Comparator (wraps BayesianAPComparator)
└── plots.py       plot_study, plot_calibration, plot_feature_importance,
                   plot_search_heatmap, plot_score_distributions
```
