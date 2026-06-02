# ml_elements — ML Experiment Blocks

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
| `ScenarioSweep` | **Sweeps** DGP design knobs (`p_pos`, `info`, …) on a grid |
| `SobolSearch` | **Searches** model hyperparameters via a quasi-random Sobol grid |
| `RandomSearch` | Searches model hyperparameters via uniform random sampling |
| `ManualSearch` | Generates trials from an explicit list of parameter dicts |
| `Comparator` | Statistical comparison using paired bootstrap AP distributions |

> **Sweep vs. search — they are not the same thing.** `p_pos` and `info` are
> *experiment design* choices: you sweep them on a grid (`ScenarioSweep`) to map
> a response curve — there is no "best" value to find. `learning_rate`, `depth`,
> etc. are *model hyperparameters*: you search them (`SobolSearch`) for the single
> best model within a fixed scenario. Don't put `p_pos`/`info` in a search space.

---

## Quick start

```python
from ml_elements import (
    GaussianBinaryDGP,
    make_logistic,
    AUC, AVG_PRECISION,
    DataBudget, Trial,
    TrialRunner, Study,
    ScenarioSweep,
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

# 4. Sweep the design knob you want to study (here: class balance)
trials = ScenarioSweep.over("p_pos", [0.02, 0.05, 0.10, 0.25, 0.50]).trials(
    dgp_fn=lambda s: GaussianBinaryDGP(
        p_pos=s["p_pos"],
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

Search **model** knobs only — keep the scenario (`p_pos`, `info`) fixed:

```python
from ml_elements import SobolSearch, make_hgb

search = SobolSearch(
    param_space={"lr": (0.01, 0.3), "max_leaf_nodes": (8, 64)},
    n_points=32,
    seed=0,
)

trials = search.trials(
    # scenario is fixed — only the model varies
    dgp_fn=lambda p: GaussianBinaryDGP(
        p_pos=0.15,
        info={"x1": 0.85, "x2": 0.55, "x3": 0.35},
    ),
    model_fn=lambda p: make_hgb(
        learning_rate=p["lr"],
        max_leaf_nodes=int(p["max_leaf_nodes"]),
    ),
)

result = study.run(trials)
```

> To study how the *scenario* affects results (e.g. how the value of `x3`
> changes with `p_pos`), sweep it with `ScenarioSweep` instead — that gives a
> clean curve. If you need both, nest a `SobolSearch` inside each scenario.

---

## Swap model — zero other changes

```python
from ml_elements import make_hgb, make_catboost

# HGB
runner_hgb = TrialRunner(setups=setups, model_factory=make_hgb(), metrics=[AUC], budget=budget)

# CatBoost
runner_cat = TrialRunner(setups=setups, model_factory=make_catboost(depth=6), metrics=[AUC], budget=budget)
```

---

## Swap DGP to real data — same runner

```python
from ml_elements import RealDataDGP

real_dgp   = RealDataDGP(df=df_historical, target_col="click")
real_trial = Trial(name="real", value=0.0, dgp=real_dgp, seed_offset=0)

result = study.run([real_trial])
```

---

## Add label noise / covariate shift

```python
from ml_elements import ShiftedDGP

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
from ml_elements import Comparator

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
from ml_elements import (
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
from ml_elements import Metric
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
ml_elements/
├── protocols.py   DGP, ModelBackend, MetricFn — structural interfaces
├── dgp.py         GaussianBinaryDGP, RealDataDGP, ShiftedDGP
├── models.py      make_logistic, make_hgb, make_catboost, make_sklearn
├── metrics.py     Metric, AUC, LOGLOSS, AVG_PRECISION, AVG_PRECISION_SMOOTH
├── trial.py       DataBudget, Trial
├── runner.py      TrialRunner, TrialResult
├── study.py       Study, StudyResult
├── sweep.py       ScenarioSweep — sweep DGP design knobs (p_pos, info)
├── search.py      SobolSearch, RandomSearch, ManualSearch — model hyperparameters
├── analysis.py    Comparator, APComparison (paired bootstrap on real predictions)
└── plots.py       plot_study, plot_calibration, plot_feature_importance,
                   plot_search_heatmap, plot_score_distributions
```
