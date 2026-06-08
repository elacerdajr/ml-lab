# Cookbook

Short recipes for common tasks. Each snippet assumes `ml_elements` is importable and a `budget` / `runner` / `study` have been set up as in the [quickstart](quickstart.md).

---

## Use a composite training objective

Reduce model complexity while keeping AP high.

```python
from ml_elements import AVG_PRECISION, N_ITER, make_hgb

# penalise extra boosting iterations
objective = AVG_PRECISION - 0.001 * N_ITER

runner = TrialRunner(
    setups=setups,
    model_factory=make_hgb(scoring=objective),
    metrics=[AUC, AVG_PRECISION],
    budget=budget,
)
```

`N_ITER` is the number of trees the model actually grew. Penalising it pushes early stopping to kick in sooner, giving a smaller model for a tiny AP cost.

---

## Compare AUC-trained vs AP-trained models

Each setup can carry its own model factory as a `(features, factory)` tuple. Setups that only specify a feature list still use the shared `model_factory`.

```python
from ml_elements import AUC, AVG_PRECISION, make_hgb, TrialRunner

runner = TrialRunner(
    setups={
        "auc_model": (["x1", "x2", "x3"], make_hgb(scoring="roc_auc")),
        "ap_model":  (["x1", "x2", "x3"], make_hgb(scoring="average_precision")),
    },
    model_factory=None,   # every setup supplies its own factory
    metrics=[AUC, AVG_PRECISION],
    budget=budget,
)
```

You can also mix the two forms — some setups with a shared factory, one with a custom one:

```python
runner = TrialRunner(
    setups={
        "default":  ["x1", "x2"],                                        # uses model_factory
        "ap_tuned": (["x1", "x2"], make_hgb(scoring="average_precision")), # overrides
    },
    model_factory=make_hgb(),
    metrics=[AUC, AVG_PRECISION],
    budget=budget,
)
```

---

## Custom metric

```python
from ml_elements import Metric
from sklearn.metrics import average_precision_score
import numpy as np

# AP computed only on the top-k predictions
def ap_at_k(y_true, y_score, k=200):
    idx = np.argsort(y_score)[::-1][:k]
    return average_precision_score(y_true[idx], y_score[idx])

AP_AT_200 = Metric(name="ap@200", direction="higher", fn=lambda y, p: ap_at_k(y, p, k=200))

runner = TrialRunner(..., metrics=[AUC, AP_AT_200], ...)
```

---

## Run on real data

```python
from ml_elements import RealDataDGP, Trial
import pandas as pd

df = pd.read_parquet("data/clickstream.parquet")

dgp   = RealDataDGP(df=df, target_col="click")
trial = Trial(name="real", value=0.0, dgp=dgp, seed_offset=0)

result = study.run([trial])
```

`RealDataDGP.sample(n, seed)` draws rows with replacement, so repeated test draws still give independent estimates.

---

## Test robustness to covariate shift

```python
from ml_elements import ShiftedDGP, GaussianBinaryDGP

base = GaussianBinaryDGP(p_pos=0.10, info={"x1": 0.8, "x2": 0.5})

# train on base, test on shifted distribution
shifted = ShiftedDGP(
    base=base,
    shift_fn=lambda df: df.assign(x1=df["x1"] + 1.0),
)

trials = [
    Trial(name="no_shift", value=0, dgp=base,    seed_offset=0),
    Trial(name="shift",    value=1, dgp=shifted,  seed_offset=0),
]

result = study.run(trials)
```

---

## Sobol grid search over two parameters

```python
from ml_elements import SobolSearch, GaussianBinaryDGP, make_hgb

trials = SobolSearch(
    param_space={
        "p_pos":    (0.02, 0.50),
        "x3_info":  (0.00, 1.50),
    },
    n_points=32,
    seed=0,
).trials(
    dgp_fn=lambda p: GaussianBinaryDGP(
        p_pos=p["p_pos"],
        info={"x1": 0.8, "x2": 0.5, "x3": p["x3_info"]},
    ),
    model_fn=lambda p: make_hgb(),
)

result = study.run(trials)
```

Sobol sampling covers the space more evenly than random, so you get better coverage with the same number of points.

---

## Parallel execution

```python
# requires: pip install joblib
study = Study(runner, primary_metric=AUC, n_jobs=8)
result = study.run(trials)   # trials run concurrently
```

---

## Bootstrap significance test

```python
from ml_elements import Comparator

cmp = Comparator(n_boot=5_000)

# Is the improvement at p_pos=0.05 statistically credible?
bac = cmp.for_trial(result, trial_value=0.05, baseline="without_x3", challenger="with_x3")

bac.ranking_report()   # P(each setup is best)
bac.pairwise_report()  # P(challenger > baseline)
```

---

## Access raw per-repeat scores

```python
# result.scores is a DataFrame with columns:
#   trial_value, setup, repeat, <metric_name>, ...
df = result.scores
df[df["trial_value"] == 0.10].groupby("setup")["auc"].describe()
```

---

## Calibration plot

```python
from ml_elements import plot_calibration

fig = plot_calibration(result.trial_results[0], setup="with_x3")
fig.savefig("calibration.png")
```

---

## Feature importance / coefficients

```python
from ml_elements import plot_feature_importance

fig = plot_feature_importance(result.trial_results[1], setup="with_x3")
fig.savefig("feature_importance.png")
```

Works for tree models (`feature_importances_`) and linear models (`coef_`).

---

## Save and reload a fitted model

```python
import joblib

model = result.trial_results[2].models["with_x3"]
joblib.dump(model, "model_p05.pkl")

# later
model = joblib.load("model_p05.pkl")
p_hat = model.predict_proba(X_new)[:, 1]
```

---

## Export scores to CSV

```python
result.scores.to_csv("scores.csv", index=False)
```

---

## Quick budget for prototyping

```python
from ml_elements import DataBudget

budget = DataBudget.quick()                   # 500 train / 100 valid / 500 test, 10 repeats
budget = DataBudget.quick(n=2000, n_repeats=20)
```

Scale up to the full budget only when you're happy with the experiment design.

---

## Get mean scores for all metrics in one table

```python
summary = study.full_summary(result, baseline="without_x3", challenger="with_x3")

# shows AUC and AP for both setups at every trial value
summary[[
    "trial_value",
    "without_x3_auc_mean", "with_x3_auc_mean",
    "without_x3_average_precision_mean", "with_x3_average_precision_mean",
]]
```
