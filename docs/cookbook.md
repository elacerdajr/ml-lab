# Cookbook

Short recipes. Each assumes the imports, `budget`, `setups` and `trials` from the [quickstart](quickstart.md).

---

## Compare two different models on the same features

Give each setup its own model factory.

```python
from ml_elements import Setup, evaluate, make_catboost, make_logistic

setups = [
    Setup("logistic", ["x1", "x2", "x3"], model=make_logistic()),
    Setup("catboost", ["x1", "x2", "x3"], model=make_catboost(depth=4)),
]
result = evaluate(trials, setups, [AUC, AVG_PRECISION], budget)
```

Setups without a `model` use `evaluate(model=...)`, or the trial's `model_factory` if it has one.

---

## Early stopping on AUC vs AP, or on a penalised objective

`make_hgb` runs a fixed number of rounds unless you ask for early stopping. Only then does `scoring` choose the criterion.

```python
from ml_elements import AVG_PRECISION, N_ITER, Setup, make_hgb

setups = [
    Setup("auc_stopped", ["x1", "x2"], model=make_hgb(early_stopping=True, scoring="roc_auc")),
    Setup("ap_stopped", ["x1", "x2"], model=make_hgb(early_stopping=True, scoring="average_precision")),
    Setup("ap_minus_trees", ["x1", "x2"], model=make_hgb(
        early_stopping=True, scoring=(AVG_PRECISION - 0.001 * N_ITER).as_scorer())),
]
```

`validation_fraction` defaults to 0.1 when early stopping is on. `metric.as_scorer()` turns any `Metric` into an sklearn scorer; lower-is-better metrics are negated, because sklearn maximises.

---

## Penalise model size in the evaluation itself

Model terms (`N_ITER`, `N_LEAVES`, `COEF_L1`, or `Metric(..., on="model")`) read the fitted model, so they work as evaluation metrics too.

```python
from ml_elements import AVG_PRECISION, N_ITER

AP_PENALISED = (AVG_PRECISION - 0.001 * N_ITER).rename("ap_penalised")
result = evaluate(trials, setups, [AVG_PRECISION, AP_PENALISED], budget, model=make_hgb())
```

---

## Custom metric

```python
import numpy as np
from sklearn.metrics import average_precision_score
from ml_elements import Metric

def ap_at_k(y_true, y_score, k=200):
    idx = np.argsort(y_score)[::-1][:k]
    return average_precision_score(y_true[idx], y_score[idx])

AP_AT_200 = Metric("ap@200", "higher", lambda y, p: ap_at_k(y, p, k=200))
```

---

## Keep going when a model fails

Some models (FIGS, greedy rule lists) can fail to fit at small sample sizes. `on_error="record"` records NaN for that setup and continues.

```python
result = evaluate(trials, setups, metrics, budget, model=make_figs(), on_error="record")
result.failures        # trial_name, trial_value, setup, stage ("fit" or "score"), error
```

---

## Run on real data

```python
import pandas as pd
from ml_elements import ResampledPopulation, Trial

df = pd.read_parquet("data/clickstream.parquet")
trial = Trial(name="real", value=0.0, population=ResampledPopulation(df, target_col="click"), seed_offset=0)
result = evaluate(trial, setups, [AUC, AVG_PRECISION], budget, model=make_logistic())
```

`ResampledPopulation.sample` draws rows with replacement. Train, validation and test draws can therefore share rows, so treat the scores as optimistic.

---

## Score under a shifted population

`evaluate` trains and tests each trial on the same population. To compare against a shifted population, run it as its own trial:

```python
from ml_elements import GaussianPopulation, ShiftedPopulation, Trial

base = GaussianPopulation(p_pos=0.10, info={"x1": 0.8, "x2": 0.5})
shifted = ShiftedPopulation(base, shift_fn=lambda df: df.assign(x1=df["x1"] + 1.0))
trials = [Trial("shift", 0, base, seed_offset=0), Trial("shift", 1, shifted, seed_offset=1)]
```

To train on one population and test on another, score the fitted models yourself:

```python
result = evaluate(Trial("base", 0, base, seed_offset=0), setups, [AUC], budget, model=make_logistic())
model = result.trial_results[0].models["with_x3"]
test = shifted.sample(5_000, seed=99)
AUC.score(test["y"], model.predict_proba(test[["x1", "x2", "x3"]])[:, 1])
```

---

## Sweep two parameters with a Sobol design

```python
trials = sweep(
    {"p_pos": (0.02, 0.50), "x3_info": (0.00, 1.50)},
    method="sobol", n=32, seed=0,
    population=lambda p: GaussianPopulation(
        p_pos=p["p_pos"], info={"x1": 0.8, "x2": 0.5, "x3": p["x3_info"]}),
)
```

Each trial's parameters are in `trial.params`; `sweep_points(space, method="sobol", n=32)` gives them as a DataFrame. `method="random"` draws uniform points instead.

---

## Parallel trials

```python
result = evaluate(trials, setups, metrics, budget, model=make_hgb(), n_jobs=8)   # needs joblib
```

---

## Per-repeat scores

```python
df = result.scores           # trial_name, trial_value, repeat, test_seed, setup, features, <metrics>
df[df["trial_value"] == 0.10].groupby("setup")["auc"].describe()

compare(result, "without_x3", "with_x3", AUC, per_repeat=True)   # one row per repeat
```

---

## Calibration and feature importance

```python
from ml_elements import plot_calibration, plot_feature_importance

plot_calibration(result.trial_results[0], setup="with_x3").savefig("calibration.png")
plot_feature_importance(result.trial_results[1], setup="with_x3").savefig("importance.png")
```

`plot_feature_importance` needs `feature_importances_` (CatBoost, decision trees) or `coef_` (linear models). `HistGradientBoostingClassifier`, which `make_hgb` builds, has neither; use `sklearn.inspection.permutation_importance` for it.

---

## Shrink scores toward the base rate

```python
from ml_elements import Setup, shrink, SCORE_ENTROPY, TIE_RATE

setups = [
    Setup("raw", ["x1", "x2"]),
    Setup("shrunk", ["x1", "x2"], scores=lambda p: shrink(p, pi=0.01, lam=0.2)),
]
result = evaluate(trials, setups, [AVG_PRECISION, SCORE_ENTROPY, TIE_RATE], budget, model=make_catboost())
```

Shrinkage is monotonic, so ranking metrics such as AP do not change; calibration and score-shape metrics do.

---

## Leaf embeddings

```python
from ml_elements import embed_leaves

leaves = embed_leaves(catboost_model, X)                   # (n, n_trees) leaf indices
one_hot = embed_leaves(catboost_model, X, one_hot=True)    # sparse, e.g. for UMAP(metric="cosine")
```

---

## Quick budget for prototyping

```python
budget = DataBudget.quick()                     # 500 train / 100 valid / 500 test, 10 repeats
budget = DataBudget.quick(n=2000, n_repeats=20)
```
