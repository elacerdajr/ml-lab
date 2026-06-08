# Quickstart — a complete example

This walks through a full experiment from scratch: does adding a weak feature (`x3`) actually help, and does the answer change with class imbalance?

## 1. Imports & budget

```python
from ml_elements import (
    GaussianBinaryDGP,
    make_hgb,
    AUC, AVG_PRECISION,
    DataBudget,
    TrialRunner, Study,
    ManualSearch,
    Comparator,
    plot_study,
)

budget = DataBudget(
    n_train=3_000,
    n_valid=500,
    n_test=2_000,
    seed_train=1,
    seed_valid=2,
    seed_test_base=10_000,
    n_repeats=25,
)
```

## 2. Define model setups

`setups` maps a name to the feature columns that setup uses. All setups share the same model factory and budget — the only difference is which columns they see.

```python
setups = {
    "without_x3": ["x1", "x2"],
    "with_x3":    ["x1", "x2", "x3"],
}

runner = TrialRunner(
    setups=setups,
    model_factory=make_hgb(),
    metrics=[AUC, AVG_PRECISION],
    budget=budget,
)

study = Study(runner, primary_metric=AVG_PRECISION)
```

## 3. Define the parameter sweep

We sweep `p_pos` — the positive-class fraction — from severely imbalanced to balanced.

```python
trials = ManualSearch(
    [{"p_pos": p} for p in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]],
    trial_name="p_pos",
).trials(
    dgp_fn=lambda p: GaussianBinaryDGP(
        p_pos=p["p_pos"],
        info={"x1": 0.80, "x2": 0.50, "x3": 0.20},  # x3 is weak
    )
)
```

## 4. Run

```python
result  = study.run(trials)
improv  = study.improvements(result, baseline="without_x3", challenger="with_x3")
summary = study.summarize(improv)

print(summary[["trial_value", "challenger_mean", "baseline_mean", "delta_mean"]])
```

Example output:

```
   trial_value  challenger_mean  baseline_mean  delta_mean
0         0.02           0.3821         0.3412      0.0409
1         0.05           0.4107         0.3780      0.0327
2         0.10           0.4553         0.4311      0.0242
3         0.20           0.5034         0.4901      0.0133
4         0.35           0.5612         0.5541      0.0071
5         0.50           0.6024         0.5980      0.0044
```

The weak feature helps more when the positives are rare — there's simply less signal to go around.

## 5. Plot

```python
fig = plot_study(summary, metric=AVG_PRECISION)
fig.savefig("x3_value_by_imbalance.png")
```

## 6. Statistical check on one condition

```python
cmp = Comparator(n_boot=3_000)

# Is the improvement at p_pos=0.05 real?
bac = cmp.for_trial(result, trial_value=0.05, baseline="without_x3", challenger="with_x3")
bac.pairwise_report()
# P(with_x3 > without_x3) = 0.94  →  likely real
```

## 7. Inspect the fitted model

```python
# trial 1 = p_pos=0.05 (second trial, 0-indexed)
trial_result = result.trial_results[1]
model = trial_result.models["with_x3"]

# feature importances
import pandas as pd
fi = pd.Series(model.feature_importances_, index=["x1", "x2", "x3"]).sort_values(ascending=False)
print(fi)
# x1    0.61
# x2    0.30
# x3    0.09
```

## What to try next

- Swap `make_hgb()` for `make_logistic()` and re-run — does the story change?
- Use `RealDataDGP` to run the same comparison on your own dataset.
- Add a `ShiftedDGP` wrapper and check whether the gain from `x3` survives covariate shift.
- Replace `ManualSearch` with `SobolSearch` to explore `p_pos` and `x3_info` jointly.
