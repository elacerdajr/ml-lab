# Quickstart — a complete example

Question: does adding a weak feature (`x3`) help a gradient-boosted model, and does the answer change with class imbalance?

## 1. Budget and setups

```python
from ml_elements import (
    AUC, AVG_PRECISION, DataBudget, GaussianPopulation, Setup,
    compare, evaluate, make_hgb, sweep,
)

budget = DataBudget(
    n_train=3_000, n_valid=500, n_test=2_000,
    seed_train=1, seed_valid=2, seed_test_base=10_000,
    n_repeats=25,
)

# Same model, different features: the only difference is which columns each setup sees.
setups = [
    Setup("without_x3", ["x1", "x2"]),
    Setup("with_x3", ["x1", "x2", "x3"]),
]
```

## 2. Sweep the positive rate

```python
trials = sweep(
    [{"p_pos": p} for p in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]],
    population=lambda p: GaussianPopulation(
        p_pos=p["p_pos"],
        info={"x1": 0.80, "x2": 0.50, "x3": 0.20},  # x3 is weak
    ),
    name="p_pos",
)
```

## 3. Evaluate and compare

```python
result = evaluate(trials, setups, [AUC, AVG_PRECISION], budget, model=make_hgb())
table = compare(result, "without_x3", "with_x3", AVG_PRECISION)

print(table[["trial_value", "mean_baseline_score", "mean_challenger_score",
             "mean_improvement", "p_improvement_gt_0"]].round(4))
```

Output (ml-lab at the commit that added this page, macOS arm64, about 25 s):

```
   trial_value  mean_baseline_score  mean_challenger_score  mean_improvement  p_improvement_gt_0
0         0.02               0.0334                 0.0357            0.0023                0.68
1         0.05               0.1156                 0.1206            0.0050                0.68
2         0.10               0.2149                 0.2203            0.0054                0.68
3         0.20               0.4037                 0.4078            0.0041                0.52
4         0.35               0.5814                 0.5851            0.0037                0.68
5         0.50               0.7148                 0.7113           -0.0035                0.32
```

How to read it: the AP gain from `x3` is small (under 0.006) at every positive rate, and the challenger wins in roughly half to two-thirds of the 25 test repeats. This run does not show a clear trend with imbalance. All repeats share one training set, so `p_improvement_gt_0` reflects test-sample variation, not training variation. To test the imbalance question properly, add training variation, for example several trials per positive rate with different `seed_offset`s, or more repeats.

## 4. Inspect a fitted model

```python
model = result.trial_results[1].models["with_x3"]   # p_pos = 0.05
model.n_iter_                                       # 200: make_hgb runs a fixed number of rounds

import pandas as pd
df_valid = result.trial_results[1].df_valid
p_hat = model.predict_proba(df_valid[["x1", "x2", "x3"]])[:, 1]
```

## 5. Plot

```python
from ml_elements import plot_study

fig = plot_study(table, metric=AVG_PRECISION)
fig.savefig("x3_value_by_imbalance.png")
```

## What to try next

- Swap `make_hgb()` for `make_logistic()`: the population is linear, so logistic regression is close to the best possible model.
- Use `ResampledPopulation` to run the same comparison on your own table.
- Wrap the population in `ShiftedPopulation` to test whether the gain from `x3` survives a shift.
- Replace the list with `sweep({"p_pos": (0.02, 0.5), "x3_info": (0.0, 1.5)}, method="sobol", n=32, ...)` to vary both at once.
