# Noisy soft labels: CrossEntropy vs Logloss

**Question:** can CatBoost still learn the decision boundary when trained on a noisy soft target instead of the clean label? Each soft target stays on the correct side of 0.5 but its distance from 0 or 1 is random, so it corrupts confidence, not the class.

At each noise level the experiment compares:

- `CatBoostClassifier(loss_function="CrossEntropy")` on the soft target,
- `CatBoostRegressor(loss_function="RMSE")` on the same target,
- `CatBoostClassifier(loss_function="Logloss")` on the clean hard label (the baseline).

## Run

```bash
make exp-noisy-label-catboost    # needs the catboost extra (umap optional)
```

Settings: `config.yaml`.

## Answer

`report.md`, with `outputs/results.csv`.
