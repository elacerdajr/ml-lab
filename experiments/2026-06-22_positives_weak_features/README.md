# Rare positives and noise features

**Question:** at 1 % positives, which CatBoost hyperparameters amplify or suppress the effect of pure-noise features on precision and AP?

With few positives, a model can latch onto noise features that happen to correlate with positives in the training draw. The experiment sweeps tree depth (6 values) and `l2_leaf_reg` (5 values). All 30 configurations see the same train, eval and test data, so differences come only from the hyperparameters.

## Run

```bash
make exp-positives-weak-features   # about 1 minute; needs the catboost extra
```

Settings: `config.yaml`.

## Answer

`outputs/report.md`, with `outputs/summary.csv`, `outputs/scores.csv` and `outputs/feature_importances.csv`.
