# Weak features × sample size

**Question:** with 100 features that each carry little signal (information 0.10), how does training size drive the gap between CatBoost and interpretable rule and tree models (FIGS, GreedyRuleList, a decision tree)? Logistic regression is the near-Bayes reference, because the population is linear in the features.

The only swept axis is `n_train`. Test sets are large and repeated, so test-side noise is small. Setups that fail to fit at small sizes are recorded as NaN rather than aborting the run.

## Run

```bash
make exp-weak-features           # about 3 minutes; needs the imodels and catboost extras
```

Settings: `config.yaml`. Design notes: `plan.md`.

## Answer

`outputs/report.md`, with `outputs/summary.csv` (means by model and size), `outputs/scores.csv` (per repeat) and `outputs/capacity.csv`.

See `2026-06-23_weak_features_beta` for the same question with heterogeneous feature strengths.
