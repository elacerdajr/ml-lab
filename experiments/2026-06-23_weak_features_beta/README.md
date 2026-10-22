# Weak features × sample size, heterogeneous strengths

**Question:** the same as `2026-06-18_weak_features_sample_size`, but each feature's information is drawn from Beta(1, 9): the same average strength (0.10), strongly right-skewed. How does training size drive the gap between CatBoost and interpretable models when a few features matter more than the rest?

## Run

```bash
make exp-weak-features-beta      # needs the imodels and catboost extras
```

Settings: `config.yaml`. Design notes: `plan.md`.

## Answer

`outputs/report.md`, with `outputs/summary.csv`, `outputs/scores.csv`, `outputs/capacity.csv` and the sampled strengths in `outputs/feature_info.csv`.
