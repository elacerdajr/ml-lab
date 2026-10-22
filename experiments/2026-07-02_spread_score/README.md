# Spread score: is neighbourhood disagreement learnable?

**Question:** building on `2026-06-23_catboost_leaf_embeddings`, the kernel score is the kernel-weighted mean label of a sample's leaf-space neighbourhood. Can the *spread* of labels in that neighbourhood also be captured, and is it a smooth function of the raw features, learnable by a plain `CatBoostRegressor` with no access to neighbours at inference time?

Four spread definitions are compared: Beta posterior variance, kernel-weighted label variance, binary entropy of the kernel score, and inverse effective neighbour count.

## Run

```bash
make exp-spread-score            # needs the catboost extra
```

Settings: `config.yaml`.

## Answer

`report.md`, with `outputs/results.csv`.
