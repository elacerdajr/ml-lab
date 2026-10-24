# Leaf-embedding UMAP reduction vs native CatBoost

**Question:** take the raw per-tree leaf indices of a CatBoost model and compress them with UMAP (Hamming metric) to k dimensions. How much downstream classification performance survives, compared with CatBoost trained natively on the raw categorical and numerical features, on an imbalanced target?

k ∈ {2, 5, 10, 20}; downstream models are logistic regression, an RBF SVM, a random forest, an MLP and CatBoost. Training time is tracked against the native baseline.

## Run

```bash
make exp-leaf-embedding-umap     # needs the catboost and umap extras
```

Settings: `config.yaml`.

## Answer

`report.md`, with `outputs/results.csv`.
