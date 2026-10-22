# Leaf-embedding residual ranking

**Question:** can leaf embeddings refine a coarse CatBoost probability into a better-spread ranking score without hurting the ranking? A Ridge model on one-hot leaf embeddings predicts the coarse model's residuals; the combined score is rank-spread through the training CDF into a soft target in (0, 1).

```
H(x)      = logit(p0(x)) + λ · zscore(Ridge(onehot(leaf(x)))) + ε · zscore(tie_break(leaf(x)))
y_soft(x) = F_train(H(x))
```

λ = 0 is the no-perturbation control.

## Run

```bash
make exp-leaf-embedding-ranking  # needs the catboost extra
```

Settings: `config.yaml`.

## Answer

`outputs/README.md`, with `outputs/results.csv`.
