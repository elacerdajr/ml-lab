# CatBoost leaf embeddings as soft targets

**Question:** can a heavily regularised CatBoost model serve as a feature extractor whose leaf indices define a useful similarity? Hamming distance between leaf vectors drives an RBF kernel that diffuses labels into soft targets, and those soft targets train a downstream model.

Two extractor configurations (light and heavy regularisation) are compared across kernel widths σ.

## Run

```bash
make exp-catboost-leaf-embeddings   # about 25 s; needs the catboost extra (umap optional)
```

Settings: `config.yaml`. `notebook.ipynb` explores the same pipeline interactively.

## Answer

`report.md`, with `outputs/results.csv`.

Follow-ups: `2026-07-02_spread_score`, `2026-07-02_leaf_embedding_ranking`, `2026-07-22_leaf_embedding_umap`.
