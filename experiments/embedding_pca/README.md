# Embedding PCA reconstruction experiment

This experiment asks how many principal components are needed to reproduce a medium-sized sentence embedding space for a diverse set of website descriptions.

The input data is `site_descriptions.csv`, with more than 100 rows where each row contains a site name and a one-sentence description.

## Run

```bash
make exp-embedding-pca
```

The script uses `sentence-transformers/all-MiniLM-L6-v2` by default and writes outputs to `experiments/embedding_pca/outputs/`. If optional dependencies are unavailable, it falls back to a deterministic 384-dimensional local hashing sentence embedding so artifacts and a report can still be generated offline.

Install the optional runtime dependencies before running the full experiment:

```bash
pip install numpy pandas scikit-learn matplotlib pyyaml sentence-transformers
```

## Outputs

- `pca_variance.csv` records per-component explained variance, cumulative explained variance, and PCA reconstruction error.
- `pca_thresholds.csv` records the number of PCA dimensions needed to retain each configured variance threshold.
- `site_pca_coordinates.csv` records PCA coordinates for the first three components for each site.
- `embedding_metadata.json` records the model name, embedding dimension, row count, and threshold summary.
- `report.md` summarizes the setup, variance retained by PCA dimensions 1-3, and threshold results.
- `fig_pca_variance.svg` plots explained and cumulative variance for PCA dimensions; `fig_pca_variance.png` is also written when matplotlib is available.
- `fig_pca_123.svg` plots sites in the first three PCA dimensions; `fig_pca_123.png` is also written when matplotlib is available.
