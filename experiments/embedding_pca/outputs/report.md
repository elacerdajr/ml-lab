# Embedding PCA reconstruction report

## Setup

- Embedding provider: `local-hashing-sentence-embedding`
- Embedding model: `signed token/ngram hashing fallback (384 dimensions)`
- Number of site descriptions: 129
- Embedding dimension: 384
- PCA dimensions reported: 50

## Key results

- PCA dimension 1 explains 0.0333 of variance, with cumulative explained variance 0.0333.
- PCA dimensions 1-3 explain 0.0701 of variance, leaving relative reconstruction error 0.9299.
- The configured variance thresholds require the following PCA dimensions:

| Variance threshold | PCA dimensions required | Cumulative explained variance | Relative reconstruction error |
|---:|---:|---:|---:|
| 80% | 96 | 0.8050 | 0.1950 |
| 90% | 111 | 0.9031 | 0.0969 |
| 95% | 119 | 0.9512 | 0.0488 |
| 99% | 126 | 0.9902 | 0.0098 |

## Artifacts

- `pca_variance.csv` contains explained variance, cumulative variance, and reconstruction error by PCA dimension.
- `pca_thresholds.csv` contains the PCA dimension count required for each configured variance threshold.
- `site_pca_coordinates.csv` contains each site projected onto PCA dimensions 1, 2, and 3.
- `embedding_metadata.json` records the embedding provider, model, dimensions, row count, and thresholds.
- `fig_pca_variance.svg` plots per-component and cumulative explained variance.
- `fig_pca_123.svg` plots pairwise views of PCA dimensions 1, 2, and 3.
