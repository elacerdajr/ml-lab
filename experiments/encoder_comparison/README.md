# Encoder Comparison

Isolates **categorical-encoding choice** as its own variable, on the same
synthetic rare-positive dataset and training discipline as the sibling
`imbalanced_classification` experiment:

```
data -> encoder_i -> model_j training -> evaluation
```

## How to run

```bash
make exp-encoder-comparison          # default profile (N_full = 300k, seconds)
make exp-encoder-comparison-smoke    # tiny smoke run for CI / verification

cd experiments/encoder_comparison
uv run --extra catboost python run_experiment.py
uv run --extra catboost python run_experiment.py --profile full_spec
uv run --extra catboost python run_experiment.py --smoke
```

## Data

Identical generative process to `experiments/imbalanced_classification`
(`enccmp/data.py` is a direct port of `imbcls/data.py` — same logit formula,
same seed handling), so the two experiments' results are comparable. Models
train on the 10%-positive `train_under`; evaluation is on `val_full`/`test_full`
at the true ~0.1% base rate.

## Encoders (encoder_i)

| Encoder | Mechanism | Unseen categories |
|---|---|---|
| `onehot` | One column per category, categories pinned to `train_full`'s domain | `handle_unknown="ignore"` (all-zero row) |
| `ordinal` | Integer code per category, same pinning | maps to `-1` |
| `frequency` | Category → its `train_under` empirical frequency (count/N) | maps to `0` |
| `target` | Cross-fitted mean target per category (`sklearn.preprocessing.TargetEncoder`, `smooth="auto"`, `cv=5`) | falls back to the global target mean |
| `hashing` | Each categorical column hashed independently into a fixed-width vector (`sklearn.feature_extraction.FeatureHasher`) | handled natively — no vocabulary at all |
| `native` (CatBoost only) | No encoding — CatBoost's own categorical split search on raw string columns | handled natively |

`target` encoding is the one place leakage is a real risk: its cross-fitting
(`cv=5`) means the *training* matrix is produced by out-of-fold statistics, so
the same rows used to fit the encoder aren't the rows whose encoded value came
from their own label. `.transform()` on val/test then uses the encoder's full
fit. `onehot`/`ordinal` are fit with categories pinned to `train_full`'s domain
(not `train_under`'s) because the undersample's negative subsample can miss
rare `cat_2` levels that do appear in val/test.

## Models (model_j)

`logistic`, `rff_logistic` (Random Fourier Features + logistic regression),
`mlp` (small sklearn `MLPClassifier`), and CatBoost run **twice** —
`catboost_encoded` (through the same shared preprocessor as every other model,
so it never sees `cat_features`) and `catboost_native` (raw categoricals +
`cat_features`, independent of the encoder axis) — to directly answer whether
CatBoost's built-in categorical handling beats explicit encoding.

## Outputs

```
outputs/
  data_summary.json         dataset + split + pi summary
  metrics.csv               one row per (encoder, model, eval_split)
  encoder_diagnostics.csv   per-encoder n_features_out + fit/transform time (model-independent)
  model_configs.json        resolved model hyperparameters
  plots/
    ap_by_encoder/          grouped bar: AP per (model x encoder) — the core comparison
    dimensionality/         n_features_out vs AP scatter
    timing/                 encode + train time per (model x encoder)
  artifacts/fitted_preprocessors/   one fitted preprocessor per encoder (joblib)
```

## Interpreting the plots

- **`ap_by_encoder.png`** — grouped bars, AP (test) per model × encoder.
  `catboost_native` appears only at its own `native` x-position since it isn't
  encoder-dependent.
- **`dimensionality_vs_ap.png`** — does a bigger representation (OneHot, ~100+
  dims for `cat_2`) actually win over compact ones (Target/Hashing, 4–66 dims)?
  **Caveat:** `native`'s `n_features_out=4` is just the raw column count, not a
  comparable "dimensionality" — CatBoost's categorical split search happens
  inside the boosting fit, not as a fixed-width numeric representation, so
  don't read it as "4 features beats 105."
- **`timing.png`** — stacked bars, encode fit time (lighter) + model train time
  (darker), per model × encoder. Answers whether a fancier encoder (Target,
  Hashing) is worth its extra cost over the trivial ones (Ordinal, Frequency).

## Research questions answered

The printed report (and `report.md`) names the best encoder per model, and
directly compares CatBoost's native categorical handling against its own best
encoded variant — the two natural questions this experiment exists to answer.
