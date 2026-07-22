# CatBoost-Encoder Model Comparison

Compares `category_encoders.CatBoostEncoder` — an **ordered target encoding**
that borrows CatBoost's internal "Ordered TS" categorical-handling trick, but
is implemented entirely outside CatBoost as a plain sklearn transformer —
across a broad model roster, on the same synthetic rare-positive dataset and
undersampling discipline as the sibling `imbalanced_classification` and
`encoder_comparison` experiments:

```
data -> catboost_encoder -> model_i training -> evaluation
```

## Why keep the CatBoost model if there's a "CatBoost encoder"?

`CatBoostEncoder` and the CatBoost gradient-boosting model are unrelated
implementations that happen to share a name. The encoder is a preprocessing
step any model can consume: it replaces each category with a running mean of
the target computed over a random permutation of the *other* training rows,
so a row never sees its own label folded into its own encoded value — this is
what makes it leakage-safe without needing cross-validation folds the way
`sklearn.preprocessing.TargetEncoder` does.

Keeping CatBoost in the model list lets this experiment answer a natural
question directly: **does feeding CatBoost this external, ordered-encoding
representation match its own native categorical handling** (which does
something conceptually similar internally)? CatBoost therefore runs twice:

- `catboost_encoded` — fed the exact same encoded numeric matrix as every
  other model (no `cat_features`).
- `catboost_native` — CatBoost's own categorical split search on raw string
  columns (`cat_features` set), bypassing the encoder entirely.

## How to run

```bash
make exp-catboost-encoder-models          # default profile (N_full = 300k, seconds)
make exp-catboost-encoder-models-smoke    # tiny smoke run for CI / verification

cd experiments/catboost_encoder_models
uv run --extra catboost --extra category_encoders python run_experiment.py
uv run --extra catboost --extra category_encoders python run_experiment.py --profile full_spec
uv run --extra catboost --extra category_encoders python run_experiment.py --smoke
```

This experiment needs one new dependency beyond the sibling experiments:
`category_encoders` (added as the `category_encoders` extra in `pyproject.toml`,
since `CatBoostEncoder` has no sklearn built-in equivalent).

## Data

Identical generative process to `experiments/imbalanced_classification` and
`experiments/encoder_comparison` (`cbenc/data.py` is a third direct port of
`imbcls/data.py`), so all three experiments' results are comparable. Models
train on the 10%-positive `train_under`; evaluation is on `val_full`/`test_full`
at the true ~0.1% base rate.

## Models (model_i)

| Model | Notes |
|---|---|
| `logistic` | Plain `LogisticRegression` on the encoded matrix |
| `rbf_svm` | `SVC(kernel="rbf")` + `CalibratedClassifierCV` (sigmoid), guarded by `models.rbf_svm.max_train_n` |
| `random_forest` | `RandomForestClassifier` — new to this repo's experiments |
| `mlp` | Small `MLPClassifier`, same config as the sibling experiments |
| `catboost_encoded` | CatBoost fed the encoded matrix, no `cat_features` |
| `catboost_native` | CatBoost's own categorical handling, raw columns + `cat_features` |

## Outputs

```
outputs/
  data_summary.json      dataset + split + pi summary
  metrics.csv            one row per (model, eval_split)
  model_configs.json     resolved model + encoder hyperparameters
  plots/
    ap_by_model/          bar chart, AP per model (native highlighted)
    precision_recall/     PR curves per model
  artifacts/fitted_preprocessors/  the fitted CatBoostEncoder+scaler pipeline
```

## Interpreting the plots

- **`ap_by_model.png`** — one bar per model, sorted by AP; `catboost_native` is
  drawn in red to set it apart as the reference point rather than one more
  encoded-feature model.
- **`precision_recall.png`** — PR curves (test); `catboost_native`'s curve is
  dashed for the same reason.

## Research question answered

The printed report (and `report.md`) gives the full model leaderboard and
directly states whether `catboost_native` beats `catboost_encoded` — the
central question this experiment exists to answer.
