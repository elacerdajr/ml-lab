# Imbalanced Binary Classification — Score Entropy, Priors, Leaf Embeddings & UMAP

Compare models for **rare-positive (~0.1%) binary classification** not just on
predictive quality (AP / AUC), but on **score smoothness / entropy, tie rate,
training speed, and usefulness for ranking**. Models train on a 10%-positive
*undersample* but are evaluated under the real base rate, so they are judged in
the deployment distribution. The experiment also studies five "prior" mechanisms
and visualises learned representations with UMAP.

## How to run

```bash
# From the repo root — uses the locked env + catboost / umap / rich extras.
make exp-imbalanced-classification          # default profile (N_full = 300k, a few minutes)
make exp-imbalanced-classification-smoke    # tiny smoke run for CI / verification

# Or directly, with profile / smoke flags:
cd experiments/imbalanced_classification
uv run --extra catboost --extra umap --extra viz python run_experiment.py
uv run --extra catboost --extra umap --extra viz python run_experiment.py --profile full_spec
uv run --extra catboost --extra umap --extra viz python run_experiment.py --smoke
```

`--profile full_spec` reproduces the literal 1,000,000-row specification (wider
RFF grid, larger GP subsample, 50k-point UMAP) and is substantially heavier.

Everything is config-driven: edit `config.yaml` to change scale, seeds, model
hyperparameters, prior sweeps or UMAP settings. No magic numbers live in code.

## Package layout

The experiment follows the repo convention (one folder under `experiments/`)
with a thin `run_experiment.py` entry, while the requested modular design lives
in the `imbcls/` package:

```
imbcls/
  config.py      typed config + profile/​smoke resolution
  data.py        DGP, splits, undersampling, pi, UMAP sampling
  priors.py      the 5 prior mechanisms
  models.py      preprocessor + model registry + soft-label fitting dispatch
  calibration.py sigmoid calibration for margin models
  scoring.py     post-hoc score transforms (priors 4 & 5)
  metrics.py     entropy / tie / gap / bucket metrics
  embeddings.py  raw / CatBoost-leaf / RFF representations for UMAP
  umap_viz.py    UMAP projection + full-vs-undersampled panels
  plots.py       figures A–F
  runner.py      orchestration + final report
  main.py        CLI entry
```

## Data & undersampling

A synthetic dataset with features `cat_1` (3 levels), `cat_2` (100 levels),
`num_1`, `num_2`. The target is a controlled **nonlinear** logit — categorical
effects, a subset of high-risk `cat_2` levels, `sin(num_1)`, a `num_2` threshold,
a categorical×numerical interaction, and noise — with the intercept calibrated by
bisection so `mean(p) ≈ 0.001`. `y ~ Bernoulli(p)`.

- `train_full` / `val_full` / `test_full` are stratified and **preserve the ~0.1%
  base rate**.
- `train_under` keeps **all** training positives and subsamples negatives so
  positives are 10%. Models fit here.
- `pi = y_train_full.mean() ≈ 0.001` — the **true** base rate — anchors every
  prior. It is *not* the 10% undersample rate.

### Why val/test must keep the real base rate

A model trained on a 10%-positive undersample outputs probabilities calibrated to
**10%**, not 0.1%. If you also evaluated on a 10%-positive set you would flatter
the model and never see the deployment behaviour. Keeping val/test at the true
base rate exposes the real precision/recall trade-off and the miscalibration —
which is exactly what the **post-hoc shrinkage** prior addresses. This is why raw
`log_loss` / `brier` on val/test look poor by design; that is a feature, not a bug.

## The five prior mechanisms

| # | Mechanism | Where it acts | Applies to |
|---|-----------|---------------|------------|
| 1 | none | — | all models |
| 2 | label smoothing `y' = (1-λ)y + λπ` | training (soft label) | soft-capable models |
| 3 | synthetic soft points (label `π`, drawn from empirical train_full) | training (soft label) | soft-capable models |
| 4 | post-hoc shrinkage `p' = (1-λ)p + λπ` | score transform | **all** models |
| 5 | deterministic noise ranking `r = αp + (1-α)u` | score transform | **all** models |

**Soft-label support.** Only CatBoost consumes soft labels natively
(`loss_function="CrossEntropy"`). Plain logistic and RFF+logistic use an *exact*
reduction: a soft label `y'` equals a positive row of weight `y'` plus a negative
row of weight `1-y'` under cross-entropy (`priors.to_weighted_rows`). Balanced
logistic, Linear/RBF SVM and the GP get priors 1/4/5 only; the MLP has an optional
`MLPRegressor` soft path (off by default).

Priors 4 and 5 are post-hoc and cost no extra training. Shrinkage is **monotonic**,
so it leaves AP unchanged (asserted in the smoke checks) while improving
calibration. The noise score is a *ranking* score, not a probability — so
`log_loss` / `brier` are `NaN` for it.

## Metrics (per model × prior × score-transform × split)

`metrics.csv` has one row per combination. Columns include `average_precision`,
`roc_auc`, `log_loss`, `brier_score` (the last two only for probability scores),
plus the score-smoothness family:

- **score_entropy / normalized_score_entropy** — Shannon entropy of a 50-bin
  histogram over [0,1], normalised by `log(50)`. Higher = scores spread across
  more of the [0,1] range (smoother, less spiky).
- **tie_rate** = `1 − n_unique_scores / n` — fraction of tied scores. Tree models
  produce many ties (few distinct leaf values); linear/RFF models produce few.
- **occupied_bins**, **max_score_gap** — coverage and the largest jump between
  adjacent sorted scores.
- **train_time_seconds / predict_time_seconds**.

`bucket_metrics.csv` gives per-bucket lift (10/20/100 buckets): count, positives,
positive rate, mean score, and lift vs the base rate — the operational ranking view.

## Interpreting the plots (`outputs/plots/`)

- **`ap_entropy/ap_vs_entropy.png`** — the core trade-off. x = normalized score
  entropy, y = AP, point size ∝ train time. Top-right is ideal (accurate *and*
  smooth). The dotted line marks the "high-entropy" threshold (0.80).
- **`ap_time/ap_vs_time.png`** — AP vs `log(1+train_time)`, coloured by entropy;
  find the cheapest model at a given accuracy.
- **`entropy_tradeoff/ap_loss_vs_entropy_gain.png`** (headline) — for the noise
  ranking score, x = entropy gain `ΔH = H(r)−H(p)`, y = AP change
  `ΔAP = AP(r)−AP(p)`. Points near the top-right buy large smoothness gains for
  little AP loss; steep drops mean noise is destroying ranking signal.
- **`bucket_lift/bucket_lift.png`** — positive rate per score decile (log scale)
  with the true base-rate line. A good ranker's top decile sits far above it.
- **`score_histograms/`** — raw vs shrinkage vs noise score distributions
  (val + test), on a log y-axis so the rare high-score tail is visible.
- **`precision_recall/`** — PR curves (the right diagnostic under heavy imbalance).

## Interpreting the UMAP plots (`outputs/plots/umap/`)

Each figure is a **full-sample vs undersampled** pair for one representation:

- **`umap_raw_features.png`** — one-hot + scaled features (euclidean). The
  model-agnostic view of how separable the classes are before any model.
- **`umap_catboost_leaf.png`** — CatBoost leaf-index one-hot (cosine). The
  *supervised* tree view: samples landing in the same leaves across trees sit
  together, so the rare positive class typically forms much tighter, more
  separated structure than in raw space.
- **`umap_rff_features.png`** — random-Fourier features (euclidean), the space the
  RFF+logistic model actually sees.

Comparing the **full** panel (true base rate — positives are a sparse minority)
against the **undersampled** panel (10% positives) shows how undersampling
reshapes the apparent class geometry the model trains on: the undersample makes
positive structure visually dominant, which is what lets simple models latch onto
it — and why evaluation must return to the real base rate.

## Research questions answered

The printed report (and `report.md`) names: the best model by AP; by normalized
entropy; the best AP among high-entropy models (`H ≥ 0.80`); the fastest "good"
model (`AP ≥ 0.95 · best_AP`); the best AP-loss/entropy-gain trade under noise;
and a Pareto-efficient **production candidate** (high AP, high entropy, low tie
rate, reasonable train time) — not a winner chosen by AP alone.

## Outputs

```
outputs/
  data_summary.json     dataset + split + pi summary
  metrics.csv           one row per model × prior × transform × split
  bucket_metrics.csv    per-bucket lift (10 / 20 / 100 buckets)
  model_configs.json    resolved model hyperparameters (incl. selected RFF)
  plots/                figures A–F + UMAP panels
  artifacts/            fitted preprocessor + linear models (CatBoost dumps off by default)
```
