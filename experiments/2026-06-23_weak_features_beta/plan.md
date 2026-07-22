# Plan — Weak Features (Beta‑Distributed Informations)

> Experiment folder: `experiments/weak_features_beta/`

## Research question

When features have **heterogeneous, Beta‑distributed information levels** —
most barely above noise, a few with non‑trivial signal — how does the
training sample size drive the gap between:

- a high‑capacity boosted model (**CatBoost**), and
- interpretable rule/tree models from `imodels` (**FIGS**, **GreedyRuleList**,
  plus a plain sklearn **DecisionTree** as a baseline)?

This replicates the `weak_features_sample_size` experiment but replaces the
constant `info_j = 0.10` with `info_j ~ Beta(1, 9)` — so the **average**
feature strength is the same (0.10), but the distribution is skewed: most
features are near‑zero, while a few carry meaningful signal.

> *Given 100 features whose information levels are drawn from Beta(1, 9) and
> a large OOS test set, at what training size does each model family recover
> the available signal — and how does the heterogeneity affect the
> interpretability cost?*

## Why this is interesting

- **Realistic heterogeneity**: real‑world features are never identically
  distributed. Beta(1, 9) is a strong "many near‑zero, few notable" pattern.
- **Variance‑model interaction**: high‑capacity ensembles pool many weak
  signals; shallow trees may latch onto the few strong features. The
  Beta‑spread tests this.
- **Comparison to the homogeneous experiment**: side‑by‑side, we can see
  whether heterogeneity helps or hurts the interpretable models relative
  to CatBoost.

## DGP

Use `ml_elements.dgp.GaussianBinaryDGP`:

```
p_pos = 0.20        # mild imbalance
sigma = 1.0
info_j ~ Beta(a=1, b=9)  sampled once per feature, seed=20250601
```

- `Beta(1, 9)` has mean 0.10 and mode 0 — identical average strength to the
  `weak_features_sample_size` experiment, but most features are much weaker
  and a few are substantially stronger.
- The Bayes‑optimal linear classifier aggregates all 100 features; the
  marginal information per feature is no longer constant.

The per‑feature info levels are **sampled once with a fixed seed** and saved
to `outputs/feature_info.csv`. The DGP is therefore deterministic across the
entire experiment — only the random draws from it vary.

## Sweep

**Single knob: `n_train`.** Same values as the homogeneous experiment for
direct comparability.

| `n_train` | rationale |
|---|---|
| 200   | far below #features → severe ill‑conditioning |
| 500   | ~5 × #features |
| 1 000 | 10 × #features |
| 2 000 | 20 × #features |
| 5 000 | comfortable regime |
| 10 000| data‑rich baseline |

Fixed across all conditions:

- `n_test = 50 000`  (large → OOS estimates are essentially noise‑free)
- `n_repeats = 30`  independent test draws per `(model, n_train)`

## Models

| Setup | Backend | Notes |
|---|---|---|
| `logistic`       | `make_logistic()` | Bayes‑optimal linear reference |
| `catboost`       | `make_catboost(iterations=400, depth=4, lr=0.05)` | high‑capacity booster |
| `hgb`            | `make_hgb(iterations=400, max_leaf_nodes=16)` | sklearn booster |
| `decision_tree`  | `DecisionTreeClassifier(max_depth=4)` | interpretable baseline |
| `figs`           | `imodels.FIGSClassifier(max_rules=25)` | interpretable tree ensemble |
| `greedy`         | `imodels.GreedyRuleListClassifier(max_depth=8)` | ordered rule list |

All models see the same 100 features; no feature selection is performed.

## Metrics

- `roc_auc` — head metric (ranking quality)
- `average_precision` — sensitive to the 0.20 base rate
- `brier_score` — calibration / squared error

## Outputs

```
experiments/weak_features_beta/outputs/
├── feature_info.csv     # per‑feature info_j values sampled from Beta(1,9)
├── scores.csv           # long‑format: model × n_train × repeat × metric
├── summary.csv          # mean ± 95% CI per (model, n_train)
├── capacity.csv         # learned rule/leaf counts per setup
├── fig_auc_vs_n.png     # AUC vs n_train
├── fig_average_precision_vs_n.png
├── fig_brier_score_vs_n.png
├── fig_gap_vs_n.png     # AUC gap vs CatBoost
├── fig_capacity_vs_n.png
├── fig_info_hist.png    # histogram of sampled feature informations
└── report.md            # generated findings
```

## Planned analyses

1. **Learning curves** — AUC, AP, Brier vs `n_train`.
2. **Interpretability cost** — ΔAUC relative to CatBoost at each `n_train`.
3. **Capacity utilisation** — n_rules / n_leaves vs `n_train`.
4. **Comparison to homogeneous experiment** — overlay or delta with the
   `weak_features_sample_size` results (saved separately).

## Risks

- **Beta(1,9) may produce a few features with info > 0.5**: these could
  dominate, making the "100 weak features" framing inaccurate. The histogram
  in `fig_info_hist.png` will diagnose this.
- **FIGS / GreedyRuleList unstable at n_train = 200**: handled by per‑setup
  try/except (same isolation strategy as `weak_features_sample_size`).
