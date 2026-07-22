# Plan — Weak Features × Sample Size

> Experiment folder: `experiments/weak_features_sample_size/`

## Research question

When the feature set is **wide but weak** — many features, each carrying
only a small amount of signal — how does the **training sample size**
drive the gap between:

- a high-capacity boosted model (**CatBoost**), and
- interpretable rule/tree models from `imodels` (**FIGS**, **GreedyRuleList**,
  plus a plain sklearn **DecisionTree** as a baseline)?

Concretely:

> *Given 100 weakly-informative features and a large OOS test set, at what
> training size does each model family recover (a near-Bayes) fraction of
> the available signal — and where do the interpretable models break down?*

## Why this is interesting

- **Realistic regime**: many applied problems (telemetry, embeddings, weak
  sensor aggregates) look like "100 features, each barely above noise".
- **Sample-size sensitivity**: tree ensembles can exploit many weak signals
  by averaging; small rule lists / shallow trees cannot, because their
  capacity is bounded by depth. We expect a clear *capacity-vs-data*
  ordering.
- **Interpretability cost**: quantify the AUC / AP the interpretable models
  give up as a function of `n_train`.

## DGP

Use the existing `ml_elements.dgp.GaussianBinaryDGP`:

```
p_pos = 0.20                          # mild imbalance
sigma = 1.0
info  = {f"x{i:02d}": 0.10 for i in range(100)}   # 100 weak features
```

- Each feature has separation `0.10` → individually near-useless, but the
  Bayes-optimal linear classifier aggregates them into a strong signal
  (expected best-achievable AUC ≳ 0.9 by the linear sum of informations).
- A `LogisticRegression` serves as the **Bayes-floor reference** (the
  DGP is linear in the features, so it is effectively optimal).
- `p_pos = 0.20` keeps both ROC-AUC and Average Precision meaningful.

## Sweep

**Single knob: `n_train`.** Everything else fixed.

| `n_train` | rationale |
|---|---|
| 200   | far below #features → severe ill-conditioning |
| 500   | ~5 × #features |
| 1 000 | 10 × #features |
| 2 000 | 20 × #features |
| 5 000 | comfortable regime |
| 10 000| data-rich baseline |

Fixed across all conditions:

- `n_test = 50 000`  (large → OOS estimates are essentially noise-free)
- `n_repeats = 30`  independent test draws per `(model, n_train)`
- `seeds.train = 101`, `seeds.test_base = 10 000` (via `DataBudget`)

## Models

| Setup | Backend | Notes |
|---|---|---|
| `logistic`   | `make_logistic()`                | Bayes-optimal linear reference |
| `catboost`   | `make_catboost(iterations=400, depth=4, learning_rate=0.05)` | high-capacity booster |
| `hgb`        | `make_hgb(iterations=400, max_leaf_nodes=16)`                | sklearn booster, no extra dep |
| `decision_tree` | `DecisionTreeClassifier(max_depth=4)` via `make_sklearn` | interpretable baseline |
| `figs`       | `imodels.FIGSClassifier(max_rules=20)`                        | interpretable tree ensemble |
| `greedy`     | `imodels.GreedyRuleListClassifier(max_depth=8)`               | ordered rule list |

> Each `imodels` estimator is wrapped through `make_sklearn`-style factory
> so it satisfies the `ModelBackend` protocol. All models see the **same
> 100 features**; no feature selection is performed (that is the point).

## Metrics

- `roc_auc`              — head metric (ranking quality)
- `average_precision`    — sensitive to the `0.20` base rate
- `brier_score`          — calibration / squared error
- `n_rules` / `n_leaves` — capacity actually used (for the interpretable models)

## Outputs

```
experiments/weak_features_sample_size/outputs/
├── scores.csv          # long-format: model × n_train × repeat × metric
├── summary.csv         # mean ± 95 % CI per (model, n_train)
├── fig_auc_vs_n.png    # AUC vs n_train, one line per model
├── fig_ap_vs_n.png     # AP  vs n_train
├── fig_gap_vs_n.png    # (catboost − model) gap vs n_train
└── report.md           # generated findings
```

## Planned analyses

1. **Learning curves** — AUC and AP vs `n_train`, one line per model.
   The logistic reference is the upper bound; expect catboost to track it,
   interpretable models to lag.
2. **Interpretability cost** — for each `n_train`, report
   `AUC(catboost) − AUC(model)`. The curve shape tells us whether the
   interpretable models *converge* or *diverge* with more data.
3. **Capacity utilisation** — plot `n_rules`/`n_leaves` learned by the
   interpretable models vs `n_train`. Hypothesis: they saturate quickly
   because their structural capacity is capped, even when more signal is
   available.
4. **Statistical CIs** — 30 repeats → bootstrap 95 % CI on each mean;
   report whether gaps are significant at each `n_train`.

## Execution checklist

- [ ] `config.yaml` — budget, DGP, model specs, metrics
- [ ] `study_sample_size.yaml` — the `n_train` sweep
- [ ] `run_experiment.py` — uses `TrialRunner` + `Study` + `ManualSearch`
- [ ] Add `imodels` factories (`make_figs`, `make_greedy`) in
      `ml_elements/models.py` if not present
- [ ] Makefile target `exp-weak-features`
- [ ] Plotting via `ml_elements.plots`

## Risks / open questions

- **`imodels` on 100 features**: FIGS and GreedyRuleList may be slow or
  unstable with 100 inputs at small `n_train`. If so, cap via a sanity
  pre-check and document.
- **DGP linearity**: because the Bayes rule is linear, logistic will likely
  win outright. That is fine — the *interesting* comparison is between
  the tree/rule models, not against logistic. Logistic is the reference
  line, not the protagonist.
- **Information level**: `0.10` per feature is a starting guess. If the
  logistic Bayes-AUC is too close to 1.0 we will lower it; if too close
  to 0.5 we will raise it. Decided empirically in a quick probe before the
  full sweep.
