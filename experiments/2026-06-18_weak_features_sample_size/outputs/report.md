# Weak Features × Sample Size — Report

Research question: with 100 weakly-informative features (info=0.1 per feature, p_pos=0.2), how does training size drive the gap between CatBoost and the interpretable rule/tree models?

Test set: n_test=50000, n_repeats=30.

## Headline: ROC-AUC at the largest training size (n_train=10000)

| Setup | mean AUC | ± 95% CI |
|---|---|---|
| LogisticRegression | 0.7546 | ± 0.0009 |
| CatBoost | 0.7074 | ± 0.0012 |
| HistGradientBoosting | 0.7075 | ± 0.0011 |
| DecisionTree (depth=4) | 0.5332 | ± 0.0010 |
| FIGS (max_rules=25) | 0.5752 | ± 0.0010 |
| GreedyRuleList (depth=8) | 0.5357 | ± 0.0010 |

## AUC gap vs CatBoost

| n_train | LogisticRegression | HistGradientBoosting | DecisionTree (depth=4) | FIGS (max_rules=25) | GreedyRuleList (depth=8) |
|---|---|---|---|---|---|
| 200 | +0.0548 | +0.0138 | -0.0314 | -0.0447 | -0.0406 |
| 500 | +0.0613 | +0.0055 | -0.0802 | -0.0956 | -0.0906 |
| 1000 | +0.0697 | +0.0189 | -0.1198 | -0.1067 | -0.1050 |
| 2000 | +0.0719 | +0.0013 | -0.1232 | -0.1048 | -0.1365 |
| 5000 | +0.0523 | -0.0039 | -0.1512 | -0.1393 | -0.1595 |
| 10000 | +0.0472 | +0.0001 | -0.1742 | -0.1322 | -0.1717 |

## Structural capacity used (rules / leaves)

| n_train | setup | capacity |
|---|---|---|
| 200 | CatBoost | 100 |
| 200 | DecisionTree (depth=4) | 6 |
| 200 | FIGS (max_rules=25) | 22 |
| 500 | CatBoost | 100 |
| 500 | DecisionTree (depth=4) | 8 |
| 500 | FIGS (max_rules=25) | 21 |
| 1000 | CatBoost | 100 |
| 1000 | DecisionTree (depth=4) | 7 |
| 1000 | FIGS (max_rules=25) | 25 |
| 2000 | CatBoost | 100 |
| 2000 | DecisionTree (depth=4) | 9 |
| 2000 | FIGS (max_rules=25) | 23 |
| 5000 | CatBoost | 100 |
| 5000 | DecisionTree (depth=4) | 14 |
| 5000 | FIGS (max_rules=25) | 23 |
| 10000 | CatBoost | 100 |
| 10000 | DecisionTree (depth=4) | 15 |
| 10000 | FIGS (max_rules=25) | 25 |

## Outputs

- `scores.csv` — long-format per-repeat scores
- `summary.csv` — mean ± 95% CI per (setup, n_train)
- `capacity.csv` — learned rule/leaf counts per setup
- `fig_auc_vs_n.png`, `fig_average_precision_vs_n.png`, `fig_brier_score_vs_n.png` — learning curves
- `fig_gap_vs_n.png` — AUC gap vs CatBoost
- `fig_capacity_vs_n.png` — capacity vs n_train
