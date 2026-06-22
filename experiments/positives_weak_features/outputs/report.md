# Positives Weak Features — Report

**Research question**: at 1% positives, which CatBoost hyperparameters amplify or suppress the effect of 20 random noise features on model AP?

## Setup

| Parameter | Value |
|---|---|
| p_pos | 0.01 |
| Informative features | 20 (info=0.1) |
| Noise features | 20 (info=0.0) |
| n_train | 5000 |
| n_eval | 2000 |
| n_test | 10000 |
| iterations | 500 |
| learning_rate | 0.05 |
| depth sweep | [2, 3, 4, 5, 6, 8] |
| l2_leaf_reg sweep | [1, 3, 10, 30, 100] |


## Best config by eval AP

depth=2, l2_leaf_reg=30.0  → eval AP=0.0129, test AP=0.0132

## Best config by test AP

depth=5, l2_leaf_reg=1.0  → eval AP=0.0092, test AP=0.0140

⚠ Eval and test disagree — eval-set optimism detected.

## Full results (by eval AP descending)

| depth | l2_leaf_reg | auc_eval | average_precision_eval | brier_score_eval | logloss_eval | auc_test | average_precision_test | brier_score_test | logloss_test |
|---|---|---|---|---|---|---|---|---|---|
| 2.0000 | 30.0000 | 0.5050 | 0.0129 | 0.0075 | 0.0453 | 0.5685 | 0.0132 | 0.0107 | 0.0607 |
| 8.0000 | 1.0000 | 0.4854 | 0.0101 | 0.0075 | 0.0608 | 0.5640 | 0.0131 | 0.0108 | 0.0846 |
| 2.0000 | 1.0000 | 0.4920 | 0.0100 | 0.0076 | 0.0479 | 0.5425 | 0.0126 | 0.0108 | 0.0636 |
| 5.0000 | 1.0000 | 0.4743 | 0.0092 | 0.0075 | 0.0545 | 0.5691 | 0.0140 | 0.0108 | 0.0741 |
| 4.0000 | 1.0000 | 0.4552 | 0.0089 | 0.0075 | 0.0523 | 0.5568 | 0.0133 | 0.0108 | 0.0693 |
| 4.0000 | 3.0000 | 0.5079 | 0.0088 | 0.0075 | 0.0476 | 0.5409 | 0.0123 | 0.0108 | 0.0649 |
| 2.0000 | 3.0000 | 0.4889 | 0.0088 | 0.0076 | 0.0477 | 0.5531 | 0.0127 | 0.0108 | 0.0627 |
| 8.0000 | 10.0000 | 0.5111 | 0.0085 | 0.0075 | 0.0467 | 0.5623 | 0.0127 | 0.0107 | 0.0638 |
| 6.0000 | 1.0000 | 0.4638 | 0.0085 | 0.0075 | 0.0572 | 0.5551 | 0.0134 | 0.0108 | 0.0784 |
| 4.0000 | 100.0000 | 0.4775 | 0.0084 | 0.0075 | 0.0455 | 0.5366 | 0.0132 | 0.0107 | 0.0604 |

## Outputs

- `scores.csv` — long format: one row per (depth, l2_leaf_reg, split)
- `summary.csv` — wide format: eval + test side-by-side, gap column
- `feature_importances.csv` — per-feature importance for each config
- `fig_heatmap_*.png` — heatmaps of each metric (eval | test)
- `fig_depth_lines_*.png` — metric vs l2_leaf_reg, one line per depth
- `fig_eval_vs_test.png` — scatter: eval vs test metric per config
- `fig_noise_importance.png` — fraction of importance on noise features
