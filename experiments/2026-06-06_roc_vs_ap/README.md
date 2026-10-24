# ROC-AUC vs Average Precision

**Question:** when and why does it matter to train on Average Precision instead of ROC-AUC?

Three HistGradientBoosting models differ only in their early-stopping criterion: ROC-AUC, AP, and AP minus a tree-count penalty. Two studies vary the conditions:

- **Imbalance sweep** (`study_imbalance.yaml`): fixed feature information, `p_pos` from 2 % to 50 %.
- **Information sweep** (`study_feature_info.yaml`): `p_pos = 0.10`, feature signal scaled from weak to strong.

## Run

```bash
make exp-roc-vs-ap               # both studies (about 25 s)
make exp-roc-vs-ap-imbalance     # imbalance sweep only
make exp-roc-vs-ap-info          # information sweep only
```

Settings: `config.yaml` (shared) and the two study YAMLs.

## Answer

`outputs/report.md`, with metrics in `outputs/metrics*.csv` and summaries in `outputs/summary*.json`.
