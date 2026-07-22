# Experiments guide

## How an experiment is structured

Each experiment lives in `experiments/<name>/` and has three parts:

```
experiments/roc_vs_ap/
├── config.yaml              # shared defaults (budget, DGP parameters, model settings)
├── study_imbalance.yaml     # overrides for Study 1
├── study_feature_info.yaml  # overrides for Study 2
├── run_experiment.py        # runs all studies
├── run_study_imbalance.py   # runs Study 1 only
├── run_study_feature_info.py
└── outputs/
    ├── metrics.csv
    ├── summary.json
    ├── report.md
    └── fig*.png
```

The YAML files separate *what to run* from *how to run it*. The Python files are thin wrappers that load config and call `ml_elements`.

---

## Running the existing experiment

```bash
make exp-roc-vs-ap               # both studies
make exp-roc-vs-ap-imbalance     # Study 1: class imbalance sweep
make exp-roc-vs-ap-info          # Study 2: feature information sweep
```

Outputs land in `experiments/roc_vs_ap/outputs/`. The generated `report.md` in that folder summarises the findings.

---

## Adding a new experiment

### Step 1 — create the folder

```bash
mkdir -p experiments/my_question/outputs
```

### Step 2 — write a config file

```yaml
# experiments/my_question/config.yaml

budget:
  n_train: 4000
  n_valid: 1000
  n_test: 2000
  n_repeats: 30

dgp:
  features:
    x1: 0.80
    x2: 0.50
    x3: 0.20

model:
  type: hgb
  learning_rate: 0.05
```

### Step 3 — write the script

```python
# experiments/my_question/run.py

import yaml
from pathlib import Path
from ml_elements import (
    GaussianBinaryDGP, make_hgb,
    AUC, AVG_PRECISION,
    DataBudget, TrialRunner, Study, ManualSearch,
    plot_study,
)

cfg = yaml.safe_load(Path("experiments/my_question/config.yaml").read_text())

budget = DataBudget(
    n_train=cfg["budget"]["n_train"],
    n_valid=cfg["budget"]["n_valid"],
    n_test=cfg["budget"]["n_test"],
    seed_train=1, seed_valid=2, seed_test_base=10_000,
    n_repeats=cfg["budget"]["n_repeats"],
)

setups = {
    "baseline":   ["x1", "x2"],
    "challenger": ["x1", "x2", "x3"],
}

runner = TrialRunner(
    setups=setups,
    model_factory=make_hgb(learning_rate=cfg["model"]["learning_rate"]),
    metrics=[AUC, AVG_PRECISION],
    budget=budget,
)

study = Study(runner, primary_metric=AVG_PRECISION)

info = cfg["dgp"]["features"]
trials = ManualSearch(
    [{"p_pos": p} for p in [0.02, 0.05, 0.10, 0.20, 0.50]],
    trial_name="p_pos",
).trials(
    dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], info=info)
)

result  = study.run(trials)
improv  = study.improvements(result, baseline="baseline", challenger="challenger")
summary = study.summarize(improv)

out = Path("experiments/my_question/outputs")
summary.to_csv(out / "summary.csv", index=False)
result.scores.to_csv(out / "scores.csv", index=False)

fig = plot_study(summary, metric=AVG_PRECISION)
fig.savefig(out / "fig_improvement.png")
print("done →", out)
```

### Step 4 — add a Makefile target

```makefile
exp-my-question:
	python experiments/my_question/run.py
```

---

## Checklist for a clean experiment

- [ ] All randomness flows through `DataBudget` seeds — no bare `random.seed()` calls
- [ ] Config values are in YAML, not hardcoded in the script
- [ ] Outputs go to `experiments/<name>/outputs/`, not the repo root
- [ ] The script is idempotent — re-running overwrites outputs cleanly
- [ ] A brief comment at the top of the script states the research question

---

## Existing experiments

### `roc_vs_ap` — ROC-AUC vs Average Precision

**Question**: when and why does optimising for Average Precision over ROC-AUC matter?

**Studies**:
- `study_imbalance`: sweep `p_pos` ∈ {0.02, 0.05, 0.10, 0.20, 0.35, 0.50}, fixed feature info
- `study_feature_info`: sweep `info_scale` ∈ {0.1, 0.3, 0.5, 0.8, 1.2, 1.8}, fixed `p_pos=0.10`

**Models**: AUC-trained HGB · AP-trained HGB · AP-penalised HGB (composite objective)

**Key findings** (full details in `experiments/roc_vs_ap/outputs/report.md`):
- ROC-AUC is insensitive to class imbalance; AP is not → in imbalanced settings, optimise for AP
- AP-penalised model recovers most of the AP gain with fewer trees
- Specialisation advantage is consistent across feature information levels

### `leaf_embedding_umap` — Leaf-Embedding UMAP Reduction vs Native CatBoost

**Question**: on an imbalanced binary target with mixed categorical + numerical features, how much downstream classification performance survives compressing a CatBoost model's raw per-tree leaf indices (no one-hot) down to k dimensions with UMAP (Hamming metric), compared to a CatBoost model trained natively on the raw features?

**Setup**: `GaussianBinaryDGP` + `ShiftedDGP` (fixed-edge binning of a subset of features into categorical bins), `p_pos=0.05`, k ∈ {2, 5, 10, 20}, downstream classifiers: logit, SVM, RF, MLP, CatBoost. Also tracks per-classifier training time vs k against the native baseline's fit time.

**Key findings** (full details in `experiments/leaf_embedding_umap/report.md`):
- An MLP (or CatBoost, depending on k) trained on the UMAP-reduced leaf embedding nearly matches the native-feature CatBoost baseline, even at low k
- Downstream classifier choice matters more than k, but doesn't rank purely by model complexity — an RBF-kernel SVM on raw UMAP coordinates is inconsistent and sometimes loses to plain logistic regression
- Fit time for MLP/RF on the compressed embedding can exceed the native CatBoost baseline's fit time on the full leaf space, despite training on far fewer dimensions
- Returns to higher k are non-monotonic — a 2-D projection already captures most of the leaf-membership signal
