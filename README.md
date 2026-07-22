# ml-lab

A personal ML experimentation lab for answering quantitative questions about model behaviour. The core idea: define a question, wire up composable building blocks, run controlled trials, and read out a statistical answer.

## What lives here

| Path | Purpose |
|---|---|
| `ml_elements/` | Reusable library — data generators, models, metrics, runners, plots |
| `experiments/` | Self-contained experiments (`YYYY-MM-DD_<slug>/`), each with config + scripts + outputs |
| `notebooks/` | Exploratory notebooks for one-off analysis |

## The mental model

Every experiment has the same four moving parts:

```
DGP  →  TrialRunner  →  Study  →  Comparator / plots
```

1. **DGP** generates labelled data under a specific condition (e.g. `p_pos=0.05`).
2. **TrialRunner** fits one or more model setups on the same training split and scores them on repeated test draws.
3. **Study** sweeps a parameter across many `Trial`s and aggregates the scores.
4. **Comparator** / plots turn the numbers into a statistical answer.

Any piece can be swapped independently — different model, different DGP, different metric — without touching the rest.

## Running an experiment

```bash
# install dependencies (uses uv; see pyproject.toml for the full set)
uv sync                         # core deps only
uv sync --extra all             # + catboost, imodels, joblib/rich, embeddings
uv sync --extra imodels         # cherry-pick just the extras you need

# run the ROC-AUC vs Average Precision experiment
make exp-roc-vs-ap

# or run individual studies
make exp-roc-vs-ap-imbalance   # class imbalance sweep
make exp-roc-vs-ap-info        # feature information sweep
```

> All commands should be run through `uv run` (e.g. `uv run python ...`) or
> inside the `.venv` that `uv sync` creates — this guarantees the exact
> locked dependency set recorded in `uv.lock`.

Outputs land in `experiments/2026-06-06_roc_vs_ap/outputs/`.

## Docs

- [Concepts & vocabulary](docs/concepts.md) — what each building block does
- [Quickstart](docs/quickstart.md) — a complete worked example from scratch
- [Cookbook](docs/cookbook.md) — short recipes for common tasks
- [Experiments guide](docs/experiments.md) — how to add a new experiment (folders named `YYYY-MM-DD_<slug>/`)

## Current experiments

### ROC-AUC vs Average Precision (`experiments/2026-06-06_roc_vs_ap/`)

**Question**: when and why does it matter to train on Average Precision instead of ROC-AUC?

Two studies:
- **Imbalance sweep** — fix feature information, vary `p_pos` from 2 % to 50 %
- **Information sweep** — fix `p_pos=0.10`, vary feature signal strength

Three model setups compared: AUC-trained · AP-trained · AP-penalised (composite objective reduces tree count).

Results are in `experiments/2026-06-06_roc_vs_ap/outputs/report.md`.

### Embedding PCA reconstruction (`experiments/2026-06-09_embedding_pca/`)

**Question**: how many PCA dimensions are needed to reproduce a medium-sized sentence embedding space for diverse website descriptions?

The experiment embeds more than 100 one-sentence site descriptions with `sentence-transformers/all-MiniLM-L6-v2`, fits PCA, and reports explained variance, reconstruction error, threshold dimensions, and plots of PCA dimensions 1, 2, and 3.

Run it with `make exp-embedding-pca` after installing the optional sentence embedding dependencies listed in the experiment README.
