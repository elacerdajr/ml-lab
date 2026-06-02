# ml-new-feature-study

Quantifying the marginal value of adding a new feature to a binary classifier,
under controlled regimes.

## What's here

- **`ml_elements/`** — composable building blocks for binary-classification
  experiments (DGPs, model factories, metrics, trial runner, study aggregation,
  scenario sweeps, hyperparameter search, bootstrap comparison, plots). See
  [`ml_elements/README.md`](ml_elements/README.md).
- **`feature_information_studies.py`** — a runnable study, built on
  `ml_elements`, comparing `{x1, x2}` vs `{x1, x2, x3}`:
  - **Study A** — vary class balance (`p_pos`), hold feature information fixed.
  - **Study B** — vary the information carried by `x3`, hold `p_pos` fixed.

## Sweep vs. search

A core distinction the library makes explicit:

- **Scenario knobs** (`p_pos`, `info`) are *experiment design* choices. You
  **sweep** them on a grid (`ScenarioSweep`) to map a response curve — there is
  no "best" value to find.
- **Model hyperparameters** (`learning_rate`, `depth`, …) are knobs you
  **search** (`SobolSearch` / `RandomSearch`) for the single best model within a
  fixed scenario.

Don't put `p_pos`/`info` in a hyperparameter search space — sweep them.

## Install & run

```bash
pip install numpy pandas scikit-learn matplotlib scipy
# optional: catboost, rich, joblib
python feature_information_studies.py
```

Artifacts (CSVs + plots) land in `artifacts_info_studies/`.
