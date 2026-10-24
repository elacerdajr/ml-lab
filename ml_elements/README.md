# ml_elements

The reusable library behind every experiment in this repo. The vocabulary, with examples, is in [docs/concepts.md](../docs/concepts.md); a worked example is in [docs/quickstart.md](../docs/quickstart.md).

```python
from ml_elements import *

trials = sweep([{"p_pos": p} for p in (0.02, 0.1, 0.5)], name="p_pos",
               population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info={"x1": 0.8, "x2": 0.3}))
result = evaluate(trials, [Setup("base", ["x1"]), Setup("both", ["x1", "x2"])],
                  [AUC, AVG_PRECISION], DataBudget.quick(), model=make_logistic())
compare(result, "base", "both", AVG_PRECISION)
```

## Modules

| Module | Holds |
|---|---|
| `population.py` | `GaussianPopulation`, `ResampledPopulation`, `ShiftedPopulation` |
| `evaluate.py` | `Setup`, `Result`, `TrialResult`, `evaluate`, `compare` |
| `sweep.py` | `sweep`, `sweep_points` |
| `trial.py` | `Trial`, `DataBudget` |
| `metrics.py` | `Metric` and the built-in metrics (`AUC`, `AVG_PRECISION`, `N_ITER`, `SCORE_ENTROPY`, …) |
| `scores.py` | Score shape (`score_entropy`, `tie_rate`, `max_score_gap`), transforms (`shrink`, `noise_rank`), labels and calibration |
| `models.py` | Model factories (`make_logistic`, `make_hgb`, `make_catboost`, …) and `embed_leaves` |
| `run.py` | `Run`, `record`, `report_figures`: the experiment record |
| `rules.py`, `rule_viz.py` | `RuleSet`, `extract_rules`, and the HTML rule matrix |
| `plots.py` | `plot_study`, `plot_calibration`, `plot_feature_importance`, … |
| `protocols.py` | `Population`, `ModelBackend`, `MetricFn` |
| `embedding_pca.py`, `rich_logger.py` | Helpers for the embedding-PCA experiment and console logging |

`dgp.py`, `objectives.py`, `runner.py`, `study.py` and `search.py` keep the former names (`GaussianBinaryDGP`, `Objective`, `TrialRunner`, `Study`, `SobolSearch`, …) working as thin wrappers over the modules above.

## Tests

```bash
make test        # from the repo root; tests/ covers this package
```
