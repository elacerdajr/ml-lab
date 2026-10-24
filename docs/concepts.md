# Concepts & vocabulary

`ml_elements` has six nouns and a handful of operators. Every experiment is the same sentence: **sweep** a parameter into trials, **evaluate** setups on them, **compare** two setups, **record** the run.

```
sweep(space, population=...) ──> [Trial] ──┐
                                            ├──> evaluate(trials, [Setup], [Metric], budget) ──> Result
     population.sample(n, seed) ────────────┘                                                    │
                                                    compare(result, baseline, challenger, metric) <┤
                                                                               record(run, tables) <┘
```

| Noun | What it is |
|---|---|
| `Population` | What a trial samples labelled rows from: anything with `sample(n, seed)` |
| `Setup` | One thing being evaluated: a name, its features, its model, an optional score transform |
| `Metric` | A named score with a direction; arithmetic on metrics returns a metric |
| `Trial` | One condition to test: a population, a label, a seed offset, optional overrides |
| `Result` | What `evaluate` returns: per-repeat scores, fitted models, data, failures |
| `Run` | One execution of an experiment folder under one profile, and its record |

`DataBudget` (sizes and seeds) and `RuleSet` (rules from an interpretable model) support these.

| Operator | Returns |
|---|---|
| `population.sample(n, seed)` | a labelled DataFrame |
| `sweep(space, population=..., method=...)` | `list[Trial]` |
| `evaluate(trials, setups, metrics, budget)` | `Result` |
| `compare(result, baseline, challenger, metric)` | one row per trial |
| `embed_leaves(model, X)` | a CatBoost model's per-tree leaf indices as features |
| `extract_rules(model, X, y, features)` | `RuleSet` |
| `record(run, tables)` | the run's output folder, with `run_meta.json` |

The former names (`GaussianBinaryDGP`, `TrialRunner`, `Study`, `SobolSearch`, `Objective`, `RuleExtractor`, …) still import and produce identical numbers; the end of this page maps them.

---

## Population

A population produces labelled DataFrames on demand. Any object with `sample(n, seed)` qualifies; the label column is `y`.

```python
from ml_elements import GaussianPopulation, ResampledPopulation, ShiftedPopulation

# Synthetic: x_j | y=1 ~ N(info_j·σ, σ), x_j | y=0 ~ N(0, σ)
population = GaussianPopulation(p_pos=0.10, info={"x1": 0.8, "x2": 0.5, "x3": 0.1})
df = population.sample(n=1000, seed=42)          # columns: y, x1, x2, x3

# A real table, sampled with replacement (target renamed to y)
population = ResampledPopulation(df=df_historical, target_col="converted", label_noise=0.05)

# Any population, transformed after sampling (shift, binning, feature engineering)
population = ShiftedPopulation(
    base=GaussianPopulation(p_pos=0.1, info={"x1": 0.8}),
    shift_fn=lambda df: df.assign(x1=df["x1"] + 1.5),
)
```

`info_j` is the class separation in units of σ: higher means more predictive.

---

## Setup

A setup names what is evaluated. Setups in one `evaluate` call share the same training data, so differences between them are differences between setups.

```python
from ml_elements import Setup, make_hgb, make_logistic, shrink

setups = [
    Setup("baseline", ["x1", "x2"]),                               # uses evaluate(model=...)
    Setup("challenger", ["x1", "x2", "x3"]),
    Setup("hgb", ["x1", "x2", "x3"], model=make_hgb(iterations=100)),
    Setup("shrunk", ["x1", "x2", "x3"], scores=lambda p: shrink(p, pi=0.1, lam=0.2)),
]
```

`model` is a zero-argument **factory**, not a fitted model, so every trial gets a fresh one. Factories: `make_logistic`, `make_hgb`, `make_catboost`, `make_decision_tree`, `make_figs`, `make_greedy_rule_list`, `make_rule_fit`, and `make_sklearn(EstimatorClass, **kwargs)` for anything else.

`scores` transforms the positive-class probabilities before metrics see them. `ml_elements.scores` has `shrink`, `noise_rank`, and the score-shape functions.

---

## Metric

A `Metric` has a `name`, a `direction` (`"higher"` or `"lower"` is better) and a scoring function.

```python
from ml_elements import AUC, AVG_PRECISION, LOGLOSS, BRIER, N_ITER, Metric

AUC.score(y_true, y_score)              # a float
LOGLOSS.improvement(0.20, 0.25)         # +0.05: positive always means "challenger better"

# Custom: on="scores" reads (y_true, y_score); on="model" reads the fitted model
F1 = Metric("f1", "higher", lambda y, p: f1_score(y, p > 0.5))
TREES = Metric("n_trees", "lower", lambda m: float(m.tree_count_), on="model")

# Arithmetic returns a metric
penalised = AVG_PRECISION - 0.001 * N_ITER      # direction "higher", needs model= when scored
blend = 0.7 * AUC + 0.3 * AVG_PRECISION
penalised = penalised.rename("ap_minus_trees")
```

Built in: `AUC`, `AVG_PRECISION`, `AVG_PRECISION_SMOOTH` (`make_smooth_ap(p0, m)`), `LOGLOSS`, `BRIER`; score shape `SCORE_ENTROPY`, `NORMALIZED_SCORE_ENTROPY`, `TIE_RATE`, `MAX_SCORE_GAP`; model terms `N_ITER`, `N_LEAVES`, `COEF_L1`.

`metric.score` clips scores to `[1e-8, 1 - 1e-8]`; `metric.fn` is the raw function for a single-term metric. `metric.as_scorer()` turns a metric into an sklearn scorer, for example to drive early stopping:

```python
make_hgb(early_stopping=True, scoring=(AVG_PRECISION - 0.001 * N_ITER).as_scorer())
```

`make_hgb` runs a fixed number of rounds unless you pass `early_stopping=True`; `scoring` has no effect without it.

---

## Trial, DataBudget and sweep

A `Trial` is one condition: a label (`name`, `value`), a population, and a `seed_offset` that keeps its data separate from other trials. It can override the model and the budget.

`DataBudget` says how much data each trial uses and where seeds start:

```python
from ml_elements import DataBudget

budget = DataBudget(n_train=4_000, n_valid=1_000, n_test=2_000,
                    seed_train=1, seed_valid=2, seed_test_base=10_000, n_repeats=30)
budget = DataBudget.quick(n=2000, n_repeats=20)   # train=test=n, valid=n//5
```

`n_repeats` fresh test draws per trial set how precise the comparison is.

`sweep` builds trials from a parameter space. Each trial keeps its parameters in `trial.params`.

```python
from ml_elements import sweep, sweep_points

# An explicit list: one trial per dict; a single parameter becomes trial.value
trials = sweep([{"p_pos": p} for p in [0.02, 0.05, 0.1, 0.25, 0.5]], name="p_pos",
               population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info={"x1": 0.8}))

# Sobol or uniform random points over bounds, optionally with a per-trial model
trials = sweep({"p_pos": (0.02, 0.5), "lr": (0.01, 0.3)}, method="sobol", n=32, seed=0,
               population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info={"x1": 0.8}),
               model=lambda p: make_hgb(learning_rate=p["lr"]))
sweep_points({"p_pos": (0.02, 0.5)}, method="sobol", n=32)   # the values, as a DataFrame
```

---

## evaluate and Result

```python
from ml_elements import evaluate

result = evaluate(trials, setups, [AUC, AVG_PRECISION], budget, model=make_logistic())
```

For each trial, `evaluate` samples train and validation data, fits every setup, then scores each on `n_repeats` fresh test draws. Seeds are `seed_train + seed_offset`, `seed_valid + seed_offset` and `seed_test_base + 1000·seed_offset + repeat`, so the same call gives the same numbers.

`Result` holds:

- `scores`: one row per (trial, setup, repeat), with a column per metric.
- `trial_results[i]`: that trial's fitted `models` (by setup name), `df_train`, `df_valid` and `scores`.
- `metrics`: the metric names, in order.
- `failures`: setups that raised, when you pass `on_error="record"`. Their metric values are NaN and the run continues, which is useful for models that are unstable at small sample sizes.

`n_jobs=-1` runs trials in parallel (joblib); `progress=True` shows a progress bar.

---

## compare

```python
from ml_elements import compare

table = compare(result, "baseline", "challenger", AVG_PRECISION)
```

One row per trial: `mean_improvement`, `std_improvement`, `p_improvement_gt_0` (the share of repeats where the challenger won), `mean_baseline_score`, `mean_challenger_score`, `n_repeats`, and `{setup}_{metric}_mean` for every metric. `improvement` is challenger minus baseline, sign-flipped for lower-is-better metrics. `per_repeat=True` returns the per-repeat rows instead.

`p_improvement_gt_0` is a share of repeats, not a probability that the effect is real. Repeats share one training set, so they measure test-side variation only.

---

## Run and record

A `Run` is one execution of an experiment folder under one profile. The default profile writes the tracked record to `outputs/`; any other profile writes to `outputs_<profile>/`, which is gitignored.

```python
from ml_elements import Run, record

run = Run(Path(__file__).parent, profile=args.profile, seed=cfg["seed"], config=cfg)
...                                      # write results into run.output_dir
record(run, {"summary": table})          # summary.csv + run_meta.json
```

See [Experiments guide → Outputs and the record](experiments.md#outputs-and-the-record).

---

## Leaves and rules

```python
from ml_elements import embed_leaves, extract_rules

leaves = embed_leaves(catboost_model, X)                 # (n, n_trees) leaf indices, float32
one_hot = embed_leaves(catboost_model, X, one_hot=True)  # sparse, one block per tree

rules = extract_rules(tree, X_train, y_train, feature_names, model_name="tree")
rules.save("outputs/rules_tree.json")
```

`extract_rules` supports sklearn `DecisionTreeClassifier` and imodels `RuleFitClassifier` and `GreedyRuleListClassifier`. `save_rule_report` renders rule sets as an interactive HTML matrix.

---

## Former names

| Former | Now |
|---|---|
| `GaussianBinaryDGP`, `RealDataDGP`, `ShiftedDGP`, `DGP` | `GaussianPopulation`, `ResampledPopulation`, `ShiftedPopulation`, `Population` |
| `TrialRunner(setups, model_factory, metrics, budget).run(trial)` | `evaluate([trial], setups, metrics, budget, model=...)` |
| `Study(runner, primary_metric).run(trials)` | `evaluate(trials, ...)` |
| `study.improvements` / `summarize` / `full_summary` | `compare(..., per_repeat=True)` / `compare(...)` |
| `StudyResult` | `Result` |
| `SobolSearch` / `RandomSearch` / `ManualSearch` `.trials(dgp_fn=...)` | `sweep(space, method=..., population=...)` |
| `Objective`, `ModelMetric`, `make_objective` | `Metric` (`on="model"`, `on="all"`) |
| `RuleExtractor(max_rules).from_model(...)` | `extract_rules(..., max_rules=...)` |
| `Trial(..., dgp=...)`, `trial.dgp` | `Trial(..., population=...)`, `trial.population` |
| `Comparator` | removed: it needed a `bayesian_ap_comparator` module that was never in the repository |
