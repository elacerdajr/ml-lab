# Concepts & vocabulary

## The four layers

```
DGP  ──►  TrialRunner  ──►  Study  ──►  Comparator / plots
 ↑              ↑               ↑
 data      model + metrics   parameter sweep
```

Every experiment is just filling in those four slots. The library enforces clean interfaces between them so you can swap any one without touching the others.

---

## DataBudget

How much data each trial uses. One object, shared across all trials in a study.

```python
from ml_elements import DataBudget

budget = DataBudget(
    n_train=4_000,
    n_valid=1_000,
    n_test=2_000,
    seed_train=1,
    seed_valid=2,
    seed_test_base=10_000,  # each repeat gets seed_test_base + repeat_index
    n_repeats=30,           # how many independent test draws to average over
)
```

`n_repeats` is the main knob for measurement precision. More repeats = tighter confidence intervals.

---

## DGP — Data-Generating Process

A DGP produces labelled DataFrames on demand. Any object with a `sample(n, seed)` method qualifies.

### Gaussian synthetic data

```python
from ml_elements import GaussianBinaryDGP

dgp = GaussianBinaryDGP(
    p_pos=0.10,                          # 10 % positive rate
    info={"x1": 0.8, "x2": 0.5, "x3": 0.1},  # per-feature separation
)
df = dgp.sample(n=1000, seed=42)
# columns: x1, x2, x3, y
```

`info` values are the mean difference between classes divided by the shared standard deviation. Higher = more predictive.

### Real data

```python
from ml_elements import RealDataDGP

dgp = RealDataDGP(df=df_historical, target_col="converted")
df  = dgp.sample(n=500, seed=7)   # samples with replacement
```

### Covariate shift

```python
from ml_elements import ShiftedDGP

dgp = ShiftedDGP(
    base=GaussianBinaryDGP(p_pos=0.1, info={"x1": 0.8}),
    shift_fn=lambda df: df.assign(x1=df["x1"] + 1.5),
)
```

The shift is applied at test time. Use it to probe how well a model generalises to distribution change.

---

## Model factories

Factories return a fresh unfitted model every time they're called. That keeps trial state independent.

```python
from ml_elements import make_logistic, make_hgb, make_catboost, make_sklearn

make_logistic(C=1.0)
make_hgb(learning_rate=0.05, max_iter=300)
make_catboost(depth=6, iterations=500)
make_sklearn(estimator)   # wrap any sklearn-compatible estimator
```

---

## Metrics

A `Metric` pairs a name, direction (`"higher"` / `"lower"`), and a function `(y_true, y_score) → float`.

```python
from ml_elements import AUC, AVG_PRECISION, LOGLOSS

# built-in
AUC.name        # "auc"
AUC.direction   # "higher"

# custom
from ml_elements import Metric
from sklearn.metrics import f1_score

F1 = Metric(name="f1", direction="higher", fn=lambda y, p: f1_score(y, p > 0.5))
```

---

## Objectives — composite scoring

An `Objective` combines metrics into a single scalar for training-time model selection.

```python
from ml_elements import AVG_PRECISION, N_ITER, N_LEAVES, COEF_L1, Objective

# AP only
objective = AVG_PRECISION

# AP minus a complexity penalty
objective = AVG_PRECISION - 0.001 * N_ITER

# Custom weighting
objective = 0.7 * AVG_PRECISION + 0.3 * AUC
```

Pass an objective as `scoring` to `make_hgb` or `make_catboost` to make it the early-stopping criterion.

---

## Trial & TrialRunner

A `Trial` is one experimental condition: which DGP to use, what the condition label is, and optional per-trial model overrides.

```python
from ml_elements import Trial

t = Trial(name="p_pos", value=0.05, dgp=my_dgp, seed_offset=0)
```

A `TrialRunner` fits one or more named model **setups** on the same training data, then scores them on many test draws.

```python
from ml_elements import TrialRunner

runner = TrialRunner(
    setups={"baseline": ["x1", "x2"], "challenger": ["x1", "x2", "x3"]},
    model_factory=make_hgb(),
    metrics=[AUC, AVG_PRECISION],
    budget=budget,
)
```

`setups` maps a name to a list of feature columns. Every setup uses the same model factory and budget.

---

## Study

A `Study` runs a list of trials through the runner and collects results.

```python
from ml_elements import Study

study  = Study(runner, primary_metric=AUC)
result = study.run(trials)           # StudyResult
improv = study.improvements(result, baseline="baseline", challenger="challenger")
summary = study.summarize(improv)    # DataFrame, one row per trial
```

`improvements` computes the per-trial delta between challenger and baseline on the primary metric.

---

## Search — generating trial lists

```python
from ml_elements import ManualSearch, SobolSearch, RandomSearch

# explicit grid
trials = ManualSearch(
    [{"p_pos": p} for p in [0.02, 0.05, 0.1, 0.25, 0.5]],
    trial_name="p_pos",
).trials(dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], info={"x1": 0.8}))

# quasi-random Sobol grid
trials = SobolSearch(
    param_space={"p_pos": (0.02, 0.5), "x3_info": (0.0, 1.5)},
    n_points=32,
    seed=0,
).trials(dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], info={"x1": 0.8, "x3": p["x3_info"]}))

# uniform random
trials = RandomSearch(
    param_space={"p_pos": (0.02, 0.5)},
    n_points=20,
    seed=99,
).trials(dgp_fn=...)
```

---

## Comparator — statistical testing

`Comparator` uses bootstrap resampling of the per-repeat scores to estimate the probability that the challenger beats the baseline.

```python
from ml_elements import Comparator

cmp = Comparator(n_boot=3_000)

# summary table — one row per trial value
report = cmp.full_report(result, baseline="baseline", challenger="challenger")

# drill into one trial
bac = cmp.for_trial(result, trial_value=0.10, baseline="baseline", challenger="challenger")
bac.ranking_report()   # P(challenger is best)
bac.pairwise_report()  # P(challenger > baseline)
```
