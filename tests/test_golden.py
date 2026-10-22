"""
The refactor into evaluate / compare / sweep must not change any number.

tests/data/golden_*.csv were written by the former TrialRunner + Study +
ManualSearch implementation (commit e0f2d07) for the setup below.
"""

from pathlib import Path

import pandas as pd

from ml_elements import (
    AUC,
    AVG_PRECISION,
    LOGLOSS,
    N_LEAVES,
    DataBudget,
    GaussianBinaryDGP,
    GaussianPopulation,
    ManualSearch,
    Setup,
    Study,
    TrialRunner,
    compare,
    evaluate,
    make_hgb,
    make_logistic,
    sweep,
)

DATA = Path(__file__).parent / "data"
BUDGET = DataBudget(n_train=600, n_valid=150, n_test=500, seed_train=101, seed_valid=202,
                    seed_test_base=10_000, n_repeats=4)
METRICS = [AUC, AVG_PRECISION, LOGLOSS, (AVG_PRECISION - 0.001 * N_LEAVES).rename("ap_penalised")]
INFO = {"x1": 0.8, "x2": 0.4, "x3": 0.6}


def _golden(name):
    return pd.read_csv(DATA / f"golden_{name}.csv")


def _check(scores, summary):
    pd.testing.assert_frame_equal(scores.reset_index(drop=True), _golden("scores"), check_dtype=False)
    pd.testing.assert_frame_equal(summary.reset_index(drop=True), _golden("summary"), check_dtype=False)


def test_evaluate_and_compare_reproduce_the_former_implementation(tmp_path):
    trials = sweep([{"p_pos": p} for p in (0.05, 0.2)], name="p_pos",
                   population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info=INFO))
    setups = [Setup("baseline", ["x1", "x2"]), Setup("challenger", ["x1", "x2", "x3"]),
              Setup("hgb", ["x1", "x2", "x3"], model=make_hgb(iterations=30))]
    result = evaluate(trials, setups, METRICS, BUDGET, model=make_logistic())
    # Round-trip through CSV exactly as the golden files were written.
    result.scores.to_csv(tmp_path / "s.csv", index=False)
    compare(result, "baseline", "challenger", AVG_PRECISION).to_csv(tmp_path / "c.csv", index=False)
    _check(pd.read_csv(tmp_path / "s.csv"), pd.read_csv(tmp_path / "c.csv"))


def test_former_names_still_produce_the_same_numbers(tmp_path):
    runner = TrialRunner(
        setups={"baseline": ["x1", "x2"], "challenger": ["x1", "x2", "x3"],
                "hgb": (["x1", "x2", "x3"], make_hgb(iterations=30))},
        model_factory=make_logistic(), metrics=METRICS, budget=BUDGET)
    trials = ManualSearch([{"p_pos": p} for p in (0.05, 0.2)], trial_name="p_pos").trials(
        dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], info=INFO))
    study = Study(runner, primary_metric=AVG_PRECISION)
    result = study.run(trials)
    result.scores.to_csv(tmp_path / "s.csv", index=False)
    study.full_summary(result, "baseline", "challenger").to_csv(tmp_path / "c.csv", index=False)
    _check(pd.read_csv(tmp_path / "s.csv"), pd.read_csv(tmp_path / "c.csv"))
