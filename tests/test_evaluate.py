"""Evaluating setups on trials, sweeping trials, and comparing setups."""

import numpy as np
import pandas as pd
import pytest

from ml_elements import (
    AUC,
    AVG_PRECISION,
    DataBudget,
    GaussianPopulation,
    Setup,
    Trial,
    compare,
    evaluate,
    make_logistic,
    shrink,
    sweep,
    sweep_points,
)

POPULATION = GaussianPopulation(p_pos=0.2, info={"x1": 1.0, "x2": 0.5, "x3": 0.0})
SETUPS = [Setup("baseline", ["x1"]), Setup("challenger", ["x1", "x2"])]
BUDGET = DataBudget.quick(n=400, n_repeats=3)
TRIAL = Trial(name="p_pos", value=0.2, population=POPULATION, seed_offset=1)


def _evaluate(trials=TRIAL, setups=SETUPS, **kw):
    return evaluate(trials, setups, [AUC, AVG_PRECISION], BUDGET, model=make_logistic(), **kw)


def test_one_trial_scores_every_setup_on_every_repeat():
    result = _evaluate()
    assert len(result.scores) == len(SETUPS) * BUDGET.n_repeats
    assert set(result.scores["setup"]) == {"baseline", "challenger"}
    assert {"auc", "average_precision", "repeat", "test_seed"} <= set(result.scores.columns)
    assert result.metrics == ["auc", "average_precision"]
    tr = result.trial_results[0]
    assert set(tr.models) == {"baseline", "challenger"} and len(tr.df_train) == BUDGET.n_train
    assert result.failures.empty


def test_evaluation_is_reproducible_for_the_same_trial():
    pd.testing.assert_frame_equal(_evaluate().scores, _evaluate().scores)


def test_setups_can_be_given_as_the_former_dict_form():
    as_dict = _evaluate(setups={"baseline": ["x1"], "challenger": (["x1", "x2"], make_logistic())})
    pd.testing.assert_frame_equal(as_dict.scores, _evaluate().scores)


def test_a_setup_score_transform_is_applied_before_scoring():
    pi = 0.2
    setups = SETUPS + [Setup("shrunk", ["x1", "x2"], scores=lambda p: shrink(p, pi, 0.5))]
    s = _evaluate(setups=setups).scores.set_index(["repeat", "setup"])
    # Shrinkage is monotonic, so ranking metrics are unchanged.
    np.testing.assert_allclose(s.xs("shrunk", level="setup")["auc"],
                               s.xs("challenger", level="setup")["auc"])


class Exploding:
    def fit(self, X, y):
        raise RuntimeError("cannot fit at this size")


def test_record_mode_keeps_going_when_one_setup_fails():
    setups = SETUPS + [Setup("unstable", ["x1"], model=Exploding)]
    with pytest.raises(RuntimeError, match="cannot fit"):
        _evaluate(setups=setups)

    result = _evaluate(setups=setups, on_error="record")
    unstable = result.scores[result.scores["setup"] == "unstable"]
    assert len(unstable) == BUDGET.n_repeats and unstable["auc"].isna().all()
    assert result.scores[result.scores["setup"] != "unstable"]["auc"].notna().all()
    assert result.failures[["setup", "stage"]].values.tolist() == [["unstable", "fit"]]
    assert "cannot fit" in result.failures["error"].iloc[0]


def test_compare_summarises_challenger_against_baseline_per_trial():
    trials = sweep([{"p_pos": 0.1}, {"p_pos": 0.3}], name="p_pos",
                   population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info={"x1": 1.0, "x2": 0.8}))
    result = _evaluate(trials=trials)
    summary = compare(result, "baseline", "challenger", AUC)
    assert list(summary["trial_value"]) == [0.1, 0.3]
    assert (summary["n_repeats"] == BUDGET.n_repeats).all()
    assert {"baseline_auc_mean", "challenger_average_precision_mean"} <= set(summary.columns)
    # x2 carries signal, so adding it should help on average.
    assert (summary["mean_improvement"] > 0).all()

    per_repeat = compare(result, "baseline", "challenger", AUC, per_repeat=True)
    assert len(per_repeat) == 2 * BUDGET.n_repeats
    np.testing.assert_allclose(per_repeat["improvement"], per_repeat["challenger"] - per_repeat["baseline"])


def test_compare_names_the_missing_setup():
    with pytest.raises(ValueError, match="'nope' not found"):
        compare(_evaluate(), "baseline", "nope", AUC)


def test_evaluate_needs_a_baseline_and_a_challenger():
    with pytest.raises(ValueError, match="at least two"):
        _evaluate(setups=[Setup("only", ["x1"])])


def test_sobol_sweep_gives_one_trial_per_point_with_its_parameters():
    space = {"p_pos": (0.05, 0.5)}
    trials = sweep(space, method="sobol", n=8, population=lambda p: GaussianPopulation(p["p_pos"], {"x1": 1.0}))
    points = sweep_points(space, method="sobol", n=8)
    assert len(trials) == 8 and points["p_pos"].between(0.05, 0.5).all()
    assert [t.population.p_pos for t in trials] == pytest.approx(points["p_pos"].tolist())
    assert [t.params["p_pos"] for t in trials] == pytest.approx(points["p_pos"].tolist())
    assert [t.seed_offset for t in trials] == list(range(8))


def test_random_sweep_is_reproducible_by_seed():
    def draw(seed):
        trials = sweep({"p_pos": (0.05, 0.5)}, method="random", n=5, seed=seed,
                       population=lambda p: GaussianPopulation(p["p_pos"], {"x1": 1.0}))
        return [t.population.p_pos for t in trials]

    assert draw(1) == draw(1) and draw(1) != draw(2)


def test_list_sweep_uses_a_single_parameter_as_the_trial_value():
    trials = sweep([{"p_pos": 0.1}, {"p_pos": 0.3}], name="p_pos",
                   population=lambda p: GaussianPopulation(p["p_pos"], {"x1": 1.0}))
    assert [(t.name, t.value) for t in trials] == [("p_pos", 0.1), ("p_pos", 0.3)]


def test_sweep_rejects_a_bounds_space_without_a_method():
    with pytest.raises(ValueError, match="sobol"):
        sweep({"p_pos": (0.1, 0.5)}, population=lambda p: None)
