"""Evaluating setups on trials, sweeping trials, and comparing setups."""

import pandas as pd
import pytest

from ml_elements import (
    AUC,
    AVG_PRECISION,
    DataBudget,
    GaussianBinaryDGP,
    ManualSearch,
    RandomSearch,
    SobolSearch,
    Study,
    Trial,
    TrialRunner,
    make_logistic,
)

POPULATION = GaussianBinaryDGP(p_pos=0.2, info={"x1": 1.0, "x2": 0.5, "x3": 0.0})
SETUPS = {"baseline": ["x1"], "challenger": ["x1", "x2"]}
BUDGET = DataBudget.quick(n=400, n_repeats=3)


def _runner():
    return TrialRunner(
        setups=SETUPS, model_factory=make_logistic(), metrics=[AUC, AVG_PRECISION], budget=BUDGET
    )


def test_one_trial_scores_every_setup_on_every_repeat():
    result = _runner().run(Trial(name="p_pos", value=0.2, dgp=POPULATION, seed_offset=1))
    assert len(result.scores) == len(SETUPS) * BUDGET.n_repeats
    assert set(result.scores["setup"]) == set(SETUPS)
    assert {"auc", "average_precision", "repeat", "test_seed"} <= set(result.scores.columns)
    assert set(result.models) == set(SETUPS)
    assert len(result.df_train) == BUDGET.n_train


def test_evaluation_is_reproducible_for_the_same_trial():
    trial = Trial(name="p_pos", value=0.2, dgp=POPULATION, seed_offset=1)
    pd.testing.assert_frame_equal(_runner().run(trial).scores, _runner().run(trial).scores)


def test_study_concatenates_trials_and_compares_challenger_to_baseline():
    trials = ManualSearch([{"p_pos": 0.1}, {"p_pos": 0.3}], trial_name="p_pos").trials(
        dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], info={"x1": 1.0, "x2": 0.8})
    )
    study = Study(_runner(), primary_metric=AUC)
    result = study.run(trials)
    assert len(result.scores) == 2 * len(SETUPS) * BUDGET.n_repeats

    summary = study.summarize(study.improvements(result, "baseline", "challenger"))
    assert list(summary["trial_value"]) == [0.1, 0.3]
    assert (summary["n_repeats"] == BUDGET.n_repeats).all()
    # x2 carries signal, so adding it should help on average.
    assert (summary["mean_improvement"] > 0).all()


def test_runner_needs_a_baseline_and_a_challenger():
    with pytest.raises(ValueError, match="at least two"):
        TrialRunner(setups={"only": ["x1"]}, model_factory=make_logistic(), metrics=[AUC],
                    budget=BUDGET)


def test_sobol_sweep_gives_one_trial_per_point_and_inspectable_parameters():
    search = SobolSearch({"p_pos": (0.05, 0.5)}, n_points=8, seed=0)
    trials = search.trials(dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], info={"x1": 1.0}))
    params = search.param_dataframe()
    assert len(trials) == 8 and len(params) == 8
    assert params["p_pos"].between(0.05, 0.5).all()
    assert [t.dgp.p_pos for t in trials] == pytest.approx(params["p_pos"].tolist())
    assert len({t.seed_offset for t in trials}) == 8


def test_random_sweep_is_reproducible_by_seed():
    def draw(seed):
        trials = RandomSearch({"p_pos": (0.05, 0.5)}, n_points=5, seed=seed).trials(
            dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], info={"x1": 1.0})
        )
        return [t.dgp.p_pos for t in trials]

    assert draw(1) == draw(1)
    assert draw(1) != draw(2)
