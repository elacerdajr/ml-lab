"""The former names still resolve to the new vocabulary."""

import ml_elements as mle


def test_former_names_are_aliases():
    assert mle.GaussianBinaryDGP is mle.GaussianPopulation
    assert mle.RealDataDGP is mle.ResampledPopulation
    assert mle.ShiftedDGP is mle.ShiftedPopulation
    assert mle.DGP is mle.Population
    assert mle.StudyResult is mle.Result
    assert mle.Objective is mle.Metric


def test_former_module_paths_still_import():
    from ml_elements.dgp import GaussianBinaryDGP
    from ml_elements.objectives import N_ITER, Objective
    from ml_elements.runner import TrialResult, TrialRunner  # noqa: F401
    from ml_elements.search import ManualSearch  # noqa: F401
    from ml_elements.study import Study, StudyResult  # noqa: F401

    assert GaussianBinaryDGP is mle.GaussianPopulation and Objective is mle.Metric
    assert N_ITER is mle.N_ITER


def test_trial_accepts_its_former_dgp_keyword():
    pop = mle.GaussianPopulation(0.1, {"x1": 1.0})
    trial = mle.Trial(name="p_pos", value=0.1, dgp=pop, seed_offset=3)
    assert trial.population is pop and trial.dgp is pop
