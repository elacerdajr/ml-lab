"""Populations: what experiments sample labelled rows from."""

import numpy as np
import pandas as pd
import pytest

from ml_elements import GaussianPopulation, ResampledPopulation, ShiftedPopulation


def test_gaussian_hits_requested_positive_rate():
    df = GaussianPopulation(p_pos=0.1, info={"x1": 0.8}).sample(n=50_000, seed=0)
    # Binomial standard error at n=50k is ~0.0013; 0.005 is ~4 SE.
    assert abs(df["y"].mean() - 0.1) < 0.005


def test_gaussian_same_seed_same_rows_and_different_seed_different_rows():
    pop = GaussianPopulation(p_pos=0.2, info={"x1": 0.8, "x2": 0.0})
    pd.testing.assert_frame_equal(pop.sample(500, seed=7), pop.sample(500, seed=7))
    assert not pop.sample(500, seed=7).equals(pop.sample(500, seed=8))


def test_gaussian_info_is_the_class_separation_in_sigma_units():
    df = GaussianPopulation(p_pos=0.5, info={"signal": 1.5, "noise": 0.0}, sigma=2.0).sample(
        n=200_000, seed=1
    )
    means = df.groupby("y").mean()
    assert means.loc[1, "signal"] - means.loc[0, "signal"] == pytest.approx(1.5 * 2.0, abs=0.05)
    assert means.loc[1, "noise"] - means.loc[0, "noise"] == pytest.approx(0.0, abs=0.05)
    assert list(df.columns) == ["y", "signal", "noise"]


@pytest.mark.parametrize("p_pos", [0.0, 1.0, -0.1])
def test_gaussian_rejects_degenerate_positive_rate(p_pos):
    with pytest.raises(ValueError, match="p_pos"):
        GaussianPopulation(p_pos=p_pos, info={"x1": 1.0})


def test_resampled_draws_rows_from_the_table_and_renames_the_target():
    table = pd.DataFrame({"clicked": [0, 1, 0, 1], "x": [10.0, 11.0, 12.0, 13.0]})
    out = ResampledPopulation(table, target_col="clicked").sample(n=100, seed=3)
    assert list(out.columns) == ["y", "x"]
    assert set(out["x"]).issubset(set(table["x"]))
    pd.testing.assert_frame_equal(out, ResampledPopulation(table, target_col="clicked").sample(100, seed=3))


def test_resampled_label_noise_flips_about_that_fraction():
    table = pd.DataFrame({"y": np.zeros(1000, dtype=int), "x": np.arange(1000)})
    out = ResampledPopulation(table, label_noise=0.2).sample(n=20_000, seed=0)
    assert out["y"].mean() == pytest.approx(0.2, abs=0.01)


def test_shifted_applies_the_transform_to_each_sample():
    base = GaussianPopulation(p_pos=0.3, info={"x1": 0.5})
    shifted = ShiftedPopulation(base, shift_fn=lambda df: df.assign(x1=df["x1"] + 10.0))
    a, b = base.sample(300, seed=5), shifted.sample(300, seed=5)
    np.testing.assert_allclose(b["x1"], a["x1"] + 10.0)
    pd.testing.assert_series_equal(a["y"], b["y"])
