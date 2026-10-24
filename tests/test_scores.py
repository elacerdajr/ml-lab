"""Score transforms, labels and calibration helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from ml_elements import noise_rank, row_noise, shrink, smooth_labels, to_weighted_rows


def test_shrink_moves_scores_toward_the_base_rate_without_reordering():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 500)
    p = rng.random(500)
    shrunk = shrink(p, pi=0.1, lam=0.3)
    np.testing.assert_allclose(shrunk, 0.7 * p + 0.03)
    assert average_precision_score(y, shrunk) == pytest.approx(average_precision_score(y, p))


def test_row_noise_is_stable_per_row_and_in_unit_interval():
    ids = np.arange(10_000)
    u = row_noise(ids, seed=42)
    np.testing.assert_array_equal(u, row_noise(ids, seed=42))
    assert not np.array_equal(u, row_noise(ids, seed=43))
    assert (u >= 0).all() and (u < 1).all()
    assert u.mean() == pytest.approx(0.5, abs=0.02)
    np.testing.assert_array_equal(row_noise(ids[::-1], seed=42), u[::-1])


def test_noise_rank_mixes_score_and_row_noise():
    p, ids = np.array([0.2, 0.8]), np.array([5, 9])
    np.testing.assert_allclose(noise_rank(p, ids, alpha=0.9, seed=1), 0.9 * p + 0.1 * row_noise(ids, 1))


def test_smooth_labels():
    np.testing.assert_allclose(smooth_labels(np.array([0, 1]), pi=0.01, lam=0.1), [0.001, 0.901])


def test_weighted_rows_reproduce_soft_labels_exactly():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    X2, y2, w2 = to_weighted_rows(X, np.array([0.0, 1.0, 0.25]))
    # Hard labels collapse to one weight-1 row; the soft row splits 0.25 / 0.75.
    assert len(X2) == 4
    for x, target in [(1.0, 0.0), (2.0, 1.0), (3.0, 0.25)]:
        rows = (X2["x"] == x).to_numpy()
        assert (w2[rows] * y2[rows]).sum() / w2[rows].sum() == pytest.approx(target)
