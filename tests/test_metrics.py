"""Metrics: scored quantities with a direction, and arithmetic on them."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from ml_elements import AUC, AVG_PRECISION, BRIER, LOGLOSS, N_ITER, Metric, make_smooth_ap

Y = np.array([0, 0, 1, 1])
P = np.array([0.1, 0.4, 0.35, 0.8])


def test_auc_matches_hand_count_of_ordered_pairs():
    # Pairs (pos, neg): (0.35,0.1)✓ (0.35,0.4)✗ (0.8,0.1)✓ (0.8,0.4)✓ → 3/4
    assert AUC.score(Y, P) == pytest.approx(0.75)


def test_average_precision_matches_hand_value():
    # Ranked: 0.8(+) 0.4(-) 0.35(+) 0.1(-). Precision at each positive: 1/1, 2/3.
    assert AVG_PRECISION.score(Y, P) == pytest.approx((1 + 2 / 3) / 2)


def test_logloss_and_brier_match_formulas():
    expected_ll = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    assert LOGLOSS.score(Y, P) == pytest.approx(expected_ll)
    assert BRIER.score(Y, P) == pytest.approx(np.mean((P - Y) ** 2))


def test_score_clips_probabilities_so_logloss_stays_finite():
    assert math.isfinite(LOGLOSS.score(np.array([0, 1]), np.array([1.0, 0.0])))


def test_improvement_is_positive_when_challenger_is_better_in_either_direction():
    assert AUC.improvement(0.8, 0.7) == pytest.approx(0.1)
    assert LOGLOSS.improvement(0.2, 0.3) == pytest.approx(0.1)
    assert LOGLOSS.improvement(0.3, 0.2) == pytest.approx(-0.1)


def test_smooth_ap_recovers_standard_ap_as_prior_strength_vanishes():
    smooth = make_smooth_ap(prior_mean=0.1, prior_strength=1e-9)
    assert smooth.score(Y, P) == pytest.approx(AVG_PRECISION.score(Y, P), abs=1e-6)


def test_smooth_ap_is_nan_without_positives():
    assert math.isnan(make_smooth_ap().score(np.zeros(4), P))


def test_metric_arithmetic_combines_prediction_and_model_terms():
    model = SimpleNamespace(n_iter_=50)
    penalised = AVG_PRECISION - 0.01 * N_ITER
    assert penalised.score(Y, P, model=model) == pytest.approx(AVG_PRECISION.score(Y, P) - 0.5)
    assert penalised.direction == "higher"


def test_blend_of_two_metrics_is_the_weighted_sum():
    blend = 0.7 * AUC + 0.3 * AVG_PRECISION
    assert blend.score(Y, P) == pytest.approx(0.7 * 0.75 + 0.3 * (1 + 2 / 3) / 2)


def test_custom_metric_wraps_any_callable():
    top1 = Metric("top1_is_positive", "higher", lambda y, p: float(y[np.argmax(p)]))
    assert top1.score(Y, P) == 1.0
