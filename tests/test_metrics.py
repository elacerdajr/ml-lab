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


def test_n_iter_reads_logistic_regressions_array_valued_n_iter():
    # LogisticRegression stores n_iter_ as array([k]); float() of it fails on numpy >= 2.
    from ml_elements import GaussianPopulation, make_logistic

    df = GaussianPopulation(0.3, {"x1": 1.0}).sample(300, seed=0)
    model = make_logistic()()
    model.fit(df[["x1"]], df["y"])
    assert N_ITER.score(Y, P, model=model) == float(model.n_iter_[0])


def test_a_model_metric_explains_when_no_model_is_given():
    with pytest.raises(ValueError, match="pass model="):
        (AVG_PRECISION - 0.01 * N_ITER).score(Y, P)


def test_metric_scorer_drives_sklearn_early_stopping():
    from ml_elements import GaussianPopulation, make_hgb

    df = GaussianPopulation(0.2, {"x1": 1.0, "x2": 0.5}).sample(1500, seed=0)
    X, y = df[["x1", "x2"]], df["y"]
    plain = make_hgb(iterations=300)()
    plain.fit(X, y)
    stopped = make_hgb(iterations=300, early_stopping=True, n_iter_no_change=5,
                       scoring=(AVG_PRECISION - 0.001 * N_ITER).as_scorer())()
    stopped.fit(X, y)
    assert plain.n_iter_ == 300
    assert stopped.n_iter_ < 300


def test_lower_is_better_metrics_are_negated_as_sklearn_scorers():
    class Fixed:
        def predict_proba(self, X):
            return np.column_stack([1 - P, P])

    assert LOGLOSS.as_scorer()(Fixed(), None, Y) == pytest.approx(-LOGLOSS.score(Y, P))


def test_score_shape_metrics():
    from ml_elements import MAX_SCORE_GAP, NORMALIZED_SCORE_ENTROPY, SCORE_ENTROPY, TIE_RATE

    flat = np.linspace(0.005, 0.995, 100)        # two scores in each of 50 bins, no ties
    spiky = np.array([0.1] * 99 + [0.9])          # nearly all tied
    assert NORMALIZED_SCORE_ENTROPY.score(None, flat) == pytest.approx(1.0)
    assert SCORE_ENTROPY.score(None, flat) > SCORE_ENTROPY.score(None, spiky)
    assert TIE_RATE.score(None, spiky) == pytest.approx(0.98)
    assert MAX_SCORE_GAP.score(None, spiky) == pytest.approx(0.8)
    assert TIE_RATE.direction == "lower" and SCORE_ENTROPY.direction == "higher"


def test_former_objective_names_build_the_same_metric():
    from ml_elements import ModelMetric, Objective, make_objective

    trees = ModelMetric("n_iter", "lower", fn=lambda m: m.n_iter_)
    model = SimpleNamespace(n_iter_=10)
    assert (AVG_PRECISION - 0.1 * trees).score(Y, P, model=model) == pytest.approx(AVG_PRECISION.score(Y, P) - 1.0)
    assert Objective is Metric
    ratio = make_objective(lambda y, s, m: AVG_PRECISION.score(y, s) / (1 + m.n_iter_), "ap_per_iter")
    assert ratio.score(Y, P, model=model) == pytest.approx(AVG_PRECISION.score(Y, P) / 11)
