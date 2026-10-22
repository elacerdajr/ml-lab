"""Model factories and leaf embeddings."""

import numpy as np
import pytest

from ml_elements import GaussianPopulation, embed_leaves, make_hgb


def test_hgb_runs_a_fixed_number_of_rounds_unless_early_stopping_is_asked_for():
    df = GaussianPopulation(0.2, {"x1": 1.0}).sample(1000, seed=0)
    fixed = make_hgb(iterations=120)()
    fixed.fit(df[["x1"]], df["y"])
    assert fixed.n_iter_ == 120
    stopped = make_hgb(iterations=120, early_stopping=True, n_iter_no_change=3)()
    stopped.fit(df[["x1"]], df["y"])
    assert stopped.validation_fraction == 0.1 and stopped.n_iter_ < 120


def test_embed_leaves_returns_one_leaf_index_per_tree():
    catboost = pytest.importorskip("catboost")
    df = GaussianPopulation(0.3, {"x1": 1.0, "x2": 0.5}).sample(400, seed=0)
    X = df[["x1", "x2"]]
    model = catboost.CatBoostClassifier(iterations=12, depth=3, verbose=0, random_seed=0)
    model.fit(X, df["y"])

    leaves = embed_leaves(model, X)
    assert leaves.shape == (400, 12) and leaves.dtype == np.float32
    assert leaves.max() < 2**3
    assert embed_leaves(model, X, dtype=np.int32).dtype == np.int32

    one_hot = embed_leaves(model, X, one_hot=True)
    assert one_hot.shape[0] == 400
    np.testing.assert_array_equal(one_hot.sum(axis=1).A.ravel(), np.full(400, 12))


def test_imodels_factories_fit_the_same_model_whatever_the_global_random_state():
    pytest.importorskip("imodels")
    from ml_elements import make_figs, make_greedy_rule_list

    df = GaussianPopulation(0.2, {f"x{i}": 0.1 for i in range(30)}).sample(400, seed=1)
    cols = [c for c in df.columns if c != "y"]
    for factory in (make_figs(), make_greedy_rule_list()):
        predictions = []
        for global_seed in (1, 2):
            np.random.seed(global_seed)
            model = factory().fit(df[cols], df["y"])
            predictions.append(model.predict_proba(df[cols])[:, 1])
            # fit leaves the caller's global random state where it was
            np.random.seed(global_seed)
            expected = np.random.random()
            np.random.seed(global_seed)
            factory().fit(df[cols], df["y"])
            assert np.random.random() == expected
        np.testing.assert_array_equal(*predictions)
