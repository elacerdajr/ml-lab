"""
cbenc
-----
CatBoost-style ordered target encoding, compared across models.

Uses ``category_encoders.CatBoostEncoder`` (an external, permutation-based
target encoding inspired by CatBoost's own "Ordered TS" categorical handling)
as the single encoder, crossed with logistic regression, RBF SVM, random
forest, MLP, and CatBoost itself — run twice, once fed the encoded features
like every other model (``catboost_encoded``) and once via its own native
categorical handling (``catboost_native``), to test whether the external
encoding matches, beats, or underperforms CatBoost's internal algorithm.

Same data-generating process and undersampling discipline as
``experiments/imbalanced_classification`` and ``experiments/encoder_comparison``.

Run via the experiment entry point::

    cd experiments/catboost_encoder_models
    uv run --extra catboost --extra category_encoders python run_experiment.py
"""

from __future__ import annotations

from .config import Config, load_config

__all__ = ["Config", "load_config"]
