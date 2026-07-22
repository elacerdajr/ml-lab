"""
encoders.py
-----------
The heart of this experiment: five ways to turn ``cat_1``/``cat_2`` into numeric
features, plus CatBoost's native categorical handling as a sixth, model-specific
path that bypasses encoding entirely.

============  ===========================================================
encoder       mechanism
============  ===========================================================
onehot        OneHotEncoder, categories pinned to ``train_full``'s domain.
ordinal       OrdinalEncoder, same pinning; unseen -> -1.
frequency     category -> its ``train_under`` frequency (count / N); unseen -> 0.
target        sklearn TargetEncoder — cross-fitted mean target per category
              (leakage-safe: ``fit_transform`` internally cross-fits with
              ``cv`` folds; plain ``transform`` on held-out data).
hashing       FeatureHasher per categorical column — fixed-width, handles
              unseen categories natively, no vocabulary to maintain.
============  ===========================================================

All five feed a shared ``ColumnTransformer`` (dense output) so every
sklearn-facing model in this experiment sees a plain numeric matrix. CatBoost's
native mode is a parallel path (:func:`to_catboost_native_X` + ``cat_features``)
that never goes through a ``ColumnTransformer`` at all.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from .config import Config
from .data import CAT_FEATURES, FEATURES, NUM_FEATURES

CAT_IDX = [FEATURES.index(c) for c in CAT_FEATURES]  # for CatBoost native mode

ENCODER_NAMES = ["onehot", "ordinal", "frequency", "target", "hashing"]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Map each category to its empirical frequency in the fitted data; unseen -> 0."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1]))
        n = len(X)
        self.freq_maps_ = []
        for col in self.columns_:
            vals = X[col] if hasattr(X, "columns") else X[:, col]
            counts = pd.Series(vals).value_counts()
            self.freq_maps_.append((counts / n).to_dict())
        return self

    def transform(self, X) -> np.ndarray:
        out = np.zeros((len(X), len(self.columns_)), dtype=np.float64)
        for i, col in enumerate(self.columns_):
            vals = X[col] if hasattr(X, "columns") else X[:, col]
            out[:, i] = pd.Series(vals).map(self.freq_maps_[i]).fillna(0.0).to_numpy()
        return out


class HashingColumnEncoder(BaseEstimator, TransformerMixin):
    """Independently hash each categorical column into ``n_features`` dims (dense)."""

    def __init__(self, n_features: int = 32):
        self.n_features = n_features

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1]))
        self.hashers_ = {
            col: FeatureHasher(n_features=self.n_features, input_type="string")
            for col in self.columns_
        }
        return self

    def transform(self, X) -> np.ndarray:
        blocks = []
        for col in self.columns_:
            vals = X[col] if hasattr(X, "columns") else X[:, col]
            raw = ([f"{col}={v}"] for v in vals)
            blocks.append(self.hashers_[col].transform(raw).toarray())
        return np.hstack(blocks)


def _cat_encoder(encoder_name: str, train_full: pd.DataFrame, cfg: Config):
    """Build the categorical-column transformer for one encoder."""
    if encoder_name == "onehot":
        categories = [list(train_full[c].cat.categories) for c in CAT_FEATURES]
        return OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)
    if encoder_name == "ordinal":
        categories = [list(train_full[c].cat.categories) for c in CAT_FEATURES]
        return OrdinalEncoder(
            categories=categories, handle_unknown="use_encoded_value", unknown_value=-1
        )
    if encoder_name == "frequency":
        return FrequencyEncoder()
    if encoder_name == "target":
        # fit_transform cross-fits internally (cv folds) so the *training* matrix
        # is leakage-safe; transform() on val/test uses the full-data fit.
        return TargetEncoder(
            target_type="binary", smooth=cfg.encoders.target_smooth, cv=cfg.encoders.target_cv
        )
    if encoder_name == "hashing":
        return HashingColumnEncoder(n_features=cfg.encoders.hashing_n_features)
    raise ValueError(f"unknown encoder {encoder_name!r}; have {ENCODER_NAMES}")


def build_preprocessor(encoder_name: str, train_full: pd.DataFrame, cfg: Config) -> ColumnTransformer:
    """
    Build the ``ColumnTransformer`` for one encoder: categorical columns through
    ``encoder_name``'s transformer, numerical columns standard-scaled. Categories
    for ``onehot``/``ordinal`` are pinned to the ``train_full`` domain so column
    layout is stable even though the transformer is *fit* on ``train_under``
    (whose negative subsample can miss rare ``cat_2`` levels).
    """
    cat_enc = _cat_encoder(encoder_name, train_full, cfg)
    return ColumnTransformer(
        [("cat", cat_enc, CAT_FEATURES), ("num", StandardScaler(), NUM_FEATURES)],
        sparse_threshold=0,
    )


def fit_preprocessor_timed(
    encoder_name: str, train_full: pd.DataFrame, train_under: pd.DataFrame, cfg: Config
) -> tuple[ColumnTransformer, np.ndarray, float]:
    """
    Build and fit the preprocessor on ``train_under`` (with ``y`` forwarded so
    the target encoder can cross-fit). Returns ``(fitted_preprocessor,
    X_train_encoded, fit_time_seconds)``.
    """
    pre = build_preprocessor(encoder_name, train_full, cfg)
    t0 = time.perf_counter()
    X_enc = pre.fit_transform(train_under[FEATURES], train_under["y"].to_numpy())
    fit_time = time.perf_counter() - t0
    return pre, np.asarray(X_enc, dtype=np.float64), fit_time


def transform_timed(pre: ColumnTransformer, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Transform (val/test) through a fitted preprocessor, timed."""
    t0 = time.perf_counter()
    out = pre.transform(X[FEATURES])
    return np.asarray(out, dtype=np.float64), time.perf_counter() - t0


def to_catboost_native_X(X: pd.DataFrame) -> pd.DataFrame:
    """Raw-categorical frame for CatBoost's native mode (bypasses encoding)."""
    out = X[FEATURES].copy()
    for c in CAT_FEATURES:
        out[c] = out[c].astype(str)
    return out
