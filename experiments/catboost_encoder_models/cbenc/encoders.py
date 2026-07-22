"""
encoders.py
-----------
The single encoder under test: ``category_encoders.CatBoostEncoder`` — an
ordered, permutation-based target encoding that borrows CatBoost's internal
"Ordered TS" categorical-handling trick (each row's encoded value is a running
mean of the target over a random permutation of the *other* training rows, so
a row never sees its own label — leakage-safe by construction) but is
implemented entirely outside CatBoost, as a plain sklearn-compatible
``fit(X, y)``/``transform(X)`` transformer any model can consume.

CatBoost's own **native** categorical handling (:func:`to_catboost_native_X`)
is a separate, parallel path that bypasses this encoder entirely — the point
of the experiment is to compare the two.
"""

from __future__ import annotations

import time

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data import CAT_FEATURES, FEATURES

CAT_IDX = [FEATURES.index(c) for c in CAT_FEATURES]  # for CatBoost native mode


def build_preprocessor(cfg: Config) -> Pipeline:
    """
    ``CatBoostEncoder`` (categorical columns -> ordered target statistics) followed
    by ``StandardScaler`` over the resulting all-numeric matrix. A plain 2-step
    ``Pipeline`` suffices here (unlike ``encoder_comparison``'s ``ColumnTransformer``)
    because ``CatBoostEncoder`` already passes numeric columns through unchanged.
    """
    encoder = ce.CatBoostEncoder(
        cols=CAT_FEATURES,
        handle_unknown="value",
        handle_missing="value",
        random_state=cfg.seed,
        sigma=cfg.encoder.sigma,
        a=cfg.encoder.a,
    )
    return Pipeline([("cbenc", encoder), ("scale", StandardScaler())])


def fit_preprocessor_timed(cfg: Config, train_under: pd.DataFrame) -> tuple[Pipeline, np.ndarray, float]:
    """
    Fit the preprocessor on ``train_under`` (``y`` forwarded so ``CatBoostEncoder``
    can compute its ordered target statistics). Returns ``(fitted_preprocessor,
    X_train_encoded, fit_time_seconds)``.
    """
    pre = build_preprocessor(cfg)
    t0 = time.perf_counter()
    X_enc = pre.fit_transform(train_under[FEATURES], train_under["y"].to_numpy())
    fit_time = time.perf_counter() - t0
    return pre, np.asarray(X_enc, dtype=np.float64), fit_time


def transform_timed(pre: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Transform (val/test) through a fitted preprocessor, timed."""
    t0 = time.perf_counter()
    out = pre.transform(X[FEATURES])
    return np.asarray(out, dtype=np.float64), time.perf_counter() - t0


def to_catboost_native_X(X: pd.DataFrame) -> pd.DataFrame:
    """Raw-categorical frame for CatBoost's native mode (bypasses the encoder)."""
    out = X[FEATURES].copy()
    for c in CAT_FEATURES:
        out[c] = out[c].astype(str)
    return out
