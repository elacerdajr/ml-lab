"""
models.py
---------
Model registry, preprocessing and the soft-label fitting dispatch.

Every model exposes a positive-class probability after fitting. sklearn models
sit behind a shared ``ColumnTransformer`` (one-hot categoricals + scaled
numericals); CatBoost uses native categorical handling on the raw frame.

Soft-label support (priors 2 & 3) follows a deliberate matrix:

============================  ============  =====================================
model                         soft_mode     mechanism
============================  ============  =====================================
logistic (plain)              row_weight    exact log-loss reduction
rff_logistic                  row_weight    exact log-loss reduction (on LR step)
catboost_conservative/aggr.   native        ``loss_function="CrossEntropy"``
logistic_balanced             None          class_weight already reweights
linear_svm / rbf_svm          None          hinge loss has no soft-label meaning
mlp                           regressor*    optional MLPRegressor (config flag)
gaussian_process              None          classification only
============================  ============  =====================================

``*`` off by default (``models.mlp.soft_label_regressor``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

from .calibration import predict_prob, wrap_sigmoid
from .config import Config
from .data import CAT_FEATURES, FEATURES, NUM_FEATURES
from .priors import to_weighted_rows

CAT_IDX = [FEATURES.index(c) for c in CAT_FEATURES]  # [0, 1] for CatBoost


# ── Preprocessing ────────────────────────────────────────────────────────────

def build_preprocessor(train_full: pd.DataFrame) -> ColumnTransformer:
    """
    One-hot categoricals + standard-scaled numericals, with categories pinned to
    the ``train_full`` domain so column layout is stable across splits and the
    undersample (which may miss rare ``cat_2`` levels). Dense output for RFF/MLP.
    """
    categories = [list(train_full[c].cat.categories) for c in CAT_FEATURES]
    ohe = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("cat", ohe, CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )


def to_catboost_X(X: pd.DataFrame) -> pd.DataFrame:
    """Frame for CatBoost: feature columns only, categoricals as strings."""
    out = X[FEATURES].copy()
    for c in CAT_FEATURES:
        out[c] = out[c].astype(str)
    return out


# ── Model spec ───────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    make_hard: Callable[[], Any]
    soft_mode: str | None = None          # None | 'row_weight' | 'native' | 'regressor'
    make_soft: Callable[[], Any] | None = None
    is_catboost: bool = False
    max_train_n: int | None = None

    @property
    def supports_soft(self) -> bool:
        return self.soft_mode is not None


# ── Builders ─────────────────────────────────────────────────────────────────

def _logistic_pipe(pre: ColumnTransformer, cfg: Config, **kw) -> Pipeline:
    return Pipeline(
        [("pre", clone(pre)), ("clf", LogisticRegression(max_iter=cfg.models["logistic"]["max_iter"], **kw))]
    )


def _rff_pipe(pre: ColumnTransformer, cfg: Config, gamma: float, n_components: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("pre", clone(pre)),
            ("rff", RBFSampler(gamma=gamma, n_components=n_components, random_state=seed)),
            ("clf", LogisticRegression(max_iter=cfg.models["rff"]["max_iter"])),
        ]
    )


def _catboost(cfg: Config, key: str, *, loss: str, seed: int):
    from catboost import CatBoostClassifier

    p = cfg.models[key]
    return CatBoostClassifier(
        iterations=p["iterations"],
        depth=p["depth"],
        learning_rate=p["learning_rate"],
        l2_leaf_reg=p["l2_leaf_reg"],
        loss_function=loss,
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def make_rff_spec(pre: ColumnTransformer, cfg: Config, gamma: float, n_components: int) -> ModelSpec:
    """Build the RFF+logistic spec for a chosen (gamma, n_components)."""
    seed = cfg.seed
    return ModelSpec(
        name="rff_logistic",
        make_hard=lambda: _rff_pipe(pre, cfg, gamma, n_components, seed),
        soft_mode="row_weight",
    )


def build_registry(pre: ColumnTransformer, cfg: Config, train_under_n: int) -> dict[str, ModelSpec]:
    """
    Build every model spec except RFF (whose hyperparameters are searched first
    in the runner). Expensive models are omitted when their size guard trips.
    """
    seed = cfg.seed
    reg: dict[str, ModelSpec] = {}

    reg["logistic"] = ModelSpec(
        name="logistic",
        make_hard=lambda: _logistic_pipe(pre, cfg),
        soft_mode="row_weight",
    )
    reg["logistic_balanced"] = ModelSpec(
        name="logistic_balanced",
        make_hard=lambda: _logistic_pipe(pre, cfg, class_weight="balanced"),
        soft_mode=None,
    )
    reg["linear_svm"] = ModelSpec(
        name="linear_svm",
        make_hard=lambda: wrap_sigmoid(
            Pipeline([("pre", clone(pre)), ("clf", LinearSVC(max_iter=cfg.models["linear_svm"]["max_iter"]))])
        ),
        soft_mode=None,
    )
    reg["catboost_conservative"] = ModelSpec(
        name="catboost_conservative",
        make_hard=lambda: _catboost(cfg, "catboost_conservative", loss="Logloss", seed=seed),
        soft_mode="native",
        make_soft=lambda: _catboost(cfg, "catboost_conservative", loss="CrossEntropy", seed=seed),
        is_catboost=True,
    )
    reg["catboost_aggressive"] = ModelSpec(
        name="catboost_aggressive",
        make_hard=lambda: _catboost(cfg, "catboost_aggressive", loss="Logloss", seed=seed),
        soft_mode="native",
        make_soft=lambda: _catboost(cfg, "catboost_aggressive", loss="CrossEntropy", seed=seed),
        is_catboost=True,
    )

    # MLP — soft path optional (MLPRegressor), off by default.
    mlp_cfg = cfg.models["mlp"]
    soft_mode = "regressor" if mlp_cfg.get("soft_label_regressor", False) else None

    def _mlp_hard() -> Pipeline:
        return Pipeline(
            [
                ("pre", clone(pre)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(mlp_cfg["hidden_layer_sizes"]),
                        alpha=mlp_cfg["alpha"],
                        learning_rate_init=mlp_cfg["learning_rate_init"],
                        max_iter=mlp_cfg["max_iter"],
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def _mlp_soft() -> Pipeline:
        return Pipeline(
            [
                ("pre", clone(pre)),
                (
                    "reg",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(mlp_cfg["hidden_layer_sizes"]),
                        alpha=mlp_cfg["alpha"],
                        learning_rate_init=mlp_cfg["learning_rate_init"],
                        max_iter=mlp_cfg["max_iter"],
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )

    reg["mlp"] = ModelSpec(
        name="mlp",
        make_hard=_mlp_hard,
        soft_mode=soft_mode,
        make_soft=_mlp_soft if soft_mode else None,
    )

    # RBF SVM — only if the undersample is small enough.
    rbf_cfg = cfg.models["rbf_svm"]
    if train_under_n <= rbf_cfg["max_train_n"]:
        reg["rbf_svm"] = ModelSpec(
            name="rbf_svm",
            make_hard=lambda: wrap_sigmoid(
                Pipeline(
                    [
                        ("pre", clone(pre)),
                        ("clf", SVC(kernel="rbf", gamma=rbf_cfg["gamma"], C=rbf_cfg["C"], probability=False)),
                    ]
                )
            ),
            soft_mode=None,
            max_train_n=rbf_cfg["max_train_n"],
        )

    # Gaussian process — trained on a subsample only (guard applied in the runner).
    reg["gaussian_process"] = ModelSpec(
        name="gaussian_process",
        make_hard=lambda: Pipeline(
            [("pre", clone(pre)), ("clf", GaussianProcessClassifier(random_state=seed))]
        ),
        soft_mode=None,
        max_train_n=cfg.models["gaussian_process"]["max_train_n"],
    )
    return reg


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit_hard(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray) -> tuple[Any, float]:
    """Fit on hard 0/1 labels. Returns ``(fitted_model, train_time_seconds)``."""
    t0 = time.perf_counter()
    model = spec.make_hard()
    if spec.is_catboost:
        model.fit(to_catboost_X(X), np.asarray(y), cat_features=CAT_IDX)
    else:
        model.fit(X, np.asarray(y))
    return model, time.perf_counter() - t0


def fit_soft(spec: ModelSpec, X: pd.DataFrame, y_soft: np.ndarray) -> tuple[Any, float]:
    """
    Fit on soft labels in ``[0,1]`` via the model's ``soft_mode``. Returns
    ``(fitted_model, train_time_seconds)``.
    """
    t0 = time.perf_counter()
    if spec.soft_mode == "native":            # CatBoost CrossEntropy
        model = spec.make_soft()
        model.fit(to_catboost_X(X), np.asarray(y_soft, dtype=float), cat_features=CAT_IDX)
    elif spec.soft_mode == "row_weight":      # exact log-loss reduction
        X2, y2, w2 = to_weighted_rows(X, y_soft)
        model = spec.make_hard()
        model.fit(X2, y2, clf__sample_weight=w2)
    elif spec.soft_mode == "regressor":       # MLPRegressor approximation
        model = spec.make_soft()
        model.fit(X, np.clip(np.asarray(y_soft, dtype=float), 0.0, 1.0))
    else:
        raise ValueError(f"model {spec.name!r} does not support soft labels")
    return model, time.perf_counter() - t0


def predict_scores(model, X: pd.DataFrame, is_catboost: bool) -> tuple[np.ndarray, float]:
    """Positive-class probability + prediction time (seconds)."""
    t0 = time.perf_counter()
    Xin = to_catboost_X(X) if is_catboost else X
    p = predict_prob(model, Xin)
    return np.asarray(p, dtype=float), time.perf_counter() - t0
