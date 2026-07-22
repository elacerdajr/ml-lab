"""
models.py
---------
The model_i registry: five models on the CatBoost-encoded feature matrix
(``logistic``, ``rbf_svm``, ``random_forest``, ``mlp``, ``catboost_encoded``),
plus ``catboost_native`` — CatBoost's own categorical handling, fit and
predicted on raw categorical columns rather than the encoded matrix.

Preprocessing lives entirely in :mod:`encoders` and is fit **once**; these
model builders take the resulting numeric matrix directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import Config

ENCODED_MODEL_NAMES = ["logistic", "rbf_svm", "random_forest", "mlp", "catboost_encoded"]


@dataclass
class ModelSpec:
    name: str
    make: Callable[[], Any]
    max_train_n: int | None = None


def _catboost_estimator(cfg: Config, seed: int):
    from catboost import CatBoostClassifier

    p = cfg.models["catboost"]
    return CatBoostClassifier(
        iterations=p["iterations"],
        depth=p["depth"],
        learning_rate=p["learning_rate"],
        l2_leaf_reg=p["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def build_registry(cfg: Config) -> dict[str, ModelSpec]:
    """Build the five encoded-input model specs (``catboost_native`` is separate)."""
    seed = cfg.seed
    reg: dict[str, ModelSpec] = {}

    reg["logistic"] = ModelSpec(
        "logistic",
        lambda: LogisticRegression(max_iter=cfg.models["logistic"]["max_iter"]),
    )

    svm_cfg = cfg.models["rbf_svm"]
    reg["rbf_svm"] = ModelSpec(
        "rbf_svm",
        lambda: CalibratedClassifierCV(
            SVC(kernel="rbf", gamma=svm_cfg["gamma"], C=svm_cfg["C"], probability=False),
            method="sigmoid",
            cv=3,
        ),
        max_train_n=svm_cfg["max_train_n"],
    )

    rf_cfg = cfg.models["random_forest"]
    reg["random_forest"] = ModelSpec(
        "random_forest",
        lambda: RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            random_state=seed,
        ),
    )

    mlp_cfg = cfg.models["mlp"]
    reg["mlp"] = ModelSpec(
        "mlp",
        lambda: MLPClassifier(
            hidden_layer_sizes=tuple(mlp_cfg["hidden_layer_sizes"]),
            alpha=mlp_cfg["alpha"],
            learning_rate_init=mlp_cfg["learning_rate_init"],
            max_iter=mlp_cfg["max_iter"],
            early_stopping=True,
            random_state=seed,
        ),
    )

    reg["catboost_encoded"] = ModelSpec("catboost_encoded", lambda: _catboost_estimator(cfg, seed))
    return reg


def build_catboost_native(cfg: Config):
    """CatBoost estimator for the native-categorical path (same hyperparameters)."""
    return _catboost_estimator(cfg, cfg.seed)


def fit_timed(spec: ModelSpec, X: np.ndarray, y: np.ndarray) -> tuple[Any, float]:
    """Fit an encoded-input model. Returns ``(fitted_model, train_time_seconds)``."""
    t0 = time.perf_counter()
    model = spec.make()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def predict_timed(model, X) -> tuple[np.ndarray, float]:
    """Positive-class probability + prediction time (seconds)."""
    t0 = time.perf_counter()
    p = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    return p, time.perf_counter() - t0
