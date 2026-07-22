"""
models.py
---------
The model_j registry: four models that consume whatever numeric matrix
``encoder_i`` produced (``logistic``, ``rff_logistic``, ``mlp``,
``catboost_encoded``), plus ``catboost_native`` — CatBoost's own categorical
handling, independent of the encoder axis, built and fit separately.

Preprocessing lives entirely in :mod:`encoders` and is fit **once per encoder**;
these model builders take plain numeric arrays, so the same encoded matrix is
reused across all four encoded models (no redundant re-fitting of the encoder).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .config import Config

ENCODED_MODEL_NAMES = ["logistic", "rff_logistic", "mlp", "catboost_encoded"]


@dataclass
class ModelSpec:
    name: str
    make: Callable[[], Any]


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
    """Build the four encoded-input model specs (``catboost_native`` is separate)."""
    seed = cfg.seed
    reg: dict[str, ModelSpec] = {}

    reg["logistic"] = ModelSpec(
        "logistic",
        lambda: LogisticRegression(max_iter=cfg.models["logistic"]["max_iter"]),
    )
    reg["rff_logistic"] = ModelSpec(
        "rff_logistic",
        lambda: Pipeline(
            [
                (
                    "rff",
                    RBFSampler(
                        gamma=cfg.models["rff"]["gamma"],
                        n_components=cfg.models["rff"]["n_components"],
                        random_state=seed,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=cfg.models["rff"]["max_iter"])),
            ]
        ),
    )
    reg["mlp"] = ModelSpec(
        "mlp",
        lambda: MLPClassifier(
            hidden_layer_sizes=tuple(cfg.models["mlp"]["hidden_layer_sizes"]),
            alpha=cfg.models["mlp"]["alpha"],
            learning_rate_init=cfg.models["mlp"]["learning_rate_init"],
            max_iter=cfg.models["mlp"]["max_iter"],
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
