"""
calibration.py
--------------
Probability calibration for the margin-based models (Linear SVM, RBF SVM).

``LinearSVC`` / ``SVC(probability=False)`` expose only ``decision_function``, so
:class:`~sklearn.calibration.CalibratedClassifierCV` (Platt sigmoid, internal CV)
turns their margins into probabilities. Using ``cv`` — not ``prefit`` — avoids a
manual holdout split.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def wrap_sigmoid(estimator, cv: int = 3) -> CalibratedClassifierCV:
    """Wrap ``estimator`` in Platt sigmoid calibration with internal ``cv``-fold CV."""
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=cv)


def predict_prob(model, X) -> np.ndarray:
    """
    Positive-class score for a fitted model.

    Uses ``predict_proba[:, 1]`` for classifiers; falls back to a clipped
    ``predict`` for regressor-based soft models.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.0, 1.0)
