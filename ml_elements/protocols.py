"""
protocols.py
------------
Structural interfaces for the ml_elements building blocks.

All blocks depend only on these protocols — never on each other's concrete
classes. This means any piece can be swapped without touching anything else.

Usage
-----
You never need to inherit from these explicitly. Any class that implements
the required methods satisfies the protocol (duck typing via typing.Protocol).

    class MyDGP:
        def sample(self, n: int, seed: int) -> pd.DataFrame: ...
        # MyDGP now satisfies DGP — no inheritance needed

"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class DGP(Protocol):
    """
    Data-generating process.

    Any object that produces a labelled DataFrame from (n, seed) satisfies
    this protocol — synthetic generators, real-data samplers, or wrappers.

    Methods
    -------
    sample(n, seed) : pd.DataFrame
        Draw n i.i.d. rows. The target column must be present. All other
        columns are treated as features. Using ``seed`` guarantees
        reproducibility without global state.
    """

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """
        Draw n rows from this data-generating process.

        Parameters
        ----------
        n : int
            Number of rows to sample.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with a target column and one or more feature columns.
        """
        ...


@runtime_checkable
class ModelBackend(Protocol):
    """
    A trainable binary classifier.

    Any sklearn-compatible estimator satisfies this protocol.
    The ``fit`` / ``predict_proba`` split maps directly onto
    ``TrialRunner``'s train-then-evaluate loop.

    Methods
    -------
    fit(X, y) : None
        Train in-place on feature matrix X and binary labels y.
    predict_proba(X) : np.ndarray, shape (n, 2)
        Return class probabilities. Column 1 is P(y=1).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model in-place.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix, shape (n_samples, n_features).
        y : pd.Series
            Binary labels (0 / 1), length n_samples.
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class-probability estimates.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Column 0 = P(y=0), Column 1 = P(y=1).
        """
        ...


@runtime_checkable
class MetricFn(Protocol):
    """
    A scoring function: (y_true, y_score) -> float.

    Used inside ``Metric`` dataclass. Any callable with this signature
    satisfies the protocol.
    """

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Compute a scalar score.

        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
            True binary labels (0 / 1).
        y_score : np.ndarray, shape (n,)
            Model scores or probabilities. Higher = more likely positive.

        Returns
        -------
        float
            Scalar metric value.
        """
        ...
