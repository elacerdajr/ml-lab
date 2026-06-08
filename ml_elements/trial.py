"""
trial.py
--------
Core experiment primitives: ``DataBudget`` and ``Trial``.

DataBudget
    Describes how much data to use and where seeds live.
    Shared across all trials in a Study. Any trial can override with
    ``budget_override``.

Trial
    One experimental condition: a (name, value) label, a DGP, and
    optional per-trial overrides for model factory and data budget.
    A list of trials is the only input ``Study.run()`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .protocols import DGP, ModelBackend

_UNSET = object()


@dataclass(frozen=True)
class DataBudget:
    """
    Data sizing and seed configuration for one experiment.

    Parameters
    ----------
    n_train : int
        Training set size.
    n_valid : int
        Validation set size (available for early stopping, calibration, etc.).
    n_test : int
        Test set size per repeat.
    seed_train : int
        Base seed for the training split.
    seed_valid : int
        Base seed for the validation split.
    seed_test_base : int
        Starting seed for test repeats. Each repeat gets a unique seed derived
        from this base plus an offset.
    n_repeats : int
        Number of fresh test draws. More repeats → tighter uncertainty estimates
        on the improvement score.

    Examples
    --------
    >>> budget = DataBudget(
    ...     n_train=2_000, n_valid=500, n_test=2_000,
    ...     seed_train=101, seed_valid=202, seed_test_base=10_000,
    ...     n_repeats=20,
    ... )
    """

    n_train: int
    n_valid: int
    n_test: int
    seed_train: int
    seed_valid: int
    seed_test_base: int
    n_repeats: int

    def __post_init__(self) -> None:
        for attr in ("n_train", "n_valid", "n_test", "n_repeats"):
            if getattr(self, attr) < 1:
                raise ValueError(f"{attr} must be >= 1, got {getattr(self, attr)}")

    @classmethod
    def quick(cls, n: int = 500, n_repeats: int = 10) -> "DataBudget":
        """
        Convenience constructor for fast prototyping.

        Parameters
        ----------
        n : int
            Train and test size. Validation is set to ``n // 5`` (min 50).
        n_repeats : int
            Number of test repeats.

        Examples
        --------
        >>> budget = DataBudget.quick()           # 500 / 100 / 500, 10 repeats
        >>> budget = DataBudget.quick(n=2000, n_repeats=20)
        """
        return cls(
            n_train=n,
            n_valid=max(50, n // 5),
            n_test=n,
            seed_train=1,
            seed_valid=2,
            seed_test_base=10_000,
            n_repeats=n_repeats,
        )


@dataclass
class Trial:
    """
    One experimental condition.

    A trial defines *what* to vary: the data-generating process, the
    condition label, and optional per-trial overrides for the model
    factory and data budget.

    Parameters
    ----------
    name : str
        Name of the axis being varied, e.g. ``"p_pos"`` or ``"x3_info"``.
        Used as a column label in result DataFrames.
    value : float
        The specific value for this trial, e.g. ``0.05`` or ``1.25``.
    dgp : DGP
        Data-generating process for this trial.
    seed_offset : int
        Added to all budget seeds to ensure non-overlapping data across
        trials. Each trial should have a unique offset.
    model_factory : Callable[[], ModelBackend] | None
        If set, overrides the ``TrialRunner``'s default model factory for
        this trial only. Useful for per-trial hyperparameter variation.
    budget_override : DataBudget | None
        If set, overrides the ``TrialRunner``'s default ``DataBudget`` for
        this trial only. Useful for train-size sweep studies.

    Examples
    --------
    Standard trial — uses runner defaults:

    >>> trial = Trial(name="p_pos", value=0.05, dgp=my_dgp, seed_offset=10_001)

    Override model for this one trial:

    >>> trial = Trial(
    ...     name="lr_search",
    ...     value=0.01,
    ...     dgp=my_dgp,
    ...     seed_offset=20_001,
    ...     model_factory=make_catboost(learning_rate=0.01),
    ... )
    """

    name: str
    value: float
    dgp: DGP
    seed_offset: int
    model_factory: Callable[[], ModelBackend] | None = field(default=None)
    budget_override: DataBudget | None = field(default=None)
