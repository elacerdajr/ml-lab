"""
models.py
---------
Model factory functions.

Each factory returns a sklearn-compatible object that satisfies the
``ModelBackend`` protocol (fit / predict_proba). Pass a factory — not an
instance — wherever ``TrialRunner`` or ``Trial.model_factory`` expects one,
so a fresh model is created for every trial.

Functions
---------
make_logistic   Fast linear baseline. Matches the Gaussian log-odds structure.
make_hgb        Lightweight sklearn histogram gradient booster.
make_catboost   CatBoost gradient booster (requires ``pip install catboost``).
make_sklearn    Generic wrapper — pass any sklearn estimator class + kwargs.

Examples
--------
>>> runner = TrialRunner(setups=..., model_factory=make_logistic(), ...)

>>> # Per-trial override — catboost on one trial, logistic on all others:
>>> trial = Trial(..., model_factory=make_catboost(depth=6))

>>> # Generic wrapper for any sklearn estimator:
>>> from sklearn.ensemble import RandomForestClassifier
>>> factory = make_sklearn(RandomForestClassifier, n_estimators=100, random_state=0)
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .protocols import ModelBackend

_DEFAULT_RANDOM_STATE = 42


def make_logistic(
    random_state: int = _DEFAULT_RANDOM_STATE,
    max_iter: int = 300,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Factory for logistic regression.

    Fast and interpretable. Optimal for the Gaussian DGP where the Bayes
    decision boundary is linear.

    Parameters
    ----------
    random_state : int
        Reproducibility seed.
    max_iter : int
        Maximum solver iterations.
    **kwargs
        Forwarded to ``LogisticRegression``.

    Returns
    -------
    Callable[[], ModelBackend]
        Zero-argument factory that produces a fresh ``LogisticRegression``.
    """
    def factory() -> ModelBackend:
        return LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
            **kwargs,
        )

    factory.__name__ = "make_logistic"
    return factory


def make_hgb(
    iterations: int = 200,
    learning_rate: float = 0.06,
    max_leaf_nodes: int = 16,
    random_state: int = _DEFAULT_RANDOM_STATE,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Factory for sklearn's HistGradientBoostingClassifier.

    Good middle ground: non-linear, fast, no external dependency.

    Parameters
    ----------
    iterations : int
        Number of boosting rounds (``max_iter``).
    learning_rate : float
        Shrinkage applied to each tree.
    max_leaf_nodes : int
        Maximum leaves per tree (controls complexity).
    random_state : int
        Reproducibility seed.
    **kwargs
        Forwarded to ``HistGradientBoostingClassifier``.

    Returns
    -------
    Callable[[], ModelBackend]
        Zero-argument factory.
    """
    def factory() -> ModelBackend:
        return HistGradientBoostingClassifier(
            max_iter=iterations,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            validation_fraction=None,
            early_stopping=False,
            **kwargs,
        )

    factory.__name__ = "make_hgb"
    return factory


def make_catboost(
    iterations: int = 200,
    learning_rate: float = 0.06,
    depth: int = 4,
    random_state: int = _DEFAULT_RANDOM_STATE,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Factory for CatBoostClassifier.

    Requires ``pip install catboost``. Import is deferred so the rest of
    the package works without CatBoost installed.

    Parameters
    ----------
    iterations : int
        Number of boosting rounds.
    learning_rate : float
        Step size shrinkage.
    depth : int
        Maximum tree depth.
    random_state : int
        Reproducibility seed.
    **kwargs
        Forwarded to ``CatBoostClassifier``.

    Returns
    -------
    Callable[[], ModelBackend]
        Zero-argument factory.

    Raises
    ------
    ImportError
        If ``catboost`` is not installed.
    """
    def factory() -> ModelBackend:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "make_catboost requires catboost. Install with: pip install catboost"
            ) from exc

        return CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_state,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            **kwargs,
        )

    factory.__name__ = "make_catboost"
    return factory


def make_decision_tree(
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    random_state: int = _DEFAULT_RANDOM_STATE,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Factory for sklearn's ``DecisionTreeClassifier``.

    Used as a small, interpretable baseline against the more elaborate
    ``imodels`` estimators.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_samples_leaf : int
        Minimum number of samples per leaf.
    random_state : int
        Reproducibility seed.
    **kwargs
        Forwarded to ``DecisionTreeClassifier``.

    Returns
    -------
    Callable[[], ModelBackend]
    """
    def factory() -> ModelBackend:
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )

    factory.__name__ = "make_decision_tree"
    return factory


def make_figs(
    max_rules: int = 25,
    random_state: int = _DEFAULT_RANDOM_STATE,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Factory for ``imodels.FIGSClassifier`` (Fast Interpretable Greedy-tree Sums).

    Requires ``pip install imodels``. Import is deferred so the rest of the
    package works without ``imodels`` installed.

    Parameters
    ----------
    max_rules : int
        Upper bound on the total number of rules (leaves across all trees).
        Controls the capacity of the additive tree-sum model.
    random_state : int
        Reproducibility seed.
    **kwargs
        Forwarded to ``FIGSClassifier``.

    Returns
    -------
    Callable[[], ModelBackend]

    Raises
    ------
    ImportError
        If ``imodels`` is not installed.
    """
    def factory() -> ModelBackend:
        try:
            from imodels import FIGSClassifier
        except ImportError as exc:
            raise ImportError(
                "make_figs requires imodels. Install with: pip install imodels"
            ) from exc

        return FIGSClassifier(
            max_rules=max_rules,
            random_state=random_state,
            **kwargs,
        )

    factory.__name__ = "make_figs"
    return factory


def make_greedy_rule_list(
    max_depth: int = 8,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Factory for ``imodels.GreedyRuleListClassifier``.

    An ordered if-then-else list of thresholds learned greedily. Highly
    interpretable but structurally limited (single chain of rules).

    Requires ``pip install imodels``.

    Parameters
    ----------
    max_depth : int
        Maximum length of the rule list.
    **kwargs
        Forwarded to ``GreedyRuleListClassifier``.

    Returns
    -------
    Callable[[], ModelBackend]

    Raises
    ------
    ImportError
        If ``imodels`` is not installed.
    """
    def factory() -> ModelBackend:
        try:
            from imodels import GreedyRuleListClassifier
        except ImportError as exc:
            raise ImportError(
                "make_greedy_rule_list requires imodels. "
                "Install with: pip install imodels"
            ) from exc

        return GreedyRuleListClassifier(
            max_depth=max_depth,
            **kwargs,
        )

    factory.__name__ = "make_greedy_rule_list"
    return factory


def make_rule_fit(
    n_estimators: int = 40,
    tree_size: int = 4,
    max_rules: int = 40,
    random_state: int = _DEFAULT_RANDOM_STATE,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Factory for ``imodels.RuleFitClassifier``.

    Sparse linear ensemble of rules generated from trees. Slower than FIGS
    and the greedy rule list but produces an L1-regularised rule set.

    Requires ``pip install imodels``.

    Parameters
    ----------
    n_estimators : int
        Number of tree generators.
    tree_size : int
        Average number of leaves per tree used for rule generation.
    max_rules : int
        Approximate cap on extracted rules.
    random_state : int
        Reproducibility seed.
    **kwargs
        Forwarded to ``RuleFitClassifier``.

    Returns
    -------
    Callable[[], ModelBackend]

    Raises
    ------
    ImportError
        If ``imodels`` is not installed.
    """
    def factory() -> ModelBackend:
        try:
            from imodels import RuleFitClassifier
        except ImportError as exc:
            raise ImportError(
                "make_rule_fit requires imodels. Install with: pip install imodels"
            ) from exc

        return RuleFitClassifier(
            n_estimators=n_estimators,
            tree_size=tree_size,
            max_rules=max_rules,
            random_state=random_state,
            **kwargs,
        )

    factory.__name__ = "make_rule_fit"
    return factory


def make_sklearn(
    cls: type,
    **kwargs: Any,
) -> Callable[[], ModelBackend]:
    """
    Generic factory for any sklearn-compatible estimator class.

    Parameters
    ----------
    cls : type
        Estimator class (not an instance). Must implement ``fit`` and
        ``predict_proba``.
    **kwargs
        Constructor arguments forwarded to ``cls``.

    Returns
    -------
    Callable[[], ModelBackend]
        Zero-argument factory.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> factory = make_sklearn(RandomForestClassifier, n_estimators=100, random_state=0)
    >>> model = factory()
    """
    def factory() -> ModelBackend:
        return cls(**kwargs)

    factory.__name__ = f"make_sklearn({cls.__name__})"
    return factory
