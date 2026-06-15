"""
rules.py
--------
Canonical rule schema and model-agnostic extractor.

A *rule* is a conjunction of feature conditions that maps to a predicted
probability, plus support, precision, and importance metadata computed from
the training set.  The schema is model-independent: the same JSON object
describes a tree path, a RuleFit rule, or a greedy rule-list step.

Classes
-------
RuleCondition   Single feature comparison  (feature, op, value).
Rule            A conjunction of conditions + metadata.
RuleSet         Collection of rules from one fitted model.
RuleExtractor   Adapter: extracts RuleSets from trained sklearn / imodels objects.

Supported backends
------------------
- sklearn.tree.DecisionTreeClassifier   (via ``tree_`` attribute)
- imodels.RuleFitClassifier             (via ``rules_`` list of Rule objects)
- imodels.GreedyRuleListClassifier      (via ``rules_`` list of dicts)

Examples
--------
>>> from sklearn.tree import DecisionTreeClassifier
>>> from ml_elements.rules import RuleExtractor
>>> model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
>>> rs = RuleExtractor().from_model(model, X_train, y_train, feature_names)
>>> print(rs.to_json(indent=2))
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ─── Schema ─────────────────────────────────────────────────────────────────


@dataclass
class RuleCondition:
    """Single feature comparison used inside a Rule."""

    feature: str
    op: str          # "<=", ">", ">=", "<", "=="
    value: float

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.value:.4g}"


@dataclass
class Rule:
    """A conjunction of RuleConditions with training-set metadata."""

    rule_id: int
    conditions: list[RuleCondition]
    prediction: float   # predicted P(positive) for covered samples
    support: int        # number of training samples covered
    precision: float    # fraction of covered samples that are positive
    importance: float   # normalized weight (0–1)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            self.description = " AND ".join(str(c) for c in self.conditions) or "(default)"


@dataclass
class RuleSet:
    """All rules extracted from one fitted model."""

    model_name: str
    model_type: str
    feature_names: list[str]
    rules: list[Rule]
    n_train: int
    positive_rate: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "n_train": self.n_train,
            "positive_rate": round(self.positive_rate, 4),
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "conditions": [
                        {"feature": c.feature, "op": c.op, "value": round(float(c.value), 4)}
                        for c in r.conditions
                    ],
                    "prediction": round(float(r.prediction), 4),
                    "support": int(r.support),
                    "precision": round(float(r.precision), 4),
                    "importance": round(float(r.importance), 4),
                    "description": r.description,
                }
                for r in self.rules
            ],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def save(self, path: str | Any) -> None:
        from pathlib import Path
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")


# ─── Extractor ──────────────────────────────────────────────────────────────


class RuleExtractor:
    """
    Model-agnostic adapter that produces a :class:`RuleSet` from any supported
    fitted estimator.

    Parameters
    ----------
    max_rules : int
        Maximum number of rules to keep (sorted by importance descending).
    """

    def __init__(self, max_rules: int = 50) -> None:
        self.max_rules = max_rules

    # ── Public entry point ─────────────────────────────────────────────────

    def from_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        model_name: str = "model",
    ) -> RuleSet:
        """
        Dispatch to the right extractor based on model type.

        Parameters
        ----------
        model : fitted estimator
            A trained sklearn or imodels classifier.
        X_train : np.ndarray, shape (n, d)
            Training features used to compute support and precision.
        y_train : np.ndarray, shape (n,)
            Binary training labels (0/1).
        feature_names : list[str]
            Column names in order, matching X_train columns.
        model_name : str
            Label shown in visualisations.

        Returns
        -------
        RuleSet
        """
        if hasattr(model, "tree_") and hasattr(model.tree_, "threshold"):
            return self._from_sklearn_tree(model, X_train, y_train, feature_names, model_name)

        if hasattr(model, "rules_"):
            rules_ = model.rules_
            if rules_ and isinstance(rules_[0], dict):
                return self._from_greedy_rule_list(model, X_train, y_train, feature_names, model_name)
            if rules_:
                return self._from_rulefit(model, X_train, y_train, feature_names, model_name)

        raise ValueError(
            f"Unsupported model type: {type(model).__name__}. "
            "Supported: DecisionTreeClassifier, imodels.RuleFitClassifier, "
            "imodels.GreedyRuleListClassifier."
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_conditions(
        conditions: list[RuleCondition],
        X: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """Boolean mask of rows satisfying all conditions (True = covered)."""
        mask = np.ones(len(X), dtype=bool)
        for cond in conditions:
            if cond.feature not in feature_names:
                continue
            idx = feature_names.index(cond.feature)
            col = X[:, idx]
            if cond.op == "<=":
                mask &= col <= cond.value
            elif cond.op == ">":
                mask &= col > cond.value
            elif cond.op == ">=":
                mask &= col >= cond.value
            elif cond.op == "<":
                mask &= col < cond.value
            elif cond.op == "==":
                mask &= col == cond.value
        return mask

    @staticmethod
    def _normalize_importance(rules: list[Rule]) -> None:
        max_imp = max((r.importance for r in rules), default=1.0)
        if max_imp > 0:
            for r in rules:
                r.importance = r.importance / max_imp

    def _trim_and_reindex(self, rules: list[Rule]) -> list[Rule]:
        rules = [r for r in rules if r.conditions]
        rules.sort(key=lambda r: r.importance, reverse=True)
        rules = rules[: self.max_rules]
        for i, r in enumerate(rules):
            r.rule_id = i + 1
        return rules

    # ── sklearn DecisionTree ───────────────────────────────────────────────

    def _from_sklearn_tree(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        model_name: str,
    ) -> RuleSet:
        tree = model.tree_
        n = len(y)
        pos_rate = float(y.mean())
        LEAF = -1
        rules: list[Rule] = []

        def traverse(node: int, conditions: list[RuleCondition]) -> None:
            if tree.children_left[node] == LEAF:
                counts = tree.value[node][0]
                support = int(tree.n_node_samples[node])
                total = float(counts.sum())
                precision = float(counts[1] / total) if total > 0 and len(counts) >= 2 else pos_rate
                importance = support / n * abs(precision - pos_rate)
                rules.append(Rule(
                    rule_id=0,
                    conditions=list(conditions),
                    prediction=precision,
                    support=support,
                    precision=precision,
                    importance=importance,
                ))
                return

            feat_idx = tree.feature[node]
            thresh = float(tree.threshold[node])
            fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"x{feat_idx}"

            traverse(
                tree.children_left[node],
                conditions + [RuleCondition(fname, "<=", thresh)],
            )
            traverse(
                tree.children_right[node],
                conditions + [RuleCondition(fname, ">", thresh)],
            )

        traverse(0, [])
        self._normalize_importance(rules)
        return RuleSet(
            model_name=model_name,
            model_type="DecisionTree",
            feature_names=feature_names,
            rules=self._trim_and_reindex(rules),
            n_train=n,
            positive_rate=pos_rate,
        )

    # ── imodels RuleFit ────────────────────────────────────────────────────

    def _from_rulefit(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        model_name: str,
    ) -> RuleSet:
        n = len(y)
        pos_rate = float(y.mean())
        rules: list[Rule] = []

        for rule_obj in model.rules_:
            agg = getattr(rule_obj, "agg_dict", {})
            if not agg:
                continue

            conditions = [
                RuleCondition(feature=fname, op=op, value=float(val))
                for (fname, op), val in agg.items()
            ]

            mask = self._apply_conditions(conditions, X, feature_names)
            support = int(mask.sum())
            precision = float(y[mask].mean()) if support > 0 else pos_rate
            coef = float(rule_obj.args[0]) if getattr(rule_obj, "args", None) else 0.0
            importance = abs(coef)

            rules.append(Rule(
                rule_id=0,
                conditions=conditions,
                prediction=precision,
                support=support,
                precision=precision,
                importance=importance,
            ))

        self._normalize_importance(rules)
        return RuleSet(
            model_name=model_name,
            model_type="RuleFit",
            feature_names=feature_names,
            rules=self._trim_and_reindex(rules),
            n_train=n,
            positive_rate=pos_rate,
        )

    # ── imodels GreedyRuleList ─────────────────────────────────────────────

    def _from_greedy_rule_list(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        model_name: str,
    ) -> RuleSet:
        n = len(y)
        pos_rate = float(y.mean())
        rules: list[Rule] = []
        remaining = np.ones(n, dtype=bool)

        for i, rd in enumerate(model.rules_):
            if "col" not in rd:
                # Default prediction for unmatched samples — no condition, skip
                continue
            col_name: str = rd["col"]
            cutoff: float = float(rd["cutoff"])
            flip: bool = bool(rd["flip"])
            val_right: float = float(rd["val_right"])

            op = ">" if flip else "<="
            cond = RuleCondition(feature=col_name, op=op, value=cutoff)

            col_idx = feature_names.index(col_name) if col_name in feature_names else None
            if col_idx is not None:
                col = X[:, col_idx]
                cond_mask = (col > cutoff) if flip else (col <= cutoff)
                fired = remaining & cond_mask
                remaining &= ~cond_mask
            else:
                fired = remaining.copy()

            support = int(fired.sum())
            precision = float(y[fired].mean()) if support > 0 else val_right
            importance = support / n * abs(precision - pos_rate)

            rules.append(Rule(
                rule_id=i + 1,
                conditions=[cond],
                prediction=val_right,
                support=support,
                precision=precision,
                importance=importance,
            ))

        self._normalize_importance(rules)
        # GRL preserves order (sequential rule list) — don't re-sort
        for i, r in enumerate(rules):
            r.rule_id = i + 1
        return RuleSet(
            model_name=model_name,
            model_type="GreedyRuleList",
            feature_names=feature_names,
            rules=rules[: self.max_rules],
            n_train=n,
            positive_rate=pos_rate,
        )
