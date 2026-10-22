"""
objectives.py
-------------
Former home of ``Objective``, ``ModelMetric`` and ``make_objective``.

They are now one type, ``Metric`` (see ``metrics.py``): arithmetic on metrics
returns a metric, and model-based terms use ``Metric(..., on="model")``.
These names remain importable from here and from ``ml_elements``.
"""

from .metrics import COEF_L1, N_ITER, N_LEAVES, Metric, ModelMetric, Objective, make_objective

__all__ = ["COEF_L1", "N_ITER", "N_LEAVES", "Metric", "ModelMetric", "Objective", "make_objective"]
