"""
imbcls
------
Modular package for the imbalanced binary classification experiment.

Compares models for rare-positive (~0.1%) classification not only on predictive
quality (AP / AUC) but on score smoothness / entropy, tie rate, training speed
and ranking usefulness. Trains on a 10%-positive undersample, evaluates under
the real base rate, studies five prior mechanisms, and visualises learned
representations with UMAP.

Run via the experiment entry point::

    cd experiments/imbalanced_classification
    uv run --extra catboost --extra umap --extra viz python run_experiment.py
"""

from __future__ import annotations

from .config import Config, load_config

__all__ = ["Config", "load_config"]
