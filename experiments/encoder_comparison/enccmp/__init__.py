"""
enccmp
------
Encoder-comparison experiment: data -> encoder_i -> model_j training -> evaluation.

Isolates categorical-encoding choice as its own variable on the same
rare-positive synthetic dataset and training discipline used by the sibling
``imbalanced_classification`` experiment (10%-positive undersample for
training, evaluation at the true ~0.1% base rate).

Run via the experiment entry point::

    cd experiments/encoder_comparison
    uv run --extra catboost python run_experiment.py
"""

from __future__ import annotations

from .config import Config, load_config

__all__ = ["Config", "load_config"]
