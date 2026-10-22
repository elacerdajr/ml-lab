"""
embeddings.py
-------------
Representations fed to UMAP.

- raw preprocessed features (one-hot cats + scaled nums) — the model-agnostic view
- CatBoost leaf-index one-hot (sparse) — the supervised tree view (cosine metric)
- RFF features — the random-Fourier view used by the RFF+logistic model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ml_elements.models import embed_leaves

from .data import FEATURES
from .models import to_catboost_X


def get_raw_preprocessed_embedding(preprocessor, X: pd.DataFrame) -> np.ndarray:
    """Dense one-hot + scaled feature matrix from a fitted preprocessor."""
    return np.asarray(preprocessor.transform(X[FEATURES]), dtype=np.float32)


def get_catboost_leaf_embedding(catboost_model, X: pd.DataFrame) -> sp.csr_matrix:
    """
    Sparse one-hot of per-tree leaf indices (suitable for UMAP ``metric="cosine"``).

    ``calc_leaf_indexes`` returns an ``(n, n_trees)`` integer matrix; each tree
    column is one-hot encoded and the blocks are concatenated.
    """
    return embed_leaves(catboost_model, to_catboost_X(X), one_hot=True)


def get_rff_embedding(rff_pipeline, X: pd.DataFrame) -> np.ndarray:
    """Random-Fourier feature matrix (all pipeline steps except the final classifier)."""
    return np.asarray(rff_pipeline[:-1].transform(X), dtype=np.float32)
