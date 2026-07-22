"""
umap_viz.py
-----------
UMAP projection and plotting, comparing the full-population representation with
the 10%-positive undersampled representation for each embedding type (raw
features, CatBoost leaf embedding, RFF features).

UMAP is imported lazily (repo pattern) so the package imports without the
``umap`` extra; callers guard on :data:`HAS_UMAP`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config

try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:  # pragma: no cover
    HAS_UMAP = False


def fit_umap(emb, metric: str, cfg: Config) -> np.ndarray:
    """Project ``emb`` to 2D with UMAP (n_neighbors/min_dist from config, fixed seed)."""
    if not HAS_UMAP:
        raise ImportError("umap-learn required. Install with the `umap` extra.")
    n = emb.shape[0]
    n_neighbors = int(min(cfg.umap.n_neighbors, max(2, n - 1)))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cfg.umap.min_dist,
        metric=metric,
        random_state=cfg.seed,
    )
    return reducer.fit_transform(emb)


def _scatter(ax, coords: np.ndarray, y: np.ndarray, title: str) -> None:
    neg = y == 0
    pos = y == 1
    ax.scatter(coords[neg, 0], coords[neg, 1], c="#4477bb", s=6, alpha=0.20,
               linewidths=0, label="neg (y=0)")
    ax.scatter(coords[pos, 0], coords[pos, 1], c="#cc4444", s=14, alpha=0.85,
               linewidths=0, label="pos (y=1)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="best")


def compare_full_vs_undersampled_umap(
    name: str,
    metric: str,
    emb_full,
    y_full: np.ndarray,
    emb_under,
    y_under: np.ndarray,
    cfg: Config,
    out_dir: Path,
) -> Path:
    """
    Run UMAP on the full-sample and undersampled embeddings for one representation
    and save a side-by-side 2-panel figure. Returns the saved path.
    """
    coords_full = fit_umap(emb_full, metric, cfg)
    coords_under = fit_umap(emb_under, metric, cfg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _scatter(ax1, coords_full, y_full, f"{name} — full sample (base rate)")
    _scatter(ax2, coords_under, y_under, f"{name} — undersampled (10% pos)")
    fig.suptitle(f"UMAP: {name}", fontsize=12)
    fig.tight_layout()

    path = out_dir / f"umap_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
