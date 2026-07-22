"""
plots.py
--------
Two figures: ``ap_by_model`` (the core comparison) and ``precision_recall``
(the right diagnostic under heavy imbalance). Simpler than the sibling
``encoder_comparison`` experiment's three plots since there is only one
encoder here — no dimensionality/timing-by-encoder comparison is needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def _despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def plot_ap_by_model(test_df: pd.DataFrame, out_dir: Path) -> Path:
    """Bar chart: AP per model (test split), CatBoost-native called out distinctly."""
    sub = test_df.sort_values("average_precision", ascending=False)
    colors = ["#cc4444" if m == "catboost_native" else "#4477bb" for m in sub["model_name"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(sub["model_name"], sub["average_precision"], color=colors, alpha=0.85)
    for x, v in enumerate(sub["average_precision"]):
        ax.text(x, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("average precision (test)")
    ax.set_title(
        "AP by model — CatBoost-encoded features vs CatBoost native (red)", fontsize=11
    )
    ax.tick_params(axis="x", rotation=20)
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "ap_by_model.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall(entries: list[tuple[str, np.ndarray, np.ndarray]], out_dir: Path) -> Path:
    """One PR curve per model (test split)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, (label, y, score) in enumerate(entries):
        prec, rec, _ = precision_recall_curve(y, score)
        style = "--" if label == "catboost_native" else "-"
        ax.plot(rec, prec, style, color=cmap(i % 10), linewidth=1.6, label=label)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall (test)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "precision_recall.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
