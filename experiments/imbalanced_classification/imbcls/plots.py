"""
plots.py
--------
All non-UMAP figures (A–F):

A score histograms (val + test; raw vs shrink vs noise)
B precision–recall curves
C AP vs normalized score entropy (point size = train time)
D AP vs log(1 + train time), coloured by entropy
E AP-loss vs entropy-gain for the deterministic noise score (headline plot)
F bucket-lift deciles with the true base-rate line

Clean Matplotlib, ``Agg`` backend, dpi 130, top/right spines hidden.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

_POS = "#cc4444"
_NEG = "#4477bb"


def _despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


# ── A. Score histograms ──────────────────────────────────────────────────────

def plot_score_histograms(
    model_name: str,
    val_variants: dict[str, np.ndarray],
    test_variants: dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    """2×3 grid: rows = val/test, cols = raw / post-hoc shrink / noise ranking."""
    variants = list(val_variants.keys())
    fig, axes = plt.subplots(2, len(variants), figsize=(4 * len(variants), 7))
    for row, (split, data) in enumerate([("validation", val_variants), ("test", test_variants)]):
        for col, key in enumerate(variants):
            ax = axes[row, col]
            ax.hist(data[key], bins=50, range=(0, 1), color="#5588cc", alpha=0.85)
            ax.set_yscale("log")
            ax.set_title(f"{split} — {key}", fontsize=9)
            ax.set_xlabel("score")
            _despine(ax)
    fig.suptitle(f"Score distributions — {model_name}", fontsize=12)
    fig.tight_layout()
    path = out_dir / f"score_hist_{model_name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── B. Precision–recall ──────────────────────────────────────────────────────

def plot_precision_recall(entries: list[tuple[str, np.ndarray, np.ndarray]], out_dir: Path) -> Path:
    """One PR curve per model (raw probability, test split)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, (label, y, score) in enumerate(entries):
        prec, rec, _ = precision_recall_curve(y, score)
        ax.plot(rec, prec, color=cmap(i % 10), linewidth=1.6, label=label)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–Recall (test, raw probability)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "precision_recall.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── C. AP vs entropy ─────────────────────────────────────────────────────────

def plot_ap_entropy(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter x=normalized entropy, y=AP, size∝train time; one point per model (raw, test)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = 40 + 400 * (df["train_time_seconds"] / df["train_time_seconds"].max())
    ax.scatter(df["normalized_score_entropy"], df["average_precision"], s=sizes,
               alpha=0.7, c="#3366aa", edgecolors="white", linewidths=0.5)
    for _, r in df.iterrows():
        ax.annotate(r["model_name"], (r["normalized_score_entropy"], r["average_precision"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.80, color="grey", linestyle=":", linewidth=1, label="high-entropy (≥0.80)")
    ax.set_xlabel("normalized score entropy")
    ax.set_ylabel("average precision")
    ax.set_title("AP vs score entropy (point size ∝ train time)", fontsize=11)
    ax.legend(fontsize=8)
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "ap_vs_entropy.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── D. AP vs training time ───────────────────────────────────────────────────

def plot_ap_time(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter x=log(1+train time), y=AP, colour = normalized entropy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(np.log1p(df["train_time_seconds"]), df["average_precision"],
                    c=df["normalized_score_entropy"], cmap="viridis", s=90,
                    vmin=0, vmax=1, edgecolors="white", linewidths=0.5)
    for _, r in df.iterrows():
        ax.annotate(r["model_name"], (np.log1p(r["train_time_seconds"]), r["average_precision"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="normalized score entropy")
    ax.set_xlabel("log(1 + train time [s])")
    ax.set_ylabel("average precision")
    ax.set_title("AP vs training time", fontsize=11)
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "ap_vs_time.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── E. AP loss vs entropy gain (headline) ────────────────────────────────────

def plot_entropy_tradeoff(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    For the noise ranking score: x = entropy gain ΔH = H(r)-H(p),
    y = AP change ΔAP = AP(r)-AP(p). Points coloured by model, sized by α.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    models = df["model_name"].unique().tolist()
    for i, m in enumerate(models):
        sub = df[df["model_name"] == m]
        ax.scatter(sub["entropy_gain"], sub["ap_delta"], color=cmap(i % 10),
                   s=30 + 200 * (1 - sub["alpha"]), alpha=0.75, label=m, edgecolors="white",
                   linewidths=0.4)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("entropy gain  ΔH = H(r) − H(p)")
    ax.set_ylabel("AP change  ΔAP = AP(r) − AP(p)")
    ax.set_title("Ranking-noise trade-off: AP cost vs entropy gain (test)", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "ap_loss_vs_entropy_gain.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── F. Bucket lift ───────────────────────────────────────────────────────────

def plot_bucket_lift(bucket_df: pd.DataFrame, base_rate: float, out_dir: Path) -> Path:
    """Decile positive-rate curves per model with the true base-rate reference line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    models = bucket_df["model_name"].unique().tolist()
    for i, m in enumerate(models):
        sub = bucket_df[bucket_df["model_name"] == m].sort_values("bucket_id")
        ax.plot(sub["bucket_id"], sub["positive_rate"], marker="o", markersize=4,
                color=cmap(i % 10), linewidth=1.4, label=m)
    ax.axhline(base_rate, color="black", linestyle="--", linewidth=1.2,
               label=f"base rate ({base_rate:.4f})")
    ax.set_yscale("log")
    ax.set_xlabel("score decile (0 = lowest)")
    ax.set_ylabel("positive rate (log)")
    ax.set_title("Bucket lift by score decile (test, raw probability)", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "bucket_lift.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
