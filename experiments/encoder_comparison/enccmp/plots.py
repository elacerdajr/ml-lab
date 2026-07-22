"""
plots.py
--------
Three figures answering "how much does the encoder matter?":

ap_by_encoder    grouped bar chart, AP per (model x encoder) — the core comparison.
dimensionality   n_features_out vs AP scatter — does dimensionality blowup help or hurt?
timing           encode + train time per (model x encoder) — is a fancier encoder worth its cost?

``catboost_native`` sits at its own ``encoder="native"`` x-position/annotation
since it bypasses the encoder axis entirely.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENCODER_ORDER = ["onehot", "ordinal", "frequency", "target", "hashing", "native"]


def _despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _ordered(df: pd.DataFrame) -> list[str]:
    present = [e for e in ENCODER_ORDER if e in set(df["encoder"])]
    return present


def plot_ap_by_encoder(test_df: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar chart: AP per (model, encoder), test split."""
    encoders = _ordered(test_df)
    models = test_df["model_name"].unique().tolist()
    x = np.arange(len(encoders))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(models):
        sub = test_df[test_df.model_name == model].set_index("encoder")
        vals = [sub.loc[e, "average_precision"] if e in sub.index else np.nan for e in encoders]
        ax.bar(x + i * width, vals, width, label=model, color=cmap(i % 10), alpha=0.85)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(encoders)
    ax.set_xlabel("encoder")
    ax.set_ylabel("average precision (test)")
    ax.set_title("AP by encoder x model", fontsize=11)
    ax.legend(fontsize=8)
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "ap_by_encoder.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dimensionality(test_df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter: n_features_out vs AP, coloured by model, annotated by encoder."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    models = test_df["model_name"].unique().tolist()
    for i, model in enumerate(models):
        sub = test_df[test_df.model_name == model]
        ax.scatter(
            sub["n_features_out"], sub["average_precision"], color=cmap(i % 10),
            s=70, alpha=0.8, label=model, edgecolors="white", linewidths=0.5,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["encoder"], (r["n_features_out"], r["average_precision"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points",
            )
    ax.set_xlabel("n_features_out (encoded dimensionality)")
    ax.set_ylabel("average precision (test)")
    ax.set_title(
        "AP vs encoded dimensionality\n"
        "(native's n_features_out=4 is raw-column width, not a fair size comparison"
        " — CatBoost searches categorical splits internally)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "dimensionality_vs_ap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_timing(test_df: pd.DataFrame, out_dir: Path) -> Path:
    """Stacked bar: encode fit time + model train time, per (model, encoder)."""
    encoders = _ordered(test_df)
    models = test_df["model_name"].unique().tolist()
    x = np.arange(len(encoders))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(models):
        sub = test_df[test_df.model_name == model].set_index("encoder")
        enc_t = [sub.loc[e, "encode_fit_time_seconds"] if e in sub.index else 0.0 for e in encoders]
        enc_t = [0.0 if pd.isna(v) else v for v in enc_t]
        train_t = [sub.loc[e, "train_time_seconds"] if e in sub.index else np.nan for e in encoders]
        pos = x + i * width
        mask = [not pd.isna(v) for v in train_t]
        pos_m = pos[mask]
        enc_m = np.array(enc_t)[mask]
        train_m = np.array(train_t)[mask]
        ax.bar(pos_m, enc_m, width, color=cmap(i % 10), alpha=0.55)
        ax.bar(pos_m, train_m, width, bottom=enc_m, color=cmap(i % 10), alpha=0.95, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(encoders)
    ax.set_xlabel("encoder")
    ax.set_ylabel("seconds (encode fit, lighter + train, darker)")
    ax.set_title("Encode + train time by encoder x model", fontsize=11)
    ax.legend(fontsize=8)
    _despine(ax)
    fig.tight_layout()
    path = out_dir / "timing.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
