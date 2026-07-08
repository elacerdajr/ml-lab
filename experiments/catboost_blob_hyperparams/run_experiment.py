"""
CatBoost Hyperparameter Sweep — Overlapping Blob Classification
=================================================================
Ten overlapping 2D blobs (x1, x2), each with its own positive rate
P(y=1) drawn from Uniform(0, 0.5) (blob id x3 is withheld from training —
it would trivially solve the task). A CatBoostClassifier is trained on
x1/x2 only and swept over `depth` and `iterations` to see how well it
recovers each blob's positive rate from coordinates alone.

Outputs (experiments/catboost_blob_hyperparams/outputs/):
  depth_sweep.png
  iterations_sweep.png
  results.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from ml_elements.dgp import BlobClassificationDGP
from ml_elements.metrics import AUC, AVG_PRECISION
from ml_elements.models import make_catboost

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import catboost  # noqa: F401
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BLOBS = 10
P_LOW = 0.0
P_HIGH = 0.5
SEED = 42
N_TRAIN = 4_000
N_PLOT = 800

DGP = BlobClassificationDGP(
    n_blobs=N_BLOBS,
    center_std=1.5,
    blob_std=1.0,
    p_low=P_LOW,
    p_high=P_HIGH,
    center_seed=0,
)

FIXED_ITERATIONS = 300
FIXED_DEPTH = 6
DEPTH_TRIALS = [2, 4, 6, 10]
ITERATION_TRIALS = [10, 50, 200, 800]

FEATURES = ["x1", "x2"]

# ---------------------------------------------------------------------------
# Fit / score
# ---------------------------------------------------------------------------


def _fit_and_score(
    train_df: pd.DataFrame, plot_df: pd.DataFrame, depth: int, iterations: int
) -> tuple[np.ndarray, float, float]:
    model = make_catboost(iterations=iterations, depth=depth, random_state=SEED)()
    model.fit(train_df[FEATURES], train_df["y"])
    y_hat = model.predict_proba(plot_df[FEATURES])[:, 1]
    auc = AUC.score(plot_df["y"].to_numpy(), y_hat)
    ap = AVG_PRECISION.score(plot_df["y"].to_numpy(), y_hat)
    return y_hat, auc, ap


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_sweep(
    rows: list[dict],
    plot_df: pd.DataFrame,
    param_name: str,
    title: str,
    out_path: Path,
) -> None:
    n_trials = len(rows)
    fig, axes = plt.subplots(n_trials, 3, figsize=(15, 4 * n_trials))

    x1, x2 = plot_df["x1"].to_numpy(), plot_df["x2"].to_numpy()
    blob_ids = sorted(plot_df["x3"].unique())
    blob_idx = plot_df["x3"].map({b: i for i, b in enumerate(blob_ids)}).to_numpy()
    blob_true_rate = plot_df.groupby("x3")["y"].transform("mean").to_numpy()

    for row_i, row in enumerate(rows):
        ax_left, ax_mid, ax_right = axes[row_i]

        ax_left.scatter(x1, x2, c=blob_idx, cmap="tab10", vmin=0, vmax=9, s=14, alpha=0.8)
        ax_left.set_title("colored by X3 (blob id)" if row_i == 0 else "", fontsize=10)
        ax_left.set_ylabel(f"{param_name}={row[param_name]}", fontsize=11)

        sc_mid = ax_mid.scatter(
            x1, x2, c=blob_true_rate, cmap="viridis", vmin=P_LOW, vmax=P_HIGH,
            s=14, alpha=0.8,
        )
        ax_mid.set_title("colored by groupby(X3).mean(y)" if row_i == 0 else "", fontsize=10)

        sc_right = ax_right.scatter(
            x1, x2, c=row["y_hat"], cmap="viridis", vmin=P_LOW, vmax=P_HIGH,
            s=14, alpha=0.8,
        )
        ax_right.set_title(
            f"model score (AUC={row['auc']:.4f}, AP={row['ap']:.3f})"
            if row_i == 0
            else f"AUC={row['auc']:.4f}, AP={row['ap']:.3f}",
            fontsize=10,
        )

        for ax in (ax_left, ax_mid, ax_right):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, alpha=0.15, linestyle=":")
            ax.set_xlabel("x1", fontsize=8)

        fig.colorbar(sc_mid, ax=ax_mid, fraction=0.046, pad=0.04)
        fig.colorbar(sc_right, ax=ax_right, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not _HAS_CATBOOST:
        log.error("catboost not installed. Run: pip install catboost")
        sys.exit(1)

    log.info(
        "Generating data — n_train=%d  n_plot=%d  n_blobs=%d  p~Uniform(%.2f, %.2f)",
        N_TRAIN, N_PLOT, N_BLOBS, P_LOW, P_HIGH,
    )
    train_df = DGP.sample(n=N_TRAIN, seed=SEED)
    plot_df = DGP.sample(n=N_PLOT, seed=SEED + 1_000)
    log.info("Train positives: %d / %d (%.1f%%)", train_df["y"].sum(), N_TRAIN, 100 * train_df["y"].mean())

    results: list[dict] = []

    log.info("=== Depth sweep (iterations=%d fixed) ===", FIXED_ITERATIONS)
    depth_rows: list[dict] = []
    for depth in DEPTH_TRIALS:
        y_hat, auc, ap = _fit_and_score(train_df, plot_df, depth, FIXED_ITERATIONS)
        log.info("  depth=%d  AUC=%.4f  AP=%.3f", depth, auc, ap)
        row = {"depth": depth, "iterations": FIXED_ITERATIONS, "y_hat": y_hat, "auc": auc, "ap": ap}
        depth_rows.append(row)
        results.append({"sweep": "depth", "depth": depth, "iterations": FIXED_ITERATIONS, "auc": auc, "ap": ap})

    _plot_sweep(
        depth_rows, plot_df, "depth",
        f"CatBoost depth sweep (iterations={FIXED_ITERATIONS} fixed)",
        OUT_DIR / "depth_sweep.png",
    )

    log.info("=== Iterations sweep (depth=%d fixed) ===", FIXED_DEPTH)
    iter_rows: list[dict] = []
    for iterations in ITERATION_TRIALS:
        y_hat, auc, ap = _fit_and_score(train_df, plot_df, FIXED_DEPTH, iterations)
        log.info("  iterations=%d  AUC=%.4f  AP=%.3f", iterations, auc, ap)
        row = {"depth": FIXED_DEPTH, "iterations": iterations, "y_hat": y_hat, "auc": auc, "ap": ap}
        iter_rows.append(row)
        results.append({"sweep": "iterations", "depth": FIXED_DEPTH, "iterations": iterations, "auc": auc, "ap": ap})

    _plot_sweep(
        iter_rows, plot_df, "iterations",
        f"CatBoost iterations sweep (depth={FIXED_DEPTH} fixed)",
        OUT_DIR / "iterations_sweep.png",
    )

    results_df = pd.DataFrame(results)
    csv_path = OUT_DIR / "results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("saved %s", csv_path.name)

    _write_report(results_df, train_df)

    log.info("\n=== Summary ===")
    print(results_df.to_string(index=False, float_format="%.4f"))


def _write_report(results_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    report_path = SCRIPT_DIR / "report.md"

    depth_sub = results_df[results_df["sweep"] == "depth"]
    iter_sub = results_df[results_df["sweep"] == "iterations"]

    lines: list[str] = [
        "# CatBoost Hyperparameter Sweep — Overlapping Blob Classification",
        "",
        "> Generated by `experiments/catboost_blob_hyperparams/run_experiment.py`",
        "",
        "---",
        "",
        "## Experimental setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| DGP | BlobClassificationDGP |",
        f"| n_blobs | {N_BLOBS} |",
        f"| Positive rate range | Uniform({P_LOW}, {P_HIGH}) per blob |",
        f"| n_train / n_plot | {N_TRAIN:,} / {N_PLOT:,} |",
        f"| Train positive rate | {train_df['y'].mean():.1%} |",
        "| Model features | x1, x2 (x3 blob id withheld) |",
        "| Model | CatBoostClassifier (Logloss, via `ml_elements.models.make_catboost`) |",
        f"| Depth sweep | {DEPTH_TRIALS} (iterations={FIXED_ITERATIONS} fixed) |",
        f"| Iterations sweep | {ITERATION_TRIALS} (depth={FIXED_DEPTH} fixed) |",
        "",
        "---",
        "",
        "## Results",
        "",
        "### Depth sweep",
        "",
        "| depth | AUC | AP |",
        "|------:|----:|---:|",
    ]
    for _, r in depth_sub.iterrows():
        lines.append(f"| {int(r['depth'])} | {r['auc']:.4f} | {r['ap']:.3f} |")

    lines += [
        "",
        "### Iterations sweep",
        "",
        "| iterations | AUC | AP |",
        "|-----------:|----:|---:|",
    ]
    for _, r in iter_sub.iterrows():
        lines.append(f"| {int(r['iterations'])} | {r['auc']:.4f} | {r['ap']:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## Figures",
        "",
        "Each row is one hyperparameter trial. Left: points colored by true blob id (X3).",
        "Middle: points colored by the true per-blob positive rate (groupby(X3).mean(y)) —",
        "this is the ground-truth surface the model has to recover from (x1, x2) alone.",
        "Right: points colored by the model's predicted probability (predict_proba), on the",
        "same color scale as the middle panel for direct visual comparison.",
        "",
        "![depth sweep](outputs/depth_sweep.png)",
        "",
        "![iterations sweep](outputs/iterations_sweep.png)",
        "",
        "---",
        "",
        "Raw data: `outputs/results.csv`",
    ]

    report_path.write_text("\n".join(lines) + "\n")
    log.info("saved %s", report_path.name)


if __name__ == "__main__":
    main()
