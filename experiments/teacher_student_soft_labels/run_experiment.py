"""
Teacher/Student Soft-Label Protocol — Train/Val Distillation Experiment
=========================================================================
Formalizes a two-stage "teacher" (p0 coarse classifier + p1 residual
regressor) that produces a rank-spread soft target y_soft, fit entirely on
df_train and only ever *transformed* (never fitted) on df_val. A "student"
CatBoostRegressor is then trained on X_train -> y_soft_train and validated
on X_val -> y_soft_val.

    p0(x)      = high-reg CatBoostClassifier(x), fit on df_train
    H0(x)      = logit(p0(x))
    r_train    = y_train - p0(X_train)
    p1(x)      = CatBoostRegressor(x, r_train), fit on df_train
    V(x)       = (p1(x) - mu_v) / sigma_v            (mu_v, sigma_v from train only)
    H(x)       = logit(p0(x)) + lambda * V(x)
    F_train    = EmpiricalCDF().fit(H_train)          (fit ONLY on H_train)
    y_soft(x)  = F_train.transform(H(x))               (applied to train AND val)

Hard rule: df_val is only ever .predict()/.transform()-ed, never .fit()-ed —
for p0, p1, the leaf one-hot encoder, TruncatedSVD, UMAP, mu_v/sigma_v, and
EmpiricalCDF.

This experiment implements ONLY the "simple" (in-sample teacher) version.
The out-of-fold (OOF) variant — fit p0/p1 per-fold within df_train, stitch
H_OOF, fit F_train on H_OOF, then refit final p0/p1 on all of df_train for
scoring df_val — is documented as future work in the report but not built
here (see outputs/README.md, "Limitations & future work").

Outputs (experiments/teacher_student_soft_labels/outputs/):
  results.csv
  teacher_train.parquet / teacher_val.parquet   (at LAMBDA_HEADLINE only)
  score_distributions_lambda{lam}.png           (x5)
  h_train_val_shift_lambda{lam}.png             (x5)
  degeneracy_unique_ratio.png
  lambda_sweep_diagnostics.png
  student_fit_scatter_lambda{headline}.png
  topk_overlap_vs_lambda.png
  auc_ap_vs_lambda.png
  event_rate_by_decile_lambda{headline}.png
  umap_leaf_embeddings.png                      (only if umap-learn installed)
  README.md
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
from scipy.stats import kendalltau, kstest, ks_2samp, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from ml_elements.dgp import GaussianBinaryDGP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

P_POS = 0.10
N_TRAIN = 2_000
N_VAL = 2_000
SEED = 42

DGP = GaussianBinaryDGP(
    p_pos=P_POS,
    info={"x1": 1.2, "x2": 0.8, "x3": 0.4, "x4": 0.15, "x5": 0.05},
    sigma=1.0,
)

LAMBDAS = [0.0, 0.02, 0.05, 0.10, 0.20]
LAMBDA_HEADLINE = 0.05  # used for parquet export + headline-only figures
EPS = 1e-6  # shared: logit clip, EmpiricalCDF eps, std floor

P0_CFG = dict(
    loss_function="Logloss", depth=3, learning_rate=0.03, iterations=500,
    l2_leaf_reg=50, random_seed=SEED, verbose=False, allow_writing_files=False,
)
P1_CFG = dict(
    loss_function="RMSE", depth=5, learning_rate=0.03, iterations=800,
    l2_leaf_reg=10, random_seed=SEED, verbose=False, allow_writing_files=False,
)
STUDENT_CFG = dict(
    loss_function="RMSE", depth=4, learning_rate=0.05, iterations=300,
    random_seed=SEED, verbose=False, allow_writing_files=False,
)

N_SVD_COMPONENTS = 40
TOPK_FRACS = [0.01, 0.05, 0.10]

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


# ---------------------------------------------------------------------------
# EmpiricalCDF — ported faithfully from the user's reference implementation.
# De-duplicates repeated x-values by averaging their u-ranks before building
# the interpolation table. The `for i, g in enumerate(inverse)` loop is O(n)
# python-level; fine at n~2-4k. A vectorized alternative would be
# `np.add.at(unique_u, inverse, u); unique_u /= np.bincount(inverse)`, but is
# intentionally not applied here to keep this an auditable, faithful port.
# ---------------------------------------------------------------------------

class EmpiricalCDF:
    def __init__(self, eps: float = EPS):
        self.eps = eps
        self.x_ = None
        self.u_ = None

    def fit(self, values: np.ndarray) -> "EmpiricalCDF":
        values = np.asarray(values)
        order = np.argsort(values)
        x = values[order]
        n = len(x)
        u = (np.arange(n) + 0.5) / n
        unique_x, inverse = np.unique(x, return_inverse=True)
        unique_u = np.zeros(len(unique_x), dtype=float)
        counts = np.zeros(len(unique_x), dtype=float)
        for i, g in enumerate(inverse):
            unique_u[g] += u[i]
            counts[g] += 1
        unique_u /= counts
        self.x_ = unique_x
        self.u_ = unique_u
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        u = np.interp(values, self.x_, self.u_, left=self.u_[0], right=self.u_[-1])
        return np.clip(u, self.eps, 1 - self.eps)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _fit_p0(X_train: np.ndarray, y_train: np.ndarray) -> object:
    model = CatBoostClassifier(**P0_CFG)
    model.fit(X_train, y_train)  # TRAIN ONLY
    return model


def _fit_p1(X_train: np.ndarray, r_train: np.ndarray) -> object:
    model = CatBoostRegressor(**P1_CFG)
    model.fit(X_train, r_train)  # TRAIN ONLY
    return model


def _fit_student(X_train: np.ndarray, y_soft_train: np.ndarray) -> object:
    model = CatBoostRegressor(**STUDENT_CFG)
    model.fit(X_train, y_soft_train)  # TRAIN ONLY
    return model


def _logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(std, EPS)


def _leaf_embeddings(model: object, X: np.ndarray) -> np.ndarray:
    return model.calc_leaf_indexes(X).astype(np.int32)  # (n, n_trees)


def _leaf_umap_pipeline(
    L_train: np.ndarray, L_val: np.ndarray, n_svd: int = N_SVD_COMPONENTS,
) -> tuple:
    """One-hot (fit train only) -> TruncatedSVD (fit train only) -> UMAP
    (fit train only, transform val) if umap-learn is installed.

    L_train/L_val = np.hstack([L0, L1]) concatenates leaf-index columns from
    both teacher components. This is safe: OneHotEncoder treats every input
    column as an independent categorical, so identical leaf-index integers
    occupying different tree-columns (whether from p0 or p1) never collide
    in the one-hot output — no special handling needed.
    """
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    E_train = enc.fit_transform(L_train)  # TRAIN ONLY
    E_val = enc.transform(L_val)

    n_svd_safe = min(n_svd, E_train.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_svd_safe, random_state=SEED)
    Z_train_svd = svd.fit_transform(E_train)  # TRAIN ONLY
    Z_val_svd = svd.transform(E_val)

    if not _HAS_UMAP:
        return E_train, E_val, Z_train_svd, Z_val_svd, None, None

    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
    Z_train_umap = reducer.fit_transform(Z_train_svd)  # TRAIN ONLY
    Z_val_umap = reducer.transform(Z_val_svd)
    return E_train, E_val, Z_train_svd, Z_val_svd, Z_train_umap, Z_val_umap


def _tie_stats(values: np.ndarray, decimals: int = 6) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)

    _, counts = np.unique(values, return_counts=True)
    n_unique = len(counts)
    max_tie_size = int(counts.max())

    rounded = np.round(values, decimals)
    _, counts_r = np.unique(rounded, return_counts=True)
    n_unique_r = len(counts_r)
    max_tie_size_r = int(counts_r.max())

    return {
        "n_unique": n_unique,
        "unique_ratio": n_unique / n,
        "max_tie_size": max_tie_size,
        "n_unique_after_rounding_1e6": n_unique_r,
        "max_tie_size_after_rounding_1e6": max_tie_size_r,
    }


def _decile_event_rate(y_soft_val: np.ndarray, y_val: np.ndarray, n_bins: int = 10) -> np.ndarray:
    bins = pd.qcut(y_soft_val, n_bins, labels=False, duplicates="drop")
    rates = np.full(n_bins, np.nan)
    for b in np.unique(bins):
        rates[int(b)] = float(y_val[bins == b].mean())
    return rates


def _student_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "kendall": float(kendalltau(y_true, y_pred).statistic),
    }


def _topk_overlap(true_scores: np.ndarray, pred_scores: np.ndarray, frac: float) -> float:
    n = len(true_scores)
    k = max(1, int(round(frac * n)))
    top_true = set(np.argsort(-true_scores)[:k])
    top_pred = set(np.argsort(-pred_scores)[:k])
    return len(top_true & top_pred) / k


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_score_distributions(
    lam: float,
    p0_train: np.ndarray,
    H_train: np.ndarray,
    y_soft_train: np.ndarray,
    y_soft_val: np.ndarray,
    y_train: np.ndarray,
    ks_uniform_pvalue: float,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    panels = [
        ("p0_train  (raw coarse score)", p0_train, y_train),
        ("H_train  (perturbed logit)", H_train, y_train),
        (f"y_soft_train  (KS-vs-uniform p={ks_uniform_pvalue:.3f})", y_soft_train, y_train),
        ("y_soft_val", y_soft_val, None),
    ]

    for ax, (title, values, y) in zip(axes, panels):
        if y is not None:
            pos = values[y == 1]
            neg = values[y == 0]
            ax.hist(neg, bins=40, alpha=0.6, color="#4477bb", label="neg (y=0)", density=True)
            ax.hist(pos, bins=40, alpha=0.6, color="#cc4444", label="pos (y=1)", density=True)
            if ax is axes[0]:
                ax.legend(fontsize=8)
        else:
            ax.hist(values, bins=40, alpha=0.75, color="#888888", density=True)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle(f"Score distributions — lambda={lam}", fontsize=11, y=1.03)
    fig.tight_layout()
    path = OUT_DIR / f"score_distributions_lambda{lam}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_h_train_val_shift(lam: float, H_train: np.ndarray, H_val: np.ndarray,
                             ks_stat: float, ks_pvalue: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(H_train, bins=40, alpha=0.6, color="#4477bb", label="H_train", density=True)
    ax.hist(H_val, bins=40, alpha=0.6, color="#cc4444", label="H_val", density=True)
    ax.set_title(f"H train vs val — lambda={lam}  KS={ks_stat:.4f} (p={ks_pvalue:.4f})", fontsize=9.5)
    ax.set_xlabel("H(x)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    path = OUT_DIR / f"h_train_val_shift_lambda{lam}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_degeneracy_unique_ratio(df: pd.DataFrame, p0_train_ratio: float, p0_val_ratio: float) -> None:
    lambdas = df["lambda"].tolist()
    x = np.arange(len(lambdas))
    width = 0.13

    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [
        ("H_train", df["H_train_unique_ratio"], "#4477bb"),
        ("H_val", df["H_val_unique_ratio"], "#88aadd"),
        ("y_soft_train", df["y_soft_train_unique_ratio"], "#cc4444"),
        ("y_soft_val", df["y_soft_val_unique_ratio"], "#dd8888"),
        ("student_pred_train", df["student_pred_train_unique_ratio"], "#44aa77"),
        ("student_pred_val", df["student_pred_val_unique_ratio"], "#88ccaa"),
    ]
    for i, (label, values, color) in enumerate(series):
        ax.bar(x + (i - 2.5) * width, values, width, label=label, color=color)

    ax.axhline(p0_train_ratio, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="p0_train baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={lam}" for lam in lambdas])
    ax.set_ylabel("unique-value ratio")
    ax.set_title("Degeneracy: unique-value ratio across the pipeline", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    fig.tight_layout()
    path = OUT_DIR / "degeneracy_unique_ratio.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_lambda_sweep(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    lam = df["lambda"]

    axes[0, 0].plot(lam, df["spearman_p0_H_train"], "o-", label="train", color="#4477bb")
    axes[0, 0].plot(lam, df["spearman_p0_H_val"], "o-", label="val", color="#cc4444")
    axes[0, 0].set_title("Spearman(p0, H)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(lam, df["H_train_unique_ratio"], "o-", label="train", color="#4477bb")
    axes[0, 1].plot(lam, df["H_val_unique_ratio"], "o-", label="val", color="#cc4444")
    axes[0, 1].set_title("unique_ratio(H)")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(lam, df["ks_uniform_y_soft_train_pvalue"], "o-", color="#44aa77")
    axes[0, 2].set_title("KS-vs-uniform p-value (y_soft_train)")
    axes[0, 2].axhline(0.05, color="black", linestyle="--", linewidth=1, alpha=0.5)

    axes[1, 0].plot(lam, df["student_r2_val"], "o-", color="#4477bb")
    axes[1, 0].set_title("Student R² (val)")

    axes[1, 1].plot(lam, df["ap_ysoft_y_val"], "o-", label="AP", color="#cc4444")
    axes[1, 1].plot(lam, df["auc_ysoft_y_val"], "o-", label="AUC", color="#4477bb")
    axes[1, 1].set_title("y_soft vs y (val)")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(lam, df["student_spearman_val"], "o-", color="#44aa77")
    axes[1, 2].set_title("Student Spearman (val)")

    for ax in axes.ravel():
        ax.set_xlabel("lambda")
        ax.grid(alpha=0.28, linestyle=":")

    fig.suptitle("Lambda-sweep diagnostics", fontsize=12)
    fig.tight_layout()
    path = OUT_DIR / "lambda_sweep_diagnostics.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_student_fit_scatter(lam: float, y_soft_val: np.ndarray, student_pred_val: np.ndarray,
                               r2: float, spearman_val: float) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_soft_val, student_pred_val, s=10, alpha=0.4, color="#4477bb", linewidths=0)
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y=x")
    ax.set_xlabel("y_soft_val (target)")
    ax.set_ylabel("student_pred_val")
    ax.set_title(f"Student fit — lambda={lam}  R²={r2:.4f}  ρ={spearman_val:.4f}", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    path = OUT_DIR / f"student_fit_scatter_lambda{lam}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_topk_overlap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    lam = df["lambda"]
    ax.plot(lam, df["top_1pct_overlap"], "o-", label="top 1%", color="#4477bb")
    ax.plot(lam, df["top_5pct_overlap"], "o-", label="top 5%", color="#cc4444")
    ax.plot(lam, df["top_10pct_overlap"], "o-", label="top 10%", color="#44aa77")
    ax.set_xlabel("lambda")
    ax.set_ylabel("overlap fraction (val)")
    ax.set_title("Student top-k% overlap with true y_soft ranking", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.28, linestyle=":")
    fig.tight_layout()
    path = OUT_DIR / "topk_overlap_vs_lambda.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_auc_ap_vs_lambda(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    lam = df["lambda"]

    ax1.plot(lam, df["auc_p0_y_val"], "o-", label="p0", color="#888888")
    ax1.plot(lam, df["auc_H_y_val"], "o-", label="H", color="#4477bb")
    ax1.plot(lam, df["auc_ysoft_y_val"], "o-", label="y_soft", color="#cc4444")
    ax1.set_title("AUC vs y (val)")
    ax1.set_xlabel("lambda")
    ax1.legend(fontsize=8)

    ax2.plot(lam, df["ap_p0_y_val"], "o-", label="p0", color="#888888")
    ax2.plot(lam, df["ap_H_y_val"], "o-", label="H", color="#4477bb")
    ax2.plot(lam, df["ap_ysoft_y_val"], "o-", label="y_soft", color="#cc4444")
    ax2.set_title("AP vs y (val)")
    ax2.set_xlabel("lambda")
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.28, linestyle=":")

    fig.tight_layout()
    path = OUT_DIR / "auc_ap_vs_lambda.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_event_rate_decile(lam: float, rates: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    deciles = np.arange(1, len(rates) + 1)
    ax.bar(deciles, rates, color="#4477bb", alpha=0.8)
    ax.set_xlabel("y_soft_val decile (1=lowest, 10=highest)")
    ax.set_ylabel("mean(y_val)")
    ax.set_title(f"Event rate by y_soft decile — lambda={lam}", fontsize=10)
    ax.set_xticks(deciles)
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    fig.tight_layout()
    path = OUT_DIR / f"event_rate_by_decile_lambda{lam}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_umap(Z_train: np.ndarray, y_train: np.ndarray, Z_val: np.ndarray, y_soft_val: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    neg_mask = y_train == 0
    pos_mask = y_train == 1
    ax1.scatter(Z_train[neg_mask, 0], Z_train[neg_mask, 1], c="#4477bb", s=8, alpha=0.55,
                linewidths=0, label="neg (y=0)")
    ax1.scatter(Z_train[pos_mask, 0], Z_train[pos_mask, 1], c="#cc4444", s=8, alpha=0.55,
                linewidths=0, label="pos (y=1)")
    ax1.set_title("Train — combined leaf embedding (L0+L1)", fontsize=9.5)
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    ax1.legend(fontsize=8)

    sc = ax2.scatter(Z_val[:, 0], Z_val[:, 1], c=y_soft_val, cmap="coolwarm", s=8, alpha=0.55,
                      vmin=0, vmax=1, linewidths=0)
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("Val — colored by y_soft_val", fontsize=9.5)
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")

    fig.suptitle("UMAP of combined teacher leaf embeddings [L0, L1]", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / "umap_leaf_embeddings.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------

def _write_teacher_parquets(
    y_train: np.ndarray, y_val: np.ndarray,
    p0_train: np.ndarray, p0_val: np.ndarray,
    logit_p0_train: np.ndarray, logit_p0_val: np.ndarray,
    v_train: np.ndarray, v_val: np.ndarray,
    V_train: np.ndarray, V_val: np.ndarray,
    H_train: np.ndarray, H_val: np.ndarray,
    y_soft_train: np.ndarray, y_soft_val: np.ndarray,
) -> None:
    # Snapshots the teacher at LAMBDA_HEADLINE only — not one file per lambda.
    train_df = pd.DataFrame({
        "row_id": np.arange(len(y_train)),
        "y": y_train,
        "p0": p0_train,
        "logit_p0": logit_p0_train,
        "p1_residual": v_train,
        "p1_residual_z": V_train,
        "H": H_train,
        "y_soft": y_soft_train,
    })
    val_df = pd.DataFrame({
        "row_id": np.arange(len(y_val)),
        "y": y_val,
        "p0": p0_val,
        "logit_p0": logit_p0_val,
        "p1_residual": v_val,
        "p1_residual_z": V_val,
        "H": H_val,
        "y_soft": y_soft_val,
    })
    train_df.to_parquet(OUT_DIR / "teacher_train.parquet", index=False)
    val_df.to_parquet(OUT_DIR / "teacher_val.parquet", index=False)
    log.info("saved teacher_train.parquet / teacher_val.parquet (lambda=%.2f)", LAMBDA_HEADLINE)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(df: pd.DataFrame, p0_train_deg: dict, p0_val_deg: dict) -> None:
    report_path = OUT_DIR / "README.md"

    headline = df[df["lambda"] == LAMBDA_HEADLINE].iloc[0]
    best_r2_row = df.loc[df["student_r2_val"].idxmax()]

    lines: list[str] = [
        "# Teacher/Student Soft-Label Protocol — Report",
        "",
        "> Generated by `experiments/teacher_student_soft_labels/run_experiment.py`",
        "",
        "---",
        "",
        "## Experimental setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| DGP | GaussianBinaryDGP |",
        f"| p_pos | {P_POS} |",
        "| Features | x1 (info=1.2), x2 (0.8), x3 (0.4), x4 (0.15), x5 (0.05) |",
        f"| n_train / n_val | {N_TRAIN:,} / {N_VAL:,} |",
        f"| Lambdas | {', '.join(str(lam) for lam in LAMBDAS)} |",
        f"| Headline lambda | {LAMBDA_HEADLINE} |",
        f"| p0 config | CatBoostClassifier(depth={P0_CFG['depth']}, iter={P0_CFG['iterations']}, "
        f"lr={P0_CFG['learning_rate']}, l2={P0_CFG['l2_leaf_reg']}) |",
        f"| p1 config | CatBoostRegressor(depth={P1_CFG['depth']}, iter={P1_CFG['iterations']}, "
        f"lr={P1_CFG['learning_rate']}, l2={P1_CFG['l2_leaf_reg']}) |",
        f"| student config | CatBoostRegressor(depth={STUDENT_CFG['depth']}, iter={STUDENT_CFG['iterations']}, "
        f"lr={STUDENT_CFG['learning_rate']}) |",
        f"| SVD components | {N_SVD_COMPONENTS} |",
        f"| umap-learn installed | {_HAS_UMAP} |",
        "",
        "---",
        "",
        "## Formula",
        "",
        "```",
        "p0(x)      = high-reg CatBoostClassifier(x), fit on df_train",
        "H0(x)      = logit(p0(x))",
        "r_train    = y_train - p0(X_train)",
        "p1(x)      = CatBoostRegressor(x, r_train), fit on df_train",
        "V(x)       = (p1(x) - mu_v) / sigma_v            (mu_v, sigma_v from train only)",
        "H(x)       = logit(p0(x)) + lambda * V(x)",
        "F_train    = EmpiricalCDF().fit(H_train)          (fit ONLY on H_train)",
        "y_soft(x)  = F_train.transform(H(x))               (applied to train AND val)",
        "```",
        "",
        "**Leakage discipline**: `df_val` is only ever passed to `.predict()` / `.transform()` — ",
        "never `.fit()` — for `p0`, `p1`, the leaf one-hot encoder, `TruncatedSVD`, `UMAP`, ",
        "`mu_v`/`sigma_v`, and `EmpiricalCDF`. All fitted objects come exclusively from `df_train`.",
        "",
        "---",
        "",
        "## 1. Degeneracy reduction",
        "",
        "**Caveat**: this DGP (`GaussianBinaryDGP`) draws continuous Gaussian features, so `p0` is ",
        "already almost always unique at float64 precision even before any perturbation — literal ",
        "ties only occur when two rows land in the exact same leaf of every tree in `p0`. The ",
        "`unique_ratio` metrics below are therefore expected to already sit near 1.0 for `p0` in this ",
        "synthetic setup; the degeneracy-lifting value of `H`/`p1` is expected to matter far more with ",
        "discrete/categorical or heavily quantized features, where raw scores collapse onto a small ",
        "set of repeated values.",
        "",
        f"Baseline `p0` unique-value ratio: train={p0_train_deg['unique_ratio']:.4f} "
        f"({p0_train_deg['n_unique']}/{N_TRAIN}), val={p0_val_deg['unique_ratio']:.4f} "
        f"({p0_val_deg['n_unique']}/{N_VAL}).",
        "",
        f"At λ={LAMBDA_HEADLINE}: unique(H_train)={int(headline['H_train_n_unique'])} vs "
        f"unique(p0_train)={p0_train_deg['n_unique']} — "
        f"{'✓ H is more granular than p0' if headline['H_train_n_unique'] > p0_train_deg['n_unique'] else '⚠ H did not increase granularity over p0'}.",
        "",
        "![degeneracy unique ratio](degeneracy_unique_ratio.png)",
        "",
        "---",
        "",
        "## 2. Distribution flatness",
        "",
        "Train `y_soft` is the empirical CDF of `H_train` by construction, so it should be close ",
        "to uniform (see KS-vs-uniform p-value in each panel title below). Val `y_soft` need not ",
        "be uniform — that is expected and informative (it reflects whatever `H_val`'s distribution ",
        "actually is relative to the train-fit CDF), not a bug.",
        "",
    ]
    for lam in LAMBDAS:
        lines += [f"![score distributions lambda={lam}](score_distributions_lambda{lam}.png)", ""]

    lines += [
        "---",
        "",
        "## 3. Train/val shift in teacher space",
        "",
        "| lambda | mean H_train | mean H_val | std H_train | std H_val | KS stat | KS p-value |",
        "|-------:|-------------:|-----------:|------------:|----------:|--------:|-----------:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['mean_H_train']:.4f} | {r['mean_H_val']:.4f} "
            f"| {r['std_H_train']:.4f} | {r['std_H_val']:.4f} "
            f"| {r['ks_H_train_vs_val_stat']:.4f} | {r['ks_H_train_vs_val_pvalue']:.4f} |"
        )
    lines.append("")
    for lam in LAMBDAS:
        lines += [f"![H train vs val shift lambda={lam}](h_train_val_shift_lambda{lam}.png)", ""]

    lines += [
        "---",
        "",
        "## 4. Perturbation strength sweep",
        "",
        "| lambda | Spearman(p0,H) train | Spearman(p0,H) val | unique_ratio(H) train | "
        "unique_ratio(H) val | KS-uniform p (y_soft_train) | Student R² val | "
        "AP(y_soft,y) val | AUC(y_soft,y) val |",
        "|-------:|----------------------:|--------------------:|------------------------:|"
        "----------------------:|------------------------------:|----------------:|"
        "-----------------:|-------------------:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['spearman_p0_H_train']:.4f} | {r['spearman_p0_H_val']:.4f} "
            f"| {r['H_train_unique_ratio']:.4f} | {r['H_val_unique_ratio']:.4f} "
            f"| {r['ks_uniform_y_soft_train_pvalue']:.4f} | {r['student_r2_val']:.4f} "
            f"| {r['ap_ysoft_y_val']:.4f} | {r['auc_ysoft_y_val']:.4f} |"
        )
    lines += [
        "",
        "![lambda sweep diagnostics](lambda_sweep_diagnostics.png)",
        "",
        "---",
        "",
        "## 5. Student learnability",
        "",
        "| lambda | R² | MAE | RMSE | Spearman | Kendall | top-1% overlap | top-5% overlap | top-10% overlap |",
        "|-------:|---:|----:|-----:|---------:|--------:|----------------:|----------------:|-----------------:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['student_r2_val']:.4f} | {r['student_mae_val']:.4f} "
            f"| {r['student_rmse_val']:.4f} | {r['student_spearman_val']:.4f} "
            f"| {r['student_kendall_val']:.4f} | {r['top_1pct_overlap']:.4f} "
            f"| {r['top_5pct_overlap']:.4f} | {r['top_10pct_overlap']:.4f} |"
        )
    lines += [
        "",
        f"![student fit scatter lambda={LAMBDA_HEADLINE}](student_fit_scatter_lambda{LAMBDA_HEADLINE}.png)",
        "",
        "![topk overlap vs lambda](topk_overlap_vs_lambda.png)",
        "",
        "---",
        "",
        "## 6. Relation to real label",
        "",
        "| lambda | AUC(p0,y) | AP(p0,y) | AUC(H,y) | AP(H,y) | AUC(y_soft,y) | AP(y_soft,y) |",
        "|-------:|----------:|---------:|---------:|--------:|---------------:|---------------:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['auc_p0_y_val']:.4f} | {r['ap_p0_y_val']:.4f} "
            f"| {r['auc_H_y_val']:.4f} | {r['ap_H_y_val']:.4f} "
            f"| {r['auc_ysoft_y_val']:.4f} | {r['ap_ysoft_y_val']:.4f} |"
        )
    decile_cols = [f"event_rate_decile_{i}" for i in range(10)]
    decile_vals = [headline[c] for c in decile_cols]
    lines += [
        "",
        f"Event rate by `y_soft_val` decile at λ={LAMBDA_HEADLINE} (decile 1 = lowest y_soft, "
        "decile 10 = highest):",
        "",
        "| " + " | ".join(f"D{i+1}" for i in range(10)) + " |",
        "|" + "---:|" * 10,
        "| " + " | ".join(f"{v:.3f}" if pd.notna(v) else "NA" for v in decile_vals) + " |",
        "",
        f"![event rate by decile lambda={LAMBDA_HEADLINE}](event_rate_by_decile_lambda{LAMBDA_HEADLINE}.png)",
        "",
        "![AUC AP vs lambda](auc_ap_vs_lambda.png)",
        "",
        "---",
        "",
        "## UMAP visualization",
        "",
    ]
    if _HAS_UMAP:
        lines += [
            "Combined teacher leaf embedding `[L0, L1]` (one-hot -> TruncatedSVD -> UMAP, all fit ",
            "on train only). Train panel colored by hard label `y`; val panel colored by `y_soft_val`.",
            "",
            "![umap leaf embeddings](umap_leaf_embeddings.png)",
            "",
        ]
    else:
        lines += [
            "umap-learn not installed — UMAP figure skipped. Install with `pip install umap-learn` ",
            "(or `uv sync --extra umap`) and rerun to generate `umap_leaf_embeddings.png`.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Headline lambda recommendation",
        "",
        f"At the proposed headline λ={LAMBDA_HEADLINE}: Spearman(p0,H) val = "
        f"{headline['spearman_p0_H_val']:.4f}, student R² val = {headline['student_r2_val']:.4f}, "
        f"AP(y_soft,y) val = {headline['ap_ysoft_y_val']:.4f}. The best student R² across the sweep ",
        f"occurred at λ={best_r2_row['lambda']:.2f} (R²={best_r2_row['student_r2_val']:.4f}). ",
        (
            f"This supports λ={LAMBDA_HEADLINE} as a reasonable default — it is within the swept range "
            "and does not sacrifice student learnability relative to the best-observed lambda."
            if abs(best_r2_row["lambda"] - LAMBDA_HEADLINE) <= 0.05
            else f"Consider λ={best_r2_row['lambda']:.2f} instead if student learnability is the "
            "primary criterion — inspect the full sweep table above before deciding."
        ),
        "",
        "---",
        "",
        "## Limitations & future work",
        "",
        "**In-sample teacher on train.** This experiment implements only the *simple* version: `p0` ",
        "and `p1` predict on the same rows they were fit on to build `y_soft_train`. This risks ",
        "`H_train` / `y_soft_train` being overly confident or overfit relative to what the teacher ",
        "would produce on genuinely unseen rows, since in-sample residuals are systematically smaller ",
        "than out-of-sample residuals.",
        "",
        "**Stronger version (not implemented here): out-of-fold (OOF) teacher.** Within `df_train`, ",
        "run K-fold cross-validation: for each fold, fit `p0`/`p1` on the other folds and predict on ",
        "the held-out fold, stitching together `H_OOF` across all folds. Fit `F_train` on `H_OOF` ",
        "instead of the in-sample `H_train`, giving `y_soft_train = F_train(H_OOF)`. For `df_val`, ",
        "still use the FINAL `p0`/`p1` models fit on all of `df_train`, transformed through the same ",
        "train-fitted `F_train` — this part is unchanged from the simple version. The OOF variant is ",
        "documented here as the recommended next step if the simple version's results look ",
        "promising (or suspiciously good) on `df_train`.",
        "",
        "---",
        "",
        f"Raw data: `results.csv`, `teacher_train.parquet`, `teacher_val.parquet` (lambda={LAMBDA_HEADLINE})",
    ]

    report_path.write_text("\n".join(lines) + "\n")
    log.info("saved %s", report_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _HAS_CATBOOST:
        log.error("catboost not installed. Run: pip install catboost")
        sys.exit(1)
    if not _HAS_PYARROW:
        log.error(
            "pyarrow not installed (required for parquet output). "
            "Run: uv sync --extra catboost --extra parquet"
        )
        sys.exit(1)
    if not _HAS_UMAP:
        log.warning("umap-learn not installed — UMAP figures will be skipped. "
                    "Install with: uv sync --extra umap")

    log.info("Generating data — n_train=%d  n_val=%d  p_pos=%.2f", N_TRAIN, N_VAL, P_POS)
    df_train = DGP.sample(n=N_TRAIN, seed=SEED)
    df_val = DGP.sample(n=N_VAL, seed=SEED + 1_000)

    y_train = df_train["y"].values
    X_train = df_train.drop(columns="y").values
    y_val = df_val["y"].values
    X_val = df_val.drop(columns="y").values

    log.info("Train positives: %d / %d (%.1f%%)", y_train.sum(), N_TRAIN, 100 * y_train.mean())

    # ---- lambda-independent: fit ONCE ----
    log.info("fitting p0 (coarse classifier) …")
    p0 = _fit_p0(X_train, y_train)
    p0_train = p0.predict_proba(X_train)[:, 1]
    p0_val = p0.predict_proba(X_val)[:, 1]  # predict only, TRAIN-fit model
    logit_p0_train = _logit(p0_train)
    logit_p0_val = _logit(p0_val)

    r_train = y_train - p0_train
    log.info("fitting p1 (residual regressor) …")
    p1 = _fit_p1(X_train, r_train)
    v_train = p1.predict(X_train)
    v_val = p1.predict(X_val)  # predict only

    mu_v, sigma_v = float(v_train.mean()), float(v_train.std())  # TRAIN ONLY
    V_train = _zscore(v_train, mu_v, sigma_v)
    V_val = _zscore(v_val, mu_v, sigma_v)

    log.info("extracting leaf embeddings from p0 and p1 …")
    L0_train, L0_val = _leaf_embeddings(p0, X_train), _leaf_embeddings(p0, X_val)
    L1_train, L1_val = _leaf_embeddings(p1, X_train), _leaf_embeddings(p1, X_val)
    L_train = np.hstack([L0_train, L1_train])
    L_val = np.hstack([L0_val, L1_val])

    log.info("building leaf one-hot -> SVD -> UMAP pipeline …")
    _, _, _, _, Z_train_umap, Z_val_umap = _leaf_umap_pipeline(L_train, L_val)

    p0_train_deg = _tie_stats(p0_train)
    p0_val_deg = _tie_stats(p0_val)

    rows: list[dict] = []
    headline_payload: dict | None = None

    for lam in LAMBDAS:
        log.info("=== lambda=%.2f ===", lam)
        H_train = logit_p0_train + lam * V_train
        H_val = logit_p0_val + lam * V_val

        ecdf = EmpiricalCDF(eps=EPS).fit(H_train)  # TRAIN ONLY
        y_soft_train = ecdf.transform(H_train)
        y_soft_val = ecdf.transform(H_val)  # transform only

        log.info("  fitting student …")
        student = _fit_student(X_train, y_soft_train)
        student_pred_train = student.predict(X_train)
        student_pred_val = student.predict(X_val)

        H_train_deg = _tie_stats(H_train)
        H_val_deg = _tie_stats(H_val)
        ysoft_train_deg = _tie_stats(y_soft_train)
        ysoft_val_deg = _tie_stats(y_soft_val)
        studpred_train_deg = _tie_stats(student_pred_train)
        studpred_val_deg = _tie_stats(student_pred_val)

        if H_train_deg["n_unique"] <= p0_train_deg["n_unique"]:
            log.warning("  unique(H_train)=%d did not exceed unique(p0_train)=%d",
                        H_train_deg["n_unique"], p0_train_deg["n_unique"])

        ks_shift = ks_2samp(H_train, H_val)
        ks_uniform = kstest(y_soft_train, "uniform")

        student_metrics = _student_metrics(y_soft_val, student_pred_val)

        row = {
            "lambda": lam,
            "p0_train_n_unique": p0_train_deg["n_unique"],
            "p0_train_unique_ratio": p0_train_deg["unique_ratio"],
            "p0_val_n_unique": p0_val_deg["n_unique"],
            "p0_val_unique_ratio": p0_val_deg["unique_ratio"],
            "H_train_n_unique": H_train_deg["n_unique"],
            "H_train_unique_ratio": H_train_deg["unique_ratio"],
            "H_val_n_unique": H_val_deg["n_unique"],
            "H_val_unique_ratio": H_val_deg["unique_ratio"],
            "y_soft_train_n_unique": ysoft_train_deg["n_unique"],
            "y_soft_train_unique_ratio": ysoft_train_deg["unique_ratio"],
            "y_soft_val_n_unique": ysoft_val_deg["n_unique"],
            "y_soft_val_unique_ratio": ysoft_val_deg["unique_ratio"],
            "student_pred_train_n_unique": studpred_train_deg["n_unique"],
            "student_pred_train_unique_ratio": studpred_train_deg["unique_ratio"],
            "student_pred_val_n_unique": studpred_val_deg["n_unique"],
            "student_pred_val_unique_ratio": studpred_val_deg["unique_ratio"],
            "mean_H_train": float(H_train.mean()),
            "mean_H_val": float(H_val.mean()),
            "std_H_train": float(H_train.std()),
            "std_H_val": float(H_val.std()),
            "ks_H_train_vs_val_stat": float(ks_shift.statistic),
            "ks_H_train_vs_val_pvalue": float(ks_shift.pvalue),
            "spearman_p0_H_train": float(spearmanr(p0_train, H_train).statistic),
            "spearman_p0_H_val": float(spearmanr(p0_val, H_val).statistic),
            "ks_uniform_y_soft_train_stat": float(ks_uniform.statistic),
            "ks_uniform_y_soft_train_pvalue": float(ks_uniform.pvalue),
            "student_r2_val": student_metrics["r2"],
            "student_mae_val": student_metrics["mae"],
            "student_rmse_val": student_metrics["rmse"],
            "student_spearman_val": student_metrics["spearman"],
            "student_kendall_val": student_metrics["kendall"],
            "top_1pct_overlap": _topk_overlap(y_soft_val, student_pred_val, 0.01),
            "top_5pct_overlap": _topk_overlap(y_soft_val, student_pred_val, 0.05),
            "top_10pct_overlap": _topk_overlap(y_soft_val, student_pred_val, 0.10),
            "auc_p0_y_val": float(roc_auc_score(y_val, p0_val)),
            "ap_p0_y_val": float(average_precision_score(y_val, p0_val)),
            "auc_H_y_val": float(roc_auc_score(y_val, H_val)),
            "ap_H_y_val": float(average_precision_score(y_val, H_val)),
            "auc_ysoft_y_val": float(roc_auc_score(y_val, y_soft_val)),
            "ap_ysoft_y_val": float(average_precision_score(y_val, y_soft_val)),
        }

        decile_rates = _decile_event_rate(y_soft_val, y_val)
        for i, rate in enumerate(decile_rates):
            row[f"event_rate_decile_{i}"] = rate

        rows.append(row)

        log.info("  unique(p0->H->y_soft->student_pred) train: %d->%d->%d->%d  val: %d->%d->%d->%d",
                 p0_train_deg["n_unique"], H_train_deg["n_unique"], ysoft_train_deg["n_unique"],
                 studpred_train_deg["n_unique"],
                 p0_val_deg["n_unique"], H_val_deg["n_unique"], ysoft_val_deg["n_unique"],
                 studpred_val_deg["n_unique"])
        log.info("  student R2 val=%.4f  AP(y_soft,y) val=%.4f", student_metrics["r2"], row["ap_ysoft_y_val"])

        _plot_score_distributions(lam, p0_train, H_train, y_soft_train, y_soft_val, y_train,
                                  row["ks_uniform_y_soft_train_pvalue"])
        _plot_h_train_val_shift(lam, H_train, H_val, row["ks_H_train_vs_val_stat"],
                                row["ks_H_train_vs_val_pvalue"])

        if lam == LAMBDA_HEADLINE:
            headline_payload = dict(
                H_train=H_train, H_val=H_val,
                y_soft_train=y_soft_train, y_soft_val=y_soft_val,
                student_pred_val=student_pred_val,
            )
            _plot_student_fit_scatter(lam, y_soft_val, student_pred_val,
                                      student_metrics["r2"], student_metrics["spearman"])
            _plot_event_rate_decile(lam, decile_rates)

    results_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("saved %s", csv_path.name)

    _plot_degeneracy_unique_ratio(results_df, p0_train_deg["unique_ratio"], p0_val_deg["unique_ratio"])
    _plot_lambda_sweep(results_df)
    _plot_topk_overlap(results_df)
    _plot_auc_ap_vs_lambda(results_df)

    if _HAS_UMAP and Z_train_umap is not None:
        _plot_umap(Z_train_umap, y_train, Z_val_umap, headline_payload["y_soft_val"])
    else:
        log.warning("skipping UMAP figure — umap-learn not installed")

    assert headline_payload is not None, "LAMBDA_HEADLINE must be a member of LAMBDAS"
    _write_teacher_parquets(
        y_train, y_val, p0_train, p0_val, logit_p0_train, logit_p0_val,
        v_train, v_val, V_train, V_val,
        headline_payload["H_train"], headline_payload["H_val"],
        headline_payload["y_soft_train"], headline_payload["y_soft_val"],
    )

    _write_report(results_df, p0_train_deg, p0_val_deg)

    log.info("\n=== Summary ===")
    print(results_df[[
        "lambda", "spearman_p0_H_val", "H_val_unique_ratio", "student_r2_val",
        "ap_ysoft_y_val", "auc_ysoft_y_val",
    ]].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
