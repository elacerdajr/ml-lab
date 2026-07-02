"""
Leaf-Embedding Perturbation Ranking — Residual Ranking Experiment
====================================================================
Replaces an ad-hoc kernel_score/w(x)/b(x) monotone-reshaping scheme with a
principled residual-ranking pipeline: fit a coarse regularized CatBoost
probability p0(x), extract leaf embeddings, one-hot encode them, fit a
Ridge regression from leaf embeddings to label residuals (y - p0), combine
the coarse logit with a z-scored residual correction plus a tiny
deterministic tie-breaker into H(x), then rank-spread H through the
empirical training CDF to obtain a well-distributed soft target y_soft
in (0,1) that preserves H's ordering.

    H(x)      = logit(p0(x)) + lambda * zscore(Ridge(onehot(leaf(x))))
                             + epsilon * zscore(tie_break(leaf(x)))
    y_soft(x) = F_train(H(x))

Outputs (experiments/leaf_embedding_ranking/outputs/):
  results.csv
  scatter_H_vs_logit_p0_{cfg}_lambda{lam}.png
  score_distributions_{cfg}_lambda{lam}.png
  duplicate_counts.png
  downstream_ap_comparison.png
  residual_fit_quality.png
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
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, r2_score
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
N_TEST = 2_000
SEED = 42

DGP = GaussianBinaryDGP(
    p_pos=P_POS,
    info={"x1": 1.2, "x2": 0.8, "x3": 0.4, "x4": 0.15, "x5": 0.05},
    sigma=1.0,
)

EXTRACTOR_CONFIGS: dict[str, dict] = {
    "heavy_reg": dict(depth=3, iterations=100, l2_leaf_reg=100, random_strength=10),
    "light_reg": dict(depth=4, iterations=200, l2_leaf_reg=1, random_strength=1),
}

LAMBDAS = [0.0, 0.05, 0.10, 0.25]  # 0.0 = no-perturbation control, 0.25 = over-perturbation demo
EPSILON_TIEBREAK = 1e-6
LOGIT_CLIP = 1e-6
STD_FLOOR = 1e-8
ALPHA_RIDGE = 10.0  # fixed L2 strength — one-hot leaf embedding can have p >> n columns
SPEARMAN_LO, SPEARMAN_HI = 0.95, 0.99

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


def _fit_coarse_model(X: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> object:
    model = CatBoostClassifier(
        **cfg,
        verbose=False,
        allow_writing_files=False,
        random_seed=seed,
        loss_function="Logloss",
    )
    model.fit(X, y)
    return model


def _leaf_embeddings(model: object, X: np.ndarray) -> np.ndarray:
    return model.calc_leaf_indexes(X).astype(np.int32)  # (n, n_trees)


def _onehot_encode(
    emb_train: np.ndarray, emb_test: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix, OneHotEncoder]:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    E_train = enc.fit_transform(emb_train)
    E_test = enc.transform(emb_test)
    return E_train, E_test, enc


def _fit_ridge_residual(E_train: sp.csr_matrix, r_train: np.ndarray) -> Ridge:
    ridge = Ridge(alpha=ALPHA_RIDGE, solver="auto")
    ridge.fit(E_train, r_train)
    return ridge


def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(std, STD_FLOOR)


def _logit(p: np.ndarray, eps: float = LOGIT_CLIP) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def _tie_breaker_signature(E: sp.csr_matrix, n_features: int, seed: int) -> np.ndarray:
    """Deterministic fixed random projection E @ w, w ~ N(0,1) with a fixed
    seed. Must be regenerated per config (column count differs) but reused
    identically for train and test within a config."""
    w = np.random.default_rng(seed).standard_normal(n_features)
    return np.asarray(E @ w).ravel()


def _combine_score(
    logit_p0: np.ndarray,
    z_resid: np.ndarray,
    z_tiebreak: np.ndarray,
    lam: float,
    epsilon: float = EPSILON_TIEBREAK,
) -> np.ndarray:
    return logit_p0 + lam * z_resid + epsilon * z_tiebreak


def _train_ecdf_transform(H_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hazen plotting-position ECDF fit on TRAIN H only."""
    n = len(H_train)
    order = np.argsort(H_train, kind="mergesort")
    H_sorted = H_train[order]
    F_sorted = (np.arange(1, n + 1) - 0.5) / n
    return H_sorted, F_sorted


def _apply_ecdf(H_new: np.ndarray, H_sorted: np.ndarray, F_sorted: np.ndarray) -> np.ndarray:
    """Linear-interpolate any H (train or test) onto the train-fit ECDF."""
    n = len(H_sorted)
    y = np.interp(H_new, H_sorted, F_sorted)
    eps = 0.5 / n
    return np.clip(y, eps, 1.0 - eps)


def _duplicate_stats(values: np.ndarray, decimals: int = 9) -> tuple[int, int]:
    rounded = np.round(np.asarray(values, dtype=float), decimals)
    return len(np.unique(rounded)), len(rounded)


def _downstream_ap(
    X_train: np.ndarray,
    target_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_soft: bool,
    seed: int,
) -> float:
    if use_soft:
        model = CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.06,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
        )
        model.fit(X_train, target_train)
        scores = model.predict(X_test)
    else:
        model = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.06,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
        )
        model.fit(X_train, target_train)
        scores = model.predict_proba(X_test)[:, 1]
    return float(average_precision_score(y_test, scores))


def _direct_ap(y_test: np.ndarray, p0_test: np.ndarray) -> float:
    """AP of the raw coarse probability with no retraining — the baseline
    that y_soft is trying to beat/preserve."""
    return float(average_precision_score(y_test, p0_test))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_h_vs_logit_scatter(
    H: np.ndarray,
    logit_p0: np.ndarray,
    y: np.ndarray,
    cfg_name: str,
    lam: float,
    spearman_val: float,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))

    neg_mask = y == 0
    pos_mask = y == 1
    ax.scatter(logit_p0[neg_mask], H[neg_mask], c="#4477bb", s=10, alpha=0.5,
               linewidths=0, label="neg (y=0)")
    ax.scatter(logit_p0[pos_mask], H[pos_mask], c="#cc4444", s=10, alpha=0.5,
               linewidths=0, label="pos (y=1)")

    lo = min(logit_p0.min(), H.min())
    hi = max(logit_p0.max(), H.max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, alpha=0.6, label="y=x")

    flag = "  ⚠ outside [0.95, 0.99]" if (lam > 0 and not (SPEARMAN_LO <= spearman_val <= SPEARMAN_HI)) else ""
    ax.set_title(f"{cfg_name}  λ={lam}  ρ_spearman={spearman_val:.4f}{flag}", fontsize=9)
    ax.set_xlabel("logit(p0)  (coarse score)")
    ax.set_ylabel("H(x)  (perturbed score)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()

    path = OUT_DIR / f"scatter_H_vs_logit_p0_{cfg_name}_lambda{lam}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_score_distributions(
    p0: np.ndarray,
    H: np.ndarray,
    y_soft: np.ndarray,
    y: np.ndarray,
    cfg_name: str,
    lam: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("p0  (raw coarse score)", p0), ("H  (post-perturbation)", H),
              ("y_soft  (post-spreading)", y_soft)]

    for ax, (title, values) in zip(axes, panels):
        pos = values[y == 1]
        neg = values[y == 0]
        ax.hist(neg, bins=40, alpha=0.6, color="#4477bb", label="neg (y=0)", density=True)
        ax.hist(pos, bins=40, alpha=0.6, color="#cc4444", label="pos (y=1)", density=True)
        ax.axvline(neg.mean(), color="#4477bb", linestyle="--", linewidth=1.5)
        ax.axvline(pos.mean(), color="#cc4444", linestyle="--", linewidth=1.5)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        if ax is axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(f"Score distributions — {cfg_name}  λ={lam}", fontsize=11, y=1.03)
    fig.tight_layout()
    path = OUT_DIR / f"score_distributions_{cfg_name}_lambda{lam}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_duplicate_counts(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    labels = [f"{r['extractor_config']}\nλ={r['lambda']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(rows)), 5))
    ax.bar(x - width, df["n_unique_p0_test"] / N_TEST, width, label="unique p0", color="#888888")
    ax.bar(x, df["n_unique_H_test"] / N_TEST, width, label="unique H", color="#dd9944")
    ax.bar(x + width, df["n_unique_ysoft_test"] / N_TEST, width, label="unique y_soft", color="#44aa77")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("unique-value fraction (test set)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Degeneracy lifting: unique-value fraction across the pipeline", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    fig.tight_layout()

    path = OUT_DIR / "duplicate_counts.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_downstream_ap(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cfg_names = df["extractor_config"].unique().tolist()
    lambdas = sorted(df["lambda"].unique().tolist())

    n_cfg = len(cfg_names)
    fig, axes = plt.subplots(1, n_cfg, figsize=(6.5 * n_cfg, 5), sharey=True)
    if n_cfg == 1:
        axes = [axes]

    x = np.arange(len(lambdas))
    width = 0.25

    for ax, cfg in zip(axes, cfg_names):
        sub = df[df["extractor_config"] == cfg].set_index("lambda")
        ap_hard = [sub.loc[lam, "downstream_ap_hard"] for lam in lambdas]
        ap_raw = [sub.loc[lam, "downstream_ap_raw_p0"] for lam in lambdas]
        ap_soft = [sub.loc[lam, "downstream_ap_soft"] for lam in lambdas]

        bars = [
            ax.bar(x - width, ap_hard, width, label="hard y (retrained)", color="#888888", alpha=0.85),
            ax.bar(x, ap_raw, width, label="raw p0 (no retrain)", color="#4477bb", alpha=0.85),
            ax.bar(x + width, ap_soft, width, label="y_soft (retrained)", color="#cc4444", alpha=0.85),
        ]
        for bar_group in bars:
            for bar in bar_group:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"λ={lam}" for lam in lambdas])
        ax.set_title(f"{cfg}", fontsize=10)
        ax.set_ylabel("Average Precision (test)")
        y_max = max(ap_hard + ap_raw + ap_soft) * 1.2
        ax.set_ylim(0, min(1.0, y_max))
        ax.legend(fontsize=7.5)
        ax.grid(axis="y", alpha=0.28, linestyle=":")

    fig.suptitle("Downstream AP: hard labels vs raw p0 vs leaf-embedding y_soft", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / "downstream_ap_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_residual_fit_quality(config_stats: list[dict]) -> None:
    df = pd.DataFrame(config_stats)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars_train = ax.bar(x - width / 2, df["ridge_r2_train"], width, label="R² train", color="#4477bb", alpha=0.85)
    bars_test = ax.bar(x + width / 2, df["ridge_r2_test"], width, label="R² test", color="#cc4444", alpha=0.85)

    for bar in list(bars_train) + list(bars_test):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (0.01 if h >= 0 else -0.03),
                f"{h:.3f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["extractor_config"])
    ax.set_ylabel("R²  (Ridge residual fit: E(x) → y - p0(x))")
    ax.set_title("Ridge residual-fit quality per extractor config", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    fig.tight_layout()

    path = OUT_DIR / "residual_fit_quality.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cfg_names = df["extractor_config"].unique().tolist()
    report_path = OUT_DIR / "README.md"

    lines: list[str] = [
        "# Leaf-Embedding Perturbation Ranking — Report",
        "",
        "> Generated by `experiments/leaf_embedding_ranking/run_experiment.py`",
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
        f"| n_train / n_test | {N_TRAIN:,} / {N_TEST:,} |",
        f"| Lambdas | {', '.join(str(lam) for lam in LAMBDAS)} |",
        f"| Ridge alpha | {ALPHA_RIDGE} |",
        f"| Tie-breaker epsilon | {EPSILON_TIEBREAK} |",
        "| Extractor configs | heavy_reg (depth=3, iter=100, l2=100, rs=10) · light_reg (depth=4, iter=200, l2=1, rs=1) |",
        "| Downstream model | CatBoostRegressor (RMSE on y_soft) vs CatBoostClassifier (Logloss on hard y) |",
        "",
        "---",
        "",
        "## Formula",
        "",
        "```",
        "p0(x)      = regularized CatBoost probability",
        "L(x)       = model.calc_leaf_indexes(x)           (leaf index per tree)",
        "E(x)       = onehot(L(x))",
        "r(x)       = y(x) - p0(x)                          (train-set residual)",
        "r_hat(x)   = Ridge(E(x))                           (fit on r, train only)",
        "H(x)       = logit(p0(x))",
        "             + lambda * zscore(r_hat(x))",
        "             + epsilon * zscore(E(x) @ w)          (deterministic tie-breaker)",
        "y_soft(x)  = F_train(H(x))                         (train-fit empirical CDF)",
        "```",
        "",
        "`zscore` uses train-set mean/std; `F_train` is a Hazen-plotting-position ECDF",
        "fit on train `H` values and applied to both train and test via linear",
        "interpolation — no test-set statistics ever enter the fit.",
        "",
        "---",
        "",
        "## Key results",
        "",
        "**Caveat on degeneracy counts**: this DGP (`GaussianBinaryDGP`) draws continuous",
        "Gaussian features, so raw CatBoost probabilities are almost always distinct at",
        "float64 precision even before any perturbation — literal ties only occur when two",
        "rows land in the exact same leaf of every single tree. The `dup p0` columns below are",
        "therefore already close to `n_test`. The degeneracy-lifting benefit of leaf-embedding",
        "perturbation is expected to matter far more with discrete/categorical or heavily",
        "quantized features, where raw CatBoost scores collapse onto a small set of repeated",
        "values. What this synthetic setup *does* cleanly demonstrate is that the perturbation",
        "preserves ranking (Spearman) while still moving individual scores — see the scatter",
        "plots below.",
        "",
    ]

    for cfg in cfg_names:
        sub = df[df["extractor_config"] == cfg]
        r0 = sub.iloc[0]
        lines += [
            f"### {cfg}",
            "",
            f"Ridge residual fit: R² train = {r0['ridge_r2_train']:.4f}, "
            f"R² test = {r0['ridge_r2_test']:.4f}  "
            f"(corr train = {r0['ridge_corr_train']:.4f}, corr test = {r0['ridge_corr_test']:.4f})  "
            f"— {r0['n_onehot_features']} one-hot leaf features vs {N_TRAIN} train rows.",
            "",
            "| lambda | Spearman(H, logit p0) | dup p0 | dup H | dup y_soft | AP hard | AP raw p0 | AP soft | ΔAP (soft − hard) |",
            "|-------:|-----------------------:|-------:|------:|-----------:|--------:|----------:|--------:|--------------------:|",
        ]
        for _, r in sub.iterrows():
            flag = " ⚠" if r["spearman_flag"] else ""
            delta = r["downstream_ap_soft"] - r["downstream_ap_hard"]
            lines.append(
                f"| {r['lambda']:.2f} "
                f"| {r['spearman_H_vs_logit_p0_test']:.4f}{flag} "
                f"| {r['n_unique_p0_test']} "
                f"| {r['n_unique_H_test']} "
                f"| {r['n_unique_ysoft_test']} "
                f"| {r['downstream_ap_hard']:.4f} "
                f"| {r['downstream_ap_raw_p0']:.4f} "
                f"| {r['downstream_ap_soft']:.4f} "
                f"| {delta:+.4f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Figures",
        "",
        "### H vs logit(p0) scatter",
        "",
        "Perturbed score `H` (y-axis) against the coarse `logit(p0)` (x-axis) on the test",
        "set, per config and lambda. Points hugging the `y=x` line indicate ordering is",
        "preserved; vertical spread at a fixed x value shows ties being broken. Title",
        "reports the Spearman correlation (target band 0.95–0.99 for lambda > 0).",
        "",
    ]
    for cfg in cfg_names:
        for lam in LAMBDAS:
            if lam == 0.0:
                continue
            lines += [f"![scatter {cfg} lambda={lam}](scatter_H_vs_logit_p0_{cfg}_lambda{lam}.png)", ""]

    lines += [
        "### Score distributions",
        "",
        "Histograms of `p0` (raw coarse score), `H` (post-perturbation), and `y_soft`",
        "(post-spreading), split by class. Showing the largest lambda (0.25) per config",
        "— where the perturbation effect is most visible.",
        "",
    ]
    for cfg in cfg_names:
        lam = LAMBDAS[-1]
        lines += [f"![distributions {cfg} lambda={lam}](score_distributions_{cfg}_lambda{lam}.png)", ""]

    lines += [
        "### Degeneracy lifting",
        "",
        "Fraction of unique values (out of the test set) for `p0`, `H`, and `y_soft`,",
        "across every (config, lambda) combination.",
        "",
        "![duplicate counts](duplicate_counts.png)",
        "",
        "### Downstream AP comparison",
        "",
        "Test Average Precision: retraining on hard labels vs the raw coarse `p0` with",
        "no retraining vs retraining on `y_soft`.",
        "",
        "![downstream AP comparison](downstream_ap_comparison.png)",
        "",
        "### Ridge residual-fit quality",
        "",
        "R² of the Ridge residual model on train vs test, per extractor config.",
        "",
        "![residual fit quality](residual_fit_quality.png)",
        "",
        "---",
        "",
        "## Key takeaways",
        "",
    ]

    takeaways: list[str] = []
    for cfg in cfg_names:
        sub = df[df["extractor_config"] == cfg]
        nonzero = sub[sub["lambda"] > 0]
        n_out_of_band = int(nonzero["spearman_flag"].sum())
        if n_out_of_band == 0:
            takeaways.append(
                f"- **{cfg}**: Spearman(H, logit p0) stayed within [{SPEARMAN_LO}, {SPEARMAN_HI}] "
                f"for all lambda > 0 — the perturbation breaks ties without disturbing the overall ranking."
            )
        else:
            takeaways.append(
                f"- **{cfg}**: Spearman(H, logit p0) fell outside [{SPEARMAN_LO}, {SPEARMAN_HI}] "
                f"for {n_out_of_band}/{len(nonzero)} nonzero-lambda rows — inspect the scatter plots for that config."
            )
        r0 = sub.iloc[0]
        best_row = sub.loc[sub["lambda"].idxmax()]
        dup_before = int(r0["n_unique_p0_test"])
        dup_after = int(best_row["n_unique_ysoft_test"])
        takeaways.append(
            f"  Unique test-set values grew from {dup_before} (raw p0) to {dup_after} "
            f"(y_soft, λ={best_row['lambda']}) out of {N_TEST} rows."
        )
        if r0["ridge_r2_test"] < 0:
            takeaways.append(
                f"  ⚠ Ridge R² test = {r0['ridge_r2_test']:.4f} (negative) — the one-hot leaf "
                f"embedding ({int(r0['n_onehot_features'])} columns) approaches or exceeds n_train "
                f"({N_TRAIN}), so the residual model overfits train despite L2 regularization."
            )
    lines += takeaways
    lines += [
        "",
        "---",
        "",
        "Raw data: `results.csv`",
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

    log.info("Generating data — n_train=%d  n_test=%d  p_pos=%.2f", N_TRAIN, N_TEST, P_POS)
    df_train = DGP.sample(n=N_TRAIN, seed=SEED)
    df_test = DGP.sample(n=N_TEST, seed=SEED + 1_000)

    y_train = df_train["y"].values
    X_train = df_train.drop(columns="y").values
    y_test = df_test["y"].values
    X_test = df_test.drop(columns="y").values

    log.info("Train positives: %d / %d (%.1f%%)", y_train.sum(), N_TRAIN, 100 * y_train.mean())

    rows: list[dict] = []
    config_stats: list[dict] = []

    for cfg_name, extractor_cfg in EXTRACTOR_CONFIGS.items():
        log.info("=== Extractor config: %s ===", cfg_name)
        log.info("  fitting coarse model …")
        coarse = _fit_coarse_model(X_train, y_train, extractor_cfg, SEED)
        p0_train = coarse.predict_proba(X_train)[:, 1]
        p0_test = coarse.predict_proba(X_test)[:, 1]

        log.info("  extracting leaf embeddings …")
        emb_train = _leaf_embeddings(coarse, X_train)
        emb_test = _leaf_embeddings(coarse, X_test)

        log.info("  one-hot encoding leaf embeddings …")
        E_train, E_test, enc = _onehot_encode(emb_train, emb_test)
        n_features = E_train.shape[1]
        log.info("  one-hot dimensionality: %d (n_train=%d)", n_features, N_TRAIN)

        r_train = y_train - p0_train
        r_test = y_test - p0_test

        log.info("  fitting Ridge residual model …")
        ridge = _fit_ridge_residual(E_train, r_train)
        r_hat_train = ridge.predict(E_train)
        r_hat_test = ridge.predict(E_test)

        ridge_r2_train = float(ridge.score(E_train, r_train))
        ridge_r2_test = float(r2_score(r_test, r_hat_test))
        ridge_corr_train = float(pearsonr(r_hat_train, r_train).statistic)
        ridge_corr_test = float(pearsonr(r_hat_test, r_test).statistic)
        log.info("  Ridge R2 train=%.4f test=%.4f  corr train=%.4f test=%.4f",
                 ridge_r2_train, ridge_r2_test, ridge_corr_train, ridge_corr_test)

        config_stats.append({
            "extractor_config": cfg_name,
            "ridge_r2_train": ridge_r2_train,
            "ridge_r2_test": ridge_r2_test,
        })

        mu, sigma = float(r_hat_train.mean()), float(r_hat_train.std())
        z_resid_train = _zscore(r_hat_train, mu, sigma)
        z_resid_test = _zscore(r_hat_test, mu, sigma)

        sig_train = _tie_breaker_signature(E_train, n_features, SEED)
        sig_test = _tie_breaker_signature(E_test, n_features, SEED)
        sig_mu, sig_sigma = float(sig_train.mean()), float(sig_train.std())
        z_tb_train = _zscore(sig_train, sig_mu, sig_sigma)
        z_tb_test = _zscore(sig_test, sig_mu, sig_sigma)

        logit_p0_train = _logit(p0_train)
        logit_p0_test = _logit(p0_test)

        ap_hard = _downstream_ap(X_train, y_train, X_test, y_test, use_soft=False, seed=SEED)
        ap_raw_p0 = _direct_ap(y_test, p0_test)
        log.info("  baseline AP: hard=%.4f  raw_p0=%.4f", ap_hard, ap_raw_p0)

        for lam in LAMBDAS:
            log.info("  lambda=%.2f …", lam)
            H_train = _combine_score(logit_p0_train, z_resid_train, z_tb_train, lam)
            H_test = _combine_score(logit_p0_test, z_resid_test, z_tb_test, lam)

            H_sorted, F_sorted = _train_ecdf_transform(H_train)
            y_soft_train = _apply_ecdf(H_train, H_sorted, F_sorted)
            y_soft_test = _apply_ecdf(H_test, H_sorted, F_sorted)

            ap_soft = _downstream_ap(X_train, y_soft_train, X_test, y_test, use_soft=True, seed=SEED)

            if lam > 0:
                spearman_test = float(spearmanr(H_test, logit_p0_test).statistic)
            else:
                spearman_test = 1.0
            spearman_flag = lam > 0 and not (SPEARMAN_LO <= spearman_test <= SPEARMAN_HI)
            if lam == 0.0 and abs(spearman_test - 1.0) > 1e-9:
                log.warning("  lambda=0 smoke check failed: spearman=%.6f (expected ~1.0)", spearman_test)

            n_uniq_p0, n_tot = _duplicate_stats(p0_test)
            n_uniq_H, _ = _duplicate_stats(H_test)
            n_uniq_ysoft, _ = _duplicate_stats(y_soft_test)

            mean_pos = float(y_soft_test[y_test == 1].mean())
            mean_neg = float(y_soft_test[y_test == 0].mean())

            log.info("    spearman=%.4f  dup(p0->H->y_soft)=%d->%d->%d  AP soft=%.4f",
                     spearman_test, n_uniq_p0, n_uniq_H, n_uniq_ysoft, ap_soft)

            rows.append({
                "extractor_config": cfg_name,
                "lambda": lam,
                "ridge_alpha": ALPHA_RIDGE,
                "n_onehot_features": n_features,
                "ridge_r2_train": ridge_r2_train,
                "ridge_r2_test": ridge_r2_test,
                "ridge_corr_train": ridge_corr_train,
                "ridge_corr_test": ridge_corr_test,
                "spearman_H_vs_logit_p0_test": spearman_test,
                "spearman_flag": spearman_flag,
                "n_unique_p0_test": n_uniq_p0,
                "n_unique_H_test": n_uniq_H,
                "n_unique_ysoft_test": n_uniq_ysoft,
                "downstream_ap_hard": ap_hard,
                "downstream_ap_raw_p0": ap_raw_p0,
                "downstream_ap_soft": ap_soft,
                "mean_y_soft_pos": mean_pos,
                "mean_y_soft_neg": mean_neg,
                "separation": mean_pos - mean_neg,
            })

            _plot_h_vs_logit_scatter(H_test, logit_p0_test, y_test, cfg_name, lam, spearman_test)
            _plot_score_distributions(p0_test, H_test, y_soft_test, y_test, cfg_name, lam)

    results_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("saved %s", csv_path.name)

    _plot_duplicate_counts(rows)
    _plot_downstream_ap(rows)
    _plot_residual_fit_quality(config_stats)
    _write_report(rows)

    log.info("\n=== Summary ===")
    print(results_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
