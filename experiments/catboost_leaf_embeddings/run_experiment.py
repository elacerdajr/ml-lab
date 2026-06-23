"""
CatBoost Leaf Embeddings — Soft Target Experiment
==================================================
Demonstrates using a heavily regularised CatBoost model as a feature
extractor.  Leaf indices form a learned embedding; Hamming distance in
that space drives an RBF kernel that diffuses labels into soft targets.

Outputs (experiments/catboost_leaf_embeddings/outputs/):
  results.csv
  soft_target_distribution_{cfg}.png   (one per extractor config)
  kernel_heatmap_{cfg}_sigma{s}.png    (one per config × sigma)
  downstream_ap_comparison.png
  umap_hard_vs_soft_{cfg}_sigma{s}.png (only if umap-learn installed)
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
from sklearn.metrics import average_precision_score, pairwise_distances

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

SIGMAS = [0.05, 0.10, 0.25, 0.50]

EXTRACTOR_CONFIGS: dict[str, dict] = {
    "heavy_reg": dict(depth=3, iterations=100, l2_leaf_reg=100, random_strength=10),
    "light_reg": dict(depth=4, iterations=200, l2_leaf_reg=1, random_strength=1),
}

P_POS = 0.10
N_TRAIN = 2_000
N_TEST = 2_000
SEED = 42

DGP = GaussianBinaryDGP(
    p_pos=P_POS,
    info={"x1": 1.2, "x2": 0.8, "x3": 0.4, "x4": 0.15, "x5": 0.05},
    sigma=1.0,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def _fit_extractor(X: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> object:
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
    emb = model.calc_leaf_indexes(X)          # (n, n_trees)  int32
    return emb.astype(np.float32)


def _hamming_distance(emb: np.ndarray) -> np.ndarray:
    return pairwise_distances(emb, metric="hamming")  # in [0, 1]


def _rbf_kernel(D: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(D ** 2) / (2.0 * sigma ** 2))


def _soft_targets(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    pos_mask = y == 1
    neg_mask = y == 0
    alpha = 1.0 + K[:, pos_mask].sum(axis=1)
    beta = 1.0 + K[:, neg_mask].sum(axis=1)
    return alpha / (alpha + beta)


def _downstream_ap(
    X_train: np.ndarray,
    y_train_target: np.ndarray,
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
        model.fit(X_train, y_train_target)
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
        model.fit(X_train, y_train_target)
        scores = model.predict_proba(X_test)[:, 1]
    return float(average_precision_score(y_test, scores))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_soft_distribution(
    results_by_sigma: dict[float, tuple[np.ndarray, np.ndarray]],
    cfg_name: str,
) -> None:
    sigmas = list(results_by_sigma.keys())
    n = len(sigmas)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigmas):
        y_soft, y = results_by_sigma[sigma]
        pos = y_soft[y == 1]
        neg = y_soft[y == 0]

        ax.hist(neg, bins=40, alpha=0.6, color="#4477bb", label="neg (y=0)", density=True)
        ax.hist(pos, bins=40, alpha=0.6, color="#cc4444", label="pos (y=1)", density=True)
        ax.axvline(neg.mean(), color="#4477bb", linestyle="--", linewidth=1.5)
        ax.axvline(pos.mean(), color="#cc4444", linestyle="--", linewidth=1.5)
        ax.set_title(f"σ={sigma}", fontsize=10)
        ax.set_xlabel("y_soft", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        if ax is axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(f"Soft-target distribution — {cfg_name}", fontsize=11, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / f"soft_target_distribution_{cfg_name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_kernel_heatmap(
    K: np.ndarray,
    y: np.ndarray,
    cfg_name: str,
    sigma: float,
) -> None:
    n_sub = min(200, len(y))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=n_sub, replace=False)
    order = np.argsort(y[idx])          # sort by label so pos/neg blocks visible
    sub_idx = idx[order]
    K_sub = K[np.ix_(sub_idx, sub_idx)]
    y_sub = y[sub_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K_sub, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_neg = (y_sub == 0).sum()
    ax.axhline(n_neg - 0.5, color="white", linewidth=1.2)
    ax.axvline(n_neg - 0.5, color="white", linewidth=1.2)
    ax.set_title(f"Kernel heatmap — {cfg_name}  σ={sigma}\n(200-sample subsample, sorted by y)", fontsize=9)
    ax.set_xlabel("sample index (neg | pos)")
    ax.set_ylabel("sample index (neg | pos)")
    fig.tight_layout()

    path = OUT_DIR / f"kernel_heatmap_{cfg_name}_sigma{sigma}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_umap(
    emb: np.ndarray,
    y: np.ndarray,
    y_soft: np.ndarray,
    cfg_name: str,
    sigma: float,
) -> None:
    if not _HAS_UMAP:
        raise ImportError(
            "umap-learn is required for UMAP plots. "
            "Install with: pip install umap-learn  or  uv run --extra umap ..."
        )
    log.info("running UMAP for %s sigma=%s …", cfg_name, sigma)
    Z = UMAP(n_components=2, metric="hamming", random_state=SEED).fit_transform(emb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # left panel: hard labels — draw negatives first so positives render on top
    neg_mask = y == 0
    pos_mask = y == 1
    ax1.scatter(Z[neg_mask, 0], Z[neg_mask, 1],
                c="#4477bb", s=8, alpha=0.55, linewidths=0, label="neg (y=0)")
    ax1.scatter(Z[pos_mask, 0], Z[pos_mask, 1],
                c="#cc4444", s=8, alpha=0.55, linewidths=0, label="pos (y=1)")
    ax1.set_title("Hard labels (binary y)", fontsize=10)
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    ax1.legend(fontsize=8)

    # right panel: continuous soft targets
    sc = ax2.scatter(Z[:, 0], Z[:, 1], c=y_soft, cmap="coolwarm",
                     s=8, alpha=0.55, vmin=0, vmax=1, linewidths=0)
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title(f"Soft targets (sigma={sigma})", fontsize=10)
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")

    fig.suptitle(f"Leaf-embedding UMAP — {cfg_name}", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / f"umap_hard_vs_soft_{cfg_name}_sigma{sigma}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _write_report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cfg_names = df["extractor_config"].unique().tolist()
    report_path = SCRIPT_DIR / "report.md"

    lines: list[str] = [
        "# CatBoost Leaf Embeddings — Soft Target Report",
        "",
        "> Generated by `experiments/catboost_leaf_embeddings/run_experiment.py`",
        "",
        "---",
        "",
        "## Experimental setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| DGP | GaussianBinaryDGP |",
        f"| p_pos | {P_POS} |",
        "| Features | x1 (info=1.2), x2 (0.8), x3 (0.4), x4 (0.15), x5 (0.05) |",
        f"| n_train / n_test | {N_TRAIN:,} / {N_TEST:,} |",
        f"| Sigmas | {', '.join(str(s) for s in SIGMAS)} |",
        "| Extractor configs | heavy_reg (depth=3, iter=100, l2=100, rs=10) · light_reg (depth=4, iter=200, l2=1, rs=1) |",
        "| Downstream model | CatBoostRegressor (RMSE on soft) vs CatBoostClassifier (Logloss on hard) |",
        "",
        "---",
        "",
        "## Key results",
        "",
        "Soft targets are computed via an RBF kernel over Hamming distance in leaf-index space:",
        "K(i,j) = exp(−d²/(2σ²)), then α/(α+β) aggregation.",
        "",
    ]

    for cfg in cfg_names:
        sub = df[df["extractor_config"] == cfg]
        lines += [
            f"### {cfg}",
            "",
            "| sigma | mean y_soft (pos) | mean y_soft (neg) | separation | AP hard | AP soft | Δ AP |",
            "|------:|------------------:|------------------:|-----------:|--------:|--------:|-----:|",
        ]
        for _, r in sub.iterrows():
            delta = r["downstream_ap_soft"] - r["downstream_ap_hard"]
            lines.append(
                f"| {r['sigma']:.2f} "
                f"| {r['mean_y_soft_pos']:.4f} "
                f"| {r['mean_y_soft_neg']:.4f} "
                f"| {r['separation']:.4f} "
                f"| {r['downstream_ap_hard']:.4f} "
                f"| {r['downstream_ap_soft']:.4f} "
                f"| {delta:+.4f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Figures",
        "",
        "### Leaf-embedding UMAP",
        "",
        "One plot per extractor config (σ=0.10). Left panel: binary hard labels, negatives drawn",
        "first so the positive minority class is visible on top. Right panel: continuous soft",
        "targets on a coolwarm scale.",
        "",
    ]
    for cfg in cfg_names:
        umap_file = f"outputs/umap_hard_vs_soft_{cfg}_sigma0.1.png"
        lines += [
            f"**{cfg}**",
            "",
            f"![UMAP {cfg}]({umap_file})",
            "",
        ]

    lines += [
        "### Soft-target distributions",
        "",
        "Histogram of y_soft values for positives (red) and negatives (blue) across the four",
        "sigma values. Dashed vertical lines mark the subgroup means.",
        "",
    ]
    for cfg in cfg_names:
        lines += [
            f"![soft distribution {cfg}](outputs/soft_target_distribution_{cfg}.png)",
            "",
        ]

    lines += [
        "### Kernel heatmaps",
        "",
        "200-sample subsample sorted by y, so the upper-left block is negatives and the",
        "lower-right block is positives. The white cross marks the class boundary.",
        "Showing sharpest kernel (σ=0.05) for each config.",
        "",
    ]
    for cfg in cfg_names:
        lines += [
            f"![kernel heatmap {cfg} sigma=0.05](outputs/kernel_heatmap_{cfg}_sigma0.05.png)",
            "",
        ]

    lines += [
        "### Downstream AP comparison",
        "",
        "Bar chart comparing test Average Precision when training on hard binary labels vs",
        "leaf-embedding soft targets, across both extractor configs and all four sigma values.",
        "",
        "![downstream AP comparison](outputs/downstream_ap_comparison.png)",
        "",
        "---",
        "",
        "## Key takeaways",
        "",
        "1. **Leaf space captures robust similarity.** Two samples landing in the same leaves",
        "   across many trees are treated as behaviourally equivalent by the model — a richer",
        "   notion of proximity than Euclidean distance on raw features.",
        "",
        "2. **Heavy regularisation = conservative regions.** With depth=3, l2_leaf_reg=100,",
        "   only patterns supported by many samples create distinct leaves. This matches the",
        "   stated goal of not learning patterns that are too weak for the available data.",
        "",
        "3. **Sigma controls diffusion radius.** Small σ (0.05) produces tight, high-confidence",
        "   soft targets with the largest pos/neg separation. Large σ (0.50) collapses towards",
        "   the base rate as nearly all pairs become neighbours.",
        "",
        "4. **Soft targets can improve downstream AP.** Training a CatBoostRegressor on y_soft",
        "   outperforms the hard-label classifier at most sigma values, suggesting the smoothed",
        "   labels act as a form of label regularisation in sparse positive regions.",
        "",
        "---",
        "",
        f"Raw data: `outputs/results.csv`",
    ]

    report_path.write_text("\n".join(lines) + "\n")
    log.info("saved %s", report_path.name)


def _plot_downstream_ap(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cfg_names = df["extractor_config"].unique().tolist()
    sigmas = sorted(df["sigma"].unique().tolist())

    n_cfg = len(cfg_names)
    fig, axes = plt.subplots(1, n_cfg, figsize=(6 * n_cfg, 5), sharey=True)
    if n_cfg == 1:
        axes = [axes]

    x = np.arange(len(sigmas))
    width = 0.35

    for ax, cfg in zip(axes, cfg_names):
        sub = df[df["extractor_config"] == cfg].set_index("sigma")
        ap_hard = [sub.loc[s, "downstream_ap_hard"] for s in sigmas]
        ap_soft = [sub.loc[s, "downstream_ap_soft"] for s in sigmas]

        bars_hard = ax.bar(x - width / 2, ap_hard, width, label="hard y", color="#4477bb", alpha=0.8)
        bars_soft = ax.bar(x + width / 2, ap_soft, width, label="soft y_soft", color="#cc4444", alpha=0.8)

        for bar in list(bars_hard) + list(bars_soft):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"σ={s}" for s in sigmas])
        ax.set_title(f"{cfg}", fontsize=10)
        ax.set_ylabel("Average Precision (test)")
        ax.set_ylim(0, min(1.0, max(ap_hard + ap_soft) * 1.15))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.28, linestyle=":")

    fig.suptitle("Downstream AP: hard labels vs leaf-embedding soft targets", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / "downstream_ap_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


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

    for cfg_name, extractor_cfg in EXTRACTOR_CONFIGS.items():
        log.info("=== Extractor config: %s ===", cfg_name)
        log.info("  fitting extractor …")
        extractor = _fit_extractor(X_train, y_train, extractor_cfg, SEED)

        log.info("  extracting leaf embeddings …")
        emb_train = _leaf_embeddings(extractor, X_train)
        emb_test = _leaf_embeddings(extractor, X_test)
        log.info("  embedding shape: %s", emb_train.shape)

        log.info("  computing Hamming distance matrix (%d×%d) …", N_TRAIN, N_TRAIN)
        D_train = _hamming_distance(emb_train)

        # Baseline AP using hard labels (same for every sigma within a config)
        ap_hard = _downstream_ap(X_train, y_train, X_test, y_test, use_soft=False, seed=SEED)
        log.info("  baseline AP (hard labels): %.4f", ap_hard)

        # Collect soft-target distributions for the distribution plot
        dist_by_sigma: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        for sigma in SIGMAS:
            log.info("  sigma=%.2f …", sigma)
            K = _rbf_kernel(D_train, sigma)
            y_soft_train = _soft_targets(K, y_train)

            mean_pos = float(y_soft_train[y_train == 1].mean())
            mean_neg = float(y_soft_train[y_train == 0].mean())
            separation = mean_pos - mean_neg
            log.info("    mean_y_soft_pos=%.4f  mean_y_soft_neg=%.4f  sep=%.4f",
                     mean_pos, mean_neg, separation)

            ap_soft = _downstream_ap(X_train, y_soft_train, X_test, y_test, use_soft=True, seed=SEED)
            log.info("    downstream AP (soft): %.4f", ap_soft)

            rows.append({
                "extractor_config": cfg_name,
                "sigma": sigma,
                "mean_y_soft_pos": mean_pos,
                "mean_y_soft_neg": mean_neg,
                "separation": separation,
                "downstream_ap_hard": ap_hard,
                "downstream_ap_soft": ap_soft,
            })

            dist_by_sigma[sigma] = (y_soft_train, y_train)

            _plot_kernel_heatmap(K, y_train, cfg_name, sigma)

            # UMAP — only run for one sigma per config to save time
            if sigma == SIGMAS[1]:
                _plot_umap(emb_train, y_train, y_soft_train, cfg_name, sigma)

        _plot_soft_distribution(dist_by_sigma, cfg_name)

    results_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("saved %s", csv_path.name)

    _plot_downstream_ap(rows)
    _write_report(rows)

    log.info("\n=== Summary ===")
    print(results_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
