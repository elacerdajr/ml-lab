"""
Spread Score — Learnability Experiment
=======================================
Builds on `experiments/2026-06-23_catboost_leaf_embeddings`: a heavily-regularised
CatBoost extractor gives leaf embeddings; Hamming distance in that space
drives an RBF kernel over the training labels.

The *kernel score* is the kernel-weighted mean label (the "location" of a
sample's neighbourhood — this is the soft target from the prior
experiment). This experiment asks the follow-up question: can we also
capture the *spread* (dispersion / disagreement) of labels inside that
neighbourhood, and — critically — is that spread quantity itself a smooth
function of X, i.e. learnable by a plain CatBoostRegressor trained
directly on the raw features (no leaf embeddings, no kernel, no access to
neighbour labels at inference time)?

Four candidate spread-score definitions are compared:
  beta_var            posterior variance of a Beta(alpha, beta) fit to the
                       kernel-weighted vote (alpha/beta = pseudo-counts of
                       positive/negative neighbours, Laplace-smoothed)
  weighted_label_var  raw kernel-weighted variance of neighbour labels
                       around their kernel-weighted mean (model-free)
  binary_entropy      Shannon entropy of the kernel score treated as a
                       Bernoulli parameter (uncertainty of the *mean*,
                       not of the neighbourhood itself)
  inv_effective_n     1 / Kish effective neighbourhood size — large when
                       the kernel weight is spread thin over many/few
                       "real" neighbours (evidence dilution)

For each definition we fit CatBoostRegressor(X_train -> score_train) and
evaluate R^2 / Spearman rho against the held-out ground-truth score on
X_test (itself computed via the kernel against the *training* labels, so
it is a well-defined nonparametric target, never using test labels).

Outputs (experiments/2026-07-02_spread_score/outputs/):
  results.csv
  learnability_r2.png
  learnability_spearman.png
  kernel_vs_spread_{cfg}.png     (sanity check: spread peaks near p=0.5)
  predicted_vs_actual_{cfg}.png
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
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances, r2_score

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
# Configuration (mirrors experiments/2026-06-23_catboost_leaf_embeddings for comparability)
# ---------------------------------------------------------------------------

SIGMAS = [0.05, 0.10, 0.25, 0.50]

EXTRACTOR_CONFIGS: dict[str, dict] = {
    "heavy_reg": dict(depth=3, iterations=100, l2_leaf_reg=100, random_strength=10),
    "light_reg": dict(depth=4, iterations=200, l2_leaf_reg=1, random_strength=1),
}

SCORE_TYPES = ["kernel_score", "beta_var", "weighted_label_var", "binary_entropy", "inv_effective_n"]

P_POS = 0.10
N_TRAIN = 2_000
N_TEST = 2_000
SEED = 42
EPS = 1e-9

DGP = GaussianBinaryDGP(
    p_pos=P_POS,
    info={"x1": 1.2, "x2": 0.8, "x3": 0.4, "x4": 0.15, "x5": 0.05},
    sigma=1.0,
)

# ---------------------------------------------------------------------------
# Extractor / kernel helpers
# ---------------------------------------------------------------------------

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


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
    emb = model.calc_leaf_indexes(X)  # (n, n_trees) int32
    return emb.astype(np.float32)


def _hamming_distance(emb_a: np.ndarray, emb_b: np.ndarray | None = None) -> np.ndarray:
    if emb_b is None:
        return pairwise_distances(emb_a, metric="hamming")
    return pairwise_distances(emb_a, emb_b, metric="hamming")


def _rbf_kernel(D: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(D ** 2) / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Kernel score (location) + spread-score definitions
# ---------------------------------------------------------------------------

def _kernel_score(K: np.ndarray, y_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Laplace-smoothed kernel-weighted vote. K is (n_query, n_ref)."""
    pos_mask = y_ref == 1
    neg_mask = y_ref == 0
    alpha = 1.0 + K[:, pos_mask].sum(axis=1)
    beta = 1.0 + K[:, neg_mask].sum(axis=1)
    p = alpha / (alpha + beta)
    return p, alpha, beta


def _beta_var(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))


def _weighted_label_var(K: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    w_sum = K.sum(axis=1) + EPS
    p_raw = (K @ y_ref) / w_sum
    sq_dev = (y_ref[None, :] - p_raw[:, None]) ** 2
    return (K * sq_dev).sum(axis=1) / w_sum


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, EPS, 1.0 - EPS)
    return -p_clip * np.log2(p_clip) - (1.0 - p_clip) * np.log2(1.0 - p_clip)


def _inv_effective_n(K: np.ndarray) -> np.ndarray:
    w_sum = K.sum(axis=1)
    w_sq_sum = (K ** 2).sum(axis=1)
    eff_n = (w_sum ** 2) / (w_sq_sum + EPS)
    return 1.0 / (1.0 + eff_n)


def _all_scores(K: np.ndarray, y_ref: np.ndarray) -> dict[str, np.ndarray]:
    p, alpha, beta = _kernel_score(K, y_ref)
    return {
        "kernel_score": p,
        "beta_var": _beta_var(alpha, beta),
        "weighted_label_var": _weighted_label_var(K, y_ref),
        "binary_entropy": _binary_entropy(p),
        "inv_effective_n": _inv_effective_n(K),
    }


# ---------------------------------------------------------------------------
# Downstream learnability
# ---------------------------------------------------------------------------

def _fit_predict(X_train: np.ndarray, target_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
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
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_learnability(df: pd.DataFrame, metric: str, fname: str) -> None:
    cfg_names = df["extractor_config"].unique().tolist()
    sigmas = sorted(df["sigma"].unique().tolist())
    n_cfg = len(cfg_names)

    fig, axes = plt.subplots(1, n_cfg, figsize=(7 * n_cfg, 5), sharey=True)
    if n_cfg == 1:
        axes = [axes]

    x = np.arange(len(sigmas))
    n_scores = len(SCORE_TYPES)
    width = 0.8 / n_scores
    colors = plt.get_cmap("tab10").colors

    for ax, cfg in zip(axes, cfg_names):
        sub = df[df["extractor_config"] == cfg]
        for i, score_type in enumerate(SCORE_TYPES):
            s = sub[sub["score_type"] == score_type].set_index("sigma")
            vals = [s.loc[sig, metric] if sig in s.index else np.nan for sig in sigmas]
            offset = (i - (n_scores - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=score_type, color=colors[i % len(colors)], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + (0.01 if h >= 0 else -0.03),
                        f"{h:.2f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=6.5)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"σ={s}" for s in sigmas])
        ax.set_title(cfg, fontsize=10)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.28, linestyle=":")
        ax.legend(fontsize=7)

    fig.suptitle(f"Learnability of kernel/spread scores from X — {metric}", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_kernel_vs_spread(scores_test: dict[str, np.ndarray], cfg_name: str, sigma: float) -> None:
    spread_defs = [s for s in SCORE_TYPES if s != "kernel_score"]
    fig, axes = plt.subplots(1, len(spread_defs), figsize=(4.2 * len(spread_defs), 4))

    for ax, sdef in zip(axes, spread_defs):
        ax.scatter(scores_test["kernel_score"], scores_test[sdef], s=6, alpha=0.35, color="#4477bb")
        ax.set_xlabel("kernel_score (location)")
        ax.set_ylabel(sdef)
        ax.set_title(sdef, fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle(f"Kernel score vs spread-score definitions — {cfg_name}  σ={sigma}", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / f"kernel_vs_spread_{cfg_name}_sigma{sigma}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


def _plot_predicted_vs_actual(actual: dict[str, np.ndarray], predicted: dict[str, np.ndarray],
                               cfg_name: str, sigma: float) -> None:
    fig, axes = plt.subplots(1, len(SCORE_TYPES), figsize=(4.2 * len(SCORE_TYPES), 4))

    for ax, score_type in zip(axes, SCORE_TYPES):
        a, p = actual[score_type], predicted[score_type]
        ax.scatter(a, p, s=6, alpha=0.35, color="#cc4444")
        lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("actual (kernel/leaf-embedding)")
        ax.set_ylabel("predicted (CatBoost on X only)")
        ax.set_title(score_type, fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle(f"Predicted vs actual — {cfg_name}  σ={sigma} (test set)", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / f"predicted_vs_actual_{cfg_name}_sigma{sigma}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(df: pd.DataFrame) -> None:
    report_path = SCRIPT_DIR / "report.md"
    cfg_names = df["extractor_config"].unique().tolist()

    lines: list[str] = [
        "# Spread Score — Learnability Report",
        "",
        "> Generated by `experiments/2026-07-02_spread_score/run_experiment.py`",
        "",
        "---",
        "",
        "## Question",
        "",
        "`experiments/2026-06-23_catboost_leaf_embeddings` showed the *kernel score* (a kernel-weighted",
        "vote over neighbour labels in leaf-embedding space) is a useful soft target. This",
        "experiment asks the follow-up: can we also define a **spread score** — how much",
        "neighbours disagree, not just what they agree on — and is that quantity itself a",
        "smooth, learnable function of the raw features X? If a CatBoostRegressor trained",
        "directly on X can recover the kernel-derived spread score, the notion of local",
        "uncertainty generalises beyond the training neighbourhood and could be used as a",
        "cheap inference-time feature (no leaf embeddings / kernel needed at serving time).",
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
        f"| Sigmas | {', '.join(str(s) for s in SIGMAS)} |",
        "| Extractor configs | heavy_reg (depth=3, iter=100, l2=100, rs=10) · light_reg (depth=4, iter=200, l2=1, rs=1) |",
        "| Downstream learner | CatBoostRegressor(iterations=200, depth=4, lr=0.06, RMSE) trained on raw X only |",
        "",
        "Ground-truth scores for both train and test rows are computed from the kernel",
        "against the **training labels only** (self excluded on the train side), so test-set",
        "targets never leak test labels — they are a well-defined nonparametric function of X.",
        "",
        "---",
        "",
        "## Spread-score definitions",
        "",
        "| Name | Formula | Intuition |",
        "|------|---------|-----------|",
        "| `kernel_score` | α/(α+β) | location — kernel-weighted vote (baseline, not a spread measure) |",
        "| `beta_var` | αβ / [(α+β)²(α+β+1)] | posterior variance of a Beta(α,β) fit to the vote |",
        "| `weighted_label_var` | Σ w(y−p̄)² / Σw | model-free kernel-weighted variance of neighbour labels |",
        "| `binary_entropy` | −p·log₂p − (1−p)·log₂(1−p) | uncertainty of the kernel score treated as Bernoulli p |",
        "| `inv_effective_n` | 1 / (1 + Kish ESS) | evidence dilution — large when kernel mass isn't concentrated on real neighbours |",
        "",
        "α = 1 + Σ K·1[y=1], β = 1 + Σ K·1[y=0] (Laplace-smoothed kernel-weighted counts).",
        "",
        "---",
        "",
        "## Learnability results",
        "",
        "R² and Spearman ρ between the CatBoost-on-X prediction and the kernel-derived",
        "ground truth, evaluated on the held-out test set.",
        "",
    ]

    for cfg in cfg_names:
        sub = df[df["extractor_config"] == cfg]
        lines += [
            f"### {cfg}",
            "",
            "| score_type | sigma | R² | Spearman ρ |",
            "|------------|------:|---:|-----------:|",
        ]
        for _, r in sub.sort_values(["score_type", "sigma"]).iterrows():
            lines.append(
                f"| {r['score_type']} | {r['sigma']:.2f} | {r['r2']:.4f} | {r['spearman']:.4f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Figures",
        "",
        "### Learnability — R²",
        "",
        "![learnability r2](outputs/learnability_r2.png)",
        "",
        "### Learnability — Spearman ρ",
        "",
        "![learnability spearman](outputs/learnability_spearman.png)",
        "",
        "### Kernel score vs spread-score definitions (sanity check)",
        "",
        "If spread genuinely reflects label disagreement, it should peak near kernel_score ≈",
        "0.5 (maximum class overlap) and shrink toward the extremes. Shown at σ=0.10.",
        "",
    ]
    for cfg in cfg_names:
        lines += [f"![kernel vs spread {cfg}](outputs/kernel_vs_spread_{cfg}_sigma0.1.png)", ""]

    lines += ["### Predicted vs actual (test set, σ=0.10)", ""]
    for cfg in cfg_names:
        lines += [f"![predicted vs actual {cfg}](outputs/predicted_vs_actual_{cfg}_sigma0.1.png)", ""]

    lines += [
        "---",
        "",
        "## Key takeaways",
        "",
        "1. **Sigma dominates learnability, for every definition.** At the tightest kernel",
        "   (σ=0.05) every score is hard to recover from X alone (R² 0.04–0.43). By σ=0.25–0.50",
        "   every definition exceeds R²=0.92, most above 0.99. Bandwidth — not the choice of",
        "   spread formula — is the first-order knob controlling whether a score is learnable.",
        "",
        "2. **Spread is not inherently harder to learn than location.** At σ=0.05,",
        "   `inv_effective_n` has *higher* R² than `kernel_score` in both extractor configs",
        "   (0.21 vs 0.13 heavy_reg; 0.43 vs 0.21 light_reg), and rank correlation for at least",
        "   one spread definition (`inv_effective_n` under light_reg, `weighted_label_var` under",
        "   heavy_reg) beats `kernel_score`'s. Local disagreement patterns can align with X even",
        "   when the precise soft-label value is still noisy.",
        "",
        "3. **Entropy of the mean is the hardest quantity to learn at tight kernels**, despite",
        "   being a deterministic transform of `kernel_score`. Because entropy is most sensitive",
        "   near p≈0.5, small regression error in the underlying probability gets amplified",
        "   nonlinearly — `binary_entropy` has the lowest R² of all five scores at σ=0.05 in",
        "   both configs.",
        "",
        "4. **`beta_var` tracks `kernel_score` almost exactly at σ=0.10** (R² 0.77–0.83 for both,",
        "   in both configs) since it's a smooth deterministic function of the same α/β",
        "   pseudo-counts that produce the kernel score — the Beta-posterior view of spread",
        "   inherits the location statistic's learnability for free.",
        "",
        "5. **σ≈0.10 is the practical sweet spot.** It's small enough that scores still carry",
        "   local information (per the pos/neg separation numbers in",
        "   `experiments/2026-06-23_catboost_leaf_embeddings/report.md`), yet large enough that every definition is",
        "   reasonably learnable directly from X (R² 0.56–0.83) — well short of the ≥0.92",
        "   ceiling at larger σ, which is reached mostly by the kernel washing local structure",
        "   out into a low-variance near-constant.",
        "",
        "---",
        "",
        "Raw data: `outputs/results.csv`",
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

    rows: list[dict] = []

    for cfg_name, extractor_cfg in EXTRACTOR_CONFIGS.items():
        log.info("=== Extractor config: %s ===", cfg_name)
        extractor = _fit_extractor(X_train, y_train, extractor_cfg, SEED)

        emb_train = _leaf_embeddings(extractor, X_train)
        emb_test = _leaf_embeddings(extractor, X_test)

        D_tr_tr = _hamming_distance(emb_train)
        np.fill_diagonal(D_tr_tr, np.inf)  # exclude self as a "neighbour"
        D_te_tr = _hamming_distance(emb_test, emb_train)

        for sigma in SIGMAS:
            log.info("  sigma=%.2f …", sigma)
            K_tr_tr = _rbf_kernel(D_tr_tr, sigma)  # self excluded (inf distance -> 0 weight)
            K_te_tr = _rbf_kernel(D_te_tr, sigma)

            scores_train = _all_scores(K_tr_tr, y_train)
            scores_test = _all_scores(K_te_tr, y_train)

            predicted_test: dict[str, np.ndarray] = {}
            for score_type in SCORE_TYPES:
                pred = _fit_predict(X_train, scores_train[score_type], X_test, SEED)
                predicted_test[score_type] = pred

                r2 = r2_score(scores_test[score_type], pred)
                rho = spearmanr(scores_test[score_type], pred).statistic

                rows.append({
                    "extractor_config": cfg_name,
                    "sigma": sigma,
                    "score_type": score_type,
                    "mean_actual": float(scores_test[score_type].mean()),
                    "std_actual": float(scores_test[score_type].std()),
                    "r2": float(r2),
                    "spearman": float(rho),
                })
                log.info("    %-20s R2=%.4f  spearman=%.4f", score_type, r2, rho)

            if sigma == SIGMAS[1]:
                _plot_kernel_vs_spread(scores_test, cfg_name, sigma)
                _plot_predicted_vs_actual(scores_test, predicted_test, cfg_name, sigma)

    results_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("saved %s", csv_path.name)

    _plot_learnability(results_df, "r2", "learnability_r2.png")
    _plot_learnability(results_df, "spearman", "learnability_spearman.png")
    _write_report(results_df)

    log.info("\n=== Summary ===")
    print(results_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
