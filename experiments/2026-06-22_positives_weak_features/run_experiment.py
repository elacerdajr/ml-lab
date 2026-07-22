"""
experiments/2026-06-22_positives_weak_features/run_experiment.py
------------------------------------------------------
Research question
~~~~~~~~~~~~~~~~~
At 1% positives, which CatBoost hyperparameters amplify or suppress the
effect of random noise features on model precision/AP?

With very few positives and many negatives, a model can latch onto noise
features that spuriously correlate with positives in the training draw.
We sweep two CatBoost axes that directly control this:

  depth        -- shallow trees use fewer features per split and are less
                  likely to build complex interactions with noise features
  l2_leaf_reg  -- higher L2 regularisation shrinks leaf weights and dampens
                  the influence of any individual feature (including noise)

Fixed iterations, no early stopping.  All 30 hyperparameter configs (6
depths × 5 regularisation levels) see the exact same train/eval/test data,
so metric differences are attributable solely to the hyperparameter choice.

Usage
~~~~~
    python run_experiment.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml_elements import (  # noqa: E402
    AUC,
    AVG_PRECISION,
    BRIER,
    LOGLOSS,
    GaussianBinaryDGP,
    make_catboost,
)

OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("positives_weak_features")

METRIC_MAP = {
    "roc_auc": AUC,
    "average_precision": AVG_PRECISION,
    "brier_score": BRIER,
    "logloss": LOGLOSS,
}


# ─── Config → objects ────────────────────────────────────────────────────────


def load_config() -> dict:
    return yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text())


def build_dgp(cfg: dict) -> tuple[GaussianBinaryDGP, list[str]]:
    dgp_cfg = cfg["dgp"]
    n_inf = dgp_cfg["n_informative"]
    n_noise = dgp_cfg["n_noise"]

    info: dict[str, float] = {}
    for i in range(n_inf):
        info[f"x{i:02d}"] = float(dgp_cfg["info_informative"])
    for i in range(n_inf, n_inf + n_noise):
        info[f"x{i:02d}"] = float(dgp_cfg["info_noise"])

    dgp = GaussianBinaryDGP(
        p_pos=dgp_cfg["p_pos"],
        info=info,
        sigma=dgp_cfg["sigma"],
    )
    return dgp, list(info.keys())


def build_metrics(cfg: dict):
    return [METRIC_MAP[name] for name in cfg["metrics"]]


# ─── Fixed data splits ────────────────────────────────────────────────────────


def sample_fixed_splits(
    dgp: GaussianBinaryDGP,
    cfg: dict,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seeds = cfg["data"]["seeds"]
    splits = [
        ("train", cfg["data"]["n_train"], seeds["train"]),
        ("eval",  cfg["data"]["n_eval"],  seeds["eval"]),
        ("test",  cfg["data"]["n_test"],  seeds["test"]),
    ]

    results = []
    for name, n, seed in splits:
        df = dgp.sample(n, seed)
        X = df[feature_cols].to_numpy()
        y = df["y"].to_numpy()
        n_pos = int(y.sum())
        log.info(
            "  %s: n=%d  positives=%d  rate=%.4f",
            name, n, n_pos, float(y.mean()),
        )
        if n_pos < 10:
            log.warning(
                "  CAUTION: %s has only %d positives — "
                "AP/AUC will be highly variable",
                name, n_pos,
            )
        results.extend([X, y])

    return tuple(results)  # type: ignore[return-value]


# ─── Scoring ─────────────────────────────────────────────────────────────────


def score_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metrics: list,
) -> dict[str, float]:
    if int(y.sum()) == 0:
        return {m.name: np.nan for m in metrics}
    try:
        p_hat = model.predict_proba(X)[:, 1]
        return {m.name: m.score(y, p_hat) for m in metrics}
    except Exception as exc:  # noqa: BLE001
        log.warning("  SCORE FAILED: %s", exc)
        return {m.name: np.nan for m in metrics}


# ─── Main sweep ──────────────────────────────────────────────────────────────


def run_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    metrics: list,
    feature_cols: list[str],
) -> pd.DataFrame:
    sw = cfg["sweep"]
    depth_values: list[int] = list(sw["depth"])
    l2_values: list[float] = list(sw["l2_leaf_reg"])
    iterations: int = sw["iterations"]
    learning_rate: float = sw["learning_rate"]
    total = len(depth_values) * len(l2_values)

    score_rows: list[dict] = []
    importance_rows: list[dict] = []

    for i, (depth, l2) in enumerate(product(depth_values, l2_values)):
        log.info(
            "  config %d/%d  depth=%d  l2_leaf_reg=%g",
            i + 1, total, depth, l2,
        )
        factory = make_catboost(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2,
        )
        try:
            model = factory()
            # No eval_set → no early stopping; runs full iterations.
            model.fit(X_train, y_train)
        except Exception as exc:  # noqa: BLE001
            log.warning("  FIT FAILED depth=%d l2=%g: %s", depth, l2, exc)
            for split in ("eval", "test"):
                row: dict[str, Any] = {
                    "depth": depth, "l2_leaf_reg": l2, "split": split,
                }
                row.update({m.name: np.nan for m in metrics})
                score_rows.append(row)
            continue

        # Per-split scores
        for split_name, X_sp, y_sp in [
            ("eval", X_eval, y_eval),
            ("test", X_test, y_test),
        ]:
            scores = score_model(model, X_sp, y_sp, metrics)
            row = {"depth": depth, "l2_leaf_reg": l2, "split": split_name}
            row.update(scores)
            score_rows.append(row)

        # Feature importances (PredictionValuesChange is the default)
        try:
            importances = model.get_feature_importance()
            for feat, imp in zip(feature_cols, importances):
                importance_rows.append({
                    "depth": depth,
                    "l2_leaf_reg": l2,
                    "feature": feat,
                    "importance": float(imp),
                })
        except Exception:  # noqa: BLE001
            pass

    scores_df = pd.DataFrame(score_rows)

    if importance_rows:
        imp_df = pd.DataFrame(importance_rows)
        imp_df.to_csv(OUT_DIR / "feature_importances.csv", index=False)
        log.info("Saved feature_importances.csv  (%d rows)", len(imp_df))

    return scores_df


# ─── Summary ─────────────────────────────────────────────────────────────────


def make_summary(scores: pd.DataFrame, metrics: list) -> pd.DataFrame:
    key_cols = ["depth", "l2_leaf_reg"]
    eval_df = scores[scores["split"] == "eval"][key_cols + [m.name for m in metrics]].copy()
    test_df = scores[scores["split"] == "test"][key_cols + [m.name for m in metrics]].copy()

    eval_df = eval_df.rename(columns={m.name: f"{m.name}_eval" for m in metrics})
    test_df = test_df.rename(columns={m.name: f"{m.name}_test" for m in metrics})

    summary = eval_df.merge(test_df, on=key_cols)

    for m in metrics:
        if m.direction == "higher":
            summary[f"{m.name}_gap"] = summary[f"{m.name}_eval"] - summary[f"{m.name}_test"]
        else:
            summary[f"{m.name}_gap"] = summary[f"{m.name}_test"] - summary[f"{m.name}_eval"]

    return summary.sort_values(key_cols).reset_index(drop=True)


# ─── Plots ───────────────────────────────────────────────────────────────────


def _pivot_for_heatmap(
    scores: pd.DataFrame,
    split: str,
    metric_name: str,
    depth_values: list,
    l2_values: list,
) -> np.ndarray:
    sub = scores[scores["split"] == split]
    pivot = (
        sub.pivot(index="depth", columns="l2_leaf_reg", values=metric_name)
        .reindex(index=depth_values, columns=l2_values)
    )
    return pivot.to_numpy(dtype=float)


def plot_heatmaps(scores: pd.DataFrame, cfg: dict, metrics: list) -> None:
    depth_values = sorted(cfg["sweep"]["depth"])
    l2_values = sorted(cfg["sweep"]["l2_leaf_reg"])

    for m in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(
            f"CatBoost — {m.name}  "
            f"(p_pos={cfg['dgp']['p_pos']}  "
            f"iter={cfg['sweep']['iterations']}  "
            f"lr={cfg['sweep']['learning_rate']})",
            fontsize=12,
        )
        cmap = "RdYlGn" if m.direction == "higher" else "RdYlGn_r"

        for ax, split in zip(axes, ["eval", "test"]):
            data = _pivot_for_heatmap(scores, split, m.name, depth_values, l2_values)
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)

            im = ax.imshow(
                data, cmap=cmap, aspect="auto",
                vmin=vmin, vmax=vmax,
            )
            plt.colorbar(im, ax=ax, label=m.name)

            ax.set_xticks(range(len(l2_values)))
            ax.set_xticklabels(l2_values)
            ax.set_yticks(range(len(depth_values)))
            ax.set_yticklabels(depth_values)
            ax.set_xlabel("l2_leaf_reg")
            ax.set_ylabel("depth")
            ax.set_title(f"{split.capitalize()} set")

            for ri in range(len(depth_values)):
                for ci in range(len(l2_values)):
                    val = data[ri, ci]
                    if not np.isnan(val):
                        ax.text(
                            ci, ri, f"{val:.3f}",
                            ha="center", va="center",
                            fontsize=7.5, color="black",
                        )

        fig.tight_layout()
        fname = f"fig_heatmap_{m.name}.png"
        fig.savefig(OUT_DIR / fname, dpi=130)
        plt.close(fig)
        log.info("  saved %s", fname)


def plot_eval_vs_test(scores: pd.DataFrame, cfg: dict, metrics: list) -> None:
    depth_values = sorted(cfg["sweep"]["depth"])

    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
    fig.suptitle(
        "Eval vs Test  — gap reveals eval-set optimism  (diagonal = no gap)",
        fontsize=12,
    )

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(depth_values), vmax=max(depth_values))

    eval_df = scores[scores["split"] == "eval"].copy()
    test_df = scores[scores["split"] == "test"].copy()
    key_cols = ["depth", "l2_leaf_reg"]
    merged = eval_df.merge(
        test_df, on=key_cols, suffixes=("_eval", "_test"),
    )

    for idx, m in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        sc = ax.scatter(
            merged[f"{m.name}_eval"],
            merged[f"{m.name}_test"],
            c=merged["depth"],
            cmap="viridis",
            norm=norm,
            s=70,
            alpha=0.85,
            zorder=3,
        )
        plt.colorbar(sc, ax=ax, label="depth")

        all_vals = pd.concat(
            [merged[f"{m.name}_eval"], merged[f"{m.name}_test"]]
        ).dropna()
        if len(all_vals):
            lo = all_vals.min() - 0.005
            hi = all_vals.max() + 0.005
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1.5, zorder=1)

        direction = "↑ better" if m.direction == "higher" else "↓ better"
        ax.set_xlabel(f"Eval {m.name}")
        ax.set_ylabel(f"Test {m.name}")
        ax.set_title(f"{m.name}  ({direction})")
        ax.grid(True, alpha=0.1)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_eval_vs_test.png", dpi=130)
    plt.close(fig)
    log.info("  saved fig_eval_vs_test.png")


def plot_depth_lines(scores: pd.DataFrame, cfg: dict, metrics: list) -> None:
    """AP and AUC vs l2_leaf_reg, one line per depth, for each split."""
    depth_values = sorted(cfg["sweep"]["depth"])
    l2_values = sorted(cfg["sweep"]["l2_leaf_reg"])

    for m in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        fig.suptitle(
            f"{m.name} vs l2_leaf_reg — one line per depth  "
            f"(p_pos={cfg['dgp']['p_pos']})",
            fontsize=12,
        )
        cmap = plt.get_cmap("plasma")
        colors = [cmap(i / max(len(depth_values) - 1, 1)) for i in range(len(depth_values))]

        for ax, split in zip(axes, ["eval", "test"]):
            sub = scores[scores["split"] == split]
            for depth, color in zip(depth_values, colors):
                g = (
                    sub[sub["depth"] == depth]
                    .sort_values("l2_leaf_reg")
                )
                ax.plot(
                    g["l2_leaf_reg"], g[m.name],
                    marker="o", linewidth=2, color=color,
                    label=f"depth={depth}",
                )
            ax.set_xscale("log")
            ax.set_xlabel("l2_leaf_reg  (log scale)")
            direction = "↑ better" if m.direction == "higher" else "↓ better"
            ax.set_ylabel(f"{m.name}  ({direction})")
            ax.set_title(f"{split.capitalize()} set")
            ax.grid(True, which="both", linewidth=1, alpha=0.1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)

        fig.tight_layout()
        fname = f"fig_depth_lines_{m.name}.png"
        fig.savefig(OUT_DIR / fname, dpi=130)
        plt.close(fig)
        log.info("  saved %s", fname)


def plot_noise_vs_signal_importance(cfg: dict) -> None:
    """Aggregate importance of noise vs informative features per config."""
    imp_path = OUT_DIR / "feature_importances.csv"
    if not imp_path.exists():
        return

    imp_df = pd.read_csv(imp_path)
    n_inf = cfg["dgp"]["n_informative"]
    informative_names = {f"x{i:02d}" for i in range(n_inf)}
    imp_df["feature_type"] = imp_df["feature"].apply(
        lambda f: "informative" if f in informative_names else "noise"
    )

    grouped = (
        imp_df.groupby(["depth", "l2_leaf_reg", "feature_type"])["importance"]
        .sum()
        .reset_index()
    )
    noise_frac = grouped[grouped["feature_type"] == "noise"].copy()
    total = grouped.groupby(["depth", "l2_leaf_reg"])["importance"].sum().reset_index()
    total = total.rename(columns={"importance": "total"})
    noise_frac = noise_frac.merge(total, on=["depth", "l2_leaf_reg"])
    noise_frac["noise_fraction"] = noise_frac["importance"] / noise_frac["total"].clip(lower=1e-9)

    depth_values = sorted(cfg["sweep"]["depth"])
    l2_values = sorted(cfg["sweep"]["l2_leaf_reg"])

    pivot = (
        noise_frac.pivot(index="depth", columns="l2_leaf_reg", values="noise_fraction")
        .reindex(index=depth_values, columns=l2_values)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    vmin, vmax = np.nanmin(pivot), np.nanmax(pivot)
    im = ax.imshow(pivot, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Fraction of importance on noise features")

    ax.set_xticks(range(len(l2_values)))
    ax.set_xticklabels(l2_values)
    ax.set_yticks(range(len(depth_values)))
    ax.set_yticklabels(depth_values)
    ax.set_xlabel("l2_leaf_reg")
    ax.set_ylabel("depth")
    ax.set_title(
        "Noise feature importance fraction  (↓ better — lower = model ignores noise)"
    )

    for ri in range(len(depth_values)):
        for ci in range(len(l2_values)):
            val = pivot[ri, ci]
            if not np.isnan(val):
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_noise_importance.png", dpi=130)
    plt.close(fig)
    log.info("  saved fig_noise_importance.png")


# ─── Markdown report ─────────────────────────────────────────────────────────


def write_report(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: dict,
    metrics: list,
) -> None:
    sw = cfg["sweep"]
    dgp_cfg = cfg["dgp"]

    lines: list[str] = []
    lines.append("# Positives Weak Features — Report\n")
    lines.append(
        f"**Research question**: at {dgp_cfg['p_pos']*100:.0f}% positives, "
        f"which CatBoost hyperparameters amplify or suppress the effect "
        f"of {dgp_cfg['n_noise']} random noise features on model AP?\n"
    )
    lines.append("## Setup\n")
    lines.append(
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| p_pos | {dgp_cfg['p_pos']} |\n"
        f"| Informative features | {dgp_cfg['n_informative']} (info={dgp_cfg['info_informative']}) |\n"
        f"| Noise features | {dgp_cfg['n_noise']} (info={dgp_cfg['info_noise']}) |\n"
        f"| n_train | {cfg['data']['n_train']} |\n"
        f"| n_eval | {cfg['data']['n_eval']} |\n"
        f"| n_test | {cfg['data']['n_test']} |\n"
        f"| iterations | {sw['iterations']} |\n"
        f"| learning_rate | {sw['learning_rate']} |\n"
        f"| depth sweep | {sw['depth']} |\n"
        f"| l2_leaf_reg sweep | {sw['l2_leaf_reg']} |\n"
    )
    lines.append("")

    ap_metric = next((m for m in metrics if m.name == "average_precision"), metrics[0])

    eval_col = f"{ap_metric.name}_eval"
    test_col = f"{ap_metric.name}_test"
    if eval_col in summary.columns and test_col in summary.columns:
        best_eval = summary.loc[summary[eval_col].idxmax()]
        best_test = summary.loc[summary[test_col].idxmax()]

        lines.append(f"## Best config by eval AP\n")
        lines.append(
            f"depth={int(best_eval['depth'])}, l2_leaf_reg={best_eval['l2_leaf_reg']}  "
            f"→ eval AP={best_eval[eval_col]:.4f}, test AP={best_eval[test_col]:.4f}\n"
        )
        lines.append(f"## Best config by test AP\n")
        lines.append(
            f"depth={int(best_test['depth'])}, l2_leaf_reg={best_test['l2_leaf_reg']}  "
            f"→ eval AP={best_test[eval_col]:.4f}, test AP={best_test[test_col]:.4f}\n"
        )

        if (int(best_eval["depth"]) == int(best_test["depth"]) and
                best_eval["l2_leaf_reg"] == best_test["l2_leaf_reg"]):
            lines.append("✓ Eval and test agree on the best config.\n")
        else:
            lines.append("⚠ Eval and test disagree — eval-set optimism detected.\n")

    lines.append("## Full results (by eval AP descending)\n")
    if eval_col in summary.columns:
        top = summary.sort_values(eval_col, ascending=False).head(10)
        cols_to_show = ["depth", "l2_leaf_reg"] + [
            c for c in summary.columns if c.endswith("_eval") or c.endswith("_test")
            if c in summary.columns
        ]
        cols_to_show = [c for c in cols_to_show if c in summary.columns]
        lines.append("| " + " | ".join(cols_to_show) + " |")
        lines.append("|" + "|".join(["---"] * len(cols_to_show)) + "|")
        for _, row in top.iterrows():
            cells = []
            for c in cols_to_show:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(int(v)) if not pd.isna(v) else "NA")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Outputs\n")
    lines.append("- `scores.csv` — long format: one row per (depth, l2_leaf_reg, split)")
    lines.append("- `summary.csv` — wide format: eval + test side-by-side, gap column")
    lines.append("- `feature_importances.csv` — per-feature importance for each config")
    lines.append("- `fig_heatmap_*.png` — heatmaps of each metric (eval | test)")
    lines.append("- `fig_depth_lines_*.png` — metric vs l2_leaf_reg, one line per depth")
    lines.append("- `fig_eval_vs_test.png` — scatter: eval vs test metric per config")
    lines.append("- `fig_noise_importance.png` — fraction of importance on noise features")
    lines.append("")

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ─── Entry point ─────────────────────────────────────────────────────────────


def run() -> None:
    cfg = load_config()
    dgp, feature_cols = build_dgp(cfg)
    metrics = build_metrics(cfg)

    log.info(
        "DGP: %d informative (info=%.2f) + %d noise features, p_pos=%.3f",
        cfg["dgp"]["n_informative"],
        cfg["dgp"]["info_informative"],
        cfg["dgp"]["n_noise"],
        cfg["dgp"]["p_pos"],
    )
    log.info(
        "Sweep: CatBoost depth=%s × l2_leaf_reg=%s  (iterations=%d, lr=%g)",
        cfg["sweep"]["depth"],
        cfg["sweep"]["l2_leaf_reg"],
        cfg["sweep"]["iterations"],
        cfg["sweep"]["learning_rate"],
    )
    log.info(
        "Fixed splits: n_train=%d  n_eval=%d  n_test=%d",
        cfg["data"]["n_train"], cfg["data"]["n_eval"], cfg["data"]["n_test"],
    )

    # 1. Sample fixed splits once
    X_train, y_train, X_eval, y_eval, X_test, y_test = sample_fixed_splits(
        dgp, cfg, feature_cols
    )

    # 2. Run the 2D hyperparameter grid
    n_configs = len(cfg["sweep"]["depth"]) * len(cfg["sweep"]["l2_leaf_reg"])
    log.info("=== CatBoost sweep (%d configs) ===", n_configs)
    scores = run_sweep(
        X_train, y_train,
        X_eval, y_eval,
        X_test, y_test,
        cfg, metrics, feature_cols,
    )

    # 3. Save raw scores
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    log.info("Saved scores.csv  (%d rows)", len(scores))

    # 4. Summary (wide format)
    summary = make_summary(scores, metrics)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    log.info("Saved summary.csv  (%d rows)", len(summary))

    # 5. Plots
    log.info("Generating plots …")
    plot_heatmaps(scores, cfg, metrics)
    plot_depth_lines(scores, cfg, metrics)
    plot_eval_vs_test(scores, cfg, metrics)
    plot_noise_vs_signal_importance(cfg)

    # 6. Report
    write_report(scores, summary, cfg, metrics)
    log.info("Saved report.md")

    log.info("=== Done → %s ===", OUT_DIR)


if __name__ == "__main__":
    run()
