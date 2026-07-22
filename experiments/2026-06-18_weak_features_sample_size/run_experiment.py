"""
experiments/weak_features_sample_size/run_experiment.py
------------------------------------------------------
Research question
~~~~~~~~~~~~~~~~
When the feature set is wide but weak (100 features, each with information
0.10), how does training sample size drive the gap between a high-capacity
booster (CatBoost) and interpretable rule/tree models (FIGS,
GreedyRuleList, plain DecisionTree)? Logistic regression serves as the
near-Bayes reference (the DGP is linear in the features).

The single sweep axis is ``n_train``; everything else is fixed. OOS metrics
are computed on a large, repeated test draw so test-side noise is negligible.

Implementation note
~~~~~~~~~~~~~~~~~~~
We do not use ``ml_elements.TrialRunner`` here because it would abort an
entire trial if any one setup's ``fit`` raised (FIGS and GreedyRuleList
can be unstable at ``n_train = 200``). Instead we run a thin manual loop
that isolates per-setup failures and records NaN, then reuse
``ml_elements``' DGPs, metrics, model factories, and plotting utilities.

Usage
~~~~~
    python run_experiment.py
    # or
    make exp-weak-features
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ml_elements as mle  # noqa: E402
from ml_elements import (  # noqa: E402
    AUC,
    AVG_PRECISION,
    BRIER,
    GaussianBinaryDGP,
)

OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("weak_features")

METRIC_MAP = {"roc_auc": AUC, "average_precision": AVG_PRECISION, "brier_score": BRIER}


# ─── Config → objects ────────────────────────────────────────────────────────


def load_config() -> dict:
    cfg_path = SCRIPT_DIR / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


def build_dgp(cfg: dict) -> tuple[GaussianBinaryDGP, list[str]]:
    n_feat = cfg["dgp"]["n_features"]
    features = [f"x{i:02d}" for i in range(n_feat)]
    info = {f: cfg["dgp"]["info_per_feature"] for f in features}
    dgp = GaussianBinaryDGP(
        p_pos=cfg["dgp"]["p_pos"],
        info=info,
        sigma=cfg["dgp"]["sigma"],
    )
    return dgp, features


def build_factories(cfg: dict) -> dict:
    factories: dict[str, Any] = {}
    for name, spec in cfg["models"].items():
        fn = getattr(mle, spec["factory"])
        params = spec.get("params", {}) or {}
        factories[name] = fn(**params)
    return factories


def build_metrics(cfg: dict):
    return [METRIC_MAP[name] for name in cfg["metrics"]]


# ─── Capacity probe (n_rules / n_leaves for interpretable models) ────────────


def extract_capacity(model) -> float | None:
    """
    Best-effort count of structural complexity (leaves / rules / depth).

    Returns None if no recognised attribute is found.
    """
    if model is None:
        return None

    # sklearn DecisionTree / tree ensembles
    if hasattr(model, "n_leaves_"):
        return float(model.n_leaves_)

    # FIGS — list of _tree.FIGSTree; each has .get_n_leaves() or .tree_
    trees = getattr(model, "trees_", None)
    if trees is not None:
        total = 0.0
        ok = True
        for t in trees:
            for attr in ("get_n_leaves", "n_leaves"):
                if hasattr(t, attr):
                    val = getattr(t, attr)
                    total += float(val() if callable(val) else val)
                    break
            else:
                ok = False
        if ok and total > 0:
            return total

    # GreedyRuleList / RuleFit — store fitted depth / rule count
    for attr in ("depth_", "n_rules_", "max_rules_"):
        if hasattr(model, attr):
            return float(getattr(model, attr))

    # Fallback: count non-zero feature importances (rough proxy)
    if hasattr(model, "feature_importances_"):
        return float(np.count_nonzero(model.feature_importances_))

    return None


# ─── Main loop ───────────────────────────────────────────────────────────────


def run() -> None:
    cfg = load_config()
    dgp, features = build_dgp(cfg)
    factories = build_factories(cfg)
    metrics = build_metrics(cfg)

    sweep_values: list[int] = list(cfg["sweep"]["values"])
    seeds = cfg["data"]["seeds"]
    n_test: int = cfg["data"]["n_test"]
    n_repeats: int = cfg["data"]["n_repeats"]

    log.info(
        "DGP: %d weak features (info=%.2f), p_pos=%.2f",
        len(features), cfg["dgp"]["info_per_feature"], cfg["dgp"]["p_pos"],
    )
    log.info(
        "Sweep n_train=%s, n_test=%d, n_repeats=%d, setups=%s",
        sweep_values, n_test, n_repeats, list(factories),
    )

    score_rows: list[dict] = []
    capacity_rows: list[dict] = []

    for i, n_train in enumerate(sweep_values):
        log.info("=== n_train = %d (%d/%d) ===", n_train, i + 1, len(sweep_values))

        train_seed = seeds["train"] + i * 7
        df_train = dgp.sample(n_train, train_seed)
        y_train = df_train["y"].to_numpy()
        log.info(
            "  train sample: n=%d  positive_rate=%.3f",
            n_train, float(y_train.mean()),
        )

        fitted: dict[str, Any] = {}
        for name, factory in factories.items():
            try:
                model = factory()
                model.fit(df_train[features], y_train)
                fitted[name] = model
                cap = extract_capacity(model)
                if cap is not None:
                    capacity_rows.append(
                        {"n_train": n_train, "setup": name, "capacity": cap}
                    )
                    log.info("  fitted %-12s capacity=%.0f", name, cap)
                else:
                    log.info("  fitted %-12s capacity=NA", name)
            except Exception as exc:  # noqa: BLE001
                log.warning("  FIT FAILED %s @ n_train=%d: %s", name, n_train, exc)
                fitted[name] = None

        for repeat in range(1, n_repeats + 1):
            test_seed = seeds["test_base"] + i * 1_000 + repeat
            df_test = dgp.sample(n_test, test_seed)
            y_test = df_test["y"].to_numpy()

            for name, model in fitted.items():
                row: dict[str, Any] = {
                    "n_train": n_train,
                    "repeat": repeat,
                    "setup": name,
                }
                if model is None:
                    for metric in metrics:
                        row[metric.name] = np.nan
                else:
                    try:
                        p_hat = model.predict_proba(df_test[features])[:, 1]
                        for metric in metrics:
                            row[metric.name] = metric.score(y_test, p_hat)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "  SCORE FAILED %s @ n_train=%d rep=%d: %s",
                            name, n_train, repeat, exc,
                        )
                        for metric in metrics:
                            row[metric.name] = np.nan
                score_rows.append(row)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    log.info("Saved scores.csv  (%d rows)", len(scores))

    if capacity_rows:
        cap_df = pd.DataFrame(capacity_rows)
        cap_df.to_csv(OUT_DIR / "capacity.csv", index=False)
        log.info("Saved capacity.csv  (%d rows)", len(cap_df))
    else:
        cap_df = pd.DataFrame()

    summary = make_summary(scores, metrics)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    log.info("Saved summary.csv  (%d rows)", len(summary))

    plot_learning_curves(scores, cfg, metrics)
    plot_gap_curves(scores, cfg)
    if not cap_df.empty:
        plot_capacity(cap_df, cfg)

    write_report(scores, summary, cap_df, cfg)
    log.info("Saved report.md")

    log.info("=== Done → %s ===", OUT_DIR)


# ─── Aggregation & plots ─────────────────────────────────────────────────────


def make_summary(scores: pd.DataFrame, metrics) -> pd.DataFrame:
    """
    Per (setup, n_train): mean and 95% bootstrap CI half-width for each metric.
    """
    rows = []
    for (setup, n_train), g in scores.groupby(["setup", "n_train"], sort=True):
        rec: dict[str, Any] = {"setup": setup, "n_train": int(n_train), "n": len(g)}
        for m in metrics:
            vals = g[m.name].dropna().to_numpy()
            if vals.size == 0:
                rec[f"{m.name}_mean"] = np.nan
                rec[f"{m.name}_ci95"] = np.nan
                continue
            rec[f"{m.name}_mean"] = float(np.mean(vals))
            rec[f"{m.name}_ci95"] = float(_bootstrap_ci_halfwidth(vals))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["setup", "n_train"]).reset_index(drop=True)


def _bootstrap_ci_halfwidth(vals: np.ndarray, n_boot: int = 1000, seed: int = 0) -> float:
    if vals.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    return float(np.diff(np.percentile(boots, [2.5, 97.5]))[0] / 2.0)


def _metric_display(metric_name: str) -> tuple[str, str]:
    """Return (axis_label, direction) for a metric name."""
    table = {
        "auc": ("ROC-AUC", "higher"),
        "average_precision": ("Average Precision", "higher"),
        "brier_score": ("Brier Score", "lower"),
    }
    return table.get(metric_name, (metric_name, "higher"))


def plot_learning_curves(scores: pd.DataFrame, cfg: dict, metrics) -> None:
    setups = list(cfg["models"].keys())
    setup_labels = {k: v["label"] for k, v in cfg["models"].items()}

    for m in metrics:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for setup in setups:
            g = (
                scores[scores["setup"] == setup]
                .groupby("n_train")[m.name]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("n_train")
            )
            if g.empty:
                continue
            x = g["n_train"].to_numpy()
            y = g["mean"].to_numpy()
            err = (g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))).to_numpy()
            ax.errorbar(
                x, y, yerr=err, marker="o", capsize=3, linewidth=2,
                label=setup_labels.get(setup, setup),
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training set size (log scale)")
        label, direction = _metric_display(m.name)
        better = "↑ better" if direction == "higher" else "↓ better"
        ax.set_ylabel(f"{label}  ({better})")
        ax.set_title(f"{label} vs training size — 100 weak features")
        ax.grid(True, which="both", linewidth=1, alpha=0.1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        fname = f"fig_{m.name}_vs_n.png"
        fig.savefig(OUT_DIR / fname, dpi=130)
        plt.close(fig)
        log.info("  saved %s", fname)


def plot_gap_curves(scores: pd.DataFrame, cfg: dict) -> None:
    """AUC gap between the reference (catboost) and every other setup."""
    baseline = cfg.get("gap_baseline", "catboost")
    setups = [s for s in cfg["models"].keys() if s != baseline]
    setup_labels = {k: v["label"] for k, v in cfg["models"].items()}

    metric = "auc"
    wide = (
        scores[scores["setup"].isin([baseline] + setups)]
        .pivot_table(index=["n_train", "repeat"], columns="setup", values=metric)
        .reset_index()
    )
    if baseline not in wide.columns:
        log.warning("gap baseline %s missing; skipping gap plot", baseline)
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for setup in setups:
        if setup not in wide.columns:
            continue
        g = (
            wide.assign(delta=wide[setup] - wide[baseline])
            .groupby("n_train")["delta"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("n_train")
        )
        x = g["n_train"].to_numpy()
        y = g["mean"].to_numpy()
        err = (g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))).to_numpy()
        ax.errorbar(
            x, y, yerr=err, marker="o", capsize=3, linewidth=2,
            label=setup_labels.get(setup, setup),
        )

    ax.axhline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (log scale)")
    ax.set_ylabel(f"Δ {metric.upper()}  ({setup_labels.get(baseline, baseline)} − model)")
    ax.set_title(f"AUC gap vs {setup_labels.get(baseline, baseline)} — interpretability cost")
    ax.grid(True, which="both", linewidth=1, alpha=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_gap_vs_n.png", dpi=130)
    plt.close(fig)
    log.info("  saved fig_gap_vs_n.png")


def plot_capacity(cap_df: pd.DataFrame, cfg: dict) -> None:
    setup_labels = {k: v["label"] for k, v in cfg["models"].items()}
    fig, ax = plt.subplots(figsize=(9, 5))
    for setup, g in cap_df.groupby("setup"):
        g = g.sort_values("n_train")
        ax.plot(
            g["n_train"], g["capacity"], marker="o", linewidth=2,
            label=setup_labels.get(setup, setup),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (log scale)")
    ax.set_ylabel("Learned capacity (rules / leaves / depth)")
    ax.set_title("Structural capacity used vs training size")
    ax.grid(True, which="both", linewidth=1, alpha=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_capacity_vs_n.png", dpi=130)
    plt.close(fig)
    log.info("  saved fig_capacity_vs_n.png")


# ─── Markdown report ─────────────────────────────────────────────────────────


def write_report(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    cap_df: pd.DataFrame,
    cfg: dict,
) -> None:
    setups = list(cfg["models"].keys())
    setup_labels = {k: v["label"] for k, v in cfg["models"].items()}
    baseline = cfg.get("gap_baseline", "catboost")

    # Pick the largest n_train block as the headline comparison.
    n_max = int(scores["n_train"].max())
    head = summary[summary["n_train"] == n_max].set_index("setup")

    lines: list[str] = []
    lines.append(f"# Weak Features × Sample Size — Report\n")
    lines.append(
        "Research question: with 100 weakly-informative features "
        f"(info={cfg['dgp']['info_per_feature']} per feature, "
        f"p_pos={cfg['dgp']['p_pos']}), how does training size drive the "
        f"gap between CatBoost and the interpretable rule/tree models?\n"
    )
    lines.append(
        f"Test set: n_test={cfg['data']['n_test']}, "
        f"n_repeats={cfg['data']['n_repeats']}.\n"
    )

    lines.append("## Headline: ROC-AUC at the largest training size "
                 f"(n_train={n_max})\n")
    lines.append("| Setup | mean AUC | ± 95% CI |")
    lines.append("|---|---|---|")
    for s in setups:
        if s not in head.index:
            continue
        row = head.loc[s]
        lines.append(
            f"| {setup_labels[s]} | "
            f"{row.get('auc_mean', float('nan')):.4f} | "
            f"± {row.get('auc_ci95', float('nan')):.4f} |"
        )
    lines.append("")

    lines.append("## AUC gap vs " + setup_labels.get(baseline, baseline) + "\n")
    wide = (
        scores[scores["setup"].isin(setups)]
        .pivot_table(index=["n_train", "repeat"], columns="setup", values="auc")
        .reset_index()
    )
    lines.append("| n_train | " + " | ".join(
        setup_labels[s] for s in setups if s in wide.columns and s != baseline
    ) + " |")
    lines.append("|---|" + "|".join(["---"] * (len(setups) - 1)) + "|")
    for n_train in sorted(wide["n_train"].unique()):
        sub = wide[wide["n_train"] == n_train]
        cells = []
        for s in setups:
            if s == baseline or s not in sub.columns:
                continue
            delta = float((sub[s] - sub[baseline]).mean())
            cells.append(f"{delta:+.4f}")
        lines.append(f"| {int(n_train)} | " + " | ".join(cells) + " |")
    lines.append("")

    if not cap_df.empty:
        lines.append("## Structural capacity used (rules / leaves)\n")
        lines.append("| n_train | setup | capacity |")
        lines.append("|---|---|---|")
        for _, r in cap_df.iterrows():
            lines.append(
                f"| {int(r['n_train'])} | "
                f"{setup_labels.get(r['setup'], r['setup'])} | "
                f"{r['capacity']:.0f} |"
            )
        lines.append("")

    lines.append("## Outputs\n")
    lines.append("- `scores.csv` — long-format per-repeat scores")
    lines.append("- `summary.csv` — mean ± 95% CI per (setup, n_train)")
    lines.append("- `capacity.csv` — learned rule/leaf counts per setup")
    lines.append("- `fig_auc_vs_n.png`, `fig_average_precision_vs_n.png`, "
                 "`fig_brier_score_vs_n.png` — learning curves")
    lines.append("- `fig_gap_vs_n.png` — AUC gap vs " + setup_labels.get(baseline, baseline))
    lines.append("- `fig_capacity_vs_n.png` — capacity vs n_train")
    lines.append("")

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
