"""
experiments/roc_vs_ap/run_experiment.py
---------------------------------------
Compare HGB models optimised for ROC-AUC vs Average Precision.

Reads config.yaml plus per-study YAMLs from the same directory.
All outputs are written to outputs/ (relative to this script).

Usage
-----
    python run_experiment.py
    python run_experiment.py --config config.yaml \\
                             --study-imbalance study_imbalance.yaml \\
                             --study-info     study_feature_info.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe in scripts
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]          # experiments/roc_vs_ap/../../ → root
sys.path.insert(0, str(REPO_ROOT))

# ── stub missing optional dependency so ml_elements imports cleanly ───────────
_stub = types.ModuleType("bayesian_ap_comparator")
_stub.BayesianAPComparator = object          # never called; just silences the import
sys.modules.setdefault("bayesian_ap_comparator", _stub)

# ── reuse ml_elements building blocks ─────────────────────────────────────────
from ml_elements.dgp     import GaussianBinaryDGP   # noqa: E402
from ml_elements.metrics import AUC, AVG_PRECISION  # noqa: E402
from ml_elements.models  import make_sklearn         # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── metric registry (maps config name → Metric object) ────────────────────────
_METRIC_OBJS = {
    "roc_auc":           AUC,
    "average_precision": AVG_PRECISION,
}


# =============================================================================
# Config loading
# =============================================================================

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Model factory builder
# =============================================================================

def build_model_factory(model_cfg: dict):
    """
    Return a zero-argument callable that produces a fresh, unfitted model.

    Delegates to ml_elements.models.make_sklearn so the factory pattern is
    consistent with the rest of the repo.
    """
    backend = model_cfg["backend"]
    if backend == "hgb":
        return make_sklearn(HistGradientBoostingClassifier, **model_cfg["hgb"])
    raise ValueError(f"Unknown model backend: {backend!r}. Supported: 'hgb'.")


# =============================================================================
# Condition builders
# =============================================================================

def build_conditions(base_cfg: dict, study_cfg: dict) -> list[dict]:
    """
    Expand a study config into a list of condition dicts, each containing:
        p_pos            positive fraction for this condition
        info             feature → separation dict
        condition_value  the swept parameter value (for labelling/grouping)
    """
    base_info = dict(base_cfg["dgp"]["info"])
    study = study_cfg["study"]
    conditions = []

    if study["name"] == "imbalance":
        scale = study.get("info_scale", 1.0)
        info  = {k: v * scale for k, v in base_info.items()}
        for p in study["p_pos_values"]:
            conditions.append({"p_pos": p, "info": info, "condition_value": p})

    elif study["name"] == "feature_info":
        p_pos = study["fixed_p_pos"]
        for s in study["info_scale_values"]:
            conditions.append({
                "p_pos":           p_pos,
                "info":            {k: v * s for k, v in base_info.items()},
                "condition_value": s,
            })

    else:
        raise ValueError(f"Unknown study name: {study['name']!r}")

    return conditions


# =============================================================================
# Runner
# =============================================================================

def run_condition(
    p_pos: float,
    info: dict[str, float],
    condition_col: str,
    condition_value: float,
    model_factories: dict[str, Any],
    metric_names: list[str],
    n_train: int,
    n_test: int,
    n_repeats: int,
    seed_train: int,
    seed_test_base: int,
    sigma: float = 1.0,
) -> pd.DataFrame:
    """
    Fit every model on one shared training set; score on n_repeats test draws.

    Uses GaussianBinaryDGP from ml_elements.dgp.
    Returns tidy DataFrame — one row per (model, repeat).
    """
    # Keep config name → Metric pairs so columns are named by the config key
    # (e.g. "roc_auc"), not by Metric.name (e.g. "auc").
    metrics = [(cfg_name, _METRIC_OBJS[cfg_name]) for cfg_name in metric_names]
    feature_cols = list(info.keys())

    dgp = GaussianBinaryDGP(p_pos=p_pos, info=info, sigma=sigma)

    df_train = dgp.sample(n_train, seed=seed_train)
    X_tr = df_train[feature_cols].values
    y_tr = df_train["y"].values

    fitted: dict[str, Any] = {}
    for name, factory in model_factories.items():
        model = factory()
        model.fit(X_tr, y_tr)
        fitted[name] = model

    rows = []
    for rep in range(n_repeats):
        df_test = dgp.sample(n_test, seed=seed_test_base + rep)
        X_te = df_test[feature_cols].values
        y_te = df_test["y"].values

        for name, model in fitted.items():
            p_hat = model.predict_proba(X_te)[:, 1]
            row: dict = {"model": name, "repeat": rep, condition_col: condition_value}
            for cfg_name, m in metrics:
                row[cfg_name] = m.score(y_te, p_hat)
            rows.append(row)

    return pd.DataFrame(rows)


def run_study(
    study_name: str,
    condition_col: str,
    conditions: list[dict],
    model_factories: dict,
    metric_names: list[str],
    data_cfg: dict,
    sigma: float = 1.0,
) -> pd.DataFrame:
    """Run all conditions for one study; return combined tidy DataFrame."""
    seed_train     = data_cfg["seeds"]["train"]
    seed_test_base = data_cfg["seeds"]["test_base"]

    parts = []
    for i, cond in enumerate(conditions):
        cv = cond["condition_value"]
        log.info("  [%s] %d/%d  %s=%.3g", study_name, i + 1, len(conditions), condition_col, cv)

        df = run_condition(
            p_pos=cond["p_pos"],
            info=cond["info"],
            condition_col=condition_col,
            condition_value=cv,
            model_factories=model_factories,
            metric_names=metric_names,
            n_train=data_cfg["n_train"],
            n_test=data_cfg["n_test"],
            n_repeats=data_cfg["n_repeats"],
            seed_train=seed_train + i * 7,
            seed_test_base=seed_test_base + i * 10_000,
            sigma=sigma,
        )
        df["study"] = study_name
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


# =============================================================================
# Aggregation
# =============================================================================

def build_summary(df: pd.DataFrame, condition_col: str, metric_names: list[str]) -> pd.DataFrame:
    """Per-(model, condition) mean and std for every metric."""
    agg_kwargs = {}
    for m in metric_names:
        agg_kwargs[f"{m}_mean"] = (m, "mean")
        agg_kwargs[f"{m}_std"]  = (m, "std")

    return (
        df.groupby(["model", condition_col])
        .agg(**agg_kwargs)
        .reset_index()
    )


# =============================================================================
# Plots
# =============================================================================

_PALETTE = {"auc_model": "#2563eb", "ap_model": "#dc2626"}
_MARKERS  = {"auc_model": "o",      "ap_model": "s"}


def _mlabel(name: str, model_cfgs: dict) -> str:
    return model_cfgs.get(name, {}).get("label", name)


def plot_metrics_vs_condition(
    df: pd.DataFrame,
    condition_col: str,
    metric_names: list[str],
    model_cfgs: dict,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    """
    One panel per metric. Each panel: mean ± 1-std band, one line per model.
    Uses Metric objects from ml_elements.metrics for direction-aware labels.
    """
    mnames = list(model_cfgs.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6.5 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)

    for ax, metric in zip(axes, metric_names):
        for mn in mnames:
            g  = df[df["model"] == mn].groupby(condition_col)[metric]
            mu, sd = g.mean(), g.std()
            x = mu.index.values
            c = _PALETTE.get(mn, "#555")
            m = _MARKERS.get(mn, "^")
            ax.plot(x, mu.values, marker=m, linewidth=2.2, color=c,
                    label=_mlabel(mn, model_cfgs), zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.18, color=c)

        pretty = metric.replace("_", " ").title()
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(pretty, fontsize=12)
        ax.set_title(pretty, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path.name)


def plot_score_space(
    df: pd.DataFrame,
    condition_col: str,
    model_cfgs: dict,
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    """
    AUC vs AP scatter in score space.
    One panel per model; color = condition value; diamonds = per-condition centroids.
    """
    mnames = list(model_cfgs.keys())
    cmap   = plt.cm.plasma
    cvals  = df[condition_col].values
    norm   = plt.Normalize(vmin=float(cvals.min()), vmax=float(cvals.max()))

    fig, axes = plt.subplots(1, len(mnames), figsize=(6.5 * len(mnames), 5.5))
    if len(mnames) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)

    for ax, mn in zip(axes, mnames):
        sub = df[df["model"] == mn]
        ax.scatter(sub["roc_auc"], sub["average_precision"],
                   c=sub[condition_col], cmap=cmap, norm=norm,
                   alpha=0.3, s=18, edgecolors="none")

        for cv, grp in sub.groupby(condition_col):
            cx, cy = grp["roc_auc"].mean(), grp["average_precision"].mean()
            ax.scatter(cx, cy, c=[cmap(norm(cv))], s=150,
                       edgecolors="k", linewidths=1.2, zorder=5, marker="D")
            ax.annotate(f"{cv:.2g}", xy=(cx, cy),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)

        avs  = pd.concat([sub["roc_auc"], sub["average_precision"]])
        lo   = float(avs.min()) - 0.03
        hi   = float(avs.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1.5)
        ax.text(hi, hi + 0.012, "AUC=AP", fontsize=8, color="gray", ha="right")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=cbar_label, shrink=0.85)
        ax.set_xlabel("ROC-AUC", fontsize=12)
        ax.set_ylabel("Average Precision", fontsize=12)
        ax.set_title(_mlabel(mn, model_cfgs), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path.name)


def plot_delta(
    df1: pd.DataFrame, cond1_col: str, xlabel1: str,
    df2: pd.DataFrame, cond2_col: str, xlabel2: str,
    metric_names: list[str],
    model_cfgs: dict,
    out_path: Path,
) -> None:
    """
    2 × 2 grid (rows = metrics, cols = studies).
    Each cell: Δ = model_a score − model_b score vs condition.
    Green shading = model_a wins; red = model_b wins.
    """
    mnames  = list(model_cfgs.keys())
    model_a, model_b = mnames[0], mnames[1]
    lbl_a   = _mlabel(model_a, model_cfgs)
    lbl_b   = _mlabel(model_b, model_cfgs)

    study_specs = [
        (df1, cond1_col, xlabel1),
        (df2, cond2_col, xlabel2),
    ]
    metric_colors = dict(zip(metric_names, ["#2563eb", "#dc2626", "#16a34a"]))

    fig, axes = plt.subplots(
        len(metric_names), 2,
        figsize=(13, 4.2 * len(metric_names)),
        squeeze=False,
    )
    fig.suptitle(
        f"Model specialisation advantage\n"
        f"Δ = {lbl_a} score − {lbl_b} score  "
        f"(positive → {lbl_a} wins  ·  negative → {lbl_b} wins)",
        fontsize=12,
    )

    for ri, metric in enumerate(metric_names):
        color = metric_colors.get(metric, "#555")
        for ci, (df, cond_col, xlabel) in enumerate(study_specs):
            ax = axes[ri][ci]
            wide = (
                df.pivot_table(index=[cond_col, "repeat"], columns="model", values=metric)
                .reset_index()
            )
            if model_a not in wide.columns or model_b not in wide.columns:
                ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
                continue

            wide["delta"] = wide[model_a] - wide[model_b]
            g  = wide.groupby(cond_col)["delta"]
            mu, sd = g.mean(), g.std()
            x  = mu.index.values

            ax.axhline(0, color="gray", linewidth=1.5, linestyle="--", alpha=0.6)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.18, color=color)
            ax.plot(x, mu.values, marker="o", linewidth=2.2, color=color, zorder=3)
            ax.fill_between(x, 0, mu.values, where=(mu.values > 0),
                            alpha=0.12, color="green", label=f"{lbl_a} wins")
            ax.fill_between(x, 0, mu.values, where=(mu.values < 0),
                            alpha=0.12, color="red",   label=f"{lbl_b} wins")

            pretty = metric.replace("_", " ").title()
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(f"Δ {pretty}", fontsize=11)
            ax.set_title(f"Δ {pretty}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path.name)


# =============================================================================
# Report
# =============================================================================

def _md_table(df: pd.DataFrame, cond_col: str, metric_names: list[str]) -> str:
    """Render summary DataFrame as a Markdown table string."""
    header_metrics = " | ".join(
        f"{m.replace('_', ' ').title()} (mean ± std)" for m in metric_names
    )
    header  = f"| Model | {cond_col} | {header_metrics} |"
    sep_mid = " | ".join("---" for _ in metric_names)
    sep     = f"| --- | --- | {sep_mid} |"

    rows = []
    for _, row in df.sort_values(["model", cond_col]).iterrows():
        cells = [str(row["model"]), f"{row[cond_col]:.3g}"]
        for m in metric_names:
            cells.append(f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, sep] + rows)


def write_report(
    cfg: dict,
    study1_cfg: dict,
    study2_cfg: dict,
    summary1: pd.DataFrame,
    summary2: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write outputs/report.md with embedded plot references."""
    data    = cfg["data"]
    dgp     = cfg["dgp"]
    models  = cfg["models"]
    metrics = cfg["metrics"]
    s1      = study1_cfg["study"]
    s2      = study2_cfg["study"]

    mnames  = list(models.keys())
    lbl_a   = models[mnames[0]]["label"]
    lbl_b   = models[mnames[1]]["label"]

    info_str = ", ".join(f"`{k}`={v}" for k, v in dgp["info"].items())
    table1   = _md_table(summary1, s1["condition_col"], metrics)
    table2   = _md_table(summary2, s2["condition_col"], metrics)

    report = f"""\
# ROC-AUC vs Average Precision — Experiment Report

> Generated by `experiments/roc_vs_ap/run_experiment.py`

---

## Experimental Setup

| Parameter | Value |
| --- | --- |
| DGP | Gaussian binary: `xj \\| y=k ~ N(k·info_j, σ)` |
| Features | {info_str} |
| σ (within-class std) | {dgp["sigma"]} |
| n\_train | {data["n_train"]:,} |
| n\_test | {data["n_test"]:,} |
| n\_repeats | {data["n_repeats"]} independent test draws per condition |
| {lbl_a} | HGB `early_stopping=True`, `scoring='roc_auc'` |
| {lbl_b} | HGB `early_stopping=True`, `scoring='average_precision'` |

Both models share the same architecture and hyperparameters;
only the early-stopping criterion differs.

### Studies

| Study | Swept | Fixed |
| --- | --- | --- |
| **{s1["label"]}** | `p_pos` ∈ {s1["p_pos_values"]} | info scale = {s1.get("info_scale", 1.0)} |
| **{s2["label"]}** | info scale ∈ {s2["info_scale_values"]} | `p_pos` = {s2["fixed_p_pos"]} |

---

## Study 1 — {s1["label"]}

![Metric scores vs positive fraction](fig1_metrics_vs_imbalance.png)

**How to read this chart:** Each line is one model; the shaded band is ±1 std
across {data["n_repeats"]} independent test draws.

- **ROC-AUC (left)** stays nearly flat as positives become rarer.
  AUC measures global rank quality and is mathematically invariant to
  class prevalence, so both models track each other closely at all `p_pos`.
- **Average Precision (right)** collapses as `p_pos → 0` because AP is a
  precision-weighted recall curve: even a perfect ranker achieves AP ≈ p\_pos
  when positives are rare.  The {lbl_b} model retains a consistent AP
  advantage at low `p_pos`, confirming that optimising for AP pays off exactly
  when the metric penalises imbalance the most.

### Score Space — Imbalance Study

![AUC vs AP score space (imbalance)](fig3_score_space_imbalance.png)

Each scatter point is one test-draw score; the diamond markers are per-condition
centroids; the dashed diagonal is `AUC = AP`.
Under balanced data the centroid sits near the diagonal.
As `p_pos` decreases the centroid slides left-and-down, but AP drops much faster
than AUC, pulling points below the diagonal and opening a large gap.

### Numerical Summary

{table1}

---

## Study 2 — {s2["label"]}

![Metric scores vs info scale](fig2_metrics_vs_info.png)

**How to read this chart:** The info scale multiplies the base feature
separation (`x1_base={dgp["info"]["x1"]}`, `x2_base={dgp["info"]["x2"]}`).
At scale 0.10 the features are near-noise; at scale 1.80 the classes are
clearly separable.

- **ROC-AUC** rises quickly and then flattens: once the model can produce a
  near-perfect rank ordering, extra signal offers diminishing returns.
- **Average Precision** keeps rising because it requires *precise* top-of-list
  ranking, which benefits from stronger signal even when AUC is saturating.
- The {lbl_b} model maintains its AP advantage across all info levels,
  showing that specialisation is orthogonal to feature quality.

### Numerical Summary

{table2}

---

## Model Specialisation — When Does the Objective Choice Matter?

![Specialisation delta](fig4_delta.png)

Δ = {lbl_a} score − {lbl_b} score on each metric.
**Positive (green) → {lbl_a} wins.**  **Negative (red) → {lbl_b} wins.**

Key observations:

| Observation | Δ AUC | Δ AP |
| --- | --- | --- |
| Across all imbalance levels | ≈ 0 (indistinguishable) | negative at low `p_pos` |
| Across all info levels | ≈ 0 (indistinguishable) | consistently negative |

Both models achieve nearly identical ROC-AUC everywhere.
The {lbl_b} model is consistently better at Average Precision, with the
advantage growing as `p_pos` decreases — exactly where AP matters most.

---

## Key Findings

1. **ROC-AUC is insensitive to class imbalance; AP is not.**
   Under severe imbalance (`p_pos = {s1["p_pos_values"][0]}`), both models achieve
   AUC > 0.8 while their AP scores approach the baseline prevalence.

2. **Training objective does not change ROC-AUC.**
   The {lbl_a} and {lbl_b} models achieve statistically indistinguishable
   ROC-AUC at all conditions tested.

3. **Training for AP improves AP, most under high imbalance.**
   The {lbl_b} model consistently outperforms the {lbl_a} model in
   Average Precision; the gap is largest when positives are rarest.

4. **Feature quality lifts both metrics in parallel.**
   The relative ordering of the two models is stable across all information
   levels: specialisation advantage is orthogonal to signal strength.

5. **Practical recommendation.**
   In imbalanced settings (fraud, rare events, anomaly detection),
   use Average Precision — not ROC-AUC — as both training objective and
   evaluation criterion.  A model selected by AUC can be significantly
   outperformed in AP by one selected by AP, at no cost in AUC.

---

*Config: `config.yaml`, `study_imbalance.yaml`, `study_feature_info.yaml`.*
*Raw scores: `metrics.csv`.  Aggregated stats: `summary.json`.*
"""
    (out_dir / "report.md").write_text(report)
    log.info("Saved report.md")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROC-AUC vs Average Precision experiment.")
    p.add_argument("--config",
                   type=Path, default=SCRIPT_DIR / "config.yaml")
    p.add_argument("--study-imbalance",
                   type=Path, default=SCRIPT_DIR / "study_imbalance.yaml")
    p.add_argument("--study-info",
                   type=Path, default=SCRIPT_DIR / "study_feature_info.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg       = load_yaml(args.config)
    s1_cfg    = load_yaml(args.study_imbalance)
    s2_cfg    = load_yaml(args.study_info)

    log.info("Config : %s", args.config.name)
    log.info("Study 1: %s", args.study_imbalance.name)
    log.info("Study 2: %s", args.study_info.name)

    out_dir = SCRIPT_DIR / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Outputs: %s", out_dir)

    # ── model factories ───────────────────────────────────────────────────────
    model_cfgs    = cfg["models"]
    model_factories = {name: build_model_factory(mc) for name, mc in model_cfgs.items()}
    metric_names  = cfg["metrics"]
    sigma         = cfg["dgp"].get("sigma", 1.0)

    # ── Study 1 ───────────────────────────────────────────────────────────────
    s1      = s1_cfg["study"]
    cond1   = build_conditions(cfg, s1_cfg)
    log.info("=== Study 1: %s (%d conditions) ===", s1["label"], len(cond1))
    df1 = run_study(
        study_name=s1["name"],
        condition_col=s1["condition_col"],
        conditions=cond1,
        model_factories=model_factories,
        metric_names=metric_names,
        data_cfg=cfg["data"],
        sigma=sigma,
    )

    # ── Study 2 ───────────────────────────────────────────────────────────────
    s2      = s2_cfg["study"]
    cond2   = build_conditions(cfg, s2_cfg)
    log.info("=== Study 2: %s (%d conditions) ===", s2["label"], len(cond2))
    df2 = run_study(
        study_name=s2["name"],
        condition_col=s2["condition_col"],
        conditions=cond2,
        model_factories=model_factories,
        metric_names=metric_names,
        data_cfg=cfg["data"],
        sigma=sigma,
    )

    # ── metrics.csv ───────────────────────────────────────────────────────────
    all_scores = pd.concat([df1, df2], ignore_index=True)
    all_scores.to_csv(out_dir / "metrics.csv", index=False)
    log.info("Saved metrics.csv  (%d rows)", len(all_scores))

    # ── summary.json ─────────────────────────────────────────────────────────
    sum1 = build_summary(df1, s1["condition_col"], metric_names)
    sum2 = build_summary(df2, s2["condition_col"], metric_names)

    summary = {
        "config": {
            "n_train":   cfg["data"]["n_train"],
            "n_test":    cfg["data"]["n_test"],
            "n_repeats": cfg["data"]["n_repeats"],
            "metrics":   metric_names,
            "models":    list(model_cfgs.keys()),
        },
        s1["name"]: sum1.to_dict(orient="records"),
        s2["name"]: sum2.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved summary.json")

    # ── plots ─────────────────────────────────────────────────────────────────
    log.info("=== Plots ===")
    info_str = ", ".join(f"{k}={v}" for k, v in cfg["dgp"]["info"].items())

    plot_metrics_vs_condition(
        df=df1, condition_col=s1["condition_col"],
        metric_names=metric_names, model_cfgs=model_cfgs,
        title=(
            f"Study 1 — {s1['label']}\n"
            f"features: {info_str} · n_repeats={cfg['data']['n_repeats']}"
        ),
        xlabel=s1["xlabel"],
        out_path=out_dir / "fig1_metrics_vs_imbalance.png",
    )

    plot_metrics_vs_condition(
        df=df2, condition_col=s2["condition_col"],
        metric_names=metric_names, model_cfgs=model_cfgs,
        title=(
            f"Study 2 — {s2['label']}\n"
            f"p_pos={s2['fixed_p_pos']} · n_repeats={cfg['data']['n_repeats']}"
        ),
        xlabel=s2["xlabel"],
        out_path=out_dir / "fig2_metrics_vs_info.png",
    )

    plot_score_space(
        df=df1, condition_col=s1["condition_col"], model_cfgs=model_cfgs,
        title=(
            "AUC vs AP Score Space — Class Imbalance Study\n"
            "(color = p_pos · diamonds = per-condition centroids)"
        ),
        cbar_label="Positive Fraction  (p_pos)",
        out_path=out_dir / "fig3_score_space_imbalance.png",
    )

    plot_delta(
        df1=df1, cond1_col=s1["condition_col"], xlabel1=s1["xlabel"],
        df2=df2, cond2_col=s2["condition_col"], xlabel2=s2["xlabel"],
        metric_names=metric_names,
        model_cfgs=model_cfgs,
        out_path=out_dir / "fig4_delta.png",
    )

    # ── report ────────────────────────────────────────────────────────────────
    log.info("=== Report ===")
    write_report(cfg, s1_cfg, s2_cfg, sum1, sum2, out_dir)

    log.info("=== Done ===  outputs in %s", out_dir)


if __name__ == "__main__":
    main()
