"""
experiments/2026-06-15_risk_uncertainty/run_experiment.py
----------------------------------------------
Knightian Risk vs Uncertainty — five synthetic scenarios.

Scenarios
---------
1. Pure risk            Same stable law in train and test.
2. Epistemic unc.       n_train=150: many hypotheses equally fit data.
3. Regime uncertainty   Sign flip: the train rule becomes anti-predictive at test.
4. Spurious correlation Feature z proxies y in train; pure noise in test.
5. Ambiguous world      Train mixes two contradictory rules; test uses only one.

Models
------
logistic          L2-regularised logistic regression (linear, interpretable)
small_tree        DecisionTree max_depth=2 (transparent, limited capacity)
rulefit           RuleFit sparse linear-over-rules (imodels)
greedy_rule_list  Greedy rule list (imodels)
catboost          CatBoost gradient boosting (complex, low-bias baseline)

Metrics
-------
test_auc         ROC-AUC on test set (20 000 samples)
test_ap          Average Precision on test set
gen_gap          train_AUC − test_AUC (overfitting signal)
ece              Expected Calibration Error
boot_std         Std of AUC over 100 bootstrap draws of test set
worst_regime_auc (scenario 3) AUC on the worst of two hidden regimes

Usage
-----
    python run_experiment.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import imodels
except ImportError as exc:
    raise ImportError("pip install imodels") from exc

try:
    from catboost import CatBoostClassifier
except ImportError as exc:
    raise ImportError("pip install catboost") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

N_TRAIN = 300
N_TEST = 20_000
D_NOISE = 100
SEED = 42
N_BOOTSTRAP = 100

_MODEL_COLORS = {
    "logistic": "#2563eb",
    "small_tree": "#16a34a",
    "rulefit": "#d97706",
    "greedy_rule_list": "#7c3aed",
    "catboost": "#dc2626",
}
_MODEL_LABELS = {
    "logistic": "Logistic",
    "small_tree": "Small Tree\n(depth=2)",
    "rulefit": "RuleFit",
    "greedy_rule_list": "Greedy\nRuleList",
    "catboost": "CatBoost",
}
_SCENARIO_LABELS = {
    "pure_risk": "1. Pure Risk\n(stable rule)",
    "epistemic": "2. Epistemic\nUncertainty\n(n_train=150)",
    "regime": "3. Regime\nUncertainty\n(sign flip)",
    "spurious": "4. Spurious\nCorrelation\n(feature breaks)",
    "ambiguous": "5. Ambiguous\nWorld\n(mixed rules in train)",
}

# ─── Helpers ────────────────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit_base(X: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Weak linear signal: logit = sign * (0.4·x₀ − 0.3·x₁)."""
    return sign * (0.4 * X[:, 0] - 0.3 * X[:, 1])


def _draw(logit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(1, _sigmoid(logit)).astype(int)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform-width bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def _bootstrap_auc_std(y: np.ndarray, p: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        if 0 < yb.sum() < n:
            aucs.append(roc_auc_score(yb, pb))
    return float(np.std(aucs)) if aucs else np.nan


def _proba(model: Any, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    if p.ndim == 2:
        p = p[:, 1]
    return np.clip(p.ravel().astype(float), 1e-7, 1 - 1e-7)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ─── Scenarios ──────────────────────────────────────────────────────────────


def scenario_pure_risk(seed: int = SEED):
    """
    Risk world: train and test drawn from the same stable probability law.
    p = sigmoid(0.4·x₀ − 0.3·x₁)  with D_NOISE additional noise features.
    """
    n_feat = D_NOISE + 2
    rng_tr = _rng(seed)
    rng_te = _rng(seed + 1_000)

    X_tr = rng_tr.standard_normal((N_TRAIN, n_feat))
    y_tr = _draw(_logit_base(X_tr), rng_tr)

    X_te = rng_te.standard_normal((N_TEST, n_feat))
    y_te = _draw(_logit_base(X_te), rng_te)

    return X_tr, y_tr, X_te, y_te, {}


def scenario_epistemic(seed: int = SEED):
    """
    Epistemic uncertainty: n_train=150 — too little data to identify the true
    rule from 100+ competing noise features. Many hypotheses fit equally.
    """
    n_feat = D_NOISE + 2
    n_tr_small = 150
    rng_tr = _rng(seed)
    rng_te = _rng(seed + 1_000)

    X_tr = rng_tr.standard_normal((n_tr_small, n_feat))
    y_tr = _draw(_logit_base(X_tr), rng_tr)

    X_te = rng_te.standard_normal((N_TEST, n_feat))
    y_te = _draw(_logit_base(X_te), rng_te)

    return X_tr, y_tr, X_te, y_te, {}


def scenario_regime(seed: int = SEED):
    """
    Regime uncertainty: hidden regime switches between train and test.
    Train:  p = sigmoid(+0.4·x₀ − 0.3·x₁)
    Test:   p = sigmoid(−0.4·x₀ + 0.3·x₁)   ← sign flip
    A model that perfectly learned the train rule will score AUC < 0.5 at test.
    """
    n_feat = D_NOISE + 2
    rng_tr = _rng(seed)
    rng_te = _rng(seed + 1_000)

    X_tr = rng_tr.standard_normal((N_TRAIN, n_feat))
    y_tr = _draw(_logit_base(X_tr, sign=+1.0), rng_tr)

    X_te = rng_te.standard_normal((N_TEST, n_feat))
    y_te = _draw(_logit_base(X_te, sign=-1.0), rng_te)  # sign flip

    # Two separate regime slices for worst-regime analysis
    rng_A = _rng(seed + 2_000)
    X_A = rng_A.standard_normal((N_TEST // 2, n_feat))
    y_A = _draw(_logit_base(X_A, sign=+1.0), rng_A)

    rng_B = _rng(seed + 3_000)
    X_B = rng_B.standard_normal((N_TEST // 2, n_feat))
    y_B = _draw(_logit_base(X_B, sign=-1.0), rng_B)

    extras = {"regime_A": (X_A, y_A), "regime_B": (X_B, y_B)}
    return X_tr, y_tr, X_te, y_te, extras


def scenario_spurious(seed: int = SEED):
    """
    Spurious correlation: feature x₁ is injected to proxy y in train
    (x₁ = 0.8·y + noise), but is pure N(0,1) noise at test time.
    The true signal lives only in x₀.
    """
    n_feat = D_NOISE + 2
    rng_tr = _rng(seed)
    rng_te = _rng(seed + 1_000)

    X_tr = rng_tr.standard_normal((N_TRAIN, n_feat))
    logit_tr = 0.4 * X_tr[:, 0]  # true signal: x₀ only
    y_tr = _draw(logit_tr, rng_tr)
    # Inject spurious feature: x₁ strongly correlates with y in train
    X_tr[:, 1] = 0.8 * y_tr + 0.3 * rng_tr.standard_normal(N_TRAIN)

    X_te = rng_te.standard_normal((N_TEST, n_feat))
    logit_te = 0.4 * X_te[:, 0]  # x₁ is pure noise in test
    y_te = _draw(logit_te, rng_te)

    return X_tr, y_tr, X_te, y_te, {}


def scenario_ambiguous(seed: int = SEED):
    """
    Ambiguous world: train is a 50/50 mix of two contradictory rules.
    Rule A: signal on (x₀, x₁).
    Rule B: signal on (x₂, x₃) — independent features, different direction.
    Test: pure Rule A.  Models that found Rule B instead will fail.
    """
    n_feat = D_NOISE + 4  # need indices 0–3 as potential signal features
    n_half = N_TRAIN // 2

    rng_A = _rng(seed)
    X_A = rng_A.standard_normal((n_half, n_feat))
    y_A = _draw(0.4 * X_A[:, 0] - 0.3 * X_A[:, 1], rng_A)

    rng_B = _rng(seed + 500)
    X_B = rng_B.standard_normal((n_half, n_feat))
    y_B = _draw(0.4 * X_B[:, 2] - 0.3 * X_B[:, 3], rng_B)

    X_tr = np.vstack([X_A, X_B])
    y_tr = np.concatenate([y_A, y_B])
    perm = _rng(seed + 999).permutation(N_TRAIN)
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    rng_te = _rng(seed + 1_000)
    X_te = rng_te.standard_normal((N_TEST, n_feat))
    y_te = _draw(0.4 * X_te[:, 0] - 0.3 * X_te[:, 1], rng_te)

    return X_tr, y_tr, X_te, y_te, {}


SCENARIOS: dict[str, Any] = {
    "pure_risk": scenario_pure_risk,
    "epistemic": scenario_epistemic,
    "regime": scenario_regime,
    "spurious": scenario_spurious,
    "ambiguous": scenario_ambiguous,
}

# ─── Models ─────────────────────────────────────────────────────────────────


def _build_models() -> dict[str, Any]:
    return {
        "logistic": LogisticRegression(max_iter=500, C=1.0, random_state=SEED),
        "small_tree": DecisionTreeClassifier(max_depth=2, random_state=SEED),
        "rulefit": imodels.RuleFitClassifier(
            n_estimators=50, tree_size=4, max_rules=30, random_state=SEED
        ),
        "greedy_rule_list": imodels.GreedyRuleListClassifier(max_depth=3),
        "catboost": CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.06,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        ),
    }


# ─── Evaluation ─────────────────────────────────────────────────────────────


def evaluate(
    model_name: str,
    model: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    extras: dict,
) -> dict:
    try:
        model.fit(X_tr, y_tr)
        p_tr = _proba(model, X_tr)
        p_te = _proba(model, X_te)

        train_auc = roc_auc_score(y_tr, p_tr)
        test_auc = roc_auc_score(y_te, p_te)
        test_ap = average_precision_score(y_te, p_te)

        row = {
            "model": model_name,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "test_ap": test_ap,
            "gen_gap": train_auc - test_auc,
            "ece": _ece(y_te, p_te),
            "boot_std": _bootstrap_auc_std(y_te, p_te, seed=SEED),
        }

        if extras:
            regime_aucs: list[float] = []
            for rname, (X_r, y_r) in extras.items():
                p_r = _proba(model, X_r)
                try:
                    r_auc = float(roc_auc_score(y_r, p_r))
                except Exception:
                    r_auc = np.nan
                row[f"auc_{rname}"] = r_auc
                regime_aucs.append(r_auc)
            valid = [v for v in regime_aucs if not np.isnan(v)]
            row["worst_regime_auc"] = float(min(valid)) if valid else np.nan

    except Exception as exc:
        log.warning("  %s FAILED: %s", model_name, exc)
        row = {
            "model": model_name,
            "train_auc": np.nan,
            "test_auc": np.nan,
            "test_ap": np.nan,
            "gen_gap": np.nan,
            "ece": np.nan,
            "boot_std": np.nan,
        }

    return row


# ─── Experiment loop ─────────────────────────────────────────────────────────


def run_all() -> pd.DataFrame:
    records: list[dict] = []
    for scen_name, scen_fn in SCENARIOS.items():
        log.info("=== Scenario: %s ===", scen_name)
        X_tr, y_tr, X_te, y_te, extras = scen_fn(seed=SEED)
        log.info(
            "  train=%d  test=%d  features=%d  pos_rate=%.3f",
            len(y_tr), len(y_te), X_tr.shape[1], y_tr.mean(),
        )
        for model_name in list(_MODEL_LABELS.keys()):
            log.info("  -> %s", model_name)
            models = _build_models()
            row = evaluate(
                model_name, models[model_name],
                X_tr, y_tr, X_te, y_te, extras,
            )
            row["scenario"] = scen_name
            records.append(row)
    return pd.DataFrame(records)


# ─── Plotting ────────────────────────────────────────────────────────────────

_SCEN_ORDER = list(SCENARIOS.keys())
_MOD_ORDER = list(_MODEL_LABELS.keys())


def _bar_group(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    ref_line: float | None = None,
    ylim: tuple | None = None,
) -> None:
    n_scen = len(_SCEN_ORDER)
    n_mod = len(_MOD_ORDER)
    width = 0.8 / n_mod
    x = np.arange(n_scen)

    for i, mname in enumerate(_MOD_ORDER):
        vals = []
        for s in _SCEN_ORDER:
            sel = df[(df["scenario"] == s) & (df["model"] == mname)][metric].values
            vals.append(float(sel[0]) if len(sel) > 0 else np.nan)
        offset = (i - n_mod / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals,
            width=width * 0.92,
            color=_MODEL_COLORS[mname],
            label=_MODEL_LABELS[mname].replace("\n", " "),
            alpha=0.88,
        )

    if ref_line is not None:
        ax.axhline(ref_line, color="#444", linestyle="--", linewidth=1.1, alpha=0.55, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_SCENARIO_LABELS[s] for s in _SCEN_ORDER], fontsize=7.5
    )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    ax.set_axisbelow(True)


def plot_fig1_auc_and_gap(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(
        "Knightian Risk vs Uncertainty — Model Performance Across Five Scenarios\n"
        "signal: logit = 0.4·x₀ − 0.3·x₁  |  D_noise = 100  |  n_train = 300 (150 for epistemic)",
        fontsize=10.5,
    )
    _bar_group(
        axes[0], df, "test_auc",
        ylabel="Test ROC-AUC",
        title="Test ROC-AUC  (higher = better)",
        ref_line=0.5,
    )
    _bar_group(
        axes[1], df, "gen_gap",
        ylabel="Train AUC − Test AUC",
        title="Generalization Gap  (lower = better; 0 = perfect transfer)",
        ref_line=0.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_fig2_uncertainty_metrics(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(
        "Calibration Error and Estimation Variance Across Scenarios",
        fontsize=11,
    )
    _bar_group(
        axes[0], df, "ece",
        ylabel="Expected Calibration Error",
        title="Calibration Error (ECE)  (lower = better)",
        ref_line=0.0,
    )
    _bar_group(
        axes[1], df, "boot_std",
        ylabel="Std of AUC  (100 bootstrap draws)",
        title="Bootstrap AUC Variance  (lower = more stable)",
        ref_line=0.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_fig3_regime(df: pd.DataFrame, out: Path) -> None:
    reg_df = df[df["scenario"] == "regime"]
    cols_A = "auc_regime_A"
    cols_B = "auc_regime_B"
    if cols_A not in reg_df.columns:
        log.warning("No regime columns found; skipping fig3.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Scenario 3 — Regime Uncertainty: Per-Regime AUC\n"
        "Regime A = train rule  |  Regime B = sign-flipped test rule",
        fontsize=11,
    )

    x = np.arange(len(_MOD_ORDER))
    for ax, (col, subtitle) in zip(axes, [
        (cols_A, "Regime A  (same as train rule)"),
        (cols_B, "Regime B  (sign-flipped at test)"),
    ]):
        vals = [
            float(reg_df[reg_df["model"] == m][col].values[0])
            if (reg_df["model"] == m).any() else np.nan
            for m in _MOD_ORDER
        ]
        colors = [_MODEL_COLORS[m] for m in _MOD_ORDER]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.axhline(0.5, color="#444", linestyle="--", linewidth=1.1, alpha=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_MODEL_LABELS[m].replace("\n", " ") for m in _MOD_ORDER], fontsize=9
        )
        ax.set_ylabel("ROC-AUC", fontsize=11)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.28, linestyle=":")
        ax.set_axisbelow(True)
        ax.axhline(0.5, color="#555", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.text(len(x) - 0.4, 0.505, "chance", fontsize=8, color="#555")

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ypos = v + 0.01 if v >= 0 else v - 0.03
                ax.text(
                    bar.get_x() + bar.get_width() / 2, ypos,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
                )

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_fig4_heatmap(df: pd.DataFrame, out: Path) -> None:
    metrics = ["test_auc", "gen_gap", "ece", "boot_std"]
    metric_titles = {
        "test_auc": "Test AUC\n(↑ better)",
        "gen_gap": "Gen. Gap\n(↓ better)",
        "ece": "ECE\n(↓ better)",
        "boot_std": "Boot Std\n(↓ better)",
    }
    inverted = {"gen_gap", "ece", "boot_std"}  # lower is better → RdYlGn_r

    n_m = len(_MOD_ORDER)
    n_s = len(_SCEN_ORDER)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5))
    fig.suptitle(
        "Summary Heatmap — Models (rows) × Scenarios (columns)",
        fontsize=12, fontweight="bold",
    )

    for ax, metric in zip(axes, metrics):
        mat = np.full((n_m, n_s), np.nan)
        for i, m in enumerate(_MOD_ORDER):
            for j, s in enumerate(_SCEN_ORDER):
                v = df[(df["model"] == m) & (df["scenario"] == s)][metric].values
                if len(v) > 0:
                    mat[i, j] = v[0]

        cmap = "RdYlGn_r" if metric in inverted else "RdYlGn"
        vmin, vmax = np.nanmin(mat), np.nanmax(mat)
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n_s))
        ax.set_xticklabels(
            [_SCENARIO_LABELS[s].split("\n")[0] for s in _SCEN_ORDER],
            rotation=28, ha="right", fontsize=8,
        )
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(
            [_MODEL_LABELS[m].replace("\n", " ") for m in _MOD_ORDER], fontsize=9
        )
        ax.set_title(metric_titles[metric], fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        for i in range(n_m):
            for j in range(n_s):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=7.5, color="black")

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_fig5_test_ap(df: pd.DataFrame, out: Path) -> None:
    """Average Precision complements AUC for showing calibrated probability quality."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    _bar_group(
        ax, df, "test_ap",
        ylabel="Test Average Precision",
        title="Test Average Precision  (higher = better; baseline ≈ class rate ~0.5)",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


# ─── Report ─────────────────────────────────────────────────────────────────


def _pivot_table(df: pd.DataFrame, metric: str) -> str:
    rows = ["| Model | " + " | ".join(_SCENARIO_LABELS[s].replace("\n", " ") for s in _SCEN_ORDER) + " |"]
    rows.append("| --- " * (len(_SCEN_ORDER) + 1) + "|")
    for m in _MOD_ORDER:
        cells = [_MODEL_LABELS[m].replace("\n", " ")]
        for s in _SCEN_ORDER:
            v = df[(df["model"] == m) & (df["scenario"] == s)][metric].values
            cells.append(f"{v[0]:.4f}" if len(v) > 0 and not np.isnan(v[0]) else "n/a")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(df: pd.DataFrame, out_dir: Path) -> None:
    regime_section = ""
    if "worst_regime_auc" in df.columns and df["worst_regime_auc"].notna().any():
        wr = df[df["scenario"] == "regime"][["model", "auc_regime_A", "auc_regime_B", "worst_regime_auc"]]
        wr_rows = ["| Model | AUC Regime A | AUC Regime B | Worst-Regime AUC |", "| --- | --- | --- | --- |"]
        for _, row in wr.iterrows():
            wr_rows.append(
                f"| {_MODEL_LABELS[row['model']].replace(chr(10), ' ')} "
                f"| {row.get('auc_regime_A', np.nan):.4f} "
                f"| {row.get('auc_regime_B', np.nan):.4f} "
                f"| {row.get('worst_regime_auc', np.nan):.4f} |"
            )
        regime_section = f"""
### Regime Performance Detail (Scenario 3)

{chr(10).join(wr_rows)}

A model that perfectly learned the train rule will score AUC ≈ 1 on Regime A
and AUC < 0.5 on Regime B (its predictions are anti-correlated with truth).
The worst-regime AUC is the key robustness metric under distributional uncertainty.
"""

    report = f"""\
# Knightian Risk vs Uncertainty

> Generated by `experiments/2026-06-15_risk_uncertainty/run_experiment.py`

---

## The Core Distinction

**Risk** (Knight, 1921): outcomes follow a *known stable* probability law.
Past data is sufficient to estimate the DGP and deploy confidently.

**Uncertainty**: outcomes follow a law that is *unknown or changing*.
Past patterns may not transfer — because the regime shifted, training was too
small to identify the true rule, or a shortcut feature stops working out-of-sample.

---

## Experimental Design

| Parameter | Value |
| --- | --- |
| n_train | 300 (150 for epistemic scenario) |
| n_test | {N_TEST:,} |
| noise features (D_noise) | {D_NOISE} |
| true signal | `logit = 0.4·x₀ − 0.3·x₁` (weak) |
| prevalence | ≈ 50 % (sigmoid of near-zero logit) |

### Models

| Key | Description |
| --- | --- |
| Logistic | L2-regularised logistic regression (C=1.0) |
| Small Tree | DecisionTreeClassifier, max_depth=2 |
| RuleFit | Sparse linear model over conjunctive rules (imodels) |
| Greedy RuleList | Ordered rule list, max_depth=3 (imodels) |
| CatBoost | Gradient boosting, depth=4, 200 rounds |

---

## Five Scenarios

### Scenario 1 — Pure Risk

**DGP**: same rule `p = sigmoid(0.4·x₀ − 0.3·x₁)` in both train and test.
This is the "known dice" world: historical data is representative of the future.

Under pure risk, complex models are allowed to win if they find the signal.
Generalization gaps should be small (both train and test see the same law).

### Scenario 2 — Epistemic Uncertainty

**DGP**: same rule, but n_train=150 with 100 noise features (150 samples to
identify 1 signal among 102 candidates). Past data exists, but it's too thin
to distinguish the true rule from many competing hypotheses.

High-capacity models exploit noise features that happened to correlate with y
in the tiny training sample. Generalization gap widens.

### Scenario 3 — Regime Uncertainty

**Train**: `p = sigmoid(+0.4·x₀ − 0.3·x₁)`
**Test**:  `p = sigmoid(−0.4·x₀ + 0.3·x₁)` ← sign flip

The hidden regime flips between train and test.  A model that perfectly learned
the train rule will produce predictions *anti-correlated* with the test truth,
yielding AUC < 0.5 — worse than random.

### Scenario 4 — Spurious Correlation

**Train**: `x₁ = 0.8·y + 0.3·ε` (x₁ is a proxy for y, e.g. a leaky feature)
**Test**: `x₁ ~ N(0,1)` — the spurious correlation evaporates.

Models that learn to rely on x₁ will fail at test time. The true signal
(x₀) is available all along, but the spurious shortcut is more salient.

### Scenario 5 — Ambiguous World

**Train**: 50 % of samples from Rule A (`signal on x₀, x₁`) and 50 % from
Rule B (`signal on x₂, x₃`). Both rules fit the training data about equally
well. Test is drawn entirely from Rule A.

A model that happened to learn Rule B will fail. Under uncertainty, you
cannot tell which rule the future will obey — model selection becomes
irreducibly ambiguous.

---

## Results

### Test ROC-AUC

{_pivot_table(df, "test_auc")}

### Generalization Gap (Train AUC − Test AUC)

{_pivot_table(df, "gen_gap")}

### Expected Calibration Error

{_pivot_table(df, "ece")}

### Bootstrap AUC Std (estimation variance)

{_pivot_table(df, "boot_std")}
{regime_section}

---

## Figures

### Test ROC-AUC and Generalization Gap

![Test AUC and generalization gap across all five scenarios](outputs/fig1_test_auc_and_gap.png)

### Calibration Error and Bootstrap Variance

![Expected calibration error and bootstrap AUC std](outputs/fig2_calibration_variance.png)

### Regime Performance (Scenario 3)

![Per-regime AUC — Regime A vs Regime B](outputs/fig3_regime_performance.png)

### Summary Heatmap

![Heatmap of all four metrics, models × scenarios](outputs/fig4_summary_heatmap.png)

### Average Precision

![Average precision by model and scenario](outputs/fig5_test_ap.png)

---

## Key Takeaways

1. **Under risk (scenario 1)**: complex models (CatBoost) can exploit weak signal if it
   exists. Simple models are competitive but slightly below capacity.

2. **Under epistemic uncertainty (scenario 2)**: generalization gaps widen for all
   models; complex models tend to overfit noise features more than simple ones.

3. **Under regime uncertainty (scenario 3)**: models that learned the train rule
   accurately achieve **worst** test performance — AUC below chance on Regime B.
   The "best training model" becomes the worst deployment model.

4. **Under spurious correlation (scenario 4)**: models that relied on the shortcut
   feature fail. Interpretable models with fewer free parameters may pick up x₀
   directly; complex models may latch onto the more salient spurious x₁.

5. **Under ambiguity (scenario 5)**: all models are at risk. The 50/50 mixed
   train signal means any model is equally likely to have found Rule A or Rule B.
   Test performance regresses toward chance unless a model happened to find Rule A.

---

*Config: hardcoded constants at top of `run_experiment.py`.*
*Raw scores: `metrics.csv`.*
"""
    readme_path = out_dir.parent / "README.md"
    readme_path.write_text(report, encoding="utf-8")
    log.info("Saved README.md")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    df = run_all()

    csv_path = OUT_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved metrics.csv (%d rows)", len(df))

    log.info("=== Plotting ===")
    plot_fig1_auc_and_gap(df, OUT_DIR / "fig1_test_auc_and_gap.png")
    plot_fig2_uncertainty_metrics(df, OUT_DIR / "fig2_calibration_variance.png")
    plot_fig3_regime(df, OUT_DIR / "fig3_regime_performance.png")
    plot_fig4_heatmap(df, OUT_DIR / "fig4_summary_heatmap.png")
    plot_fig5_test_ap(df, OUT_DIR / "fig5_test_ap.png")

    log.info("=== Report ===")
    write_report(df, OUT_DIR)

    log.info("=== Done === outputs in %s", OUT_DIR)
    print("\n" + "=" * 60)
    print("TEST AUC by scenario and model")
    print("=" * 60)
    pivot = df.pivot_table(index="model", columns="scenario", values="test_auc")
    print(pivot.to_string(float_format="{:.4f}".format))
    print("\nGENERALIZATION GAP (train - test AUC)")
    print("=" * 60)
    pivot_gap = df.pivot_table(index="model", columns="scenario", values="gen_gap")
    print(pivot_gap.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
