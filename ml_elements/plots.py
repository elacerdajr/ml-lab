"""
plots.py
--------
Visualization functions for study results.

All functions return a ``matplotlib.Figure``. The caller decides what to
do with it (``plt.show()``, ``fig.savefig(...)``, inline in a notebook).

Functions
---------
plot_study
    Improvement curve across trial values: mean ± std, P(Δ>0) annotated.

plot_calibration
    Calibration curve (predicted vs. observed probability) for one setup
    within a ``TrialResult``.

plot_feature_importance
    Horizontal bar chart of feature importances (or coefficients for
    logistic regression) for one setup in a ``TrialResult``.

plot_search_heatmap
    2D heatmap of a metric value over two parameter axes. Useful after
    ``SobolSearch`` or ``RandomSearch`` to spot interaction effects.

plot_score_distributions
    Violin / box plot of per-repeat scores for baseline vs. challenger,
    grouped by trial value.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

from .metrics import Metric
from .runner import TrialResult
from .study import StudyResult


def plot_study(
    summary: pd.DataFrame,
    metric: Metric,
    title: str = "",
    x_label: str = "trial_value",
    figsize: tuple[float, float] = (9, 5),
) -> Figure:
    """
    Plot mean improvement ± std across trial values.

    The zero line is drawn for reference. Data points where
    ``p_improvement_gt_0 >= 0.8`` are highlighted in green; below 0.5 in red.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of ``Study.summarize()``. Expected columns:
        ``trial_value``, ``mean_improvement``, ``std_improvement``,
        ``p_improvement_gt_0``.
    metric : Metric
        Used for the y-axis label.
    title : str
        Plot title. Defaults to a generic description.
    x_label : str
        x-axis label.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = summary["trial_value"].values
    y = summary["mean_improvement"].values
    err = summary["std_improvement"].fillna(0).values
    p_win = summary.get("p_improvement_gt_0", pd.Series(np.full(len(x), np.nan))).values

    ax.axhline(0, color="gray", linewidth=1.5, alpha=0.4, linestyle="--")
    ax.fill_between(x, y - err, y + err, alpha=0.15, color="steelblue")
    ax.plot(x, y, marker="o", linewidth=2.5, color="steelblue", zorder=3)

    for xi, yi, pi in zip(x, y, p_win):
        if not np.isnan(pi):
            color = "#2ca02c" if pi >= 0.8 else ("#d62728" if pi < 0.5 else "gray")
            ax.scatter(xi, yi, color=color, s=80, zorder=4)
            ax.annotate(
                f"P={pi:.0%}",
                xy=(xi, yi),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )

    direction_label = "↑ better" if metric.direction == "higher" else "↓ better"
    ax.set_ylabel(f"Mean improvement in {metric.name.upper()}  ({direction_label})")
    ax.set_xlabel(x_label)
    ax.set_title(title or f"Improvement in {metric.name.upper()} across {x_label}")
    ax.grid(True, linewidth=1, alpha=0.1)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def plot_calibration(
    trial_result: TrialResult,
    setup: str,
    n_bins: int = 10,
    figsize: tuple[float, float] = (7, 6),
) -> Figure:
    """
    Calibration curve: predicted probability vs. observed fraction of positives.

    Uses the validation set from ``TrialResult.df_valid`` and the fitted model
    from ``TrialResult.models[setup]``.

    Parameters
    ----------
    trial_result : TrialResult
    setup : str
        Setup name, e.g. ``"challenger"``.
    n_bins : int
        Number of calibration bins.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the setup is not found or feature columns are missing.
    """
    if setup not in trial_result.models:
        raise ValueError(
            f"Setup {setup!r} not in trial_result.models. "
            f"Available: {list(trial_result.models.keys())}"
        )

    model = trial_result.models[setup]
    df_val = trial_result.df_valid

    features = [c for c in df_val.columns if c != "y"]
    y_true = df_val["y"].to_numpy()
    p_hat = model.predict_proba(df_val[features])[:, 1]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p_hat, bin_edges[1:-1])

    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(float(p_hat[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, label="Perfect calibration")
    ax_cal.plot(mean_pred, frac_pos, "o-", linewidth=2, markersize=6, label=f"{setup}")
    ax_cal.set_ylabel("Observed fraction of positives")
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.set_title(
        f"Calibration — {setup}  "
        f"(trial {trial_result.trial.name}={trial_result.trial.value:.4g})"
    )
    ax_cal.legend(frameon=False)
    ax_cal.grid(True, alpha=0.1)
    ax_cal.spines[["top", "right"]].set_visible(False)

    ax_hist.bar(mean_pred, counts, width=0.07, color="steelblue", alpha=0.6)
    ax_hist.set_xlabel("Mean predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def plot_feature_importance(
    trial_result: TrialResult,
    setup: str,
    top_n: int = 20,
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """
    Horizontal bar chart of feature importances or regression coefficients.

    Supports:
    - Tree models with ``feature_importances_`` (HGB, CatBoost, RF, …)
    - Linear models with ``coef_`` (LogisticRegression, …)

    Parameters
    ----------
    trial_result : TrialResult
    setup : str
        Setup name.
    top_n : int
        Maximum number of features to display.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the model has neither ``feature_importances_`` nor ``coef_``.
    """
    if setup not in trial_result.models:
        raise ValueError(
            f"Setup {setup!r} not found. Available: {list(trial_result.models.keys())}"
        )

    model = trial_result.models[setup]
    df = trial_result.df_train
    features = [c for c in df.columns if c != "y"]

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        importance_type = "Feature importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
        importance_type = "|Coefficient|"
    else:
        raise ValueError(
            f"Model for setup {setup!r} has neither feature_importances_ nor coef_. "
            "Cannot plot importance."
        )

    df_imp = (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#d62728" if v < 0 else "steelblue" for v in df_imp["importance"]]
    ax.barh(df_imp["feature"], df_imp["importance"], color=colors)
    ax.set_xlabel(importance_type)
    ax.set_title(
        f"{importance_type} — {setup}  "
        f"(trial {trial_result.trial.name}={trial_result.trial.value:.4g})"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_search_heatmap(
    result: StudyResult,
    param_x: str,
    param_y: str,
    metric_col: str = "mean_improvement",
    summary: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """
    2D heatmap of a metric value over two parameter axes.

    Useful after ``SobolSearch`` or ``RandomSearch`` to visualise interaction
    effects between two parameters.

    Parameters
    ----------
    result : StudyResult
        Used to extract trial names/values if ``summary`` is not provided.
    param_x : str
        Column name for the x-axis parameter (in ``summary`` or trial metadata).
    param_y : str
        Column name for the y-axis parameter.
    metric_col : str
        Column in ``summary`` to use as the heatmap colour.
    summary : pd.DataFrame | None
        Pre-computed summary (output of ``Study.summarize()`` or custom).
        If None, uses ``result.scores`` directly.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    This function expects ``summary`` to have columns ``param_x``, ``param_y``,
    and ``metric_col``. If your study used a single-axis sweep, the heatmap
    degenerates to a single row/column — use ``plot_study`` instead.
    """
    from scipy.interpolate import griddata

    if summary is None:
        summary = result.scores

    if param_x not in summary.columns or param_y not in summary.columns:
        raise ValueError(
            f"Columns {param_x!r} and {param_y!r} not found in summary. "
            f"Available: {list(summary.columns)}"
        )

    x = summary[param_x].values.astype(float)
    y = summary[param_y].values.astype(float)
    z = summary[metric_col].values.astype(float)

    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi_grid, yi_grid), method="linear")

    fig, ax = plt.subplots(figsize=figsize)
    hm = ax.contourf(xi_grid, yi_grid, zi, levels=12, cmap="RdYlGn")
    ax.scatter(x, y, c=z, cmap="RdYlGn", edgecolors="black", linewidths=0.5, s=40, zorder=5)
    plt.colorbar(hm, ax=ax, label=metric_col)

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"{metric_col} over ({param_x}, {param_y})")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def plot_score_distributions(
    result: StudyResult,
    metric_name: str,
    setups: Sequence[str] | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """
    Box plot of per-repeat metric scores, grouped by trial value.

    Useful for spotting variance across trials and comparing distributions
    between setups side by side.

    Parameters
    ----------
    result : StudyResult
    metric_name : str
        Name of the metric column to plot.
    setups : list[str] | None
        Subset of setups to include. None = all setups.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = result.scores.copy()

    if metric_name not in df.columns:
        raise ValueError(
            f"Metric {metric_name!r} not in scores. Available: {list(df.columns)}"
        )

    all_setups = df["setup"].unique().tolist()
    if setups is not None:
        for s in setups:
            if s not in all_setups:
                raise ValueError(f"Setup {s!r} not found. Available: {all_setups}")
        df = df[df["setup"].isin(setups)]

    trial_values = sorted(df["trial_value"].unique())
    setup_names = df["setup"].unique().tolist()
    n_setups = len(setup_names)
    n_trials = len(trial_values)

    fig, ax = plt.subplots(figsize=figsize)
    width = 0.7 / n_setups
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for si, setup in enumerate(setup_names):
        offset = (si - (n_setups - 1) / 2) * width
        color = palette[si % len(palette)]

        for ti, tv in enumerate(trial_values):
            vals = df.loc[
                (df["setup"] == setup) & (df["trial_value"] == tv), metric_name
            ].values
            if len(vals) == 0:
                continue
            bp = ax.boxplot(
                vals,
                positions=[ti + offset],
                widths=width * 0.85,
                patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.6},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1},
                capprops={"linewidth": 1},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
            )

    ax.set_xticks(range(n_trials))
    ax.set_xticklabels([f"{v:.4g}" for v in trial_values])
    ax.set_xlabel("trial_value")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Score distributions: {metric_name}")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[i % len(palette)], alpha=0.6)
        for i in range(n_setups)
    ]
    ax.legend(handles, setup_names, frameon=False, title="setup")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
