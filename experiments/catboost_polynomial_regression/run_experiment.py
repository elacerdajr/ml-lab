"""
experiments/catboost_polynomial_regression/run_experiment.py
------------------------------------------------------------
CatBoost regression on a degree-5 polynomial DGP with two features.

  x1  — carries all the signal  (poly-5 relationship with y)
  x2  — pure uniform noise       (no dependence on y)

Sobol sequence samples (depth, iterations) hyperparameter pairs, giving a
low-discrepancy spread from tiny to large models.  Each fitted model's
predictions are overlaid as a coloured line on the (x1, y) scatter so you
can directly see how complexity affects the fit.

Usage
-----
    python run_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── experiment constants ───────────────────────────────────────────────────────
SEED        = 42
N_TRAIN     = 300
N_SOBOL     = 16          # power of 2 → cleanest Sobol coverage
X1_RANGE    = (-2.0, 2.0)
NOISE_STD   = 0.5

DEPTH_RANGE = (1, 10)     # CatBoost tree depth
ITER_RANGE  = (10, 500)   # boosting rounds


# ── data-generating process ────────────────────────────────────────────────────
# y = x1^5 - 4*x1^3 + 3*x1 + ε
# roots at x1 ∈ {-1, 0, 1}; global range ≈ [-6, 6] over [-2, 2]
_COEFFS = [0.0, 3.0, 0.0, -4.0, 0.0, 1.0]   # a0 … a5


def _poly5(x: np.ndarray) -> np.ndarray:
    return sum(c * x**i for i, c in enumerate(_COEFFS))


def generate_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1  = rng.uniform(*X1_RANGE, size=n)
    x2  = rng.uniform(-3.0, 3.0, size=n)   # irrelevant feature
    y   = _poly5(x1) + rng.normal(0.0, NOISE_STD, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# ── Sobol hyperparameter sampling ──────────────────────────────────────────────

def sample_sobol_params(n_points: int, seed: int) -> list[dict]:
    """Return n_points dicts with integer 'depth' and 'iterations' keys."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    raw     = sampler.random(n_points)
    scaled  = qmc.scale(
        raw,
        l_bounds=[DEPTH_RANGE[0], ITER_RANGE[0]],
        u_bounds=[DEPTH_RANGE[1], ITER_RANGE[1]],
    )
    return [
        {"depth": int(round(row[0])), "iterations": int(round(row[1]))}
        for row in scaled
    ]


# ── model training ─────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, depth: int, iterations: int):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        depth=depth,
        iterations=iterations,
        learning_rate=0.1,
        random_seed=SEED,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(df[["x1", "x2"]].values, df["y"].values)
    return model


def predict_on_grid(model, x1_grid: np.ndarray) -> np.ndarray:
    x2_zeros = np.zeros_like(x1_grid)
    return model.predict(np.column_stack([x1_grid, x2_zeros]))


# ── plots ──────────────────────────────────────────────────────────────────────

def _complexity(depth: int, iterations: int) -> float:
    return depth * np.log1p(iterations)


def plot_trials(
    df_train: pd.DataFrame,
    trials: list[dict],
    x1_grid: np.ndarray,
    out_path: Path,
) -> None:
    """
    Scatter of training data with one prediction line per Sobol trial.
    Lines are coloured by depth × ln(1 + iterations).
    """
    cvals = np.array([_complexity(t["depth"], t["iterations"]) for t in trials])
    norm  = plt.Normalize(vmin=cvals.min(), vmax=cvals.max())
    cmap  = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(
        df_train["x1"], df_train["y"],
        s=16, alpha=0.35, color="#5b9bd5", edgecolors="none",
        label="Training data  (x1 relevant, x2 noise)", zorder=2,
    )

    ax.plot(
        x1_grid, _poly5(x1_grid),
        "k--", linewidth=2.5, label="True polynomial", zorder=6,
    )

    for t in trials:
        c = cmap(norm(_complexity(t["depth"], t["iterations"])))
        ax.plot(x1_grid, t["preds"], color=c, linewidth=1.5, alpha=0.85, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Complexity  =  depth × ln(1 + iterations)", fontsize=10)

    # annotate min / max complexity trials
    lo_idx = int(np.argmin(cvals))
    hi_idx = int(np.argmax(cvals))
    for idx, label in [(lo_idx, "smallest"), (hi_idx, "largest")]:
        t = trials[idx]
        mid = len(x1_grid) // 2
        ax.annotate(
            f"{label}\nd={t['depth']}, it={t['iterations']}",
            xy=(x1_grid[mid], t["preds"][mid]),
            xytext=(20, 18), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
            fontsize=8, color="dimgray",
        )

    ax.set_xlabel("x1  (the relevant feature)", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"CatBoost Regressor — {N_SOBOL} Sobol trials\n"
        f"depth ∈ {DEPTH_RANGE}  ·  iterations ∈ {ITER_RANGE}  ·  "
        f"DGP: y = x₁⁵ − 4x₁³ + 3x₁ + ε",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_hyperparam_space(trials: list[dict], out_path: Path) -> None:
    """Scatter of (depth, iterations) Sobol points coloured by complexity."""
    depths = [t["depth"] for t in trials]
    iters  = [t["iterations"] for t in trials]
    cvals  = [_complexity(t["depth"], t["iterations"]) for t in trials]
    norm   = plt.Normalize(vmin=min(cvals), vmax=max(cvals))
    cmap   = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        depths, iters, c=cvals, cmap=cmap, norm=norm,
        s=140, edgecolors="k", linewidths=0.8, zorder=3,
    )
    for i, (d, it) in enumerate(zip(depths, iters)):
        ax.annotate(
            str(i + 1), (d, it),
            textcoords="offset points", xytext=(7, 5), fontsize=8,
        )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Complexity  =  depth × ln(1 + iterations)", fontsize=10)
    ax.set_xlabel("depth", fontsize=12)
    ax.set_ylabel("iterations", fontsize=12)
    ax.set_title(f"Sobol hyperparameter sampling  ({N_SOBOL} points)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_predictions_grid(
    df_train: pd.DataFrame,
    trials: list[dict],
    x1_grid: np.ndarray,
    out_path: Path,
) -> None:
    """
    One small panel per trial so individual fits are easy to inspect.
    Panels arranged in a 4×4 grid, ordered by increasing complexity.
    """
    cvals   = np.array([_complexity(t["depth"], t["iterations"]) for t in trials])
    order   = np.argsort(cvals)
    norm    = plt.Normalize(vmin=cvals.min(), vmax=cvals.max())
    cmap    = plt.cm.plasma

    ncols = 4
    nrows = (len(trials) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    true_curve = _poly5(x1_grid)

    for plot_rank, trial_idx in enumerate(order):
        ax = axes_flat[plot_rank]
        t  = trials[trial_idx]
        c  = cmap(norm(cvals[trial_idx]))

        ax.scatter(df_train["x1"], df_train["y"],
                   s=6, alpha=0.2, color="#5b9bd5", edgecolors="none", zorder=1)
        ax.plot(x1_grid, true_curve, "k--", linewidth=1.2, alpha=0.5, zorder=2)
        ax.plot(x1_grid, t["preds"], color=c, linewidth=1.8, zorder=3)
        ax.set_title(
            f"d={t['depth']}  it={t['iterations']}",
            fontsize=8, color=c, fontweight="bold",
        )
        ax.grid(True, alpha=0.2, linestyle=":")

    for ax in axes_flat[len(trials):]:
        ax.set_visible(False)

    fig.suptitle(
        "Per-trial fits  (sorted by complexity: depth × ln(1 + iterations))\n"
        "dashed = true polynomial",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = SCRIPT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating training data ...")
    df_train = generate_data(N_TRAIN, seed=SEED)
    x1_grid  = np.linspace(*X1_RANGE, 400)

    print(f"Sampling {N_SOBOL} Sobol hyperparameter points ...")
    params_list = sample_sobol_params(N_SOBOL, seed=SEED)

    print("Training CatBoost regressors ...")
    trials: list[dict] = []
    for i, params in enumerate(params_list):
        print(f"  [{i+1:2d}/{N_SOBOL}]  depth={params['depth']:2d}  "
              f"iterations={params['iterations']:4d}")
        model = train_model(df_train, **params)
        preds = predict_on_grid(model, x1_grid)
        trials.append({**params, "preds": preds})

    print("Plotting ...")
    plot_trials(df_train, trials, x1_grid, out_dir / "fig_trials_overlay.png")
    plot_hyperparam_space(trials, out_dir / "fig_hyperparam_space.png")
    plot_predictions_grid(df_train, trials, x1_grid, out_dir / "fig_predictions_grid.png")

    meta = pd.DataFrame([
        {"trial": i + 1, "depth": t["depth"], "iterations": t["iterations"],
         "complexity": _complexity(t["depth"], t["iterations"])}
        for i, t in enumerate(trials)
    ]).sort_values("complexity").reset_index(drop=True)
    meta.to_csv(out_dir / "trials.csv", index=False)
    print("  Saved trials.csv")

    print(f"\nDone.  Outputs in {out_dir}/")
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
