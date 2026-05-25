
"""
feature_information_studies.py

Goal
----
Study the marginal value of adding x3 under two controlled regimes.

Setup:

    setup_1 = {x1, x2}
    setup_2 = {x1, x2, x3}

    improvement = score(setup_2) - score(setup_1)

For logloss:

    improvement = logloss(setup_1) - logloss(setup_2)

So:

    improvement > 0  => setup_2 is better


Two studies
-----------

Study A: class-balance sensitivity

    keep information fixed
    vary positive fraction

    I(x1), I(x2), I(x3) = constant
    p_pos varies

Study B: new-feature information sensitivity

    keep positive fraction fixed
    vary information in x3

    p_pos = constant
    I(x1), I(x2) = constant
    I(x3) varies


Simple information proxy
------------------------

For this Gaussian experiment:

    x_j | y=0 ~ Normal(0, sigma)
    x_j | y=1 ~ Normal(separation_j, sigma)

Information proxy:

    info_j = separation_j = |mu_1j - mu_0j| / sigma

Higher info_j means less overlap between classes.

This is not literal mutual information, but it is a clean signal-strength knob.


Install
-------

    pip install numpy pandas scikit-learn matplotlib catboost rich

Run
---

    python feature_information_studies.py

Outputs
-------

    artifacts_info_studies/
        study_A_results.csv
        study_B_results.csv
        study_A_positive_fraction.png
        study_B_x3_information.png
        config_snapshot.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


# =============================================================================
# 0. Config
# =============================================================================

CONFIG = {
    "output_dir": "artifacts_info_studies",

    "target_col": "y",

    # Choose:
    #   "auc"
    #   "average_precision"
    #   "logloss"
    "primary_metric": "auc",

    "model": {
        # "logistic" is very fast and matches the Gaussian/log-odds structure.
        # "hgb" is a lightweight sklearn tree booster.
        # "catboost" uses CatBoostClassifier (all numeric features here).
        "backend": "logistic",

        "random_state": 42,

        # shared by hgb and catboost
        "iterations": 200,
        "learning_rate": 0.06,

        # hgb only
        "max_leaf_nodes": 16,

        # catboost only
        "depth": 4,
    },

    "sample_sizes": {
        "train": 2_000,
        "valid": 500,
        "test": 2_000,
    },

    "seeds": {
        "train": 101,
        "valid": 202,
        "test_base": 10_000,
    },

    "test_sampling": {
        # Fresh test worlds:
        #
        #   D_test_r ~ P_true(X, y)
        #
        # Training data is fixed per condition.
        "n_repeats": 20,
    },

    "setups": {
        "setup_1_x1_x2": ["x1", "x2"],
        "setup_2_x1_x2_x3": ["x1", "x2", "x3"],
    },

    # Study A:
    #   p_pos varies
    #   info levels fixed
    "study_A_positive_fraction": {
        "p_pos_grid": [0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
        "sigma": 1.0,
        "fixed_info": {
            "x1": 0.85,
            "x2": 0.55,
            "x3": 0.35,
        },
    },

    # Study B:
    #   p_pos fixed
    #   info_x3 varies
    "study_B_x3_information": {
        "p_pos": 0.15,
        "sigma": 1.0,
        "fixed_info": {
            "x1": 0.85,
            "x2": 0.55,
        },
        "x3_info_grid": [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25],
    },
}


METRIC_DIRECTION = {
    "auc": "higher",
    "average_precision": "higher",
    "logloss": "lower",
}


# =============================================================================
# 1. Data generator
# =============================================================================

class GaussianBinaryDGP:
    """
    DGP = data-generating process.

    Simple notation:

        y ~ Bernoulli(p)

        x_j | y=0 ~ N(0, sigma)
        x_j | y=1 ~ N(info_j * sigma, sigma)

    Since:

        info_j = |mu_1j - mu_0j| / sigma

    we can control information by changing info_j.
    """

    def __init__(
        self,
        p_pos: float,
        info: Dict[str, float],
        sigma: float = 1.0,
    ):
        self.p_pos = p_pos
        self.info = info
        self.sigma = sigma

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        y = rng.binomial(n=1, p=self.p_pos, size=n)
        df = pd.DataFrame({"y": y})

        for feature, info_j in self.info.items():
            mu0 = 0.0
            mu1 = info_j * self.sigma
            mu = np.where(y == 1, mu1, mu0)

            df[feature] = rng.normal(loc=mu, scale=self.sigma, size=n)

        return df


# =============================================================================
# 2. Model factory
# =============================================================================

class ModelFactory:
    """
    The model is kept constant across setup_1 and setup_2.

    Only feature set changes:

        setup_1 = {x1, x2}
        setup_2 = {x1, x2, x3}
    """

    def __init__(self, config: Dict):
        self.config = config

    def create(self):
        backend = self.config["backend"]

        if backend == "logistic":
            return LogisticRegression(
                max_iter=300,
                solver="lbfgs",
                random_state=self.config["random_state"],
            )

        if backend == "hgb":
            return HistGradientBoostingClassifier(
                max_iter=self.config["iterations"],
                learning_rate=self.config["learning_rate"],
                max_leaf_nodes=self.config["max_leaf_nodes"],
                random_state=self.config["random_state"],
                validation_fraction=None,
                early_stopping=False,
            )

        if backend == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                raise ImportError(
                    "CatBoost backend requires catboost. Install with: pip install catboost"
                ) from exc

            return CatBoostClassifier(
                iterations=self.config["iterations"],
                learning_rate=self.config["learning_rate"],
                depth=self.config["depth"],
                random_seed=self.config["random_state"],
                loss_function="Logloss",
                verbose=False,
                allow_writing_files=False,
            )

        raise ValueError(f"Unknown model backend: {backend}")


# =============================================================================
# 3. Evaluation engine
# =============================================================================

class SetupComparator:
    """
    Fixed train/valid within one condition.

    Repeated true test sampling:

        D_train fixed
        D_valid fixed
        D_test_r ~ P_true(X, y)

    For each condition c and repeat r:

        score_1(c, r) = score(model({x1,x2}), D_test_r)
        score_2(c, r) = score(model({x1,x2,x3}), D_test_r)

        improvement(c, r) = score_2(c, r) - score_1(c, r)

    For lower-is-better metrics:

        improvement(c, r) = score_1(c, r) - score_2(c, r)
    """

    def __init__(
        self,
        setups: Dict[str, List[str]],
        target_col: str,
        primary_metric: str,
        model_factory: ModelFactory,
    ):
        if primary_metric not in METRIC_DIRECTION:
            raise ValueError(f"Unknown metric: {primary_metric}")

        self.setups = setups
        self.target_col = target_col
        self.primary_metric = primary_metric
        self.model_factory = model_factory

    def _score_all(self, y_true: np.ndarray, p_hat: np.ndarray) -> Dict[str, float]:
        p_hat = np.clip(p_hat, 1e-8, 1 - 1e-8)

        return {
            "auc": roc_auc_score(y_true, p_hat),
            "average_precision": average_precision_score(y_true, p_hat),
            "logloss": log_loss(y_true, p_hat),
        }

    def fit_models(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> Dict[str, object]:
        models = {}

        for setup_name, features in self.setups.items():
            X_train = df_train[features]
            y_train = df_train[self.target_col]

            # df_valid is kept fixed for future extension:
            # early stopping, calibration, threshold selection, etc.
            _ = df_valid

            model = self.model_factory.create()
            model.fit(X_train, y_train)

            models[setup_name] = model

        return models

    def evaluate_once(
        self,
        models: Dict[str, object],
        df_test: pd.DataFrame,
        condition_name: str,
        condition_value: float,
        repeat: int,
        seed: int,
    ) -> pd.DataFrame:
        y_test = df_test[self.target_col]
        rows = []

        for setup_name, features in self.setups.items():
            model = models[setup_name]
            p_hat = model.predict_proba(df_test[features])[:, 1]
            scores = self._score_all(y_test, p_hat)

            rows.append({
                "condition_name": condition_name,
                "condition_value": condition_value,
                "repeat": repeat,
                "test_seed": seed,
                "setup": setup_name,
                "features": ",".join(features),
                **scores,
            })

        return pd.DataFrame(rows)

    def add_improvement_column(self, raw_results: pd.DataFrame) -> pd.DataFrame:
        metric = self.primary_metric
        direction = METRIC_DIRECTION[metric]

        setup_names = list(self.setups.keys())
        baseline = setup_names[0]
        challenger = setup_names[1]

        key_cols = ["condition_name", "condition_value", "repeat"]

        wide = raw_results.pivot_table(
            index=key_cols,
            columns="setup",
            values=metric,
        ).reset_index()

        raw_delta = wide[challenger] - wide[baseline]

        if direction == "higher":
            improvement = raw_delta
        else:
            improvement = -raw_delta

        wide["baseline"] = baseline
        wide["challenger"] = challenger
        wide["metric"] = metric
        wide["raw_delta_challenger_minus_baseline"] = raw_delta
        wide["improvement"] = improvement

        return wide

    def summarize(self, improvement_results: pd.DataFrame) -> pd.DataFrame:
        return (
            improvement_results
            .groupby(["condition_name", "condition_value"])
            .agg(
                mean_improvement=("improvement", "mean"),
                std_improvement=("improvement", "std"),
                p_improvement_gt_0=("improvement", lambda x: float((x > 0).mean())),
                mean_baseline_score=(list(self.setups.keys())[0], "mean"),
                mean_challenger_score=(list(self.setups.keys())[1], "mean"),
                n_repeats=("improvement", "count"),
            )
            .reset_index()
        )


# =============================================================================
# 4. Study runner
# =============================================================================

class InformationStudyRunner:
    """
    Runs both studies.

    Study A:

        variable = p_pos
        fixed info = {info_x1, info_x2, info_x3}

    Study B:

        fixed p_pos
        variable = info_x3
        fixed info_x1, info_x2
    """

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.comparator = SetupComparator(
            setups=config["setups"],
            target_col=config["target_col"],
            primary_metric=config["primary_metric"],
            model_factory=ModelFactory(config["model"]),
        )

    def _run_condition(
        self,
        dgp: GaussianBinaryDGP,
        condition_name: str,
        condition_value: float,
        condition_seed_offset: int,
    ) -> pd.DataFrame:
        n_train = self.config["sample_sizes"]["train"]
        n_valid = self.config["sample_sizes"]["valid"]
        n_test = self.config["sample_sizes"]["test"]

        train_seed = self.config["seeds"]["train"] + condition_seed_offset
        valid_seed = self.config["seeds"]["valid"] + condition_seed_offset

        df_train = dgp.sample(n_train, train_seed)
        df_valid = dgp.sample(n_valid, valid_seed)

        models = self.comparator.fit_models(df_train, df_valid)

        parts = []

        for repeat in range(1, self.config["test_sampling"]["n_repeats"] + 1):
            test_seed = (
                self.config["seeds"]["test_base"]
                + condition_seed_offset * 1000
                + repeat
            )

            df_test = dgp.sample(n_test, test_seed)

            parts.append(
                self.comparator.evaluate_once(
                    models=models,
                    df_test=df_test,
                    condition_name=condition_name,
                    condition_value=condition_value,
                    repeat=repeat,
                    seed=test_seed,
                )
            )

        return pd.concat(parts, ignore_index=True)

    def run_study_A_positive_fraction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config["study_A_positive_fraction"]
        p_pos_grid = cfg["p_pos_grid"]
        n_train = self.config["sample_sizes"]["train"]
        n_repeats = self.config["test_sampling"]["n_repeats"]
        metric = self.config["primary_metric"]
        backend = self.config["model"]["backend"]
        fixed_info = cfg["fixed_info"]

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            from rich.table import Table
            from rich.text import Text

            console = Console()
            use_rich = True
        except ImportError:
            console = None
            use_rich = False

        if use_rich:
            console.print(
                Panel(
                    "\n".join(
                        [
                            f"[bold]metric[/bold]       {metric}",
                            f"[bold]model[/bold]        {backend}",
                            f"[bold]train[/bold]        {n_train:,}",
                            f"[bold]valid[/bold]        {self.config['sample_sizes']['valid']:,}",
                            f"[bold]test[/bold]         {self.config['sample_sizes']['test']:,}",
                            f"[bold]repeats[/bold]      {n_repeats} per condition",
                            f"[bold]conditions[/bold]   {len(p_pos_grid)} p_pos values",
                            (
                                "[bold]info[/bold]         "
                                f"x1={fixed_info['x1']}, "
                                f"x2={fixed_info['x2']}, "
                                f"x3={fixed_info['x3']}"
                            ),
                        ]
                    ),
                    title="Study A · positive fraction sweep",
                    border_style="cyan",
                )
            )
        else:
            print(
                f"Study A: {len(p_pos_grid)} conditions, "
                f"metric={metric}, model={backend}, train={n_train:,}"
            )

        parts = []

        def run_grid(progress=None, task_id=None) -> None:
            for i, p_pos in enumerate(p_pos_grid):
                expected_pos = n_train * p_pos

                if progress is not None and task_id is not None:
                    progress.update(
                        task_id,
                        description=(
                            f"p_pos={p_pos:.4g}  "
                            f"E[pos]≈{expected_pos:.0f}"
                        ),
                    )

                dgp = GaussianBinaryDGP(
                    p_pos=p_pos,
                    info=fixed_info,
                    sigma=cfg["sigma"],
                )

                part = self._run_condition(
                    dgp=dgp,
                    condition_name="p_pos",
                    condition_value=p_pos,
                    condition_seed_offset=10_000 + i,
                )
                parts.append(part)

                if use_rich:
                    improved_part = self.comparator.add_improvement_column(part)
                    mean_imp = improved_part["improvement"].mean()
                    p_win = (improved_part["improvement"] > 0).mean()
                    style = "green" if mean_imp > 0 else "red" if mean_imp < 0 else "yellow"
                    console.print(
                        f"  [{style}]Δ {metric}[/] = {mean_imp:+.5f}  "
                        f"[dim]P(improvement>0)={p_win:.0%}[/dim]"
                    )
                else:
                    print(f"  p_pos={p_pos:.4g} done")

                if progress is not None and task_id is not None:
                    progress.advance(task_id)

        if use_rich:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("conditions", total=len(p_pos_grid))
                run_grid(progress=progress, task_id=task_id)
        else:
            run_grid()

        raw = pd.concat(parts, ignore_index=True)
        improved = self.comparator.add_improvement_column(raw)
        summary = self.comparator.summarize(improved)

        raw.to_csv(self.output_dir / "study_A_raw_scores.csv", index=False)
        improved.to_csv(self.output_dir / "study_A_improvements.csv", index=False)
        summary.to_csv(self.output_dir / "study_A_summary.csv", index=False)

        if use_rich:
            table = Table(
                title="Study A summary",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("p_pos", justify="right")
            table.add_column(f"Δ {metric}", justify="right")
            table.add_column("± std", justify="right")
            table.add_column("P(Δ>0)", justify="right")
            table.add_column("baseline", justify="right")
            table.add_column("+x3", justify="right")

            for _, row in summary.iterrows():
                imp = row["mean_improvement"]
                style = "green" if imp > 0 else "red" if imp < 0 else ""
                table.add_row(
                    f"{row['condition_value']:.4g}",
                    Text(f"{imp:+.5f}", style=style),
                    f"{row['std_improvement']:.5f}",
                    f"{row['p_improvement_gt_0']:.0%}",
                    f"{row['mean_baseline_score']:.5f}",
                    f"{row['mean_challenger_score']:.5f}",
                )

            console.print(table)
            console.print(
                f"[dim]Saved[/dim] study_A_raw_scores.csv, "
                f"study_A_improvements.csv, study_A_summary.csv "
                f"[dim]→[/dim] {self.output_dir}"
            )

        return improved, summary

    def run_study_B_x3_information(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config["study_B_x3_information"]

        parts = []

        for i, x3_info in enumerate(cfg["x3_info_grid"]):
            info = {
                "x1": cfg["fixed_info"]["x1"],
                "x2": cfg["fixed_info"]["x2"],
                "x3": x3_info,
            }

            dgp = GaussianBinaryDGP(
                p_pos=cfg["p_pos"],
                info=info,
                sigma=cfg["sigma"],
            )

            parts.append(
                self._run_condition(
                    dgp=dgp,
                    condition_name="x3_info",
                    condition_value=x3_info,
                    condition_seed_offset=20_000 + i,
                )
            )

        raw = pd.concat(parts, ignore_index=True)
        improved = self.comparator.add_improvement_column(raw)
        summary = self.comparator.summarize(improved)

        raw.to_csv(self.output_dir / "study_B_raw_scores.csv", index=False)
        improved.to_csv(self.output_dir / "study_B_improvements.csv", index=False)
        summary.to_csv(self.output_dir / "study_B_summary.csv", index=False)

        return improved, summary

    def save_config(self) -> None:
        with open(self.output_dir / "config_snapshot.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def plot_study_summary(
        self,
        summary: pd.DataFrame,
        x_col: str,
        title: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))

        x = summary["condition_value"].values
        y = summary["mean_improvement"].values
        err = summary["std_improvement"].fillna(0).values

        ax.axhline(0, linewidth=2, alpha=0.35)
        ax.plot(x, y, marker="o", linewidth=3)
        ax.fill_between(x, y - err, y + err, alpha=0.18)

        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(f"Mean improvement in {self.config['primary_metric'].upper()}")
        ax.grid(True, linewidth=2, alpha=0.1)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160)
        plt.close(fig)

    def run_all(self) -> None:
        self.save_config()

        study_A_improvements, study_A_summary = self.run_study_A_positive_fraction()
        study_B_improvements, study_B_summary = self.run_study_B_x3_information()

        self.plot_study_summary(
            summary=study_A_summary,
            x_col="positive fraction p_pos",
            title="Study A: fixed feature information, varying positive fraction",
            filename="study_A_positive_fraction.png",
        )

        self.plot_study_summary(
            summary=study_B_summary,
            x_col="x3 information level",
            title="Study B: fixed positive fraction, varying x3 information",
            filename="study_B_x3_information.png",
        )

        print("\nStudy A summary:")
        print(study_A_summary.round(5).to_string(index=False))

        print("\nStudy B summary:")
        print(study_B_summary.round(5).to_string(index=False))

        print("\nArtifacts saved in:", self.output_dir)
        for p in sorted(self.output_dir.glob("*")):
            print(" -", p.name)


def main():
    runner = InformationStudyRunner(CONFIG)
    runner.run_all()


if __name__ == "__main__":
    main()
