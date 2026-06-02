"""
feature_information_studies.py

Goal
----
Study the marginal value of adding x3 under two controlled regimes.

Setup:

    setup_1 = {x1, x2}
    setup_2 = {x1, x2, x3}

    improvement = score(setup_2) - score(setup_1)     (higher-is-better metrics)
    improvement = score(setup_1) - score(setup_2)     (logloss)

So:

    improvement > 0  => setup_2 is better


Two studies
-----------

Study A: class-balance sensitivity
    keep information fixed, vary the positive fraction p_pos.

Study B: new-feature information sensitivity
    keep the positive fraction fixed, vary the information in x3.

Both are *scenario sweeps*: p_pos and the info levels are experiment-design
choices, not knobs to optimize. We sweep them on a grid (``ScenarioSweep``)
to map the improvement curve — there is no "best" p_pos to search for.


Implementation
--------------
This script is a thin driver over the ``ml_elements`` package — it owns no DGP,
model, or evaluation logic of its own. The building blocks are:

    GaussianBinaryDGP   the data-generating process
    TrialRunner/Study   fit per setup, score on repeated fresh test draws
    ScenarioSweep       generate one trial per scenario on the grid
    plot_study          render the improvement curve


Install
-------

    pip install numpy pandas scikit-learn matplotlib scipy
    # optional: catboost (catboost backend), rich (progress bars), joblib (n_jobs)

Run
---

    python feature_information_studies.py

Outputs
-------

    artifacts_info_studies/
        study_A_raw_scores.csv
        study_A_improvements.csv
        study_A_summary.csv
        study_B_raw_scores.csv
        study_B_improvements.csv
        study_B_summary.csv
        study_A_positive_fraction.png
        study_B_x3_information.png
        config_snapshot.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ml_elements import (
    AUC,
    AVG_PRECISION,
    LOGLOSS,
    DataBudget,
    GaussianBinaryDGP,
    Metric,
    ScenarioSweep,
    Study,
    StudyResult,
    TrialRunner,
    make_catboost,
    make_hgb,
    make_logistic,
    plot_study,
)


# =============================================================================
# 0. Config
# =============================================================================

CONFIG = {
    "output_dir": "artifacts_info_studies",

    "target_col": "y",

    # Choose: "auc" | "average_precision" | "logloss"
    "primary_metric": "auc",

    "model": {
        # "logistic" is fast and matches the Gaussian/log-odds structure.
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
        # Fresh test worlds per repeat: D_test_r ~ P_true(X, y).
        # Training data is fixed per condition.
        "n_repeats": 20,
    },

    "setups": {
        "setup_1_x1_x2": ["x1", "x2"],
        "setup_2_x1_x2_x3": ["x1", "x2", "x3"],
    },

    # Study A: p_pos varies, info levels fixed.
    "study_A_positive_fraction": {
        "p_pos_grid": [0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
        "sigma": 1.0,
        "fixed_info": {"x1": 0.85, "x2": 0.55, "x3": 0.35},
    },

    # Study B: p_pos fixed, info_x3 varies.
    "study_B_x3_information": {
        "p_pos": 0.15,
        "sigma": 1.0,
        "fixed_info": {"x1": 0.85, "x2": 0.55},
        "x3_info_grid": [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25],
    },
}


_METRICS: dict[str, Metric] = {
    "auc": AUC,
    "average_precision": AVG_PRECISION,
    "logloss": LOGLOSS,
}


# =============================================================================
# 1. Wiring helpers — translate CONFIG into ml_elements building blocks
# =============================================================================

def build_model_factory(model_cfg: dict) -> Callable:
    """Map the ``model`` config block to a zero-arg model factory."""
    backend = model_cfg["backend"]
    rs = model_cfg["random_state"]

    if backend == "logistic":
        return make_logistic(random_state=rs)
    if backend == "hgb":
        return make_hgb(
            iterations=model_cfg["iterations"],
            learning_rate=model_cfg["learning_rate"],
            max_leaf_nodes=model_cfg["max_leaf_nodes"],
            random_state=rs,
        )
    if backend == "catboost":
        return make_catboost(
            iterations=model_cfg["iterations"],
            learning_rate=model_cfg["learning_rate"],
            depth=model_cfg["depth"],
            random_state=rs,
        )
    raise ValueError(f"Unknown model backend: {backend}")


def build_budget(config: dict) -> DataBudget:
    """Map the sample-size / seed / repeat config into a DataBudget."""
    return DataBudget(
        n_train=config["sample_sizes"]["train"],
        n_valid=config["sample_sizes"]["valid"],
        n_test=config["sample_sizes"]["test"],
        seed_train=config["seeds"]["train"],
        seed_valid=config["seeds"]["valid"],
        seed_test_base=config["seeds"]["test_base"],
        n_repeats=config["test_sampling"]["n_repeats"],
    )


# =============================================================================
# 2. Study runner
# =============================================================================

class InformationStudyRunner:
    """
    Runs both scenario sweeps on top of a single shared ``Study``.

    Study A:  sweep p_pos, hold the info levels fixed.
    Study B:  sweep info_x3, hold p_pos and info_x1/x2 fixed.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config["primary_metric"] not in _METRICS:
            raise ValueError(f"Unknown primary_metric: {config['primary_metric']}")
        self.primary_metric = _METRICS[config["primary_metric"]]

        self.setups = config["setups"]
        self.baseline, self.challenger = list(self.setups.keys())[:2]

        runner = TrialRunner(
            setups=self.setups,
            model_factory=build_model_factory(config["model"]),
            metrics=[AUC, AVG_PRECISION, LOGLOSS],
            budget=build_budget(config),
            target_col=config["target_col"],
        )
        self.study = Study(runner, primary_metric=self.primary_metric)

    def _finalize(self, result: StudyResult, prefix: str):
        """Compute improvements + summary, persist CSVs, return (improved, summary)."""
        improved = self.study.improvements(
            result, baseline=self.baseline, challenger=self.challenger
        )
        summary = self.study.summarize(improved)

        result.scores.to_csv(self.output_dir / f"{prefix}_raw_scores.csv", index=False)
        improved.to_csv(self.output_dir / f"{prefix}_improvements.csv", index=False)
        summary.to_csv(self.output_dir / f"{prefix}_summary.csv", index=False)

        return improved, summary

    def run_study_A_positive_fraction(self):
        cfg = self.config["study_A_positive_fraction"]
        fixed_info = cfg["fixed_info"]
        sigma = cfg["sigma"]

        trials = ScenarioSweep.over("p_pos", cfg["p_pos_grid"]).trials(
            dgp_fn=lambda s: GaussianBinaryDGP(
                p_pos=s["p_pos"],
                info=fixed_info,
                sigma=sigma,
            ),
            seed_offset=10_000,
        )

        result = self.study.run(trials)
        return self._finalize(result, "study_A")

    def run_study_B_x3_information(self):
        cfg = self.config["study_B_x3_information"]
        base_info = cfg["fixed_info"]
        p_pos = cfg["p_pos"]
        sigma = cfg["sigma"]

        trials = ScenarioSweep.over("x3_info", cfg["x3_info_grid"]).trials(
            dgp_fn=lambda s: GaussianBinaryDGP(
                p_pos=p_pos,
                info={**base_info, "x3": s["x3_info"]},
                sigma=sigma,
            ),
            seed_offset=20_000,
        )

        result = self.study.run(trials)
        return self._finalize(result, "study_B")

    def save_config(self) -> None:
        with open(self.output_dir / "config_snapshot.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def run_all(self) -> None:
        self.save_config()

        _, study_A_summary = self.run_study_A_positive_fraction()
        _, study_B_summary = self.run_study_B_x3_information()

        fig_a = plot_study(
            study_A_summary,
            metric=self.primary_metric,
            title="Study A: fixed feature information, varying positive fraction",
            x_label="positive fraction p_pos",
        )
        fig_a.savefig(self.output_dir / "study_A_positive_fraction.png", dpi=160)

        fig_b = plot_study(
            study_B_summary,
            metric=self.primary_metric,
            title="Study B: fixed positive fraction, varying x3 information",
            x_label="x3 information level",
        )
        fig_b.savefig(self.output_dir / "study_B_x3_information.png", dpi=160)

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
