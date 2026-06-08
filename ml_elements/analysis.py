"""
analysis.py
-----------
``Comparator``: statistical comparison of setups across a study.

Wraps ``BayesianAPComparator`` (from ``bayesian_ap_comparator.py``) and
connects it to ``StudyResult`` without modifying the comparator itself.

Two entry points
----------------
``for_trial(result, trial_value, baseline, challenger)``
    Returns a fitted ``BayesianAPComparator`` for a single trial value.
    Use it to drill into one condition and access the full report suite:
    ``model_report``, ``pairwise_report``, ``ranking_report``, ``best_model_report``.

``full_report(result, baseline, challenger)``
    Runs the comparison for every trial value and returns a single DataFrame
    with one row per trial — a bird's-eye view of the whole study.

Both methods compute bootstrap AP distributions, so they work on the
probability scale (Average Precision). For other metrics use the
``Study.improvements()`` / ``Study.summarize()`` path.

Example
-------
>>> comparator = Comparator()

# Full study overview:
>>> report = comparator.full_report(result, baseline="baseline", challenger="challenger")
>>> report[["trial_value", "challenger_ap_observed", "p_challenger_beats_baseline"]]

# Drill into one trial:
>>> bac = comparator.for_trial(result, trial_value=0.10, baseline="baseline", challenger="challenger")
>>> bac.ranking_report()
>>> bac.pairwise_report()
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .study import StudyResult

# ---------------------------------------------------------------------------
# Locate BayesianAPComparator — it lives one level up from this package.
# Optional dependency: Comparator raises ImportError at call time if missing.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent

if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

try:
    from bayesian_ap_comparator import BayesianAPComparator  # noqa: E402
    _BAC_AVAILABLE = True
except ImportError:
    BayesianAPComparator = None  # type: ignore[assignment,misc]
    _BAC_AVAILABLE = False


class Comparator:
    """
    Statistical setup comparison using ``BayesianAPComparator``.

    Uses stratified bootstrap AP distributions to estimate:
    - P(challenger AP > baseline AP)
    - 90% CI on AP difference
    - log-ratio and z-score summaries

    Parameters
    ----------
    n_boot : int
        Bootstrap samples per comparison.
    stratified : bool
        If True, bootstrap positives and negatives separately (recommended).
    random_state : int
        Reproducibility seed.

    Notes
    -----
    This comparator operates on **Average Precision** because
    ``BayesianAPComparator`` is designed for that metric. For AUC or
    log-loss comparisons, use ``Study.improvements()`` + ``Study.summarize()``.
    """

    def __init__(
        self,
        n_boot: int = 3_000,
        stratified: bool = True,
        random_state: int = 42,
    ) -> None:
        if not _BAC_AVAILABLE:
            raise ImportError(
                "Comparator requires bayesian_ap_comparator. "
                "Place bayesian_ap_comparator.py in the repo root."
            )
        self.n_boot = n_boot
        self.stratified = stratified
        self.random_state = random_state

    def for_trial(
        self,
        result: StudyResult,
        trial_value: float,
        baseline: str,
        challenger: str,
    ) -> BayesianAPComparator:
        """
        Build and fit a ``BayesianAPComparator`` for one trial value.

        Aggregates all repeat scores for the given trial value into a single
        pooled evaluation by concatenating scores (i.e., treats all repeats
        as independent observations for the bootstrap).

        Parameters
        ----------
        result : StudyResult
        trial_value : float
            The ``trial_value`` to filter on.
        baseline : str
            Setup name for the reference model.
        challenger : str
            Setup name for the model under evaluation.

        Returns
        -------
        BayesianAPComparator
            Fitted comparator. Call ``.ranking_report()``, ``.pairwise_report()``,
            ``.best_model_report()``, etc.

        Raises
        ------
        ValueError
            If trial_value is not found, or setups are missing.
        """
        trial_result = self._find_trial_result(result, trial_value)
        y_true, scores = self._extract_scores(trial_result, [baseline, challenger])

        bac = BayesianAPComparator(
            y_true=y_true,
            scores=scores,
            n_boot=self.n_boot,
            stratified=self.stratified,
            random_state=self.random_state,
        )
        bac.fit()
        return bac

    def full_report(
        self,
        result: StudyResult,
        baseline: str,
        challenger: str,
    ) -> pd.DataFrame:
        """
        Run ``BayesianAPComparator`` for every trial value in the study.

        Parameters
        ----------
        result : StudyResult
        baseline : str
        challenger : str

        Returns
        -------
        pd.DataFrame
            One row per trial. Columns:
            ``trial_name``, ``trial_value``,
            ``baseline_ap_observed``, ``challenger_ap_observed``,
            ``diff_post_mean``, ``diff_ci_05``, ``diff_ci_95``,
            ``p_challenger_beats_baseline``, ``z_log_ratio``.
        """
        rows = []

        for trial_result in result.trial_results:
            trial_value = trial_result.trial.value
            trial_name = trial_result.trial.name

            try:
                y_true, scores = self._extract_scores(trial_result, [baseline, challenger])
                bac = BayesianAPComparator(
                    y_true=y_true,
                    scores=scores,
                    n_boot=self.n_boot,
                    stratified=self.stratified,
                    random_state=self.random_state,
                )
                bac.fit()
                model_report = bac.model_report_
                pairwise = bac.pairwise_report_

                ch_row = model_report[model_report["model"] == challenger].iloc[0]
                bl_row = model_report[model_report["model"] == baseline].iloc[0]
                pair_row = pairwise[
                    (pairwise["model_a"] == challenger) & (pairwise["model_b"] == baseline)
                ].iloc[0]

                rows.append({
                    "trial_name": trial_name,
                    "trial_value": trial_value,
                    "n": int(bac.n),
                    "n_pos": int(bac.n_pos),
                    "baseline_ap_observed": float(bl_row["ap_observed"]),
                    "challenger_ap_observed": float(ch_row["ap_observed"]),
                    "diff_observed": float(ch_row["ap_observed"] - bl_row["ap_observed"]),
                    "diff_post_mean": float(pair_row["diff_post_mean"]),
                    "diff_ci_05": float(pair_row["diff_ci_05"]),
                    "diff_ci_95": float(pair_row["diff_ci_95"]),
                    "p_challenger_beats_baseline": float(pair_row["p_a_better"]),
                    "z_log_ratio": float(pair_row["z_log_ratio"]),
                })

            except Exception as exc:
                rows.append({
                    "trial_name": trial_name,
                    "trial_value": trial_value,
                    "error": str(exc),
                })

        return pd.DataFrame(rows).sort_values("trial_value").reset_index(drop=True)

    def _find_trial_result(self, result: StudyResult, trial_value: float):
        matches = [tr for tr in result.trial_results if tr.trial.value == trial_value]
        if not matches:
            available = [tr.trial.value for tr in result.trial_results]
            raise ValueError(
                f"trial_value {trial_value} not found. Available: {available}"
            )
        return matches[0]

    def _extract_scores(
        self,
        trial_result,
        setup_names: list[str],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Extract real observation-level predictions from the stored validation set.

        Uses ``trial_result.df_valid`` (already split, never seen during training)
        and the stored fitted models to obtain genuine prediction arrays for
        ``BayesianAPComparator``. This is exact, not an approximation.
        """
        scores_df = trial_result.scores

        for setup in setup_names:
            if setup not in scores_df["setup"].values:
                available = scores_df["setup"].unique().tolist()
                raise ValueError(
                    f"Setup {setup!r} not in trial scores. Available: {available}"
                )

        df = trial_result.df_valid

        # Infer target column: the one column in df_valid that is not a feature.
        all_features: set[str] = set()
        for setup in setup_names:
            feat_str = scores_df.loc[scores_df["setup"] == setup, "features"].iloc[0]
            all_features.update(f.strip() for f in feat_str.split(","))
        non_feature = [c for c in df.columns if c not in all_features]
        target_col = non_feature[0] if non_feature else "y"

        y_true = df[target_col].to_numpy()

        model_scores: dict[str, np.ndarray] = {}
        for setup in setup_names:
            feat_str = scores_df.loc[scores_df["setup"] == setup, "features"].iloc[0]
            features = [f.strip() for f in feat_str.split(",")]
            model = trial_result.models[setup]
            model_scores[setup] = model.predict_proba(df[features])[:, 1]

        return y_true, model_scores
