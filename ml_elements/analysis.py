"""
analysis.py
-----------
``Comparator``: statistical comparison of setups across a study.

What it does
------------
For a given trial it reconstructs the **real** observation-level predictions —
it re-draws the exact test sets the runner used (same DGP, same seeds, same
size, recorded on each ``TrialResult``) and re-runs ``predict_proba`` with the
fitted models — then pools them across repeats. A paired stratified bootstrap
over Average Precision gives:

    - P(challenger AP > baseline AP)
    - a CI on the AP difference
    - observed and bootstrap-mean AP per setup

This is a genuine bootstrap on the model outputs; nothing is synthesised.

Two entry points
----------------
``for_trial(result, trial_value, baseline, challenger)``
    Returns an ``APComparison`` for one trial value. Call ``.model_report()``,
    ``.ranking_report()``, ``.pairwise_report()`` on it.

``full_report(result, baseline, challenger)``
    Runs the comparison for every trial and returns one row per trial — a
    bird's-eye view of the whole study.

Both work on the Average Precision scale. For AUC or log-loss, use the
``Study.improvements()`` / ``Study.summarize()`` path.

Example
-------
>>> comparator = Comparator()
>>> report = comparator.full_report(result, baseline="baseline", challenger="challenger")
>>> report[["trial_value", "challenger_ap_observed", "p_challenger_beats_baseline"]]

>>> ac = comparator.for_trial(result, trial_value=0.10, baseline="baseline", challenger="challenger")
>>> ac.ranking_report()
>>> ac.pairwise_report()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .runner import TrialResult
from .study import StudyResult


@dataclass
class APComparison:
    """
    Bootstrap Average-Precision comparison for the setups of a single trial.

    Attributes
    ----------
    setups : list[str]
        Setup names compared, baseline first.
    y_true : np.ndarray
        Pooled true labels across all test repeats.
    scores : dict[str, np.ndarray]
        Pooled predicted P(y=1) per setup, aligned with ``y_true``.
    boot_ap : dict[str, np.ndarray]
        Bootstrap AP distribution per setup (paired across setups: the same
        resampled indices are used for every setup on each draw).
    observed_ap : dict[str, float]
        AP on the full pooled sample per setup.
    n : int
        Pooled sample size.
    n_pos : int
        Number of positives in the pooled sample.
    """

    setups: list[str]
    y_true: np.ndarray
    scores: dict[str, np.ndarray]
    boot_ap: dict[str, np.ndarray]
    observed_ap: dict[str, float]
    n: int
    n_pos: int

    def model_report(self) -> pd.DataFrame:
        """One row per setup: observed AP plus bootstrap mean and 90% CI."""
        rows = []
        for setup in self.setups:
            boot = self.boot_ap[setup]
            rows.append({
                "model": setup,
                "ap_observed": self.observed_ap[setup],
                "ap_boot_mean": float(np.mean(boot)),
                "ap_ci_05": float(np.quantile(boot, 0.05)),
                "ap_ci_95": float(np.quantile(boot, 0.95)),
            })
        return pd.DataFrame(rows)

    def ranking_report(self) -> pd.DataFrame:
        """Setups ranked by observed AP (best first)."""
        report = self.model_report().sort_values(
            "ap_observed", ascending=False
        ).reset_index(drop=True)
        report.insert(0, "rank", report.index + 1)
        return report

    def pairwise_report(self) -> pd.DataFrame:
        """
        All ordered setup pairs (a, b) with paired-bootstrap difference stats.

        ``p_a_better`` is the fraction of bootstrap draws where a's AP exceeds
        b's. ``diff_*`` summarise the paired AP difference (a − b).
        """
        rows = []
        for a in self.setups:
            for b in self.setups:
                if a == b:
                    continue
                diff = self.boot_ap[a] - self.boot_ap[b]
                rows.append({
                    "model_a": a,
                    "model_b": b,
                    "diff_observed": self.observed_ap[a] - self.observed_ap[b],
                    "diff_post_mean": float(np.mean(diff)),
                    "diff_ci_05": float(np.quantile(diff, 0.05)),
                    "diff_ci_95": float(np.quantile(diff, 0.95)),
                    "p_a_better": float(np.mean(diff > 0)),
                })
        return pd.DataFrame(rows)


class Comparator:
    """
    Statistical setup comparison via a paired stratified AP bootstrap.

    Parameters
    ----------
    n_boot : int
        Bootstrap resamples per comparison.
    stratified : bool
        If True, resample positives and negatives separately so the class
        balance is preserved on every draw (recommended for imbalanced data).
    random_state : int
        Reproducibility seed.

    Notes
    -----
    Operates on **Average Precision**. For AUC or log-loss comparisons, use
    ``Study.improvements()`` + ``Study.summarize()``.
    """

    def __init__(
        self,
        n_boot: int = 3_000,
        stratified: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_boot = n_boot
        self.stratified = stratified
        self.random_state = random_state

    def for_trial(
        self,
        result: StudyResult,
        trial_value: float,
        baseline: str,
        challenger: str,
    ) -> APComparison:
        """
        Build an ``APComparison`` for one trial value.

        Pools every test repeat for that trial into a single evaluation by
        reconstructing the real predictions, then bootstraps.

        Raises
        ------
        ValueError
            If ``trial_value`` is not found or a setup is missing.
        """
        trial_result = self._find_trial_result(result, trial_value)
        return self._compare(trial_result, [baseline, challenger])

    def full_report(
        self,
        result: StudyResult,
        baseline: str,
        challenger: str,
    ) -> pd.DataFrame:
        """
        Run the comparison for every trial and return one row per trial.

        Returns
        -------
        pd.DataFrame
            Columns: ``trial_name``, ``trial_value``, ``n``, ``n_pos``,
            ``baseline_ap_observed``, ``challenger_ap_observed``,
            ``diff_observed``, ``diff_post_mean``, ``diff_ci_05``,
            ``diff_ci_95``, ``p_challenger_beats_baseline``.
            Rows that fail are reported with an ``error`` column instead.
        """
        rows = []

        for trial_result in result.trial_results:
            trial_value = trial_result.trial.value
            trial_name = trial_result.trial.name

            try:
                ac = self._compare(trial_result, [baseline, challenger])
                pair = ac.pairwise_report()
                pair_row = pair[
                    (pair["model_a"] == challenger) & (pair["model_b"] == baseline)
                ].iloc[0]

                rows.append({
                    "trial_name": trial_name,
                    "trial_value": trial_value,
                    "n": ac.n,
                    "n_pos": ac.n_pos,
                    "baseline_ap_observed": ac.observed_ap[baseline],
                    "challenger_ap_observed": ac.observed_ap[challenger],
                    "diff_observed": float(pair_row["diff_observed"]),
                    "diff_post_mean": float(pair_row["diff_post_mean"]),
                    "diff_ci_05": float(pair_row["diff_ci_05"]),
                    "diff_ci_95": float(pair_row["diff_ci_95"]),
                    "p_challenger_beats_baseline": float(pair_row["p_a_better"]),
                })
            except Exception as exc:
                rows.append({
                    "trial_name": trial_name,
                    "trial_value": trial_value,
                    "error": str(exc),
                })

        return pd.DataFrame(rows).sort_values("trial_value").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _compare(
        self,
        trial_result: TrialResult,
        setups: list[str],
    ) -> APComparison:
        y_true, scores = self._pool_predictions(trial_result, setups)

        n = len(y_true)
        pos_idx = np.flatnonzero(y_true == 1)
        neg_idx = np.flatnonzero(y_true == 0)
        if len(pos_idx) == 0:
            raise ValueError("No positive labels in pooled test data; AP is undefined.")

        observed_ap = {
            s: float(average_precision_score(y_true, scores[s])) for s in setups
        }

        rng = np.random.default_rng(self.random_state)
        boot_ap = {s: np.empty(self.n_boot, dtype=float) for s in setups}

        for b in range(self.n_boot):
            if self.stratified:
                idx = np.concatenate([
                    rng.choice(pos_idx, size=len(pos_idx), replace=True),
                    rng.choice(neg_idx, size=len(neg_idx), replace=True),
                ])
            else:
                idx = rng.integers(0, n, size=n)

            y_b = y_true[idx]
            for s in setups:
                boot_ap[s][b] = average_precision_score(y_b, scores[s][idx])

        return APComparison(
            setups=setups,
            y_true=y_true,
            scores=scores,
            boot_ap=boot_ap,
            observed_ap=observed_ap,
            n=n,
            n_pos=int(len(pos_idx)),
        )

    def _pool_predictions(
        self,
        trial_result: TrialResult,
        setups: list[str],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Reconstruct and pool real observation-level predictions.

        Re-draws each test set from the trial's DGP using the exact seeds
        recorded in ``trial_result.scores`` and the size in
        ``trial_result.budget``, then re-runs ``predict_proba`` with the
        stored fitted models. Predictions are concatenated across repeats.
        """
        scores_df = trial_result.scores
        available = scores_df["setup"].unique().tolist()
        for setup in setups:
            if setup not in available:
                raise ValueError(
                    f"Setup {setup!r} not in trial scores. Available: {available}"
                )

        dgp = trial_result.trial.dgp
        n_test = trial_result.budget.n_test
        target_col = trial_result.target_col
        test_seeds = sorted(int(s) for s in scores_df["test_seed"].unique())

        # Feature columns per setup are recorded on each score row.
        features = {
            setup: scores_df.loc[scores_df["setup"] == setup, "features"]
            .iloc[0]
            .split(",")
            for setup in setups
        }

        y_parts: list[np.ndarray] = []
        score_parts: dict[str, list[np.ndarray]] = {s: [] for s in setups}

        for seed in test_seeds:
            df_test = dgp.sample(n_test, seed)
            y_parts.append(df_test[target_col].to_numpy())
            for setup in setups:
                model = trial_result.models[setup]
                p_hat = model.predict_proba(df_test[features[setup]])[:, 1]
                score_parts[setup].append(p_hat)

        y_true = np.concatenate(y_parts).astype(int)
        scores = {s: np.concatenate(score_parts[s]) for s in setups}
        return y_true, scores

    def _find_trial_result(
        self,
        result: StudyResult,
        trial_value: float,
    ) -> TrialResult:
        matches = [tr for tr in result.trial_results if tr.trial.value == trial_value]
        if not matches:
            available = [tr.trial.value for tr in result.trial_results]
            raise ValueError(
                f"trial_value {trial_value} not found. Available: {available}"
            )
        return matches[0]
