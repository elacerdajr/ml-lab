"""
search.py
---------
Trial generators for hyperparameter search.

All three classes share the same interface: call ``.trials(dgp_fn)`` and
get back a ``list[Trial]`` that can be passed directly to ``Study.run()``.

Classes
-------
SobolSearch
    Quasi-random low-discrepancy sequences (scipy Sobol). Better coverage
    than uniform random for the same number of points. Ideal for 2–10
    continuous parameters.

RandomSearch
    Independent uniform random sampling. Simple and reproducible.

ManualSearch
    Explicit list of parameter dicts. Use when you already know the points
    you want to evaluate (grid search, hand-crafted ablations, etc.).

Interface
---------
All three accept:
    - ``dgp_fn(params) -> DGP``   required — wires sampled params to a DGP
    - ``model_fn(params) -> ModelBackend | None``  optional — per-trial model
    - ``seed_offset``  base offset for trial seeds

The caller is in full control of which parameters flow into the DGP vs the
model factory, because they write the lambda/function.

Examples
--------
Sobol search over (p_pos, x3_info, learning_rate):

    search = SobolSearch(
        param_space={"p_pos": (0.02, 0.5), "x3_info": (0.0, 1.5), "lr": (0.01, 0.3)},
        n_points=32,
        seed=0,
    )
    trials = search.trials(
        dgp_fn=lambda p: GaussianBinaryDGP(
            p_pos=p["p_pos"],
            info={"x1": 0.85, "x2": 0.55, "x3": p["x3_info"]},
        ),
        model_fn=lambda p: make_hgb(learning_rate=p["lr"]),
    )
    result = study.run(trials)

Manual ablation — three explicit points:

    search = ManualSearch([
        {"label_noise": 0.0},
        {"label_noise": 0.05},
        {"label_noise": 0.10},
    ])
    trials = search.trials(
        dgp_fn=lambda p: ShiftedDGP(base_dgp, lambda df: inject_noise(df, p["label_noise"])),
    )
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .protocols import DGP, ModelBackend
from .trial import Trial


_DGPFactory = Callable[[dict[str, float]], DGP]
_ModelFactory = Callable[[], ModelBackend]
_ModelFn = Optional[Callable[[dict[str, float]], _ModelFactory]]


def _make_trial(
    name: str,
    value: float,
    params: dict[str, float],
    seed_offset: int,
    dgp_fn: _DGPFactory,
    model_fn: _ModelFn,
) -> Trial:
    dgp = dgp_fn(params)
    model_factory = model_fn(params) if model_fn is not None else None
    return Trial(
        name=name,
        value=value,
        dgp=dgp,
        seed_offset=seed_offset,
        model_factory=model_factory,
    )


class SobolSearch:
    """
    Quasi-random Sobol sequence search over a continuous parameter space.

    Points are evenly distributed across the unit hypercube (low discrepancy),
    giving better coverage than plain random for the same budget.

    Requires ``scipy >= 1.7``.

    Parameters
    ----------
    param_space : dict[str, tuple[float, float]]
        Parameter name → (low, high) bounds.
    n_points : int
        Number of points to generate.
    seed : int
        Scrambling seed for the Sobol sequence.

    Examples
    --------
    >>> search = SobolSearch({"p_pos": (0.02, 0.5), "lr": (0.01, 0.3)}, n_points=16)
    >>> trials = search.trials(dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], ...))
    """

    def __init__(
        self,
        param_space: dict[str, tuple[float, float]],
        n_points: int,
        seed: int = 0,
    ) -> None:
        if not param_space:
            raise ValueError("param_space must not be empty.")
        if n_points < 1:
            raise ValueError("n_points must be >= 1.")

        self.param_space = param_space
        self.n_points = n_points
        self.seed = seed

    def trials(
        self,
        dgp_fn: _DGPFactory,
        model_fn: _ModelFn = None,
        seed_offset: int = 0,
    ) -> list[Trial]:
        """
        Generate trials from Sobol-sampled parameters.

        Parameters
        ----------
        dgp_fn : Callable[[dict[str, float]], DGP]
            Maps a parameter dict to a DGP instance.
        model_fn : Callable[[dict[str, float]], ModelBackend] | None
            Maps a parameter dict to a model factory. ``None`` = use Study default.
        seed_offset : int
            Added to each trial's ``seed_offset`` to avoid data overlap.

        Returns
        -------
        list[Trial]
        """
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise ImportError(
                "SobolSearch requires scipy >= 1.7. Install with: pip install scipy"
            ) from exc

        names = list(self.param_space.keys())
        bounds = list(self.param_space.values())
        l_bounds = [b[0] for b in bounds]
        u_bounds = [b[1] for b in bounds]

        sampler = qmc.Sobol(d=len(names), scramble=True, seed=self.seed)
        raw = sampler.random(self.n_points)
        scaled = qmc.scale(raw, l_bounds=l_bounds, u_bounds=u_bounds)

        result = []
        for i, row in enumerate(scaled):
            params = dict(zip(names, row.tolist()))
            trial = _make_trial(
                name="sobol",
                value=float(i),
                params=params,
                seed_offset=seed_offset + i,
                dgp_fn=dgp_fn,
                model_fn=model_fn,
            )
            result.append(trial)

        return result

    def param_dataframe(self) -> "pd.DataFrame":
        """
        Return a DataFrame of the sampled parameter values (for inspection).
        Requires the same scipy version as ``trials()``.
        """
        import pandas as pd
        from scipy.stats import qmc

        names = list(self.param_space.keys())
        bounds = list(self.param_space.values())

        sampler = qmc.Sobol(d=len(names), scramble=True, seed=self.seed)
        raw = sampler.random(self.n_points)
        scaled = qmc.scale(raw, [b[0] for b in bounds], [b[1] for b in bounds])
        return pd.DataFrame(scaled, columns=names)


class RandomSearch:
    """
    Independent uniform random search over a continuous parameter space.

    Simpler than Sobol but still reproducible via ``seed``. Good for
    quick exploratory searches.

    Parameters
    ----------
    param_space : dict[str, tuple[float, float]]
        Parameter name → (low, high) bounds.
    n_points : int
        Number of random points to sample.
    seed : int
        NumPy random seed.
    """

    def __init__(
        self,
        param_space: dict[str, tuple[float, float]],
        n_points: int,
        seed: int = 0,
    ) -> None:
        if not param_space:
            raise ValueError("param_space must not be empty.")
        if n_points < 1:
            raise ValueError("n_points must be >= 1.")

        self.param_space = param_space
        self.n_points = n_points
        self.seed = seed

    def trials(
        self,
        dgp_fn: _DGPFactory,
        model_fn: _ModelFn = None,
        seed_offset: int = 0,
    ) -> list[Trial]:
        """
        Generate trials from uniformly random parameter samples.

        Parameters
        ----------
        dgp_fn : Callable[[dict[str, float]], DGP]
        model_fn : Callable[[dict[str, float]], ModelBackend] | None
        seed_offset : int

        Returns
        -------
        list[Trial]
        """
        rng = np.random.default_rng(self.seed)
        names = list(self.param_space.keys())

        result = []
        for i in range(self.n_points):
            params = {
                name: float(rng.uniform(low, high))
                for name, (low, high) in self.param_space.items()
            }
            trial = _make_trial(
                name="random_search",
                value=float(i),
                params=params,
                seed_offset=seed_offset + i,
                dgp_fn=dgp_fn,
                model_fn=model_fn,
            )
            result.append(trial)

        return result


class ManualSearch:
    """
    Explicit list of parameter dicts — full user control.

    Use for grid search, hand-crafted ablations, or any case where you
    know exactly which points you want to evaluate.

    Parameters
    ----------
    param_list : list[dict[str, float]]
        Each dict is one set of parameters for one trial.
    trial_name : str
        Label for the trial axis (appears in result DataFrames).

    Examples
    --------
    >>> search = ManualSearch(
    ...     [{"p_pos": 0.05}, {"p_pos": 0.10}, {"p_pos": 0.25}],
    ...     trial_name="p_pos",
    ... )
    >>> trials = search.trials(dgp_fn=lambda p: GaussianBinaryDGP(p_pos=p["p_pos"], ...))
    """

    def __init__(
        self,
        param_list: list[dict[str, float]],
        trial_name: str = "manual",
    ) -> None:
        if not param_list:
            raise ValueError("param_list must not be empty.")

        self.param_list = param_list
        self.trial_name = trial_name

    def trials(
        self,
        dgp_fn: _DGPFactory,
        model_fn: _ModelFn = None,
        seed_offset: int = 0,
    ) -> list[Trial]:
        """
        Generate trials from the explicit parameter list.

        Parameters
        ----------
        dgp_fn : Callable[[dict[str, float]], DGP]
        model_fn : Callable[[dict[str, float]], ModelBackend] | None
        seed_offset : int

        Returns
        -------
        list[Trial]
        """
        result = []
        for i, params in enumerate(self.param_list):
            value = list(params.values())[0] if len(params) == 1 else float(i)
            trial = _make_trial(
                name=self.trial_name,
                value=value,
                params=params,
                seed_offset=seed_offset + i,
                dgp_fn=dgp_fn,
                model_fn=model_fn,
            )
            result.append(trial)

        return result
