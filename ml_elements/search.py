"""
search.py
---------
Trial generators for **model hyperparameter search**.

Scope — read this first
-----------------------
These generators are for *optimizing a model* (``learning_rate``, ``depth``,
regularization, …): knobs that have a single best value you want to find.

They are **not** the tool for varying the *experiment* itself — class balance
(``p_pos``), feature signal strength (``info``), label noise, and the like.
Those are design knobs you *sweep* on a grid to map a response curve; there is
no "best" value to search for. Use ``ScenarioSweep`` (``sweep.py``) for them.
See ``sweep.py`` for the full distinction. If you need both, nest a
hyperparameter search inside each scenario.

All classes share the same interface: call ``.trials(dgp_fn)`` and get back a
``list[Trial]`` that can be passed directly to ``Study.run()``.

Classes
-------
SobolSearch
    Quasi-random low-discrepancy sequences (scipy Sobol). Better coverage
    than uniform random for the same number of points. Ideal for 2–10
    continuous hyperparameters.

RandomSearch
    Independent uniform random sampling. Simple and reproducible.

ManualSearch
    Explicit list of parameter dicts — full control over the points
    evaluated. General-purpose; for pure DGP scenario sweeps prefer the
    clearer ``ScenarioSweep``.

Interface
---------
All accept:
    - ``dgp_fn(params) -> DGP``   required — wires the fixed scenario to a DGP
    - ``model_fn(params) -> ModelBackend | None``  optional — per-trial model
    - ``seed_offset``  base offset for trial seeds

The caller controls which parameters flow into the DGP vs the model factory,
because they write the lambda/function.

Examples
--------
Sobol search over model hyperparameters, with the scenario held fixed:

    search = SobolSearch(
        param_space={"lr": (0.01, 0.3), "max_leaf_nodes": (8, 64)},
        n_points=32,
        seed=0,
    )
    trials = search.trials(
        # scenario is fixed — only the model varies
        dgp_fn=lambda p: GaussianBinaryDGP(
            p_pos=0.15,
            info={"x1": 0.85, "x2": 0.55, "x3": 0.35},
        ),
        model_fn=lambda p: make_hgb(
            learning_rate=p["lr"],
            max_leaf_nodes=int(p["max_leaf_nodes"]),
        ),
    )
    result = study.run(trials)

Manual ablation — three explicit hyperparameter points:

    search = ManualSearch([
        {"lr": 0.01},
        {"lr": 0.06},
        {"lr": 0.20},
    ])
    trials = search.trials(
        dgp_fn=lambda p: base_dgp,
        model_fn=lambda p: make_hgb(learning_rate=p["lr"]),
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
    >>> search = SobolSearch({"lr": (0.01, 0.3), "depth": (3, 8)}, n_points=16)
    >>> trials = search.trials(
    ...     dgp_fn=lambda p: fixed_dgp,
    ...     model_fn=lambda p: make_catboost(learning_rate=p["lr"], depth=int(p["depth"])),
    ... )
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

    Use for hand-crafted hyperparameter ablations, or any case where you
    know exactly which points you want to evaluate.

    Note
    ----
    If the parameters you are listing are DGP *scenario* knobs (``p_pos``,
    ``info``, label noise, …) rather than model hyperparameters, prefer
    ``ScenarioSweep`` — it carries the same mechanics but names the intent
    correctly (you are sweeping an experiment axis, not searching for a best
    value).

    Parameters
    ----------
    param_list : list[dict[str, float]]
        Each dict is one set of parameters for one trial.
    trial_name : str
        Label for the trial axis (appears in result DataFrames).

    Examples
    --------
    >>> search = ManualSearch(
    ...     [{"lr": 0.01}, {"lr": 0.06}, {"lr": 0.20}],
    ...     trial_name="lr",
    ... )
    >>> trials = search.trials(
    ...     dgp_fn=lambda p: fixed_dgp,
    ...     model_fn=lambda p: make_hgb(learning_rate=p["lr"]),
    ... )
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
