"""
search.py
---------
``SobolSearch``, ``RandomSearch`` and ``ManualSearch``: the former trial
generators, kept as thin wrappers over ``sweep``.

    SobolSearch(space, n_points=32, seed=0).trials(dgp_fn=f)
        == sweep(space, method="sobol", n=32, seed=0, population=f)
    RandomSearch(space, n_points=20, seed=1).trials(dgp_fn=f)
        == sweep(space, method="random", n=20, seed=1, population=f)
    ManualSearch(points, trial_name="p_pos").trials(dgp_fn=f)
        == sweep(points, name="p_pos", population=f)

New code should call ``sweep`` directly; see ``sweep.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .protocols import ModelBackend, Population
from .sweep import sweep, sweep_points
from .trial import Trial

if TYPE_CHECKING:
    import pandas as pd

_PopulationFn = Callable[[dict[str, float]], Population]
_ModelFn = Callable[[dict[str, float]], Callable[[], ModelBackend]] | None


class SobolSearch:
    """Scrambled Sobol points over ``param_space``. Prefer ``sweep(method="sobol")``."""

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

    def trials(self, dgp_fn: _PopulationFn, model_fn: _ModelFn = None, seed_offset: int = 0) -> list[Trial]:
        return sweep(self.param_space, method="sobol", n=self.n_points, seed=self.seed,
                     population=dgp_fn, model=model_fn, seed_offset=seed_offset)

    def param_dataframe(self) -> pd.DataFrame:
        """The sampled parameter values, one row per trial."""
        return sweep_points(self.param_space, method="sobol", n=self.n_points, seed=self.seed)


class RandomSearch:
    """Uniform random points over ``param_space``. Prefer ``sweep(method="random")``."""

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

    def trials(self, dgp_fn: _PopulationFn, model_fn: _ModelFn = None, seed_offset: int = 0) -> list[Trial]:
        return sweep(self.param_space, method="random", n=self.n_points, seed=self.seed,
                     population=dgp_fn, model=model_fn, seed_offset=seed_offset)


class ManualSearch:
    """An explicit list of parameter dicts. Prefer ``sweep(points, name=...)``."""

    def __init__(self, param_list: list[dict[str, float]], trial_name: str = "manual") -> None:
        if not param_list:
            raise ValueError("param_list must not be empty.")
        self.param_list = param_list
        self.trial_name = trial_name

    def trials(self, dgp_fn: _PopulationFn, model_fn: _ModelFn = None, seed_offset: int = 0) -> list[Trial]:
        return sweep(self.param_list, name=self.trial_name,
                     population=dgp_fn, model=model_fn, seed_offset=seed_offset)
