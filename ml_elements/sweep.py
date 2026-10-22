"""
sweep.py
--------
``sweep``: turn a parameter space into a list of ``Trial`` objects.

    trials = sweep([{"p_pos": p} for p in (0.02, 0.1, 0.5)],
                   population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info=INFO),
                   name="p_pos")
    trials = sweep({"p_pos": (0.02, 0.5), "lr": (0.01, 0.3)}, method="sobol", n=32,
                   population=lambda p: GaussianPopulation(p_pos=p["p_pos"], info=INFO),
                   model=lambda p: make_hgb(learning_rate=p["lr"]))

Methods
-------
"list"    ``space`` is a list of parameter dicts, one trial each (the default
          when ``space`` is a list). A single-parameter dict uses its value as
          ``trial.value``; otherwise the index.
"sobol"   ``space`` maps names to ``(low, high)``; ``n`` scrambled Sobol points.
"random"  Same space; ``n`` independent uniform points.

Each trial gets ``seed_offset = seed_offset + i`` and keeps its parameters in
``trial.params``. Trials match the former ``ManualSearch``, ``SobolSearch``
and ``RandomSearch`` exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from .protocols import ModelBackend, Population
from .trial import Trial

Params = dict[str, float]
Method = Literal["list", "sobol", "random"]

_DEFAULT_NAMES = {"list": "manual", "sobol": "sobol", "random": "random_search"}


def sweep(
    space: list[Params] | dict[str, tuple[float, float]],
    *,
    population: Callable[[Params], Population],
    method: Method | None = None,
    n: int | None = None,
    model: Callable[[Params], Callable[[], ModelBackend]] | None = None,
    seed: int = 0,
    seed_offset: int = 0,
    name: str | None = None,
) -> list[Trial]:
    """
    Build one ``Trial`` per point of ``space``.

    Parameters
    ----------
    space : list of dicts, or dict of ``name -> (low, high)``
    population : Callable[[params], Population]
        Builds each trial's population from its parameters.
    method : {"list", "sobol", "random"}, optional
        Defaults to "list" for a list ``space``; required otherwise.
    n : int
        Number of points for "sobol" and "random".
    model : Callable[[params], model factory], optional
        Per-trial model; ``None`` uses ``evaluate``'s default.
    seed : int
        Sobol scrambling seed or random-number seed.
    seed_offset : int
        Base for each trial's data seeds; trial i gets ``seed_offset + i``.
    name : str, optional
        ``trial.name`` (the axis label in result tables).
    """
    points, values = _points(space, method, n, seed)
    method = method or "list"
    label = name or _DEFAULT_NAMES[method]
    return [
        Trial(
            name=label,
            value=value,
            population=population(params),
            seed_offset=seed_offset + i,
            model_factory=model(params) if model is not None else None,
            params=params,
        )
        for i, (params, value) in enumerate(zip(points, values, strict=True))
    ]


def sweep_points(
    space: dict[str, tuple[float, float]], *, method: Method, n: int, seed: int = 0
) -> pd.DataFrame:
    """The parameter values ``sweep`` would use, one row per trial."""
    points, _ = _points(space, method, n, seed)
    return pd.DataFrame(points, columns=list(space))


def _points(space, method, n, seed) -> tuple[list[Params], list[float]]:
    if isinstance(space, list):
        if method not in (None, "list"):
            raise ValueError(f"A list space needs method='list', got {method!r}.")
        if not space:
            raise ValueError("space must not be empty.")
        values = [list(p.values())[0] if len(p) == 1 else float(i) for i, p in enumerate(space)]
        return list(space), values

    if method not in ("sobol", "random"):
        raise ValueError(f"A bounds space needs method='sobol' or 'random', got {method!r}.")
    if not space:
        raise ValueError("space must not be empty.")
    if n is None or n < 1:
        raise ValueError(f"n must be >= 1 for method={method!r}.")

    names = list(space)
    if method == "sobol":
        from scipy.stats import qmc

        bounds = list(space.values())
        sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
        scaled = qmc.scale(sampler.random(n), [b[0] for b in bounds], [b[1] for b in bounds])
        points = [dict(zip(names, row.tolist(), strict=True)) for row in scaled]
    else:
        rng = np.random.default_rng(seed)
        points = [
            {k: float(rng.uniform(low, high)) for k, (low, high) in space.items()}
            for _ in range(n)
        ]
    return points, [float(i) for i in range(n)]
