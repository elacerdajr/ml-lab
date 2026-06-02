"""
sweep.py
--------
Scenario sweeps: deliberately *varying the experiment*, not optimizing a model.

Why this is separate from ``search.py``
---------------------------------------
There are two fundamentally different kinds of knob in these experiments, and
they should never be conflated:

============   ==========================   ============================
                Scenario / design knobs       Model hyperparameters
============   ==========================   ============================
examples        ``p_pos``, ``info``           ``learning_rate``, ``depth``
role            define *which world* you      tune the model *within* a
                study                          fixed world
you want        the whole response curve       the single best value
method          **sweep** on a grid            **search** / optimize
"best" value?   no — there is nothing to       yes
                maximize
tool            ``ScenarioSweep`` (here)       ``SobolSearch`` (search.py)
============   ==========================   ============================

A class-balance (``p_pos``) or signal-strength (``info``) value is *user
input* that picks the regime under study. The deliverable of such a study is
``Δmetric vs p_pos`` — a curve — so you want clean, evenly spaced grid points,
not a Sobol scatter. Treating ``p_pos``/``info`` as a search space (looking for
the "best" one) is a category error: use ``ScenarioSweep`` to sweep them.

Use ``search.py`` (Sobol/Random) only for genuine model hyperparameters, and,
if you need both, nest a hyperparameter search *inside* a scenario.

Interface
---------
``ScenarioSweep`` shares the trial-generator interface with ``search.py``:
``.trials(dgp_fn, model_fn=None, seed_offset=0) -> list[Trial]``.
``dgp_fn`` wires each scenario's parameter dict into a concrete DGP.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Optional, Sequence

from .protocols import DGP, ModelBackend
from .trial import Trial

_DGPFactory = Callable[[dict[str, float]], DGP]
_ModelFactory = Callable[[], ModelBackend]
_ModelFn = Optional[Callable[[dict[str, float]], _ModelFactory]]


class ScenarioSweep:
    """
    A grid of experiment scenarios — one ``Trial`` per scenario.

    Each scenario is a dict of *design* parameters (e.g. ``{"p_pos": 0.1}``)
    that ``dgp_fn`` turns into a DGP. Scenarios are swept deliberately and
    exhaustively; they are not optimized.

    Construct directly with an explicit list, or use the convenience
    constructors :meth:`over` (single axis) and :meth:`grid` (cartesian
    product of several axes).

    Parameters
    ----------
    scenarios : list[dict[str, float]]
        Each dict is one scenario's design parameters.
    axis_name : str
        Label for the swept axis (appears as ``trial_name`` in results).

    Examples
    --------
    Single-axis class-balance sweep:

    >>> sweep = ScenarioSweep.over("p_pos", [0.02, 0.05, 0.10, 0.25, 0.50])
    >>> trials = sweep.trials(
    ...     dgp_fn=lambda s: GaussianBinaryDGP(
    ...         p_pos=s["p_pos"],
    ...         info={"x1": 0.85, "x2": 0.55, "x3": 0.35},
    ...     )
    ... )

    Vary the information carried by a new feature:

    >>> sweep = ScenarioSweep.over("x3_info", [0.0, 0.25, 0.5, 1.0])
    >>> trials = sweep.trials(
    ...     dgp_fn=lambda s: GaussianBinaryDGP(
    ...         p_pos=0.15,
    ...         info={"x1": 0.85, "x2": 0.55, "x3": s["x3_info"]},
    ...     )
    ... )
    """

    def __init__(
        self,
        scenarios: list[dict[str, float]],
        axis_name: str = "scenario",
    ) -> None:
        if not scenarios:
            raise ValueError("scenarios must not be empty.")

        self.scenarios = scenarios
        self.axis_name = axis_name

    @classmethod
    def over(cls, axis_name: str, values: Sequence[float]) -> "ScenarioSweep":
        """
        Sweep a single design knob over an explicit list of values.

        Parameters
        ----------
        axis_name : str
            Name of the knob, e.g. ``"p_pos"`` or ``"x3_info"``. Each scenario
            dict will contain exactly this key.
        values : Sequence[float]
            The grid of values to sweep.

        Returns
        -------
        ScenarioSweep
        """
        values = list(values)
        if not values:
            raise ValueError("values must not be empty.")
        return cls([{axis_name: v} for v in values], axis_name=axis_name)

    @classmethod
    def grid(cls, axes: dict[str, Sequence[float]], axis_name: str = "grid") -> "ScenarioSweep":
        """
        Sweep the full cartesian product of several design knobs.

        Each scenario contains one value for every key in ``axes``. The
        per-trial ``value`` is the scenario's index (multi-axis grids have no
        single scalar to plot against — recover the axes from the scenario
        dict via ``trial.dgp`` or by carrying them yourself).

        Parameters
        ----------
        axes : dict[str, Sequence[float]]
            Knob name → grid of values.
        axis_name : str
            Label for the combined axis.

        Returns
        -------
        ScenarioSweep
        """
        if not axes:
            raise ValueError("axes must not be empty.")
        names = list(axes.keys())
        scenarios = [
            dict(zip(names, combo))
            for combo in product(*(list(axes[n]) for n in names))
        ]
        return cls(scenarios, axis_name=axis_name)

    def trials(
        self,
        dgp_fn: _DGPFactory,
        model_fn: _ModelFn = None,
        seed_offset: int = 0,
    ) -> list[Trial]:
        """
        Generate one ``Trial`` per scenario.

        Parameters
        ----------
        dgp_fn : Callable[[dict[str, float]], DGP]
            Maps a scenario dict to a DGP instance.
        model_fn : Callable[[dict[str, float]], ModelBackend] | None
            Optional per-scenario model factory. Usually ``None`` for a pure
            scenario sweep — the model is held fixed so only the *world*
            changes.
        seed_offset : int
            Added to each trial's seed offset to keep data non-overlapping.

        Returns
        -------
        list[Trial]
        """
        trials = []
        for i, scenario in enumerate(self.scenarios):
            value = (
                list(scenario.values())[0]
                if len(scenario) == 1
                else float(i)
            )
            dgp = dgp_fn(scenario)
            model_factory = model_fn(scenario) if model_fn is not None else None
            trials.append(
                Trial(
                    name=self.axis_name,
                    value=value,
                    dgp=dgp,
                    seed_offset=seed_offset + i,
                    model_factory=model_factory,
                )
            )
        return trials
