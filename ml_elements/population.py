"""
population.py
-------------
Populations: what an experiment samples labelled rows from.

Any object with ``sample(n, seed) -> DataFrame`` is a population (see
``protocols.Population``). No inheritance required. A population can be
synthetic, a resampled real table, or a transformed version of another one.

Classes
-------
GaussianPopulation
    Synthetic Gaussian classes. x_j | y=c ~ N(mu_cj, sigma).
    Information knob: ``info_j = |mu_1j - mu_0j| / sigma``.

ResampledPopulation
    Draws rows with replacement from a real DataFrame. Optional
    ``label_noise`` flips a fraction of labels.

ShiftedPopulation
    Wraps any population and applies ``shift_fn(df) -> df`` to each sample,
    for distribution shift, rescaling or feature engineering.

These were called ``GaussianBinaryDGP``, ``RealDataDGP`` and ``ShiftedDGP``
("data-generating process"); those names remain as aliases.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


class GaussianPopulation:
    """
    Synthetic binary classification population with Gaussian classes.

    Data model
    ----------
    y ~ Bernoulli(p_pos)

    x_j | y=0 ~ Normal(0, sigma)
    x_j | y=1 ~ Normal(info_j * sigma, sigma)

    So ``info_j = separation_j = |mu_1j - mu_0j| / sigma`` acts as a
    clean signal-strength knob: higher info → less class overlap.

    Parameters
    ----------
    p_pos : float
        Probability of a positive label.
    info : dict[str, float]
        Feature name → information level (separation). Zero means the
        feature carries no signal.
    sigma : float
        Within-class standard deviation shared by all features.

    Examples
    --------
    >>> population = GaussianPopulation(p_pos=0.1, info={"x1": 0.8, "x2": 0.3})
    >>> df = population.sample(n=1000, seed=42)
    >>> df.columns.tolist()
    ['y', 'x1', 'x2']
    """

    def __init__(
        self,
        p_pos: float,
        info: dict[str, float],
        sigma: float = 1.0,
    ) -> None:
        if not (0.0 < p_pos < 1.0):
            raise ValueError(f"p_pos must be in (0, 1), got {p_pos}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.p_pos = p_pos
        self.info = info
        self.sigma = sigma

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """
        Draw n labelled rows.

        Returns
        -------
        pd.DataFrame
            Columns: ``y`` (0/1) + one column per feature in ``info``.
        """
        rng = np.random.default_rng(seed)

        y = rng.binomial(n=1, p=self.p_pos, size=n)
        df = pd.DataFrame({"y": y})

        for feature, info_j in self.info.items():
            mu1 = info_j * self.sigma
            mu = np.where(y == 1, mu1, 0.0)
            df[feature] = rng.normal(loc=mu, scale=self.sigma, size=n)

        return df


class ResampledPopulation:
    """
    Population backed by a real DataFrame, sampled with replacement.

    Sampling with replacement (bootstrap-style) keeps sample sizes flexible.
    Optionally injects label noise.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset. Must contain ``target_col``.
    target_col : str
        Name of the binary label column. Samples rename it to ``y``.
    label_noise : float
        Fraction of labels to randomly flip after sampling (in [0, 1)).

    Examples
    --------
    >>> population = ResampledPopulation(df=my_dataframe, target_col="click")
    >>> sample = population.sample(n=500, seed=7)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        label_noise: float = 0.0,
    ) -> None:
        if target_col not in df.columns:
            raise ValueError(f"target_col {target_col!r} not found in DataFrame columns.")
        if not (0.0 <= label_noise < 1.0):
            raise ValueError(f"label_noise must be in [0, 1), got {label_noise}")

        self._df = df.reset_index(drop=True)
        self.target_col = target_col
        self.label_noise = label_noise

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Draw n rows from the backing DataFrame (with replacement)."""
        rng = np.random.default_rng(seed)

        idx = rng.integers(0, len(self._df), size=n)
        out = self._df.iloc[idx].copy().reset_index(drop=True)

        if self.target_col != "y":
            out = out.rename(columns={self.target_col: "y"})

        if self.label_noise > 0.0:
            flip_mask = rng.random(n) < self.label_noise
            out.loc[flip_mask, "y"] = 1 - out.loc[flip_mask, "y"]

        return out


class ShiftedPopulation:
    """
    A population whose samples pass through ``shift_fn``.

    Use it to inject distribution shift, rescale features, add outliers,
    or test robustness without modifying the base population.

    Parameters
    ----------
    base : Population
        Any object with ``sample(n, seed)``.
    shift_fn : Callable[[pd.DataFrame], pd.DataFrame]
        Transform applied to each sample. Must keep the ``y`` column.

    Examples
    --------
    >>> shifted = ShiftedPopulation(
    ...     base=GaussianPopulation(p_pos=0.1, info={"x1": 0.8}),
    ...     shift_fn=lambda df: df.assign(x1=df["x1"] + 2.0),
    ... )
    >>> shifted.sample(n=100, seed=0)
    """

    def __init__(
        self,
        base: object,
        shift_fn: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> None:
        self.base = base
        self.shift_fn = shift_fn

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample from the base population, then apply the shift."""
        df = self.base.sample(n, seed)
        return self.shift_fn(df)
