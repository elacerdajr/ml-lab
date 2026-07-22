"""
config.py
---------
Typed configuration for the imbalanced-classification experiment.

The YAML file carries a ``default`` set of values plus named ``profiles`` whose
keys are deep-merged on top of the defaults. ``load_config`` resolves a profile
(and the ``--smoke`` override) into a nested :class:`Config` dataclass so the
rest of the package never touches raw dictionaries.

Examples
--------
>>> cfg = load_config("config.yaml")                     # default profile
>>> cfg = load_config("config.yaml", profile="full_spec")
>>> cfg = load_config("config.yaml", smoke=True)
>>> cfg.data.n_full
300000
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class DataConfig:
    n_full: int
    positive_rate: float
    cat_1_values: int
    cat_2_values: int
    cat_2_high_risk: int
    num_2_threshold: float
    noise_sigma: float


@dataclass
class SplitConfig:
    val_fraction: float
    test_fraction: float


@dataclass
class UndersampleConfig:
    positive_rate: float


@dataclass
class PriorConfig:
    label_smoothing_lambdas: list[float]
    synthetic_rhos: list[float]
    synthetic_num_perturb: float
    shrinkage_lambdas: list[float]
    noise_alphas: list[float]


@dataclass
class UmapConfig:
    n_umap: int
    n_neighbors: int
    min_dist: float


@dataclass
class OutputConfig:
    save_preprocessors: bool
    save_linear_models: bool
    save_catboost_models: bool
    bucket_sizes: list[int]


@dataclass
class Config:
    seed: int
    data: DataConfig
    splits: SplitConfig
    undersample: UndersampleConfig
    priors: PriorConfig
    models: dict[str, Any]
    umap: UmapConfig
    output: OutputConfig
    profile: str = "default"
    raw: dict[str, Any] = field(default_factory=dict)


def _build(merged: dict, profile: str) -> Config:
    return Config(
        seed=int(merged["seed"]),
        data=DataConfig(**merged["data"]),
        splits=SplitConfig(**merged["splits"]),
        undersample=UndersampleConfig(**merged["undersample"]),
        priors=PriorConfig(**merged["priors"]),
        models=merged["models"],
        umap=UmapConfig(**merged["umap"]),
        output=OutputConfig(**merged["output"]),
        profile=profile,
        raw=merged,
    )


def load_config(
    path: str | Path,
    *,
    profile: str = "default",
    smoke: bool = False,
) -> Config:
    """
    Load and resolve the experiment configuration.

    Parameters
    ----------
    path : str or Path
        Path to ``config.yaml``.
    profile : str
        Named profile under ``profiles:`` to merge over the defaults. The
        literal ``"default"`` applies no profile override.
    smoke : bool
        If True, the ``smoke`` profile override is applied last (tiny run for
        CI / verification), regardless of ``profile``.

    Returns
    -------
    Config
    """
    doc = yaml.safe_load(Path(path).read_text())
    profiles = doc.pop("profiles", {}) or {}

    merged = doc
    applied = "default"
    if profile and profile != "default":
        if profile not in profiles:
            raise KeyError(f"unknown profile {profile!r}; have {sorted(profiles)}")
        merged = _deep_merge(merged, profiles[profile])
        applied = profile
    if smoke:
        merged = _deep_merge(merged, profiles.get("smoke", {}))
        applied = "smoke" if applied == "default" else f"{applied}+smoke"

    return _build(merged, applied)
