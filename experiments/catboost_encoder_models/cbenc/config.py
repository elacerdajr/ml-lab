"""
config.py
---------
Typed configuration for the CatBoost-encoder experiment. Same profile-merge
design as the sibling experiments: the YAML top level is the default, named
``profiles`` are deep-merged on top, and ``--smoke`` applies the ``smoke``
profile last regardless of ``--profile``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
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
class EncoderConfig:
    sigma: float | None
    a: float


@dataclass
class OutputConfig:
    save_preprocessors: bool


@dataclass
class Config:
    seed: int
    data: DataConfig
    splits: SplitConfig
    undersample: UndersampleConfig
    encoder: EncoderConfig
    models: dict[str, Any]
    output: OutputConfig
    profile: str = "default"
    raw: dict[str, Any] = field(default_factory=dict)


def _build(merged: dict, profile: str) -> Config:
    return Config(
        seed=int(merged["seed"]),
        data=DataConfig(**merged["data"]),
        splits=SplitConfig(**merged["splits"]),
        undersample=UndersampleConfig(**merged["undersample"]),
        encoder=EncoderConfig(**merged["encoder"]),
        models=merged["models"],
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
    """Load and resolve the CatBoost-encoder experiment configuration."""
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
