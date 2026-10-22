"""
run.py
------
``Run`` and ``record``: one execution of an experiment folder, and its record.

A ``Run`` names the experiment folder, the profile it ran under, its seed and
its resolved config. The profile decides where outputs go:

    profile "default"  ->  <experiment>/outputs/          (the tracked record)
    any other profile  ->  <experiment>/outputs_<profile>/  (gitignored)

so a smoke or exploratory run can never overwrite the committed results.

``record(run, tables)`` writes the given tables as CSV plus ``run_meta.json``
(profile, seed, config, git revision, library versions). ``run_meta.json`` has
no timestamp, so rerunning the same code with the same seed leaves it unchanged.

Example
-------
>>> run = Run(Path(__file__).parent, profile="smoke", seed=42, config=cfg_dict)
>>> run.output_dir
PosixPath('.../outputs_smoke')
>>> record(run, {"metrics": metrics_df})
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

RECORD_PROFILE = "default"
META_FILE = "run_meta.json"


class ProfileMismatch(RuntimeError):
    """An output folder already holds results from a different profile."""


@dataclass
class Run:
    """
    One execution of one experiment folder under one profile.

    Parameters
    ----------
    experiment_dir : Path
        The dated experiment folder, e.g. ``experiments/2026-07-21_imbalanced_classification``.
    profile : str
        ``"default"`` writes the tracked record into ``outputs/``; any other
        name writes into ``outputs_<profile>/``.
    seed : int | None
        The run's base seed, recorded in ``run_meta.json``.
    config : dict
        The resolved configuration, recorded in ``run_meta.json``.

    Creating a ``Run`` creates its output folder. It raises ``ProfileMismatch``
    if that folder's ``run_meta.json`` names a different profile.
    """

    experiment_dir: Path
    profile: str = RECORD_PROFILE
    seed: int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.experiment_dir = Path(self.experiment_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.output_dir / META_FILE
        if meta_path.exists():
            previous = json.loads(meta_path.read_text()).get("profile")
            if previous is not None and previous != self.profile:
                raise ProfileMismatch(
                    f"{self.output_dir} holds results from profile {previous!r}, "
                    f"not {self.profile!r}. Delete the folder or run with profile {previous!r}."
                )

    @property
    def is_record(self) -> bool:
        """True when this run writes the tracked record (profile ``"default"``)."""
        return self.profile == RECORD_PROFILE

    @property
    def output_dir(self) -> Path:
        if self.is_record:
            return self.experiment_dir / "outputs"
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", self.profile)
        return self.experiment_dir / f"outputs_{slug}"


def record(run: Run, tables: dict[str, pd.DataFrame] | None = None) -> Path:
    """
    Write ``tables`` as ``<name>.csv`` and ``run_meta.json`` into the run's output folder.

    Returns the output folder.
    """
    out = run.output_dir
    for name, table in (tables or {}).items():
        table.to_csv(out / f"{name}.csv", index=False)

    meta = {
        "experiment": run.experiment_dir.name,
        "profile": run.profile,
        "seed": run.seed,
        "git_revision": _git_revision(run.experiment_dir),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "config": run.config,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    return out


FIGURE_SUFFIXES = (".png", ".svg")


def report_figures(experiment_dir: Path) -> list[Path]:
    """
    Figures in ``outputs/`` that the experiment's Markdown embeds, names, or matches by glob.

    These are part of the record and stay tracked; other figures are gitignored.
    A figure counts when any ``*.md`` in the experiment (outside ``outputs_*``)
    embeds it (``![..](path)``), names its file in backticks, or names a
    backticked glob such as ``fig_heatmap_*.png`` that matches it.
    """
    import fnmatch

    experiment_dir = Path(experiment_dir)
    outputs = experiment_dir / "outputs"
    if not outputs.exists():
        return []
    figures = sorted(p for p in outputs.rglob("*") if p.suffix in FIGURE_SUFFIXES)

    names: set[str] = set()
    patterns: set[str] = set()
    for md in experiment_dir.rglob("*.md"):
        if any(part.startswith("outputs_") for part in md.relative_to(experiment_dir).parts):
            continue
        text = md.read_text(errors="ignore")
        for target in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
            names.add(Path(target).name)
        for token in re.findall(r"`([^`\s]+)`", text):
            (patterns if "*" in token else names).add(Path(token).name)

    return [
        p for p in figures
        if p.name in names or any(fnmatch.fnmatch(p.name, pat) for pat in patterns)
    ]


def _git_revision(path: Path) -> str | None:
    """
    HEAD commit, suffixed ``-dirty`` when tracked code has uncommitted changes.

    Experiment outputs and generated reports are excluded, so a run's own
    freshly written results do not mark it dirty.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git", "status", "--porcelain", "--untracked-files=no", "--", ":/",
                ":(exclude,glob,top)experiments/*/outputs*/**",
                ":(exclude,glob,top)experiments/*/report.md",
            ],
            cwd=path, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{rev}-dirty" if dirty else rev


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "umap-learn", "imodels"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions
