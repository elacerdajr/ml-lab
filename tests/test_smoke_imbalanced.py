"""End-to-end: the imbalanced-classification smoke profile runs from a clean copy."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("catboost")
pytest.importorskip("umap")

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "2026-07-21_imbalanced_classification"


@pytest.mark.slow
def test_smoke_run_writes_only_its_own_folder(tmp_path):
    # A fresh copy has no outputs/ at all, like a clone under the record policy.
    copy = tmp_path / EXPERIMENT.name
    copy.mkdir()
    shutil.copytree(EXPERIMENT / "imbcls", copy / "imbcls", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("config.yaml", "run_experiment.py"):
        shutil.copy(EXPERIMENT / name, copy / name)

    subprocess.run([sys.executable, "run_experiment.py", "--smoke"], cwd=copy, check=True,
                   capture_output=True, text=True, timeout=600)

    out = copy / "outputs_smoke"
    assert not (copy / "outputs").exists() and not (copy / "report.md").exists()
    assert json.loads((out / "run_meta.json").read_text())["profile"] == "smoke"
    metrics = pd.read_csv(out / "metrics.csv")
    assert "train_time_seconds" not in metrics.columns
    assert metrics["average_precision"].between(0, 1).all()
    assert "](plots/" in (out / "report.md").read_text()
