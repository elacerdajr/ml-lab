"""
run_experiment.py
-----------------
Thin entry point (repo convention) for the encoder-comparison experiment.

Prepends this folder to ``sys.path`` (so ``import enccmp`` works) and the repo
root (so ``import ml_elements`` works), then delegates to ``enccmp.main.main``.

Usage
-----
    uv run --extra catboost python run_experiment.py
    uv run --extra catboost python run_experiment.py --smoke
    uv run --extra catboost python run_experiment.py --profile full_spec
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))   # import enccmp
sys.path.insert(0, str(REPO_ROOT))    # import ml_elements

from enccmp.main import main  # noqa: E402

if __name__ == "__main__":
    main()
