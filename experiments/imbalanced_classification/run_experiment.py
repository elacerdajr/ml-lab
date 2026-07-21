"""
run_experiment.py
-----------------
Thin entry point (repo convention) for the imbalanced-classification experiment.

Prepends this folder to ``sys.path`` (so ``import imbcls`` works) and the repo
root (so ``import ml_elements`` works), then delegates to ``imbcls.main.main``.

Usage
-----
    uv run --extra catboost --extra umap --extra viz python run_experiment.py
    uv run --extra catboost --extra umap --extra viz python run_experiment.py --smoke
    uv run --extra catboost --extra umap --extra viz python run_experiment.py --profile full_spec
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))   # import imbcls
sys.path.insert(0, str(REPO_ROOT))    # import ml_elements

from imbcls.main import main  # noqa: E402

if __name__ == "__main__":
    main()
