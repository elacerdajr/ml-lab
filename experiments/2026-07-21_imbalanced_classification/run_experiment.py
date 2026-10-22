"""
run_experiment.py
-----------------
Thin entry point (repo convention) for the imbalanced-classification experiment.

Run it from this folder so ``import imbcls`` resolves (Python puts the script's
folder on the path); ``ml_elements`` comes from the installed package. It
delegates to ``imbcls.main.main``.

Usage
-----
    uv run --extra catboost --extra umap --extra viz python run_experiment.py
    uv run --extra catboost --extra umap --extra viz python run_experiment.py --smoke
    uv run --extra catboost --extra umap --extra viz python run_experiment.py --profile full_spec
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

from imbcls.main import main  # noqa: E402

if __name__ == "__main__":
    main()
