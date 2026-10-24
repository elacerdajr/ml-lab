.PHONY: help lint test check track-figures \
	exp-roc-vs-ap exp-roc-vs-ap-imbalance exp-roc-vs-ap-info exp-embedding-pca \
	exp-risk-uncertainty exp-rule-viz exp-weak-features exp-positives-weak-features \
	exp-catboost-leaf-embeddings exp-weak-features-beta exp-noisy-label-catboost \
	exp-leaf-embedding-ranking exp-spread-score exp-imbalanced-classification \
	exp-imbalanced-classification-smoke exp-leaf-embedding-umap

# Run everything through `uv run` so it uses the locked environment from uv.lock.
# Each experiment target sets EXTRAS to the optional dependency groups it needs
# (see [project.optional-dependencies] in pyproject.toml); uv installs them on demand.
# Override with `PYTHON=...` to bypass uv.
EXTRAS ?=
PYTHON ?= uv run $(foreach extra,$(EXTRAS),--extra $(extra)) python

.DEFAULT_GOAL := help

help:  ## List the targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-38s %s\n", $$1, $$2}'

lint:  ## Lint the library, experiments, notebooks and tests with ruff
	uv run ruff check .

test:  ## Run the test suite (the slow smoke test needs the catboost and umap extras)
	uv run --extra catboost --extra umap pytest -q

check: lint test  ## Lint, then test

track-figures:  ## Force-add every output figure a report embeds or names (other figures are gitignored)
	$(PYTHON) -c "from pathlib import Path; from ml_elements.run import report_figures; print('\n'.join(str(p) for d in sorted(Path('experiments').iterdir()) if d.is_dir() for p in report_figures(d)))" | xargs git add -f

# ── Experiments, in date order ───────────────────────────────────────────────

exp-roc-vs-ap:  ## ROC-AUC vs AP early stopping: both studies
	cd experiments/2026-06-06_roc_vs_ap && $(PYTHON) run_experiment.py

exp-roc-vs-ap-imbalance:  ## ROC-AUC vs AP: class-imbalance sweep only
	cd experiments/2026-06-06_roc_vs_ap && $(PYTHON) run_study_imbalance.py

exp-roc-vs-ap-info:  ## ROC-AUC vs AP: feature-information sweep only
	cd experiments/2026-06-06_roc_vs_ap && $(PYTHON) run_study_feature_info.py

exp-embedding-pca: EXTRAS = embedding
exp-embedding-pca:  ## Sentence-embedding PCA: how many dimensions reproduce the space
	cd experiments/2026-06-09_embedding_pca && $(PYTHON) run_experiment.py

exp-risk-uncertainty: EXTRAS = catboost imodels
exp-risk-uncertainty:  ## Risk vs uncertainty: model behaviour across data scenarios
	cd experiments/2026-06-15_risk_uncertainty && $(PYTHON) run_experiment.py

exp-rule-viz: EXTRAS = imodels
exp-rule-viz:  ## Train interpretable models, extract rules, build an interactive D3 HTML report
	cd experiments/2026-06-15_rule_viz && $(PYTHON) run_experiment.py

exp-weak-features: EXTRAS = catboost imodels
exp-weak-features:  ## 100 weak features: CatBoost vs FIGS / GreedyRuleList / DecisionTree across training size
	cd experiments/2026-06-18_weak_features_sample_size && $(PYTHON) run_experiment.py

exp-positives-weak-features: EXTRAS = catboost
exp-positives-weak-features:  ## 1% positives: which CatBoost depth / L2 settings resist noise features
	cd experiments/2026-06-22_positives_weak_features && $(PYTHON) run_experiment.py

exp-catboost-leaf-embeddings: EXTRAS = catboost umap
exp-catboost-leaf-embeddings:  ## CatBoost leaf embeddings + Hamming RBF kernel as soft targets
	cd experiments/2026-06-23_catboost_leaf_embeddings && $(PYTHON) run_experiment.py

exp-weak-features-beta: EXTRAS = catboost imodels viz
exp-weak-features-beta:  ## Same as exp-weak-features but info_j ~ Beta(1,9) instead of constant 0.10
	cd experiments/2026-06-23_weak_features_beta && $(PYTHON) run_experiment.py

exp-noisy-label-catboost: EXTRAS = catboost umap
exp-noisy-label-catboost:  ## CatBoost CrossEntropy on noisy soft labels vs Logloss on hard labels, across noise levels
	cd experiments/2026-07-01_noisy_label_catboost && $(PYTHON) run_experiment.py

exp-leaf-embedding-ranking: EXTRAS = catboost
exp-leaf-embedding-ranking:  ## Leaf-embedding residual ranking: Ridge-on-leaves + ECDF rank spreading vs raw CatBoost probability
	cd experiments/2026-07-02_leaf_embedding_ranking && $(PYTHON) run_experiment.py

exp-spread-score: EXTRAS = catboost
exp-spread-score:  ## Spread score: is leaf-neighbourhood label disagreement learnable from raw features
	cd experiments/2026-07-02_spread_score && $(PYTHON) run_experiment.py

exp-imbalanced-classification: EXTRAS = catboost umap viz
exp-imbalanced-classification:  ## Rare-positive (0.1%) model comparison: AP vs score entropy, priors, leaf/RFF UMAP
	cd experiments/2026-07-21_imbalanced_classification && $(PYTHON) run_experiment.py

exp-imbalanced-classification-smoke: EXTRAS = catboost umap viz
exp-imbalanced-classification-smoke:  ## Fast smoke run of the imbalanced-classification experiment (writes outputs_smoke/)
	cd experiments/2026-07-21_imbalanced_classification && $(PYTHON) run_experiment.py --smoke

exp-leaf-embedding-umap: EXTRAS = catboost umap viz
exp-leaf-embedding-umap:  ## Leaf-index UMAP (Hamming) reduction vs native CatBoost: logit/SVM/RF/MLP/CatBoost across k
	cd experiments/2026-07-22_leaf_embedding_umap && $(PYTHON) run_experiment.py
