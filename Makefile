.PHONY: exp-roc-vs-ap exp-roc-vs-ap-imbalance exp-roc-vs-ap-info exp-embedding-pca exp-rule-viz exp-weak-features exp-weak-features-beta exp-noisy-label-catboost exp-imbalanced-classification exp-imbalanced-classification-smoke exp-leaf-embedding-umap

# Run every experiment through `uv run` so it always uses the locked
# environment from uv.lock. Override with `PYTHON=` if you want to bypass.
PYTHON ?= uv run python

exp-roc-vs-ap:  ## Run both studies (full experiment)
	cd experiments/2026-06-06_roc_vs_ap && $(PYTHON) run_experiment.py

exp-roc-vs-ap-imbalance:  ## Run Study 1 only: class imbalance sweep
	cd experiments/2026-06-06_roc_vs_ap && $(PYTHON) run_study_imbalance.py

exp-roc-vs-ap-info:  ## Run Study 2 only: feature information sweep
	cd experiments/2026-06-06_roc_vs_ap && $(PYTHON) run_study_feature_info.py

exp-embedding-pca:  ## Run sentence embedding PCA reconstruction experiment
	cd experiments/2026-06-09_embedding_pca && $(PYTHON) run_experiment.py

exp-rule-viz:  ## Train interpretable models, extract rules, build interactive D3 HTML report
	cd experiments/2026-06-15_rule_viz && $(PYTHON) run_experiment.py

exp-weak-features:  ## Sweep training size with 100 weak features; compare CatBoost vs FIGS / Greedy / DecisionTree
	cd experiments/2026-06-18_weak_features_sample_size && $(PYTHON) run_experiment.py

exp-weak-features-beta:  ## Same as exp-weak-features but info_j ~ Beta(1,9) instead of constant 0.10
	cd experiments/2026-06-23_weak_features_beta && $(PYTHON) run_experiment.py

exp-leaf-embedding-ranking:  ## Leaf-embedding residual ranking: Ridge-on-leaves + ECDF rank spreading vs raw CatBoost probability
	cd experiments/2026-07-02_leaf_embedding_ranking && $(PYTHON) run_experiment.py
exp-noisy-label-catboost:  ## CatBoost CrossEntropy on noisy soft labels vs Logloss on hard labels, across noise levels
	cd experiments/2026-07-01_noisy_label_catboost && $(PYTHON) run_experiment.py

exp-imbalanced-classification:  ## Rare-positive (0.1%) model comparison: AP vs score entropy, priors, leaf/RFF UMAP
	cd experiments/2026-07-21_imbalanced_classification && uv run --extra catboost --extra umap --extra viz python run_experiment.py

exp-imbalanced-classification-smoke:  ## Fast smoke run of the imbalanced-classification experiment (CI/verification)
	cd experiments/2026-07-21_imbalanced_classification && uv run --extra catboost --extra umap --extra viz python run_experiment.py --smoke

exp-leaf-embedding-umap:  ## Leaf-index UMAP (Hamming) reduction vs native CatBoost: logit/SVM/RF/MLP/CatBoost across k, incl. training time
	cd experiments/2026-07-22_leaf_embedding_umap && uv run --extra catboost --extra umap --extra viz python run_experiment.py
