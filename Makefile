.PHONY: exp-roc-vs-ap exp-roc-vs-ap-imbalance exp-roc-vs-ap-info exp-embedding-pca exp-rule-viz exp-weak-features

# Run every experiment through `uv run` so it always uses the locked
# environment from uv.lock. Override with `PYTHON=` if you want to bypass.
PYTHON ?= uv run python

exp-roc-vs-ap:  ## Run both studies (full experiment)
	cd experiments/roc_vs_ap && $(PYTHON) run_experiment.py

exp-roc-vs-ap-imbalance:  ## Run Study 1 only: class imbalance sweep
	cd experiments/roc_vs_ap && $(PYTHON) run_study_imbalance.py

exp-roc-vs-ap-info:  ## Run Study 2 only: feature information sweep
	cd experiments/roc_vs_ap && $(PYTHON) run_study_feature_info.py

exp-embedding-pca:  ## Run sentence embedding PCA reconstruction experiment
	cd experiments/embedding_pca && $(PYTHON) run_experiment.py

exp-rule-viz:  ## Train interpretable models, extract rules, build interactive D3 HTML report
	cd experiments/rule_viz && $(PYTHON) run_experiment.py

exp-weak-features:  ## Sweep training size with 100 weak features; compare CatBoost vs FIGS / Greedy / DecisionTree
	cd experiments/weak_features_sample_size && $(PYTHON) run_experiment.py
