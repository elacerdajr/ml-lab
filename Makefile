.PHONY: exp-roc-vs-ap exp-roc-vs-ap-imbalance exp-roc-vs-ap-info

exp-roc-vs-ap:  ## Run both studies (full experiment)
	cd experiments/roc_vs_ap && python run_experiment.py

exp-roc-vs-ap-imbalance:  ## Run Study 1 only: class imbalance sweep
	cd experiments/roc_vs_ap && python run_study_imbalance.py

exp-roc-vs-ap-info:  ## Run Study 2 only: feature information sweep
	cd experiments/roc_vs_ap && python run_study_feature_info.py
