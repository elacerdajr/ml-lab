.PHONY: exp-roc-vs-ap

exp-roc-vs-ap:  ## Run the ROC-AUC vs Average Precision experiment
	cd experiments/roc_vs_ap && python run_experiment.py
