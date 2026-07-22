"""
experiments/2026-06-15_rule_viz/run_experiment.py
---------------------------------------
Train interpretable models on a synthetic credit-fraud dataset, extract their
rules into a canonical JSON schema, and render a self-contained interactive
HTML report using D3.js.

The data-generating process has named, meaningful features so the extracted
rules read like real business logic rather than raw array indices.

DGP
---
Features: age, income_k, credit_score, n_transactions, is_mobile (binary)
Target:   fraud (binary)
True rule (approximate):
  logit = -0.03·age − 0.01·income_k − 0.003·credit_score
          + 0.12·n_transactions + 0.9·is_mobile + 1.5

Models
------
DecisionTree    max_depth=4   sklearn interpretable baseline
RuleFit         imodels       sparse linear rule ensemble
GreedyRuleList  imodels       ordered greedy rule list

Outputs
-------
outputs/rules_<model>.json    per-model rule schemas
outputs/rule_report.html      interactive D3 rule matrix (open in browser)
README.md                     description + embedded screenshot hint

Usage
-----
    python run_experiment.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml_elements.rules import RuleExtractor
from ml_elements.rule_viz import save_rule_report

try:
    import imodels
except ImportError as exc:
    raise ImportError("pip install imodels") from exc

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 42
N_TRAIN = 2_000
N_TEST = 10_000

FEATURE_NAMES = ["age", "income_k", "credit_score", "n_transactions", "is_mobile"]


# ─── DGP ────────────────────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_credit_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic credit-fraud dataset with human-readable features.

    True signal (approximate):
      younger + lower income + lower credit score + more transactions + mobile
      → higher fraud probability
    """
    rng = np.random.default_rng(seed)

    age = rng.uniform(18.0, 75.0, n)
    income_k = np.exp(rng.normal(3.8, 0.8, n))          # log-normal ≈ 45k
    credit_score = rng.normal(680.0, 90.0, n).clip(300, 850)
    n_transactions = rng.poisson(8, n).astype(float)
    is_mobile = rng.binomial(1, 0.55, n).astype(float)

    logit = (
        -0.030 * age
        - 0.008 * income_k
        - 0.003 * credit_score
        + 0.120 * n_transactions
        + 0.900 * is_mobile
        - 1.0   # intercept → ~25% base fraud rate
    )
    fraud = rng.binomial(1, _sigmoid(logit))
    X = np.column_stack([age, income_k, credit_score, n_transactions, is_mobile])
    return X, fraud.astype(int)


# ─── Models ─────────────────────────────────────────────────────────────────


def build_models() -> dict:
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, random_state=SEED
        ),
        "RuleFit": imodels.RuleFitClassifier(
            n_estimators=40, tree_size=4, max_rules=40, random_state=SEED
        ),
        "GreedyRuleList": imodels.GreedyRuleListClassifier(max_depth=8),
    }


# ─── Experiment ─────────────────────────────────────────────────────────────


def run() -> None:
    log.info("=== Generating data ===")
    X_tr, y_tr = make_credit_data(N_TRAIN, seed=SEED)
    X_te, y_te = make_credit_data(N_TEST, seed=SEED + 1)
    log.info("  train=%d  test=%d  fraud_rate_train=%.3f  fraud_rate_test=%.3f",
             N_TRAIN, N_TEST, y_tr.mean(), y_te.mean())

    extractor = RuleExtractor(max_rules=40)
    rulesets = []
    models_info = []

    for model_name, model in build_models().items():
        log.info("=== %s ===", model_name)
        if "imodels" in type(model).__module__:
            model.fit(X_tr, y_tr, feature_names=FEATURE_NAMES)
        else:
            model.fit(X_tr, y_tr)

        try:
            auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        except Exception:
            auc = float("nan")
        log.info("  test AUC = %.4f", auc)

        log.info("  Extracting rules ...")
        rs = extractor.from_model(
            model, X_tr, y_tr, FEATURE_NAMES, model_name=model_name
        )
        log.info("  %d rules extracted", len(rs.rules))

        # Save per-model JSON
        json_path = OUT_DIR / f"rules_{model_name.lower().replace(' ', '_')}.json"
        rs.save(json_path)
        log.info("  Saved %s", json_path.name)

        rulesets.append(rs)
        models_info.append({
            "model": model_name,
            "test_auc": round(auc, 4),
            "n_rules": len(rs.rules),
        })

    # ── HTML report ──────────────────────────────────────────────────────────
    log.info("=== Building HTML report ===")
    html_path = save_rule_report(
        rulesets,
        OUT_DIR / "rule_report.html",
        title="Credit Fraud — Rule Explorer",
    )
    log.info("  Saved %s  (%d bytes)", html_path.name, html_path.stat().st_size)

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(models_info, indent=2), encoding="utf-8")
    log.info("  Saved summary.json")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Model':<18} {'Test AUC':>10} {'N Rules':>9}")
    print("-" * 55)
    for info in models_info:
        print(f"{info['model']:<18} {info['test_auc']:>10.4f} {info['n_rules']:>9}")
    print("=" * 55)
    print(f"\nHTML report: {html_path}")

    # ── README ───────────────────────────────────────────────────────────────
    _write_readme(models_info)
    log.info("  Saved README.md")

    log.info("=== Done ===")


# ─── README ─────────────────────────────────────────────────────────────────


def _write_readme(models_info: list[dict]) -> None:
    rows = "\n".join(
        f"| {m['model']} | {m['test_auc']:.4f} | {m['n_rules']} |"
        for m in models_info
    )
    readme = f"""\
# Rule Visualization — Interactive D3 Explorer

> Generated by `experiments/2026-06-15_rule_viz/run_experiment.py`

---

## What this experiment does

Three interpretable models are trained on a synthetic **credit-fraud** dataset
with human-readable features:

| Feature | Description |
| --- | --- |
| `age` | Customer age (18–75) |
| `income_k` | Annual income in thousands (log-normal) |
| `credit_score` | Credit score 300–850 |
| `n_transactions` | Transactions in the last month |
| `is_mobile` | 1 if mobile device, 0 otherwise |

**True DGP (approximate):**
```
logit = −0.03·age − 0.008·income_k − 0.003·credit_score
        + 0.12·n_transactions + 0.9·is_mobile + 4.0
```
→ Younger, lower-income, lower-credit, high-transaction, mobile users
  are more likely to be fraud.

---

## Models

| Model | Test AUC | Rules extracted |
| --- | --- | --- |
{rows}

Each model's rules are exported to a canonical JSON schema:

```json
{{
  "rule_id": 3,
  "conditions": [
    {{"feature": "n_transactions", "op": ">",  "value": 14.5}},
    {{"feature": "credit_score",   "op": "<=", "value": 612.0}}
  ],
  "prediction": 0.72,
  "support":    148,
  "precision":  0.72,
  "importance": 0.85,
  "description": "n_transactions > 14.5 AND credit_score <= 612.0"
}}
```

---

## Interactive report

Open **[outputs/rule_report.html](outputs/rule_report.html)** in any browser.

### Features

- **Model tabs** — switch between DecisionTree, RuleFit, GreedyRuleList
- **Rule matrix** — rows = rules, columns = features used by that model
  - Condition cells are colour-coded by precision (red → yellow → green)
  - Dash cells mean the rule does not mention that feature
- **Sortable columns** — click Precision / Support / Importance header
- **Filter sliders** — hide low-confidence or low-coverage rules
- **Hover tooltip** — full rule description and all metrics

### Reading the matrix

```
Rule   age     income_k   credit_score   n_trans   is_mobile   Precision  Support  Importance
R1      —         —          ≤ 612.0     > 14.5       —          72.0%      148      100%
R2      —      ≤ 18.3        ≤ 591.0       —          —          68.3%       89       74%
R3    ≤ 28.4     —             —          > 9.5        1          65.1%      203       61%
```

A ✓ (coloured condition) means the rule *requires* that feature condition.
A — means the rule does not restrict that feature.

---

## Architecture

```
ml_elements/
    rules.py          # RuleCondition, Rule, RuleSet, RuleExtractor
    rule_viz.py       # rule_matrix_html(), save_rule_report()

experiments/2026-06-15_rule_viz/
    run_experiment.py
    outputs/
        rules_decisiontree.json
        rules_rulefit.json
        rules_greedyrulelist.json
        rule_report.html       ← open this in a browser
        summary.json
```

---

*Raw rules: `outputs/rules_*.json` — machine-readable, model-independent.*
*Run:* `make exp-rule-viz`
"""
    (SCRIPT_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    run()
