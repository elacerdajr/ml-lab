"""
run_study_feature_info.py
-------------------------
Run Study 2 only: effect of feature information level (info-scale sweep).

Imports all shared logic from run_experiment.py so there is no duplication.

Usage
-----
    python run_study_feature_info.py
    python run_study_feature_info.py --config config.yaml \\
                                     --study-config study_feature_info.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))   # repo root  → ml_elements
sys.path.insert(0, str(SCRIPT_DIR))              # this dir   → run_experiment

# ── stub missing optional dep (must precede run_experiment import) ─────────────
_stub = types.ModuleType("bayesian_ap_comparator")
_stub.BayesianAPComparator = object
sys.modules.setdefault("bayesian_ap_comparator", _stub)

from run_experiment import (  # noqa: E402
    load_yaml,
    build_model_factory,
    build_conditions,
    run_study,
    build_summary,
    plot_metrics_vs_condition,
    plot_score_space,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study 2: feature information sweep.")
    p.add_argument("--config",       type=Path, default=SCRIPT_DIR / "config.yaml")
    p.add_argument("--study-config", type=Path, default=SCRIPT_DIR / "study_feature_info.yaml")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = load_yaml(args.config)
    s_cfg  = load_yaml(args.study_config)
    s      = s_cfg["study"]

    log.info("Config      : %s", args.config.name)
    log.info("Study config: %s", args.study_config.name)

    out_dir = SCRIPT_DIR / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Outputs     : %s", out_dir)

    model_cfgs      = cfg["models"]
    model_factories = {name: build_model_factory(mc) for name, mc in model_cfgs.items()}
    metric_names    = cfg["metrics"]
    sigma           = cfg["dgp"].get("sigma", 1.0)

    conditions = build_conditions(cfg, s_cfg)
    log.info("=== Study 2: %s (%d conditions) ===", s["label"], len(conditions))

    df = run_study(
        study_name=s["name"],
        condition_col=s["condition_col"],
        conditions=conditions,
        model_factories=model_factories,
        metric_names=metric_names,
        data_cfg=cfg["data"],
        sigma=sigma,
    )

    # ── metrics CSV ───────────────────────────────────────────────────────────
    csv_path = out_dir / f"metrics_{s['name']}.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s  (%d rows)", csv_path.name, len(df))

    # ── summary JSON ──────────────────────────────────────────────────────────
    summary = build_summary(df, s["condition_col"], metric_names)
    summary_dict = {
        "study":   s["name"],
        "config": {
            "n_train":   cfg["data"]["n_train"],
            "n_test":    cfg["data"]["n_test"],
            "n_repeats": cfg["data"]["n_repeats"],
            "metrics":   metric_names,
            "models":    list(model_cfgs.keys()),
        },
        "results": summary.to_dict(orient="records"),
    }
    json_path = out_dir / f"summary_{s['name']}.json"
    json_path.write_text(json.dumps(summary_dict, indent=2))
    log.info("Saved %s", json_path.name)

    # ── plots ─────────────────────────────────────────────────────────────────
    p_pos = s["fixed_p_pos"]

    plot_metrics_vs_condition(
        df=df, condition_col=s["condition_col"],
        metric_names=metric_names, model_cfgs=model_cfgs,
        title=(
            f"Study 2 — {s['label']}\n"
            f"p_pos={p_pos} · n_repeats={cfg['data']['n_repeats']}"
        ),
        xlabel=s["xlabel"],
        out_path=out_dir / "fig2_metrics_vs_info.png",
    )

    plot_score_space(
        df=df, condition_col=s["condition_col"], model_cfgs=model_cfgs,
        title=(
            "AUC vs AP Score Space — Feature Information\n"
            "(color = info scale · diamonds = per-condition centroids)"
        ),
        cbar_label="Info Scale",
        out_path=out_dir / "fig_score_space_info.png",
    )

    log.info("=== Done ===  outputs in %s", out_dir)


if __name__ == "__main__":
    main()
