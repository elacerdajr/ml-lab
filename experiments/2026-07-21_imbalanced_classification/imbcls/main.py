"""
main.py
-------
Entry point: parse arguments, resolve the config profile, run the experiment,
write ``report.md`` and print the final research-question report.

Usage
-----
    python -m imbcls.main [--config PATH] [--profile default|full_spec] [--smoke]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ml_elements.run import Run, record

from .config import load_config
from .runner import Results, final_report, run

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # experiments/2026-07-21_imbalanced_classification/


def _img(title: str, path: Path, report_dir: Path, caption: str) -> list[str]:
    """Markdown block: a rendered image (linked relative to report.md) plus an italic caption.

    ``path`` is only embedded if the file exists, so a partial run (e.g. UMAP
    skipped without the extra) still produces a valid report.
    """
    if not path.exists():
        return []
    rel_path = Path(os.path.relpath(path, report_dir)).as_posix()
    return [f"![{title}]({rel_path})", "", f"*{caption}*", ""]


def _plots_section(out_dir: Path, report_dir: Path, model_order: list[str]) -> list[str]:
    """Assemble the plots section: paths + rendered images + interpretive captions."""
    plots = out_dir / "plots"
    lines: list[str] = ["## Plots", ""]

    # Core scatter / curve figures.
    lines += ["### AP vs score entropy — the core trade-off", ""]
    lines += _img(
        "AP vs entropy", plots / "ap_entropy/ap_vs_entropy.png", report_dir,
        "x = normalized score entropy, y = average precision, point size ∝ train time. "
        "Top-right (accurate *and* smooth) is ideal; the dotted line marks the "
        "high-entropy threshold (H ≥ 0.80). Tree models sit left (spiky scores), "
        "linear / balanced models sit right.",
    )

    lines += ["### Ranking-noise trade-off — headline plot", ""]
    lines += _img(
        "AP loss vs entropy gain", plots / "entropy_tradeoff/ap_loss_vs_entropy_gain.png", report_dir,
        "For the deterministic noise score r = αp + (1-α)u: x = entropy gain "
        "ΔH = H(r) − H(p), y = AP change ΔAP = AP(r) − AP(p). Points near the top-right "
        "buy large smoothness gains for negligible ranking loss; steep drops mean the "
        "noise is destroying signal.",
    )

    lines += ["### AP vs training time", ""]
    lines += _img(
        "AP vs time", plots / "ap_time/ap_vs_time.png", report_dir,
        "x = log(1 + train time [s]), y = AP, colour = normalized entropy — find the "
        "cheapest model at a given accuracy.",
    )

    lines += ["### Bucket lift by score decile", ""]
    lines += _img(
        "Bucket lift", plots / "bucket_lift/bucket_lift.png", report_dir,
        "Positive rate per score decile (log y) with the true base-rate line. A good "
        "ranker's top decile sits far above the base rate.",
    )

    lines += ["### Precision–recall curves", ""]
    lines += _img(
        "Precision-recall", plots / "precision_recall/precision_recall.png", report_dir,
        "PR curves (test, raw probability) — the right diagnostic under heavy imbalance, "
        "where ROC-AUC looks deceptively high.",
    )

    # Per-model score histograms (collapsed to keep the report scannable).
    hist_dir = out_dir / "plots" / "score_histograms"
    hist_blocks: list[str] = []
    for name in model_order:
        path = plots / "score_histograms" / f"score_hist_{name}.png"
        hist_blocks += _img(name, path, report_dir, f"`{name}` — raw vs shrinkage vs noise score.")
    if hist_blocks:
        lines += [
            "### Score distributions (per model)",
            "",
            "Raw probability vs post-hoc shrinkage vs deterministic-noise ranking score, "
            "validation and test, on a log y-axis so the rare high-score tail is visible.",
            "",
            "<details><summary>Show per-model score histograms</summary>",
            "",
            *hist_blocks,
            "</details>",
            "",
        ]
    elif hist_dir.exists():  # pragma: no cover
        pass

    # UMAP panels.
    umap_blocks: list[str] = []
    umap_blocks += _img(
        "UMAP raw features", plots / "umap/umap_raw_features.png", report_dir,
        "Raw one-hot + scaled features (euclidean). The model-agnostic view of class "
        "separability before any model — full sample (true base rate) vs undersample.",
    )
    umap_blocks += _img(
        "UMAP CatBoost leaf", plots / "umap/umap_catboost_leaf.png", report_dir,
        "CatBoost leaf-index one-hot (cosine) — the *supervised* tree view. The rare "
        "positive class forms much tighter, more separated structure than in raw space.",
    )
    umap_blocks += _img(
        "UMAP RFF features", plots / "umap/umap_rff_features.png", report_dir,
        "Random-Fourier feature space (euclidean) that the RFF+logistic model actually "
        "sees.",
    )
    if umap_blocks:
        lines += [
            "### UMAP representations (full sample vs undersampled)",
            "",
            "Each figure pairs the full-population view (positives are a sparse minority) "
            "with the 10%-positive undersample the model trains on.",
            "",
            *umap_blocks,
        ]

    return lines


def _write_report(results: Results, report_text: str, out_dir: Path, path: Path) -> None:
    s = results.summary
    raw_test = results.metrics[
        (results.metrics.prior_method == "none")
        & (results.metrics.score_transform == "raw")
        & (results.metrics.eval_split == "test")
    ].sort_values("average_precision", ascending=False)

    cols = ["model_name", "average_precision", "roc_auc", "normalized_score_entropy",
            "tie_rate", "train_time_seconds"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, r in raw_test[cols].iterrows():
        cells = [str(r["model_name"])] + [f"{r[c]:.4f}" for c in cols[1:]]
        body.append("| " + " | ".join(cells) + " |")
    table = "\n".join([header, sep, *body])
    model_order = raw_test["model_name"].tolist()

    lines = [
        "# Imbalanced Binary Classification — Report",
        "",
        "> Generated by `experiments/2026-07-21_imbalanced_classification/run_experiment.py`.",
        f"> Profile: **{s['profile']}**.",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| N_full | {s['full']['n']:,} |",
        f"| Full positive rate | {s['full']['positive_rate']:.5f} ({s['full']['n_positive']} positives) |",
        f"| pi (true train base rate) | {s['pi_true_train_base_rate']:.5f} |",
        f"| train_full / val_full / test_full | {s['train_full']['n']:,} / {s['val_full']['n']:,} / {s['test_full']['n']:,} |",
        f"| train_under | {s['train_under']['n']:,} (pos rate {s['train_under']['positive_rate']:.3f}) |",
        "",
        "Models train on the 10%-positive **undersample**; evaluation is on **val_full / "
        "test_full**, which preserve the real ~0.1% base rate (the deployment distribution).",
        "",
        "## Final report",
        "",
        "```",
        report_text,
        "```",
        "",
        "## Model leaderboard (prior=none, raw score, test split)",
        "",
        table,
        "",
        *_plots_section(out_dir, path.parent, model_order),
        "See `README.md` for how to interpret each metric and plot.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Imbalanced binary classification experiment")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, smoke=args.smoke)
    # Non-default profiles (smoke, full_spec) write to outputs_<profile>/ with their
    # own report, so they never overwrite the tracked record in outputs/ and report.md.
    experiment = Run(SCRIPT_DIR, profile=cfg.profile, seed=cfg.seed, config=cfg.raw)
    out_dir = experiment.output_dir
    report_path = SCRIPT_DIR / "report.md" if experiment.is_record else out_dir / "report.md"

    results = run(cfg, out_dir)
    report_text = final_report(results.metrics, results.pi)

    _write_report(results, report_text, out_dir, report_path)
    record(experiment)

    print("\n" + "=" * 70)
    print(report_text)
    print("=" * 70)


if __name__ == "__main__":
    main()
