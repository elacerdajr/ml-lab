"""
runner.py
---------
Orchestration: generate data, split, undersample, search the RFF grid, fit every
model under each applicable prior, evaluate all score transforms on val/test,
build the metric / bucket tables, render the plots and UMAPs, and print the
final report answering the experiment's research questions.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import data as datamod
from . import embeddings as emb
from . import plots, umap_viz
from .config import Config
from .data import FEATURES
from .metrics import (
    compute_bucket_metrics,
    compute_score_entropy,
    eval_row,
    stratified_ratio_table,
)
from .models import (
    build_preprocessor,
    build_registry,
    fit_hard,
    fit_soft,
    make_rff_spec,
    predict_scores,
)
from .priors import label_smoothing, make_synthetic_prior_points
from .scoring import apply_transforms

warnings.filterwarnings("ignore")


# ── lightweight logger (RichLogger if available) ─────────────────────────────

def _make_logger():
    try:
        from ml_elements.rich_logger import RichLogger

        return RichLogger("imbalanced_classification")
    except Exception:  # pragma: no cover
        class _Fallback:
            def section(self, m):
                print(f"\n== {m} ==")

            def info(self, m):
                print(m)

            ok = saved = warn = done = info

        return _Fallback()


@dataclass
class Results:
    metrics: pd.DataFrame
    buckets: pd.DataFrame
    summary: dict
    model_configs: dict
    pi: float
    gp_ratios: pd.DataFrame | None = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _transform_meta(name: str) -> tuple[str, float]:
    """Map a transform key like ``shrink@0.1`` / ``noise@0.95`` / ``raw`` to (base, param)."""
    if name == "raw":
        return "raw", float("nan")
    base, val = name.split("@")
    mapping = {"shrink": "shrinkage", "noise": "noise_rank"}
    return mapping[base], float(val)


def _gp_subsample(train_under: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    """Stratified subsample for the Gaussian process (keep all positives)."""
    if len(train_under) <= max_n:
        return train_under
    rng = np.random.default_rng(seed)
    pos = train_under[train_under["y"] == 1]
    neg = train_under[train_under["y"] == 0]
    n_neg = max(0, max_n - len(pos))
    neg_idx = rng.choice(len(neg), size=min(n_neg, len(neg)), replace=False)
    return pd.concat([pos, neg.iloc[neg_idx]], ignore_index=True)


def _fit_tasks(spec, train_under, train_full, pi, cfg):
    """
    Yield ``(prior_method, prior_lambda, rho, X, target, is_soft)`` for one model.
    Prior 1 always; priors 2 & 3 only for soft-capable models.
    """
    X_under = train_under[FEATURES]
    y_under = train_under["y"].to_numpy()
    yield ("none", float("nan"), float("nan"), X_under, y_under, False)

    if not spec.supports_soft:
        return

    for lam in cfg.priors.label_smoothing_lambdas:
        yield (
            "label_smoothing", lam, float("nan"),
            X_under, label_smoothing(y_under, pi, lam), True,
        )

    for rho in cfg.priors.synthetic_rhos:
        m = int(rho * len(train_under))
        if m == 0:
            continue
        synth = make_synthetic_prior_points(train_full, m, pi, cfg, cfg.seed + int(1000 * rho))
        X_aug = pd.concat([X_under, synth[FEATURES]], ignore_index=True)
        y_aug = np.concatenate([y_under.astype(float), synth["y"].to_numpy(dtype=float)])
        yield ("synthetic_soft", float("nan"), rho, X_aug, y_aug, True)


def _search_rff(pre, cfg, train_under, val_full, log):
    """Grid-search RFF (gamma × n_components) under prior 1; pick best val AP."""
    from ml_elements.metrics import AVG_PRECISION

    X_under, y_under = train_under[FEATURES], train_under["y"].to_numpy()
    y_val = val_full["y"].to_numpy()
    best = None
    for gamma in cfg.models["rff"]["gammas"]:
        for ncomp in cfg.models["rff"]["n_components"]:
            spec = make_rff_spec(pre, cfg, gamma, ncomp)
            model, _ = fit_hard(spec, X_under, y_under)
            p_val, _ = predict_scores(model, val_full[FEATURES], False)
            ap = float(AVG_PRECISION.fn(y_val, p_val))
            log.info(f"  RFF gamma={gamma} n_components={ncomp}: val AP={ap:.4f}")
            if best is None or ap > best[0]:
                best = (ap, gamma, ncomp)
    log.info(f"  RFF best: gamma={best[1]} n_components={best[2]} (val AP={best[0]:.4f})")
    return make_rff_spec(pre, cfg, best[1], best[2]), {"gamma": best[1], "n_components": best[2]}


# ── main run ─────────────────────────────────────────────────────────────────

def run(cfg: Config, out_dir: Path, log=None) -> Results:
    log = log or _make_logger()
    rng_seed = cfg.seed
    plots_dir = out_dir / "plots"

    # 1. Data ----------------------------------------------------------------
    log.section("Generating data")
    df = datamod.generate_full(cfg)
    splits = datamod.split_dataset(df, cfg)
    train_full, val_full, test_full = splits["train_full"], splits["val_full"], splits["test_full"]
    pi = datamod.compute_pi(train_full)
    train_under = datamod.make_undersampled_train(train_full, cfg)
    log.info(
        f"  n_full={len(df):,}  pi(true train base rate)={pi:.5f}  "
        f"train_under={len(train_under):,} (pos rate {train_under['y'].mean():.3f})"
    )
    assert abs(pi - cfg.data.positive_rate) < 5e-4, "pi should be ≈ target positive rate"
    assert int(train_full["y"].sum()) == int(train_under["y"].sum()), "all train positives kept"

    summary = datamod.data_summary(df, splits, train_under, pi, cfg)
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))

    # 2. Preprocessor + registry --------------------------------------------
    pre = build_preprocessor(train_full)
    pre_emb = clone(pre).fit(train_full[FEATURES])
    registry = build_registry(pre, cfg, len(train_under))

    log.section("Searching RFF hyperparameters")
    rff_spec, rff_choice = _search_rff(pre, cfg, train_under, val_full, log)
    registry["rff_logistic"] = rff_spec

    model_configs = {name: dict(cfg.models.get(name, {})) for name in registry}
    model_configs["rff_logistic"] = {**cfg.models["rff"], "selected": rff_choice}
    (out_dir / "model_configs.json").write_text(json.dumps(model_configs, indent=2, default=str))

    # 3. Fit + evaluate ------------------------------------------------------
    eval_splits = {"val": val_full, "test": test_full}
    metric_rows: list[dict] = []
    bucket_rows: list[pd.DataFrame] = []
    kept: dict[str, dict] = {}   # per-model artifacts for plots/UMAP
    run_id = 0

    for name, spec in registry.items():
        log.section(f"Model: {name}")
        calib = "sigmoid" if name in ("linear_svm", "rbf_svm") else "none"

        for prior_method, lam, rho, X_fit, target, is_soft in _fit_tasks(
            spec, train_under, train_full, pi, cfg
        ):
            # GP guard: subsample training data.
            if name == "gaussian_process":
                sub = _gp_subsample(train_under, spec.max_train_n, rng_seed)
                X_fit, target = sub[FEATURES], sub["y"].to_numpy()

            try:
                if is_soft:
                    model, t_train = fit_soft(spec, X_fit, target)
                else:
                    model, t_train = fit_hard(spec, X_fit, target)
            except Exception as exc:  # record failure, keep going
                log.warn(f"  {prior_method}: fit failed ({exc})")
                continue

            tag = prior_method + (f"@{lam:g}" if not np.isnan(lam) else "") \
                + (f"@rho{rho:g}" if not np.isnan(rho) else "")
            log.info(f"  fit [{tag}] in {t_train:.2f}s")

            for split_name, split_df in eval_splits.items():
                p, t_pred = predict_scores(model, split_df[FEATURES], spec.is_catboost)
                y_eval = split_df["y"].to_numpy()
                row_ids = split_df["row_id"].to_numpy()
                transforms = apply_transforms(p, row_ids, pi, cfg, rng_seed)

                for tname, (score, is_prob) in transforms.items():
                    base, param = _transform_meta(tname)
                    keys = {
                        "run_id": run_id,
                        "model_name": name,
                        "prior_method": prior_method,
                        "prior_lambda": lam,
                        "synthetic_prior_rho": rho,
                        "calibration_method": calib,
                        "score_transform": base,
                        "score_alpha": param,
                        "eval_split": split_name,
                    }
                    timings = {"train_time_seconds": t_train, "predict_time_seconds": t_pred}
                    counts = {
                        "n_train": int(len(X_fit)),
                        "n_eval": int(len(split_df)),
                        "positive_rate_train": float(np.mean(target) if is_soft else target.mean()),
                        "positive_rate_eval": float(y_eval.mean()),
                    }
                    metric_rows.append(eval_row(y_eval, score, is_prob, keys, timings, counts))
                    run_id += 1

            # Keep prior-none artifacts for plots / UMAP / buckets.
            if prior_method == "none":
                p_val, _ = predict_scores(model, val_full[FEATURES], spec.is_catboost)
                p_test, _ = predict_scores(model, test_full[FEATURES], spec.is_catboost)
                kept[name] = {"model": model, "spec": spec, "p_val": p_val, "p_test": p_test}

    metrics_df = pd.DataFrame(metric_rows)

    # 4. Bucket metrics (prior none; raw + noise) ---------------------------
    log.section("Bucket metrics")
    mid_alpha = cfg.priors.noise_alphas[len(cfg.priors.noise_alphas) // 2]
    y_test = test_full["y"].to_numpy()
    base_rate = float(y_test.mean())
    for name, art in kept.items():
        transforms = apply_transforms(art["p_test"], test_full["row_id"].to_numpy(), pi, cfg, rng_seed)
        for tname in ("raw", f"noise@{mid_alpha:g}"):
            base, _ = _transform_meta(tname)
            score = transforms[tname][0]
            for nb in cfg.output.bucket_sizes:
                bdf = compute_bucket_metrics(y_test, score, nb, base_rate)
                bdf.insert(0, "score_transform", base)
                bdf.insert(0, "model_name", name)
                bucket_rows.append(bdf)
    buckets_df = pd.concat(bucket_rows, ignore_index=True)

    # 4b. Stratified ratio tables (every model's metrics / Gaussian process's) ---
    ratio_frames = []
    for split_name in ("val", "test"):
        rt = stratified_ratio_table(metrics_df, baseline_model="gaussian_process", split=split_name)
        if not rt.empty:
            rt.insert(0, "eval_split", split_name)
            ratio_frames.append(rt)
    ratio_df = pd.concat(ratio_frames, ignore_index=True) if ratio_frames else pd.DataFrame()

    # 5. Save tables ---------------------------------------------------------
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    buckets_df.to_csv(out_dir / "bucket_metrics.csv", index=False)
    if not ratio_df.empty:
        ratio_df.to_csv(out_dir / "gp_ratio_metrics.csv", index=False)
    log.saved(
        f"metrics.csv ({len(metrics_df)} rows), bucket_metrics.csv ({len(buckets_df)} rows), "
        f"gp_ratio_metrics.csv ({len(ratio_df)} rows)"
    )

    # 6. Plots ---------------------------------------------------------------
    log.section("Plots")
    _make_plots(cfg, metrics_df, buckets_df, kept, val_full, test_full, pi, base_rate, plots_dir, log)

    # 7. UMAP ----------------------------------------------------------------
    if umap_viz.HAS_UMAP:
        log.section("UMAP")
        _make_umaps(cfg, df, train_under, val_full, pre_emb, kept, plots_dir / "umap", log)
    else:
        log.warn("umap-learn not installed — skipping UMAP panels")

    # 8. Artifacts -----------------------------------------------------------
    _save_artifacts(cfg, pre_emb, kept, out_dir / "artifacts", log)

    return Results(metrics_df, buckets_df, summary, model_configs, pi, ratio_df)


# ── plotting orchestration ───────────────────────────────────────────────────

def _make_plots(cfg, metrics_df, buckets_df, kept, val_full, test_full, pi, base_rate, plots_dir, log):
    mid_lam = cfg.priors.shrinkage_lambdas[len(cfg.priors.shrinkage_lambdas) // 2]
    mid_alpha = cfg.priors.noise_alphas[len(cfg.priors.noise_alphas) // 2]

    # A. Score histograms (per model)
    for name, art in kept.items():
        tv = apply_transforms(art["p_val"], val_full["row_id"].to_numpy(), pi, cfg, cfg.seed)
        tt = apply_transforms(art["p_test"], test_full["row_id"].to_numpy(), pi, cfg, cfg.seed)
        vv = {"raw": tv["raw"][0], "shrink": tv[f"shrink@{mid_lam:g}"][0], "noise": tv[f"noise@{mid_alpha:g}"][0]}
        vt = {"raw": tt["raw"][0], "shrink": tt[f"shrink@{mid_lam:g}"][0], "noise": tt[f"noise@{mid_alpha:g}"][0]}
        plots.plot_score_histograms(name, vv, vt, plots_dir / "score_histograms")

    # B. Precision–recall
    y_test = test_full["y"].to_numpy()
    plots.plot_precision_recall(
        [(n, y_test, a["p_test"]) for n, a in kept.items()], plots_dir / "precision_recall"
    )

    # C/D. AP vs entropy / time — prior none, raw, test
    raw_test = metrics_df[
        (metrics_df.prior_method == "none")
        & (metrics_df.score_transform == "raw")
        & (metrics_df.eval_split == "test")
    ].copy()
    plots.plot_ap_entropy(raw_test, plots_dir / "ap_entropy")
    plots.plot_ap_time(raw_test, plots_dir / "ap_time")

    # E. Entropy trade-off (noise rows vs raw, prior none, test)
    trade = []
    for name, art in kept.items():
        tt = apply_transforms(art["p_test"], test_full["row_id"].to_numpy(), pi, cfg, cfg.seed)
        from ml_elements.metrics import AVG_PRECISION

        ap_raw = float(AVG_PRECISION.fn(y_test, tt["raw"][0]))
        h_raw = compute_score_entropy(tt["raw"][0])[1]
        for alpha in cfg.priors.noise_alphas:
            sc = tt[f"noise@{alpha:g}"][0]
            trade.append(
                {
                    "model_name": name,
                    "alpha": alpha,
                    "ap_delta": float(AVG_PRECISION.fn(y_test, sc)) - ap_raw,
                    "entropy_gain": compute_score_entropy(sc)[1] - h_raw,
                }
            )
    plots.plot_entropy_tradeoff(pd.DataFrame(trade), plots_dir / "entropy_tradeoff")

    # F. Bucket lift (decile, raw)
    dec = buckets_df[(buckets_df.n_buckets == 10) & (buckets_df.score_transform == "raw")]
    plots.plot_bucket_lift(dec, base_rate, plots_dir / "bucket_lift")
    log.saved("plots A–F")


# ── UMAP orchestration ───────────────────────────────────────────────────────

def _make_umaps(cfg, df, train_under, val_full, pre_emb, kept, umap_dir, log):
    n = cfg.umap.n_umap
    full_s = datamod.sample_for_umap(df, n, cfg.seed)
    under_s = datamod.sample_for_umap(train_under, n, cfg.seed + 1)

    # 1–2 raw features
    umap_viz.compare_full_vs_undersampled_umap(
        "raw_features", "euclidean",
        emb.get_raw_preprocessed_embedding(pre_emb, full_s), full_s["y"].to_numpy(),
        emb.get_raw_preprocessed_embedding(pre_emb, under_s), under_s["y"].to_numpy(),
        cfg, umap_dir,
    )
    log.saved("umap_raw_features.png")

    # 3–4 CatBoost leaf embedding (aggressive)
    if "catboost_aggressive" in kept:
        cb = kept["catboost_aggressive"]["model"]
        umap_viz.compare_full_vs_undersampled_umap(
            "catboost_leaf", "cosine",
            emb.get_catboost_leaf_embedding(cb, full_s), full_s["y"].to_numpy(),
            emb.get_catboost_leaf_embedding(cb, under_s), under_s["y"].to_numpy(),
            cfg, umap_dir,
        )
        log.saved("umap_catboost_leaf.png")

    # 5–6 RFF features
    if "rff_logistic" in kept:
        rff = kept["rff_logistic"]["model"]
        umap_viz.compare_full_vs_undersampled_umap(
            "rff_features", "euclidean",
            emb.get_rff_embedding(rff, full_s), full_s["y"].to_numpy(),
            emb.get_rff_embedding(rff, under_s), under_s["y"].to_numpy(),
            cfg, umap_dir,
        )
        log.saved("umap_rff_features.png")


# ── artifacts ────────────────────────────────────────────────────────────────

def _save_artifacts(cfg, pre_emb, kept, art_dir, log):
    try:
        import joblib
    except Exception:
        log.warn("joblib not installed — skipping artifact dumps")
        return
    if cfg.output.save_preprocessors:
        joblib.dump(pre_emb, art_dir / "fitted_preprocessors" / "preprocessor.joblib")
    if cfg.output.save_linear_models:
        for name in ("logistic", "rff_logistic", "linear_svm"):
            if name in kept:
                joblib.dump(kept[name]["model"], art_dir / "fitted_models" / f"{name}.joblib")
    if cfg.output.save_catboost_models:
        for name in ("catboost_conservative", "catboost_aggressive"):
            if name in kept:
                kept[name]["model"].save_model(str(art_dir / "fitted_models" / f"{name}.cbm"))
    log.saved("artifacts")


# ── final report ─────────────────────────────────────────────────────────────

def final_report(metrics_df: pd.DataFrame, pi: float) -> str:
    """Answer the research questions from the test-split metrics."""
    raw = metrics_df[
        (metrics_df.prior_method == "none")
        & (metrics_df.score_transform == "raw")
        & (metrics_df.eval_split == "test")
    ].copy()
    if raw.empty:
        return "No results."

    best_ap = raw["average_precision"].max()
    best_ap_model = raw.loc[raw["average_precision"].idxmax(), "model_name"]
    best_ent_model = raw.loc[raw["normalized_score_entropy"].idxmax(), "model_name"]
    best_ent = raw["normalized_score_entropy"].max()

    high_ent = raw[raw["normalized_score_entropy"] >= 0.80]
    if not high_ent.empty:
        he = high_ent.loc[high_ent["average_precision"].idxmax()]
        best_ap_high_ent = f"{he['model_name']} (AP={he['average_precision']:.4f}, H={he['normalized_score_entropy']:.3f})"
    else:
        best_ap_high_ent = "none reach H≥0.80"

    good = raw[raw["average_precision"] >= 0.95 * best_ap]
    fastest = good.loc[good["train_time_seconds"].idxmin()]

    # noise trade-off: best entropy gain with minimal AP loss
    noise = metrics_df[
        (metrics_df.prior_method == "none")
        & (metrics_df.score_transform == "noise_rank")
        & (metrics_df.eval_split == "test")
    ]
    tradeoff_line = "n/a"
    if not noise.empty:
        base = raw.set_index("model_name")["average_precision"]
        base_h = raw.set_index("model_name")["normalized_score_entropy"]
        nz = noise.copy()
        nz["ap_delta"] = nz.apply(lambda r: r["average_precision"] - base.get(r["model_name"], np.nan), axis=1)
        nz["h_gain"] = nz.apply(lambda r: r["normalized_score_entropy"] - base_h.get(r["model_name"], np.nan), axis=1)
        nz["eff"] = nz["h_gain"] + nz["ap_delta"]  # reward entropy gain, penalise AP loss
        b = nz.loc[nz["eff"].idxmax()]
        tradeoff_line = (
            f"{b['model_name']} @α={b['score_alpha']:g} "
            f"(ΔH=+{b['h_gain']:.3f}, ΔAP={b['ap_delta']:+.4f})"
        )

    # Pareto production candidate
    cand = raw.copy()
    cand["score"] = (
        cand["average_precision"] / best_ap
        + cand["normalized_score_entropy"]
        - cand["tie_rate"]
    )
    prod = cand.loc[cand["score"].idxmax()]

    lines = [
        f"Best model by AP:                    {best_ap_model} (AP={best_ap:.4f})",
        f"Best model by normalized entropy:    {best_ent_model} (H={best_ent:.3f})",
        f"Best AP among high-entropy models:   {best_ap_high_ent}",
        f"Fastest good model (AP≥0.95·best):   {fastest['model_name']} "
        f"({fastest['train_time_seconds']:.2f}s, AP={fastest['average_precision']:.4f})",
        f"Best AP-loss / entropy-gain trade:   {tradeoff_line}",
        f"Best candidate for production:       {prod['model_name']} "
        f"(AP={prod['average_precision']:.4f}, H={prod['normalized_score_entropy']:.3f}, "
        f"tie={prod['tie_rate']:.3f})",
    ]
    return "\n".join(lines)
