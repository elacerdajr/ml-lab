"""
runner.py
---------
Orchestration: generate data (same DGP as the sibling experiments), undersample,
fit the CatBoostEncoder-based preprocessor once, fan it out to five models,
separately fit CatBoost's native-categorical path, and answer "best model on
CatBoost-encoded features" and "does catboost_native beat catboost_encoded?".
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as datamod
from . import plots
from .config import Config
from .encoders import CAT_IDX, fit_preprocessor_timed, to_catboost_native_X, transform_timed
from .metrics import eval_row
from .models import (
    ENCODED_MODEL_NAMES,
    build_catboost_native,
    build_registry,
    fit_timed,
    predict_timed,
)

warnings.filterwarnings("ignore")


def _make_logger():
    try:
        from ml_elements.rich_logger import RichLogger

        return RichLogger("catboost_encoder_models")
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
    summary: dict
    model_configs: dict


def run(cfg: Config, out_dir: Path, log=None) -> Results:
    log = log or _make_logger()
    plots_dir = out_dir / "plots"

    # 1. Data --------------------------------------------------------------
    log.section("Generating data")
    df = datamod.generate_full(cfg)
    splits = datamod.split_dataset(df, cfg)
    train_full, val_full, test_full = splits["train_full"], splits["val_full"], splits["test_full"]
    pi = datamod.compute_pi(train_full)
    train_under = datamod.make_undersampled_train(train_full, cfg)
    log.info(
        f"  n_full={len(df):,}  pi={pi:.5f}  train_under={len(train_under):,} "
        f"(pos rate {train_under['y'].mean():.3f})"
    )
    assert abs(pi - cfg.data.positive_rate) < 5e-4
    assert int(train_full["y"].sum()) == int(train_under["y"].sum())

    summary = datamod.data_summary(df, splits, train_under, pi, cfg)
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))

    y_train = train_under["y"].to_numpy()
    eval_splits = {"val": val_full, "test": test_full}

    registry = build_registry(cfg)
    model_configs = {
        "logistic": dict(cfg.models["logistic"]),
        "rbf_svm": dict(cfg.models["rbf_svm"]),
        "random_forest": dict(cfg.models["random_forest"]),
        "mlp": dict(cfg.models["mlp"]),
        "catboost_encoded": dict(cfg.models["catboost"]),
        "catboost_native": dict(cfg.models["catboost"]),
        "encoder": {"sigma": cfg.encoder.sigma, "a": cfg.encoder.a},
    }
    (out_dir / "model_configs.json").write_text(json.dumps(model_configs, indent=2, default=str))

    metric_rows: list[dict] = []
    run_id = 0

    # 2. Fit the CatBoostEncoder preprocessor once, fan out to 5 models -----
    log.section("Fitting CatBoostEncoder preprocessor")
    pre, X_train_enc, fit_time = fit_preprocessor_timed(cfg, train_under)
    log.info(f"  fit_time={fit_time:.3f}s  n_features_out={X_train_enc.shape[1]}")

    X_eval_enc: dict[str, tuple[np.ndarray, float]] = {}
    for split_name, split_df in eval_splits.items():
        X_enc, t_transform = transform_timed(pre, split_df)
        X_eval_enc[split_name] = (X_enc, t_transform)

    kept: dict[str, dict] = {}
    n_under = len(train_under)

    for model_name in ENCODED_MODEL_NAMES:
        spec = registry[model_name]
        if spec.max_train_n is not None and n_under > spec.max_train_n:
            log.warn(f"  {model_name}: skipped (train_under {n_under} > max_train_n {spec.max_train_n})")
            continue

        log.section(f"Model: {model_name}")
        try:
            model, t_train = fit_timed(spec, X_train_enc, y_train)
        except Exception as exc:
            log.warn(f"  fit failed ({exc})")
            continue
        log.info(f"  fit in {t_train:.3f}s")

        p_test_for_plot = None
        for split_name, split_df in eval_splits.items():
            X_enc, t_transform = X_eval_enc[split_name]
            p, t_pred = predict_timed(model, X_enc)
            y_eval = split_df["y"].to_numpy()
            if split_name == "test":
                p_test_for_plot = p

            keys = {"run_id": run_id, "model_name": model_name, "eval_split": split_name}
            timings = {
                "train_time_seconds": t_train,
                "predict_time_seconds": t_pred,
                "encode_fit_time_seconds": fit_time,
                "encode_transform_time_seconds": t_transform,
            }
            counts = {
                "n_train": int(len(X_train_enc)),
                "n_eval": int(len(split_df)),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_eval": float(y_eval.mean()),
            }
            metric_rows.append(eval_row(y_eval, p, keys, timings, counts))
            run_id += 1

        kept[model_name] = {"p_test": p_test_for_plot}

    # 3. CatBoost native path (encoder-independent) -------------------------
    log.section("CatBoost native categorical handling")
    native_model = build_catboost_native(cfg)
    t0 = time.perf_counter()
    native_model.fit(to_catboost_native_X(train_under), y_train, cat_features=CAT_IDX)
    t_train = time.perf_counter() - t0
    log.info(f"  fit in {t_train:.3f}s")

    p_test_native = None
    for split_name, split_df in eval_splits.items():
        Xn = to_catboost_native_X(split_df)
        p, t_pred = predict_timed(native_model, Xn)
        y_eval = split_df["y"].to_numpy()
        if split_name == "test":
            p_test_native = p

        keys = {"run_id": run_id, "model_name": "catboost_native", "eval_split": split_name}
        timings = {
            "train_time_seconds": t_train,
            "predict_time_seconds": t_pred,
            "encode_fit_time_seconds": np.nan,
            "encode_transform_time_seconds": np.nan,
        }
        counts = {
            "n_train": int(len(train_under)),
            "n_eval": int(len(split_df)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_eval": float(y_eval.mean()),
        }
        metric_rows.append(eval_row(y_eval, p, keys, timings, counts))
        run_id += 1

    kept["catboost_native"] = {"p_test": p_test_native}

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    log.saved(f"metrics.csv ({len(metrics_df)} rows)")

    # 4. Plots ---------------------------------------------------------------
    log.section("Plots")
    test_df = metrics_df[metrics_df.eval_split == "test"]
    plots.plot_ap_by_model(test_df, plots_dir / "ap_by_model")
    y_test = test_full["y"].to_numpy()
    pr_entries = [(name, y_test, art["p_test"]) for name, art in kept.items() if art["p_test"] is not None]
    plots.plot_precision_recall(pr_entries, plots_dir / "precision_recall")
    log.saved("plots")

    # 5. Artifacts -------------------------------------------------------------
    if cfg.output.save_preprocessors:
        try:
            import joblib

            art_dir = out_dir / "artifacts" / "fitted_preprocessors"
            art_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pre, art_dir / "catboost_encoder.joblib")
            log.saved("fitted preprocessor")
        except Exception as exc:  # pragma: no cover
            log.warn(f"artifact save skipped: {exc}")

    return Results(metrics_df, summary, model_configs)


def final_report(metrics_df: pd.DataFrame) -> str:
    """Answer: best model on CatBoost-encoded features, and does native beat encoded?"""
    test = metrics_df[metrics_df.eval_split == "test"]
    if test.empty:
        return "No results."

    lines = ["Model leaderboard (AP, test):"]
    for _, r in test.sort_values("average_precision", ascending=False).iterrows():
        lines.append(f"  {r['model_name']:20s} AP={r['average_precision']:.4f}")

    cb_encoded = test[test.model_name == "catboost_encoded"]
    cb_native = test[test.model_name == "catboost_native"]
    if not cb_encoded.empty and not cb_native.empty:
        ap_encoded = cb_encoded["average_precision"].iloc[0]
        ap_native = cb_native["average_precision"].iloc[0]
        delta = ap_native - ap_encoded
        verdict = "beats" if delta > 0 else "loses to"
        lines += [
            "",
            f"CatBoost native (AP={ap_native:.4f}) {verdict} CatBoost fed the "
            f"CatBoostEncoder features (AP={ap_encoded:.4f}) by {delta:+.4f}.",
        ]

    best = test.loc[test["average_precision"].idxmax()]
    lines += ["", f"Best overall: {best['model_name']} (AP={best['average_precision']:.4f})"]
    return "\n".join(lines)
