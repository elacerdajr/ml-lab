"""
runner.py
---------
Orchestration: generate data (same DGP as the sibling experiment), undersample,
then for each encoder build+fit the shared preprocessor once and fan it out to
every encoded model; separately fit CatBoost's native-categorical path. Collect
metrics, write CSVs and plots, and print/write a report answering "best encoder
per model" and "does CatBoost-native beat every encoded variant?".
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
from .data import FEATURES
from .encoders import (
    CAT_IDX,
    ENCODER_NAMES,
    fit_preprocessor_timed,
    to_catboost_native_X,
    transform_timed,
)
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

        return RichLogger("encoder_comparison")
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
    diagnostics: pd.DataFrame
    summary: dict
    model_configs: dict


def run(cfg: Config, out_dir: Path, log=None) -> Results:
    log = log or _make_logger()
    plots_dir = out_dir / "plots"

    # 1. Data ------------------------------------------------------------
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
    model_configs = {name: dict(cfg.models.get(name.split("_")[0], {})) for name in registry}
    model_configs["catboost_native"] = dict(cfg.models["catboost"])
    (out_dir / "model_configs.json").write_text(json.dumps(model_configs, indent=2, default=str))

    metric_rows: list[dict] = []
    diag_rows: list[dict] = []
    run_id = 0

    # 2. Encoded models, one preprocessor fit per encoder ------------------
    for encoder_name in ENCODER_NAMES:
        log.section(f"Encoder: {encoder_name}")
        pre, X_train_enc, fit_time = fit_preprocessor_timed(encoder_name, train_full, train_under, cfg)
        n_features_out = X_train_enc.shape[1]

        X_eval_enc: dict[str, tuple[np.ndarray, float]] = {}
        for split_name, split_df in eval_splits.items():
            X_enc, t_transform = transform_timed(pre, split_df)
            X_eval_enc[split_name] = (X_enc, t_transform)

        log.info(f"  n_features_out={n_features_out}  fit_time={fit_time:.3f}s")
        diag_rows.append(
            {
                "encoder": encoder_name,
                "n_features_out": n_features_out,
                "fit_time_seconds": fit_time,
                "val_transform_time_seconds": X_eval_enc["val"][1],
                "test_transform_time_seconds": X_eval_enc["test"][1],
            }
        )

        for model_name in ENCODED_MODEL_NAMES:
            spec = registry[model_name]
            try:
                model, t_train = fit_timed(spec, X_train_enc, y_train)
            except Exception as exc:
                log.warn(f"  {model_name}: fit failed ({exc})")
                continue
            log.info(f"  {model_name}: fit in {t_train:.3f}s")

            for split_name, split_df in eval_splits.items():
                X_enc, t_transform = X_eval_enc[split_name]
                p, t_pred = predict_timed(model, X_enc)
                y_eval = split_df["y"].to_numpy()

                keys = {
                    "run_id": run_id,
                    "encoder": encoder_name,
                    "model_name": model_name,
                    "eval_split": split_name,
                }
                encoder_diag = {
                    "n_features_out": n_features_out,
                    "encode_fit_time_seconds": fit_time,
                    "encode_transform_time_seconds": t_transform,
                }
                timings = {"train_time_seconds": t_train, "predict_time_seconds": t_pred}
                counts = {
                    "n_train": int(len(X_train_enc)),
                    "n_eval": int(len(split_df)),
                    "positive_rate_train": float(y_train.mean()),
                    "positive_rate_eval": float(y_eval.mean()),
                }
                metric_rows.append(eval_row(y_eval, p, keys, encoder_diag, timings, counts))
                run_id += 1

    # 3. CatBoost native path (encoder-independent) ------------------------
    log.section("CatBoost native categorical handling")
    native_model = build_catboost_native(cfg)
    t0 = time.perf_counter()
    native_model.fit(to_catboost_native_X(train_under), y_train, cat_features=CAT_IDX)
    t_train = time.perf_counter() - t0
    log.info(f"  catboost_native: fit in {t_train:.3f}s")

    diag_rows.append(
        {
            "encoder": "native",
            "n_features_out": len(FEATURES),
            "fit_time_seconds": np.nan,
            "val_transform_time_seconds": np.nan,
            "test_transform_time_seconds": np.nan,
        }
    )

    for split_name, split_df in eval_splits.items():
        Xn = to_catboost_native_X(split_df)
        p, t_pred = predict_timed(native_model, Xn)
        y_eval = split_df["y"].to_numpy()
        keys = {
            "run_id": run_id,
            "encoder": "native",
            "model_name": "catboost_native",
            "eval_split": split_name,
        }
        encoder_diag = {
            "n_features_out": len(FEATURES),
            "encode_fit_time_seconds": np.nan,
            "encode_transform_time_seconds": np.nan,
        }
        timings = {"train_time_seconds": t_train, "predict_time_seconds": t_pred}
        counts = {
            "n_train": int(len(train_under)),
            "n_eval": int(len(split_df)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_eval": float(y_eval.mean()),
        }
        metric_rows.append(eval_row(y_eval, p, keys, encoder_diag, timings, counts))
        run_id += 1

    metrics_df = pd.DataFrame(metric_rows)
    diag_df = pd.DataFrame(diag_rows)

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    diag_df.to_csv(out_dir / "encoder_diagnostics.csv", index=False)
    log.saved(f"metrics.csv ({len(metrics_df)} rows), encoder_diagnostics.csv ({len(diag_df)} rows)")

    # 4. Plots --------------------------------------------------------------
    log.section("Plots")
    test_df = metrics_df[metrics_df.eval_split == "test"]
    plots.plot_ap_by_encoder(test_df, plots_dir / "ap_by_encoder")
    plots.plot_dimensionality(test_df, plots_dir / "dimensionality")
    plots.plot_timing(test_df, plots_dir / "timing")
    log.saved("plots")

    # 5. Artifacts ------------------------------------------------------------
    if cfg.output.save_preprocessors:
        try:
            import joblib

            art_dir = out_dir / "artifacts" / "fitted_preprocessors"
            art_dir.mkdir(parents=True, exist_ok=True)
            # Refit + save one preprocessor per encoder for downstream reuse.
            for encoder_name in ENCODER_NAMES:
                pre, _, _ = fit_preprocessor_timed(encoder_name, train_full, train_under, cfg)
                joblib.dump(pre, art_dir / f"{encoder_name}.joblib")
            log.saved("fitted preprocessors")
        except Exception as exc:  # pragma: no cover
            log.warn(f"artifact save skipped: {exc}")

    return Results(metrics_df, diag_df, summary, model_configs)


def final_report(metrics_df: pd.DataFrame) -> str:
    """Answer: best encoder per model, and does CatBoost-native beat the encoded variants?"""
    test = metrics_df[metrics_df.eval_split == "test"]
    if test.empty:
        return "No results."

    lines = ["Best encoder per model (AP, test):"]
    for model_name, sub in test.groupby("model_name"):
        best = sub.loc[sub["average_precision"].idxmax()]
        lines.append(f"  {model_name:20s} {best['encoder']:10s} AP={best['average_precision']:.4f}")

    cb_encoded = test[test.model_name == "catboost_encoded"]
    cb_native = test[test.model_name == "catboost_native"]
    if not cb_encoded.empty and not cb_native.empty:
        best_encoded = cb_encoded.loc[cb_encoded["average_precision"].idxmax()]
        native_ap = cb_native["average_precision"].iloc[0]
        delta = native_ap - best_encoded["average_precision"]
        verdict = "beats" if delta > 0 else "loses to"
        lines += [
            "",
            f"CatBoost native (AP={native_ap:.4f}) {verdict} its best encoded variant "
            f"({best_encoded['encoder']}, AP={best_encoded['average_precision']:.4f}) "
            f"by {delta:+.4f}.",
        ]

    best_overall = test.loc[test["average_precision"].idxmax()]
    lines += [
        "",
        f"Best overall: {best_overall['model_name']} x {best_overall['encoder']} "
        f"(AP={best_overall['average_precision']:.4f})",
    ]
    return "\n".join(lines)
