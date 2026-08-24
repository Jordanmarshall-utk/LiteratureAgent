#!/usr/bin/env python3
"""Evaluate corrected versions of the pre-Jordan notebook PCE models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_model_module(path: Path):
    spec = importlib.util.spec_from_file_location("pce_pipeline_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def catboost_frames(train: pd.DataFrame, test: pd.DataFrame, helpers, cfg):
    train_work = train.copy()
    num_cols, cat_cols = helpers.split_num_cat(train_work, cfg)
    keep = []
    for column in train_work.columns:
        if train_work[column].notna().any() and train_work[column].astype(str).nunique() > 1:
            keep.append(column)
    num_cols = [column for column in num_cols if column in keep]
    cat_cols = [column for column in cat_cols if column in keep]
    keep = num_cols + cat_cols
    train_out = pd.DataFrame(index=train.index)
    test_out = pd.DataFrame(index=test.index)

    for column in num_cols:
        train_values = helpers.to_numeric_series(train[column])
        test_values = helpers.to_numeric_series(test[column])
        median = train_values.median()
        if not np.isfinite(median):
            median = 0.0
        train_out[column] = train_values.fillna(median).astype(float)
        test_out[column] = test_values.fillna(median).astype(float)

    for column in cat_cols:
        train_out[column] = (
            train[column].fillna("__MISSING__").astype(str).str.slice(0, 180)
        )
        test_out[column] = (
            test[column].fillna("__MISSING__").astype(str).str.slice(0, 180)
        )

    return train_out[keep], test_out[keep], cat_cols


def chemistry_numeric_frames(train: pd.DataFrame, test: pd.DataFrame, helpers, cfg):
    train_work = train.copy()
    num_cols, _ = helpers.split_num_cat(train_work, cfg)
    num_cols = [
        column
        for column in num_cols
        if train[column].notna().mean() > 0.70
        and helpers.to_numeric_series(train[column]).nunique(dropna=True) > 1
    ]
    if not num_cols:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    train_num = pd.DataFrame(
        {column: helpers.to_numeric_series(train[column]) for column in num_cols},
        index=train.index,
    )
    test_num = pd.DataFrame(
        {column: helpers.to_numeric_series(test[column]) for column in num_cols},
        index=test.index,
    )
    imputer = SimpleImputer(strategy="median")
    return (
        pd.DataFrame(imputer.fit_transform(train_num), index=train.index, columns=num_cols),
        pd.DataFrame(imputer.transform(test_num), index=test.index, columns=num_cols),
    )


def catboost_model(iterations: int, depth: int, learning_rate: float, random_state: int):
    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def score(y: pd.Series, pred: np.ndarray) -> dict:
    return {
        "r2": r2_score(y, pred),
        "rmse": rmse(y, pred),
        "mae": mean_absolute_error(y, pred),
    }


def plot_predictions(y: pd.Series, pred: np.ndarray, title: str, path: Path):
    metrics = score(y, pred)
    fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=220)
    ax.scatter(y, pred, s=8, alpha=0.28, color="#4C78A8")
    lo = min(float(y.min()), float(np.min(pred)))
    hi = max(float(y.max()), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="#E45756", linewidth=1.2)
    ax.set_xlabel("Measured PCE (%)")
    ax.set_ylabel("Out-of-fold predicted PCE (%)")
    ax.set_title(title)
    ax.text(
        0.04,
        0.96,
        f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\nn = {len(y):,}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model-script",
        default=str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "pce_then_stability_same_approach.py"
        ),
    )
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-publication-year", type=int, default=2018)
    parser.add_argument(
        "--source-notebook",
        default=None,
        help="Optional source notebook path recorded for provenance only.",
    )
    args = parser.parse_args()

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    helpers = load_model_module(Path(args.model_script).resolve())
    cfg = helpers.Config(
        INPUT_CSV=str(Path(args.csv).resolve()),
        OUTPUT_DIR=str(output),
        MIN_PUBLICATION_YEAR=args.min_publication_year,
        CV_STRATEGY="grouped",
    )

    raw = helpers.clean_df(helpers.robust_read_csv(cfg.INPUT_CSV))
    raw = helpers.apply_publication_year_filter(raw, cfg, output)
    group_col = helpers.find_group_col(raw)
    raw = helpers.add_additive_descriptor_features(raw, output)
    data, target = helpers.prepare_pce_target(raw.copy(), cfg)
    features = helpers.build_base_feature_columns(data, target, purpose="pce")
    for column in [c for c in data.columns if c.startswith("ADD_DESC_")]:
        if column not in features:
            features.append(column)
    blocks = helpers.infer_feature_blocks(features)
    data = helpers.apply_row_completeness(data, features, cfg, output)

    required = features + [target, group_col]
    model_data = data[required].copy()
    model_data[target] = helpers.to_numeric_series(model_data[target])
    model_data = model_data[model_data[target].notna()].copy()
    y = model_data[target].astype(float)
    groups = model_data[group_col].fillna("NO_GROUP").astype(str)
    folds = list(helpers.make_cv(y, groups, cfg))

    ordered_stages = [
        ("chemistry_hist_gradient_boosting", blocks["chemistry_architecture"]),
        ("perovskite_processing_catboost", blocks["process"]),
        ("transport_interfaces_catboost", blocks["interfaces"]),
        ("device_context_catboost", blocks["device_context"]),
    ]
    direct_oof = np.full(len(model_data), np.nan)
    hybrid_oof = np.full(len(model_data), np.nan)
    fold_rows = []
    stage_rows = []

    print(
        f"Notebook-inspired grouped PCE evaluation | rows={len(model_data):,} | "
        f"features={len(features)} | groups={groups.nunique():,}"
    )
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        direct_train, direct_test, direct_cat = catboost_frames(
            model_data.iloc[train_idx][features],
            model_data.iloc[test_idx][features],
            helpers,
            cfg,
        )
        direct_model = catboost_model(
            args.iterations, args.depth, args.learning_rate, cfg.RANDOM_STATE
        )
        direct_model.fit(direct_train, y_train, cat_features=direct_cat)
        direct_pred = direct_model.predict(direct_test)
        direct_oof[test_idx] = direct_pred
        direct_metrics = score(y_test, direct_pred)
        print(
            f"fold {fold}: direct CatBoost complete, R²={direct_metrics['r2']:.3f}",
            flush=True,
        )
        fold_rows.append(
            {"fold": fold, "model_family": "catboost_direct", **direct_metrics}
        )

        residual_train = y_train.to_numpy(dtype=float).copy()
        cumulative_test = np.zeros(len(test_idx), dtype=float)
        for stage_number, (stage_name, stage_features) in enumerate(
            ordered_stages, start=1
        ):
            train_stage = model_data.iloc[train_idx][stage_features]
            test_stage = model_data.iloc[test_idx][stage_features]
            if stage_number == 1:
                stage_train, stage_test = chemistry_numeric_frames(
                    train_stage, test_stage, helpers, cfg
                )
                if stage_train.empty:
                    continue
                stage_model = HistGradientBoostingRegressor(
                    max_iter=1000,
                    learning_rate=0.03,
                    max_depth=6,
                    random_state=cfg.RANDOM_STATE,
                )
                stage_model.fit(stage_train, residual_train)
                train_delta = stage_model.predict(stage_train)
                test_delta = stage_model.predict(stage_test)
                feature_count = stage_train.shape[1]
            else:
                stage_train, stage_test, stage_cat = catboost_frames(
                    train_stage, test_stage, helpers, cfg
                )
                if stage_train.empty:
                    continue
                stage_model = catboost_model(
                    args.iterations, args.depth, args.learning_rate, cfg.RANDOM_STATE
                )
                stage_model.fit(
                    stage_train, residual_train, cat_features=stage_cat
                )
                train_delta = stage_model.predict(stage_train)
                test_delta = stage_model.predict(stage_test)
                feature_count = stage_train.shape[1]

            residual_train -= train_delta
            cumulative_test += test_delta
            cumulative_metrics = score(y_test, cumulative_test)
            print(
                f"fold {fold}: stage {stage_number}/4 {stage_name} complete, "
                f"cumulative R²={cumulative_metrics['r2']:.3f}",
                flush=True,
            )
            stage_rows.append(
                {
                    "fold": fold,
                    "stage_number": stage_number,
                    "stage": stage_name,
                    "feature_count": feature_count,
                    **cumulative_metrics,
                }
            )

        hybrid_oof[test_idx] = cumulative_test
        hybrid_metrics = score(y_test, cumulative_test)
        fold_rows.append(
            {
                "fold": fold,
                "model_family": "notebook_block_hybrid",
                **hybrid_metrics,
            }
        )
        print(
            f"fold {fold}: CatBoost direct R²={direct_metrics['r2']:.3f}; "
            f"corrected block hybrid R²={hybrid_metrics['r2']:.3f}",
            flush=True,
        )

    predictions = model_data[[target, group_col]].copy()
    predictions["catboost_direct_oof"] = direct_oof
    predictions["notebook_block_hybrid_oof"] = hybrid_oof
    predictions.to_csv(output / "notebook_block_pce_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "notebook_block_pce_fold_metrics.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output / "notebook_block_pce_stage_metrics.csv", index=False)

    metric_rows = []
    for family, pred in [
        ("catboost_direct", direct_oof),
        ("notebook_block_hybrid", hybrid_oof),
    ]:
        metric_rows.append(
            {
                "model_family": family,
                "cv_strategy": "doi_grouped",
                "n": len(y),
                "n_features_raw": len(features),
                **score(y, pred),
            }
        )
        plot_predictions(
            y,
            pred,
            f"{family}: DOI-grouped PCE prediction",
            output / f"{family}_oof_pred_vs_actual.png",
        )
    metrics = pd.DataFrame(metric_rows).sort_values("r2", ascending=False)
    metrics.to_csv(output / "notebook_block_pce_metrics.csv", index=False)

    fingerprint_payload = (
        predictions[[target, group_col]]
        .fillna("<NA>")
        .astype(str)
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    summary = {
        "source_notebook": (
            str(Path(args.source_notebook).resolve())
            if args.source_notebook
            else "not supplied; historical notebook block reproduced by this script"
        ),
        "input_csv": cfg.INPUT_CSV,
        "target": target,
        "group_column": group_col,
        "cv_strategy": "doi_grouped",
        "rows": len(y),
        "features": len(features),
        "input_fingerprint_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
        "best_model": metrics.iloc[0]["model_family"],
        "best_oof_r2": metrics.iloc[0]["r2"],
        "notebook_corrections": [
            "All preprocessing is fitted inside each outer training fold.",
            "No feature selection uses the test fold.",
            "Residual stages are trained only on training-fold residuals.",
            "Metrics use DOI-grouped out-of-fold predictions only.",
            "No full-data in-sample predictions are reported as test performance.",
        ],
    }
    (output / "notebook_block_pce_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nCorrected notebook-inspired comparison complete")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
