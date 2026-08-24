#!/usr/bin/env python3
"""Benchmark PCE regressors with one frozen DOI-grouped holdout.

Hyperparameters are selected using grouped CV on the original-database training
partition only. The selected configurations are then refit on each successive
dataset version and evaluated against the exact same original-database rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_SCRIPT = PROJECT_ROOT / "models" / "pce_then_stability_same_approach.py"
DEFAULT_ORIGINAL_CSV = PROJECT_ROOT / "data" / "Perovskite_database_content_all_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-csv", type=Path, default=DEFAULT_ORIGINAL_CSV)
    parser.add_argument("--google-drive-csv", type=Path, required=True)
    parser.add_argument("--integrated-csv", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-tuning", action="store_true")
    return parser.parse_args()


def load_model_module(path: Path):
    spec = importlib.util.spec_from_file_location("pce_stability_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_groups(df: pd.DataFrame, group_col: str) -> pd.Series:
    values = df[group_col].fillna("").astype(str).str.strip().str.lower()
    missing = values.eq("") | values.isin({"nan", "none", "null"})
    # Missing DOI values must not collapse thousands of unrelated papers into one group.
    values.loc[missing] = [f"missing_group_row_{idx}" for idx in values.index[missing]]
    return values


def prepare_dataset(path: Path, module, cfg, out_dir: Path, feature_cols: list[str] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = module.robust_read_csv(path)
    df = module.clean_df(df)
    df = module.apply_publication_year_filter(df, cfg, out_dir)
    group_col = module.find_group_col(df)
    if not group_col:
        raise RuntimeError(f"No DOI/reference grouping column found in {path}")
    df = module.add_additive_descriptor_features(df, out_dir)
    df, target = module.prepare_pce_target(df, cfg)
    if feature_cols is None:
        feature_cols = module.build_base_feature_columns(df, target, purpose="pce")
        for col in [c for c in df.columns if c.startswith("ADD_DESC_")]:
            if col not in feature_cols and col != target:
                feature_cols.append(col)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = module.apply_row_completeness(df, feature_cols, cfg, out_dir)
    groups = normalized_groups(df, group_col)
    return df, target, feature_cols, groups, group_col


def one_hot_preprocessor(X: pd.DataFrame, module, cfg) -> ColumnTransformer:
    num_cols, cat_cols = module.split_num_cat(X, cfg)
    try:
        num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    except TypeError:
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=5)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True, min_frequency=5)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("to_numeric", FunctionTransformer(module.coerce_numeric_frame, validate=False)),
                        ("imputer", num_imputer),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            ),
            ("cat", Pipeline([("imputer", cat_imputer), ("encode", encoder)]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def ordinal_preprocessor(X: pd.DataFrame, module, cfg) -> ColumnTransformer:
    num_cols, cat_cols = module.split_num_cat(X, cfg)
    try:
        num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    except TypeError:
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("to_numeric", FunctionTransformer(module.coerce_numeric_frame, validate=False)),
                        ("imputer", num_imputer),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", cat_imputer),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def candidate_models(seed: int):
    candidates = {
        "extra_trees": [
            ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1, max_features="sqrt", random_state=seed, n_jobs=-1),
            ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1),
            ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1, max_features=0.5, random_state=seed, n_jobs=-1),
        ],
        "random_forest": [
            RandomForestRegressor(n_estimators=350, min_samples_leaf=1, max_features="sqrt", random_state=seed, n_jobs=-1),
            RandomForestRegressor(n_estimators=350, min_samples_leaf=2, max_features=0.5, random_state=seed, n_jobs=-1),
        ],
        "hist_gradient_boosting": [
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed),
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=5.0, random_state=seed),
        ],
        "ridge": [Ridge(alpha=1.0, solver="lsqr"), Ridge(alpha=10.0, solver="lsqr")],
    }
    if XGBRegressor is not None:
        candidates["xgboost"] = [
            XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=3,
                reg_lambda=3.0,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            ),
            XGBRegressor(
                n_estimators=650,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=5.0,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            ),
        ]
    return candidates


def pipeline_for(model_name: str, model, X: pd.DataFrame, module, cfg) -> Pipeline:
    if model_name == "hist_gradient_boosting":
        preprocessor = ordinal_preprocessor(X, module, cfg)
    else:
        preprocessor = one_hot_preprocessor(X, module, cfg)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def model_parameters(model) -> dict:
    keep = {
        "n_estimators",
        "min_samples_leaf",
        "max_features",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
        "max_iter",
        "max_leaf_nodes",
        "l2_regularization",
        "alpha",
    }
    return {k: v for k, v in model.get_params().items() if k in keep}


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def tune_models(X, y, groups, module, cfg, candidates, folds: int, out_dir: Path, skip_tuning: bool):
    rows = []
    selected = {}
    splitter = GroupKFold(n_splits=folds)
    for model_name, variants in candidates.items():
        use_variants = variants[:1] if skip_tuning else variants
        best = None
        for variant_index, model in enumerate(use_variants, start=1):
            started = time.time()
            oof = np.full(len(y), np.nan)
            print(f"[TUNE] {model_name} variant {variant_index}/{len(use_variants)}", flush=True)
            for fold, (tr, va) in enumerate(splitter.split(X, y, groups), start=1):
                pipe = pipeline_for(model_name, clone(model), X.iloc[tr], module, cfg)
                pipe.fit(X.iloc[tr], y.iloc[tr])
                oof[va] = pipe.predict(X.iloc[va])
                print(f"  fold {fold}/{folds} complete", flush=True)
            metrics = regression_metrics(y, oof)
            row = {
                "model": model_name,
                "variant": variant_index,
                **metrics,
                "elapsed_seconds": round(time.time() - started, 3),
                "parameters": json.dumps(model_parameters(model), sort_keys=True),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "inner_cv_model_tuning.csv", index=False)
            if best is None or metrics["rmse"] < best[0]:
                best = (metrics["rmse"], clone(model), variant_index)
        selected[model_name] = best[1]
    return selected, pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, out_dir: Path) -> None:
    order = list(dict.fromkeys(results["model"].tolist()))
    versions = list(dict.fromkeys(results["dataset_version"].tolist()))
    x = np.arange(len(order))
    width = 0.8 / max(len(versions), 1)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    for i, version in enumerate(versions):
        sub = results[results["dataset_version"] == version].set_index("model").reindex(order)
        ax.bar(x + (i - (len(versions) - 1) / 2) * width, sub["test_r2"], width, label=version)
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylabel("Frozen-holdout R2")
    ax.set_title("PCE regressor comparison on identical DOI-grouped test rows")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "frozen_holdout_r2_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "01_original": args.original_csv,
        "02_after_google_drive": args.google_drive_csv,
        "03_after_google_drive_plus_expansion": args.integrated_csv,
    }
    missing = [str(p) for p in [args.model_script, *datasets.values()] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))

    module = load_model_module(args.model_script)
    cfg = module.Config()
    cfg.MIN_PUBLICATION_YEAR = 2018
    cfg.RANDOM_STATE = args.random_state
    cfg.ENABLE_SHAP = False

    prepared = {}
    original, target, feature_cols, original_groups, group_col = prepare_dataset(
        args.original_csv, module, cfg, args.out_dir / "prepared" / "01_original"
    )
    prepared["01_original"] = (original, original_groups)
    for name, path in list(datasets.items())[1:]:
        df, found_target, _, groups, found_group_col = prepare_dataset(
            path, module, cfg, args.out_dir / "prepared" / name, feature_cols
        )
        if found_target != target or found_group_col != group_col:
            raise RuntimeError(f"Target/group mismatch for {name}: {found_target}, {found_group_col}")
        prepared[name] = (df, groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_pos, test_pos = next(splitter.split(original, original[target], original_groups))
    frozen_test = original.iloc[test_pos].copy()
    frozen_test_groups = set(original_groups.iloc[test_pos])
    original_train = original.iloc[train_pos].copy()
    original_train_groups = original_groups.iloc[train_pos].copy()

    split_manifest = {
        "target": target,
        "group_column": group_col,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "inner_folds": args.inner_folds,
        "original_model_rows": len(original),
        "original_train_rows": len(original_train),
        "frozen_test_rows": len(frozen_test),
        "frozen_test_groups": len(frozen_test_groups),
        "feature_count": len(feature_cols),
        "model_script": str(args.model_script),
        "model_script_sha256": hashlib.sha256(args.model_script.read_bytes()).hexdigest(),
        "datasets": {k: str(v) for k, v in datasets.items()},
        "config": asdict(cfg),
    }
    (args.out_dir / "frozen_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame({"row_index": frozen_test.index, "group": original_groups.iloc[test_pos].values}).to_csv(
        args.out_dir / "frozen_test_rows.csv", index=False
    )
    pd.Series(feature_cols, name="feature_column").to_csv(args.out_dir / "feature_columns.csv", index=False)

    X_train = original_train[feature_cols]
    y_train = original_train[target]
    candidates = candidate_models(args.random_state)
    selected, tuning = tune_models(
        X_train,
        y_train,
        original_train_groups,
        module,
        cfg,
        candidates,
        args.inner_folds,
        args.out_dir,
        args.skip_tuning,
    )
    selected_payload = {name: model_parameters(model) for name, model in selected.items()}
    (args.out_dir / "selected_model_parameters.json").write_text(
        json.dumps(selected_payload, indent=2, default=str), encoding="utf-8"
    )

    result_rows = []
    X_test = frozen_test[feature_cols]
    y_test = frozen_test[target]
    for dataset_name, (df, groups) in prepared.items():
        # Exclude every row sharing a DOI/reference group with the frozen test set.
        train_mask = ~groups.isin(frozen_test_groups)
        train_df = df.loc[train_mask].copy()
        lit_mask = module.literature_agent_source_mask(train_df)
        for model_name, model in selected.items():
            print(f"[FINAL] {dataset_name} | {model_name} | train={len(train_df):,}", flush=True)
            started = time.time()
            pipe = pipeline_for(model_name, clone(model), train_df[feature_cols], module, cfg)
            pipe.fit(train_df[feature_cols], train_df[target])
            pred = pipe.predict(X_test)
            metrics = regression_metrics(y_test, pred)
            result_rows.append(
                {
                    "dataset_version": dataset_name,
                    "model": model_name,
                    "train_rows": len(train_df),
                    "literatureagent_train_rows": int(lit_mask.sum()),
                    "test_rows": len(frozen_test),
                    "test_groups": len(frozen_test_groups),
                    "test_r2": metrics["r2"],
                    "test_rmse": metrics["rmse"],
                    "test_mae": metrics["mae"],
                    "elapsed_seconds": round(time.time() - started, 3),
                    "parameters": json.dumps(model_parameters(model), sort_keys=True),
                }
            )
            pd.DataFrame(result_rows).to_csv(args.out_dir / "frozen_holdout_model_comparison.csv", index=False)
            pd.DataFrame(
                {
                    "row_index": frozen_test.index,
                    "group": original_groups.iloc[test_pos].values,
                    "actual_pce": y_test.values,
                    "predicted_pce": pred,
                }
            ).to_csv(args.out_dir / f"predictions__{dataset_name}__{model_name}.csv", index=False)

    results = pd.DataFrame(result_rows)
    plot_results(results, args.out_dir)
    best = results.sort_values(["dataset_version", "test_rmse"]).groupby("dataset_version", as_index=False).first()
    best.to_csv(args.out_dir / "best_model_by_dataset.csv", index=False)
    print("\nBest frozen-holdout model by dataset:")
    print(best[["dataset_version", "model", "test_r2", "test_rmse", "test_mae", "train_rows", "literatureagent_train_rows"]].to_string(index=False))
    print(f"\nOutputs: {args.out_dir}")


if __name__ == "__main__":
    main()
