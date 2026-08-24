#!/usr/bin/env python3
"""Compare dataset versions on fixed DOI-grouped and row-random PCE holdouts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("pce_stability_model_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(path: Path, module, cfg, audit_dir: Path, feature_cols=None):
    audit_dir.mkdir(parents=True, exist_ok=True)
    df = module.robust_read_csv(path)
    df = module.clean_df(df)
    df = module.apply_publication_year_filter(df, cfg, audit_dir)
    group_col = module.find_group_col(df)
    if not group_col:
        raise RuntimeError(f"No DOI/reference grouping column found in {path}")
    df = module.add_additive_descriptor_features(df, audit_dir)
    df, target = module.prepare_pce_target(df, cfg)
    if feature_cols is None:
        feature_cols = module.build_base_feature_columns(df, target, purpose="pce")
        for col in [c for c in df.columns if c.startswith("ADD_DESC_")]:
            if col not in feature_cols and col != target:
                feature_cols.append(col)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = module.apply_row_completeness(df, feature_cols, cfg, audit_dir)
    # Match the established production workflow exactly.
    groups = df[group_col].astype(str).fillna("NO_GROUP")
    return df, target, feature_cols, groups, group_col


def normalize_identity_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else f"{float(value):.12g}"
    return str(value).strip().lower()


def build_row_keys(
    df: pd.DataFrame,
    group_col: str,
    identity_cols: list[str] | None = None,
) -> tuple[pd.Series, list[str]]:
    if identity_cols is None:
        identity_cols = [
            col
            for col in ("Ref_ID", "Ref_ID_temp", "Ref_internal_sample_id")
            if col in df.columns
        ]
    if not identity_cols:
        raise RuntimeError("No stable row-identity columns were found.")
    missing = [col for col in identity_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing row-identity columns: {missing}")
    normalized = pd.DataFrame(index=df.index)
    normalized["group"] = df[group_col].map(normalize_identity_value)
    for col in identity_cols:
        normalized[col] = df[col].map(normalize_identity_value)
    if normalized[identity_cols].eq("").all(axis=1).any():
        count = int(normalized[identity_cols].eq("").all(axis=1).sum())
        raise RuntimeError(f"Stable row identity is missing for {count} model-ready rows.")
    keys = normalized[["group", *identity_cols]].agg("|".join, axis=1)
    return keys, identity_cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-csv", type=Path, required=True)
    parser.add_argument("--google-drive-csv", type=Path, required=True)
    parser.add_argument("--integrated-csv", type=Path, required=True)
    parser.add_argument(
        "--model-script",
        type=Path,
        default=PROJECT_ROOT / "models" / "pce_then_stability_same_approach.py",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    module = load_module(args.model_script)
    cfg = module.Config()
    cfg.MIN_PUBLICATION_YEAR = 2018
    cfg.RANDOM_STATE = args.random_state
    cfg.N_ESTIMATORS = args.n_estimators
    cfg.MODEL_BACKEND = "extra_trees"
    cfg.ENABLE_SHAP = False

    datasets = [
        ("01_original_database", args.original_csv),
        ("02_after_google_drive", args.google_drive_csv),
        ("03_after_google_drive_plus_expansion", args.integrated_csv),
    ]
    prepared = {}
    original, target, feature_cols, original_groups, group_col = prepare(
        args.original_csv, module, cfg, args.out_dir / "audit" / "01_original_database"
    )
    original_row_keys, row_key_columns = build_row_keys(original, group_col)
    duplicate_original_keys = int(original_row_keys.duplicated(keep=False).sum())
    if duplicate_original_keys:
        raise RuntimeError(
            f"Stable row keys are not unique for {duplicate_original_keys} original rows."
        )
    prepared["01_original_database"] = (
        original,
        original_groups,
        original_row_keys,
    )
    for label, path in datasets[1:]:
        df, found_target, _, groups, found_group = prepare(
            path, module, cfg, args.out_dir / "audit" / label, feature_cols
        )
        if found_target != target or found_group != group_col:
            raise RuntimeError(f"Target/group mismatch for {label}")
        row_keys, _ = build_row_keys(df, group_col, row_key_columns)
        prepared[label] = (df, groups, row_keys)

    doi_splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_state
    )
    doi_train_pos, doi_test_pos = next(
        doi_splitter.split(original[feature_cols], original[target], original_groups)
    )
    row_splitter = ShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_state
    )
    row_train_pos, row_test_pos = next(
        row_splitter.split(original[feature_cols], original[target])
    )
    splits = {
        "doi_grouped": (doi_train_pos, doi_test_pos),
        "row_random": (row_train_pos, row_test_pos),
    }

    rows = []
    split_audit = {}
    for split_strategy, (_, test_pos) in splits.items():
        frozen_test = original.iloc[test_pos].copy()
        frozen_test_groups = original_groups.iloc[test_pos]
        frozen_groups = set(frozen_test_groups)
        frozen_test_keys = original_row_keys.iloc[test_pos]
        frozen_key_set = set(frozen_test_keys)
        test_filename = (
            "frozen_original_test_rows.csv"
            if split_strategy == "doi_grouped"
            else "frozen_row_random_test_rows.csv"
        )
        frozen_test.assign(
            _frozen_group=frozen_test_groups.to_numpy(),
            _frozen_row_key=frozen_test_keys.to_numpy(),
        ).to_csv(args.out_dir / test_filename, index=True)

        split_audit[split_strategy] = {
            "test_rows": int(len(frozen_test)),
            "test_doi_groups": int(len(frozen_groups)),
            "unique_test_row_keys": int(len(frozen_key_set)),
        }
        for label, path in datasets:
            df, groups, row_keys = prepared[label]
            matched_test_keys = len(frozen_key_set.intersection(set(row_keys)))
            if matched_test_keys != len(frozen_key_set):
                raise RuntimeError(
                    f"{label} preserves only {matched_test_keys}/{len(frozen_key_set)} "
                    f"frozen {split_strategy} row keys."
                )
            if split_strategy == "doi_grouped":
                train_mask = ~groups.isin(frozen_groups)
            else:
                train_mask = ~row_keys.isin(frozen_key_set)
            train = df.loc[train_mask].copy()
            train_groups = set(groups.loc[train_mask])
            row_key_overlap = int(row_keys.loc[train_mask].isin(frozen_key_set).sum())
            group_overlap = len(train_groups.intersection(frozen_groups))
            if row_key_overlap:
                raise RuntimeError(
                    f"{label} has {row_key_overlap} frozen test row keys in training."
                )
            if split_strategy == "doi_grouped" and group_overlap:
                raise RuntimeError(
                    f"{label} has {group_overlap} frozen DOI groups in training."
                )

            preprocessor, _, _ = module.make_preprocessor(train[feature_cols].copy(), cfg)
            model = module.make_model("regression", cfg)
            pipe = Pipeline([("pre", preprocessor), ("model", model)])
            print(
                f"[FROZEN:{split_strategy}] {label}: "
                f"train={len(train):,}, test={len(frozen_test):,}",
                flush=True,
            )
            pipe.fit(train[feature_cols], train[target])
            pred = pipe.predict(frozen_test[feature_cols])
            rows.append({
                "split_strategy": split_strategy,
                "dataset_version": label,
                "csv_path": str(path),
                "model_backend": "extra_trees",
                "train_rows": len(train),
                "frozen_test_rows": len(frozen_test),
                "frozen_test_groups": len(frozen_groups),
                "test_groups_also_in_training": group_overlap,
                "frozen_row_keys_matched": matched_test_keys,
                "frozen_row_keys_in_training": row_key_overlap,
                "r2": r2_score(frozen_test[target], pred),
                "rmse": mean_squared_error(frozen_test[target], pred) ** 0.5,
                "mae": mean_absolute_error(frozen_test[target], pred),
            })
            prediction_name = (
                f"{label}_frozen_predictions.csv"
                if split_strategy == "doi_grouped"
                else f"{label}_frozen_row_random_predictions.csv"
            )
            pd.DataFrame({
                "actual": frozen_test[target].to_numpy(),
                "predicted": pred,
                "group": frozen_test_groups.to_numpy(),
                "row_key": frozen_test_keys.to_numpy(),
            }).to_csv(args.out_dir / prediction_name, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "frozen_production_pce_comparison.csv", index=False)
    manifest = {
        "target": target,
        "group_column": group_col,
        "feature_count": len(feature_cols),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "n_estimators": args.n_estimators,
        "model_backend": "extra_trees",
        "model_script": str(args.model_script),
        "split_strategies": {
            "doi_grouped": "Frozen DOI groups are absent from every training stage.",
            "row_random": (
                "Frozen original-database row keys are absent from every training "
                "stage; other rows from the same DOI may remain in training."
            ),
        },
        "row_key_columns": row_key_columns,
        "split_audit": split_audit,
    }
    (args.out_dir / "frozen_production_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
