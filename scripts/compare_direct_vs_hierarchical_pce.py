#!/usr/bin/env python
"""Compare unchanged pce_direct with the separate staged residual PCE model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "pce_then_stability_same_approach.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("pce_model_comparison_runtime", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model module: {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Dataset must be NAME=CSV_PATH")
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--model-backend", choices=["extra_trees", "xgboost"], default="extra_trees")
    parser.add_argument("--min-publication-year", type=int, default=2018)
    args = parser.parse_args()

    model = load_model_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []

    for dataset_name, csv_path in args.dataset:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        run_dir = model.ensure_dir(args.out_dir / dataset_name)
        pce_dir = model.ensure_dir(run_dir / "pce")
        cfg = model.Config()
        cfg.INPUT_CSV = str(csv_path)
        cfg.OUTPUT_DIR = str(run_dir)
        cfg.N_ESTIMATORS = args.n_estimators
        cfg.MODEL_BACKEND = args.model_backend
        cfg.MIN_PUBLICATION_YEAR = args.min_publication_year

        df = model.clean_df(model.robust_read_csv(str(csv_path)))
        df = model.apply_publication_year_filter(df, cfg, run_dir)
        group_col = model.find_group_col(df)
        df = model.add_additive_descriptor_features(df, run_dir)
        df_pce, target = model.prepare_pce_target(df.copy(), cfg)
        features = model.build_base_feature_columns(df_pce, target, purpose="pce")
        for col in [c for c in df_pce.columns if c.startswith("ADD_DESC_")]:
            if col not in features and col != target:
                features.append(col)
        blocks = model.infer_feature_blocks(features)
        df_pce = model.apply_row_completeness(df_pce, features, cfg, pce_dir)

        _, direct_metrics, _ = model.train_oof_model(
            df_pce, features, target, "regression", group_col, cfg,
            model.ensure_dir(pce_dir / "direct"), "pce_direct",
        )
        _, hierarchical_metrics, stage_metrics = model.train_hierarchical_residual_pce(
            df_pce, blocks, target, group_col, cfg,
            model.ensure_dir(pce_dir / "hierarchical_residual"),
        )

        for family, metrics in [
            ("pce_direct", direct_metrics),
            ("pce_hierarchical_residual", hierarchical_metrics),
        ]:
            comparison_rows.append({
                "dataset": dataset_name,
                "csv_path": str(csv_path),
                "model_family": family,
                "model_backend": args.model_backend,
                "min_publication_year": args.min_publication_year,
                "group_column": group_col,
                **metrics,
            })
        stage_metrics.assign(dataset=dataset_name).to_csv(
            run_dir / "hierarchical_stage_metrics.csv", index=False
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.out_dir / "direct_vs_hierarchical_pce_metrics.csv", index=False)
    summary = {
        "model_script": str(MODEL_PATH),
        "model_backend": args.model_backend,
        "n_estimators": args.n_estimators,
        "min_publication_year": args.min_publication_year,
        "datasets": [{"name": name, "csv": str(path)} for name, path in args.dataset],
        "metrics_csv": str(args.out_dir / "direct_vs_hierarchical_pce_metrics.csv"),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
