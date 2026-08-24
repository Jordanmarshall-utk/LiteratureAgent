#!/usr/bin/env python3
"""Run PCE/stability modeling on original, extracted-only, and integrated CSVs.

This is meant for clean LiteratureAgent campaigns:

1. Original Perovskite Database baseline.
2. LiteratureAgent extracted rows alone, usually integration/literature_update_accepted_rows.csv.
3. Integrated database after adding accepted LiteratureAgent rows.

The extracted-only run is useful as a diagnostic. It may have too few rows for
some targets; failures are captured instead of stopping the whole comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


KEY_METRICS = [
    ("PCE_hierarchical_residual", "stage_4_final", "JV_default_PCE"),
    ("PCE", "design_only", "JV_default_PCE"),
    ("empirical_baseline", "design_only", "target_end_retention_pct"),
    ("empirical_baseline", "design_only", "target_log1p_T80_hours"),
    ("empirical_baseline", "design_only", "target_end_retention_ge80"),
    ("empirical_baseline", "design_only", "target_T80_ge100h"),
    ("physical_condition_features", "physical_condition_features", "target_log1p_T80_hours"),
    ("condition_normalized_hybrid", "condition_normalized_hybrid", "target_phys_log1p_T80_ref_hours"),
]

SUMMARY_COLUMNS = [
    "dataset_version",
    "csv_path",
    "model_out_dir",
    "return_code",
    "model_family",
    "mode",
    "target",
    "task",
    "status",
    "n",
    "oof_r2",
    "oof_rmse",
    "oof_mae",
    "oof_accuracy",
    "oof_balanced_accuracy",
    "oof_f1",
    "oof_roc_auc",
]


def run_model(
    label: str,
    csv_path: Path,
    out_dir: Path,
    model_script: Path,
    min_year: int,
    n_estimators: int,
    min_completeness: float,
    min_completeness_column_coverage: float,
    model_backend: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(model_script),
        "--csv",
        str(csv_path),
        "--out",
        str(out_dir),
        "--min-publication-year",
        str(min_year),
        "--n-estimators",
        str(n_estimators),
        "--min-completeness",
        str(min_completeness),
        "--min-completeness-column-coverage",
        str(min_completeness_column_coverage),
        "--model-backend",
        model_backend,
    ]
    print(f"\n[{label}] Running:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(model_script.parent.parent), text=True)
    return int(completed.returncode)


def read_selected_metrics(label: str, csv_path: Path, out_dir: Path, return_code: int) -> list[dict]:
    metrics_path = out_dir / "model_comparison_pce_then_stability.csv"
    if not metrics_path.exists():
        return [{
            "dataset_version": label,
            "csv_path": str(csv_path),
            "model_out_dir": str(out_dir),
            "return_code": return_code,
            "model_family": "not_found",
            "mode": "",
            "target": "",
            "task": "",
            "status": "model_metrics_csv_not_found",
            "n": "",
            "oof_r2": "",
            "oof_rmse": "",
            "oof_mae": "",
            "oof_accuracy": "",
            "oof_balanced_accuracy": "",
            "oof_f1": "",
            "oof_roc_auc": "",
        }]

    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = []
    for family, mode, target in KEY_METRICS:
        match = next(
            (
                row for row in rows
                if row.get("model_family") == family
                and row.get("mode") == mode
                and row.get("target") == target
            ),
            None,
        )
        if match is None:
            selected.append({
                "dataset_version": label,
                "csv_path": str(csv_path),
                "model_out_dir": str(out_dir),
                "return_code": return_code,
                "model_family": family,
                "mode": mode,
                "target": target,
                "task": "",
                "status": "metric_not_found",
                "n": "",
                "oof_r2": "",
                "oof_rmse": "",
                "oof_mae": "",
                "oof_accuracy": "",
                "oof_balanced_accuracy": "",
                "oof_f1": "",
                "oof_roc_auc": "",
            })
            continue
        selected.append({
            "dataset_version": label,
            "csv_path": str(csv_path),
            "model_out_dir": str(out_dir),
            "return_code": return_code,
            "model_family": family,
            "mode": mode,
            "target": target,
            "task": match.get("task", ""),
            "status": match.get("status", ""),
            "n": match.get("n", ""),
            "oof_r2": match.get("oof_r2", ""),
            "oof_rmse": match.get("oof_rmse", ""),
            "oof_mae": match.get("oof_mae", ""),
            "oof_accuracy": match.get("oof_accuracy", ""),
            "oof_balanced_accuracy": match.get("oof_balanced_accuracy", ""),
            "oof_f1": match.get("oof_f1", ""),
            "oof_roc_auc": match.get("oof_roc_auc", ""),
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-csv", required=True, type=Path)
    parser.add_argument("--extracted-csv", required=True, type=Path)
    parser.add_argument("--integrated-csv", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--model-script", default="models/pce_then_stability_same_approach.py", type=Path)
    parser.add_argument("--min-publication-year", type=int, default=2018)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--model-backend",
        choices=["extra_trees", "xgboost"],
        default="extra_trees",
        help="Explicit model backend. Defaults to the established ExtraTrees production workflow.",
    )
    parser.add_argument("--min-completeness", type=float, default=0.15,
                        help="Row input-fill threshold for original and integrated datasets.")
    parser.add_argument("--extracted-min-completeness", type=float, default=0.05,
                        help="Row input-fill threshold for extracted-only diagnostics. Default lower because extracted rows are sparse by design.")
    parser.add_argument("--min-completeness-column-coverage", type=float, default=0.005,
                        help="Only count model feature columns with at least this non-null fraction in each dataset.")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("01_original_database", args.original_csv, args.out_root / "01_original_database", args.min_completeness),
        ("02_literatureagent_extracted_only", args.extracted_csv, args.out_root / "02_literatureagent_extracted_only", args.extracted_min_completeness),
        ("03_integrated_database", args.integrated_csv, args.out_root / "03_integrated_database", args.min_completeness),
    ]

    all_rows = []
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "min_publication_year": args.min_publication_year,
        "n_estimators": args.n_estimators,
        "model_backend": args.model_backend,
        "min_completeness": args.min_completeness,
        "extracted_min_completeness": args.extracted_min_completeness,
        "min_completeness_column_coverage": args.min_completeness_column_coverage,
        "model_script": str(args.model_script),
        "datasets": [],
    }

    for label, csv_path, out_dir, min_completeness in datasets:
        if not csv_path.exists():
            rc = 2
            print(f"[WARN] Missing CSV for {label}: {csv_path}")
        else:
            rc = run_model(
                label,
                csv_path,
                out_dir,
                args.model_script,
                args.min_publication_year,
                args.n_estimators,
                min_completeness,
                args.min_completeness_column_coverage,
                args.model_backend,
            )
        manifest["datasets"].append({
            "label": label,
            "csv_path": str(csv_path),
            "out_dir": str(out_dir),
            "return_code": rc,
            "min_completeness": min_completeness,
            "min_completeness_column_coverage": args.min_completeness_column_coverage,
            "model_backend": args.model_backend,
        })
        all_rows.extend(read_selected_metrics(label, csv_path, out_dir, rc))

    summary_csv = args.out_root / "three_way_model_metric_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    manifest_path = args.out_root / "three_way_model_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote metric summary: {summary_csv}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
