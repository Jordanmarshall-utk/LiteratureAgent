#!/usr/bin/env python
"""Compare PCE/stability metrics across sequential dataset states."""

from __future__ import annotations

import argparse
import csv
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


def read_metrics(run_name: str, run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "model_comparison_pce_then_stability.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing model comparison CSV for {run_name}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for family, mode, target in KEY_METRICS:
        match = next(
            (
                row
                for row in rows
                if row.get("model_family") == family
                and row.get("mode") == mode
                and row.get("target") == target
            ),
            None,
        )
        if not match:
            continue
        out.append({
            "run": run_name,
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
            "source_csv": str(path),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--google-drive-dir", type=Path, required=True)
    parser.add_argument("--expansion-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    rows.extend(read_metrics("01_original_before_literature", args.before_dir))
    rows.extend(read_metrics("02_after_google_drive", args.google_drive_dir))
    rows.extend(read_metrics("03_after_expansion", args.expansion_dir))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
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
        "source_csv",
    ]
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison: {args.out}")


if __name__ == "__main__":
    main()
