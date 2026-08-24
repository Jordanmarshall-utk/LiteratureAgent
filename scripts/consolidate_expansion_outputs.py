#!/usr/bin/env python
"""Consolidate retained LiteratureAgent expansion outputs into one clean work folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


BLOCKED_CSV_PREFIXES = (
    "all_records",
    "paper_summaries",
    "accepted",
    "rejected",
    "audit",
    "duplicate",
)


def iter_per_paper_csvs(root: Path):
    csv_dir = root / "csv"
    if not csv_dir.exists():
        return
    for path in sorted(csv_dir.glob("*.csv")):
        if path.name.lower().startswith(BLOCKED_CSV_PREFIXES):
            continue
        yield path


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("Ref_DOI_number") or "").strip().lower(),
        str(row.get("Ref_internal_sample_id") or "").strip().lower(),
        str(row.get("Ref_original_filename_data_upload") or "").strip().lower(),
        str(row.get("Cell_stack_sequence") or "").strip().lower(),
        str(row.get("JV_default_PCE") or row.get("JV_reverse_scan_PCE") or row.get("JV_forward_scan_PCE") or "").strip().lower(),
    )


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def safe_copy(source: Path, target_dir: Path, source_label: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / source.name
    if not candidate.exists():
        shutil.copy2(source, candidate)
        return candidate
    stem = source.stem
    suffix = source.suffix
    candidate = target_dir / f"{stem}__from_{source_label}{suffix}"
    n = 2
    while candidate.exists():
        candidate = target_dir / f"{stem}__from_{source_label}_{n}{suffix}"
        n += 1
    shutil.copy2(source, candidate)
    return candidate


def copy_matching_summary_files(source_root: Path, out_root: Path, csv_path: Path, source_label: str) -> int:
    copied = 0
    summary_dirs = [
        "paper_summaries_json",
        "paper_summaries_text",
        "figure_reports/json",
        "figure_reports/text",
        "reasoning_logs",
    ]
    slug = csv_path.stem
    for rel in summary_dirs:
        src_dir = source_root / rel
        if not src_dir.exists():
            continue
        out_dir = out_root / rel
        for path in src_dir.glob(f"{slug}*"):
            if path.is_file():
                safe_copy(path, out_dir, source_label)
                copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--out-work-dir", type=Path, required=True)
    parser.add_argument("--processed-registry", type=Path)
    parser.add_argument("--copy-sidecars", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    out_csv_dir = args.out_work_dir / "csv"
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    seen_rows: set[tuple[str, str, str, str, str]] = set()
    manifest_rows: list[dict[str, str | int]] = []
    copied_csvs = 0
    copied_sidecars = 0
    duplicate_rows = 0

    for source_root in args.source_root:
        if not source_root.exists():
            manifest_rows.append({
                "source_root": str(source_root),
                "status": "missing_source_root",
                "per_paper_csv": "",
                "rows_read": 0,
                "rows_kept": 0,
                "rows_duplicate": 0,
                "copied_csv": "",
            })
            continue
        source_label = "".join(ch if ch.isalnum() else "_" for ch in source_root.name).strip("_")
        for csv_path in iter_per_paper_csvs(source_root) or []:
            local_fields, local_rows = read_rows(csv_path)
            kept_here = 0
            duplicate_here = 0
            for field in local_fields:
                if field not in seen_fields:
                    seen_fields.add(field)
                    fieldnames.append(field)
            for row in local_rows:
                key = row_key(row)
                if key in seen_rows:
                    duplicate_rows += 1
                    duplicate_here += 1
                    continue
                seen_rows.add(key)
                rows.append(row)
                kept_here += 1
            copied_path = safe_copy(csv_path, out_csv_dir, source_label)
            copied_csvs += 1
            if args.copy_sidecars:
                copied_sidecars += copy_matching_summary_files(source_root, args.out_work_dir, csv_path, source_label)
            manifest_rows.append({
                "source_root": str(source_root),
                "status": "ok",
                "per_paper_csv": str(csv_path),
                "rows_read": len(local_rows),
                "rows_kept": kept_here,
                "rows_duplicate": duplicate_here,
                "copied_csv": str(copied_path),
            })

    if not fieldnames:
        fieldnames = ["Ref_DOI_number", "Ref_original_filename_data_upload", "Ref_internal_sample_id"]

    all_records = out_csv_dir / "all_records.csv"
    with all_records.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    all_records_from_manager = out_csv_dir / "all_records_from_manager.csv"
    shutil.copy2(all_records, all_records_from_manager)

    manifest_dir = args.out_work_dir / "consolidation_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifest_dir / "source_file_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "source_root",
            "status",
            "per_paper_csv",
            "rows_read",
            "rows_kept",
            "rows_duplicate",
            "copied_csv",
        ])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    unique_papers = {
        (
            str(row.get("Ref_DOI_number") or "").strip().lower(),
            str(row.get("Ref_internal_sample_id") or "").strip().lower(),
            str(row.get("Ref_original_filename_data_upload") or "").strip().lower(),
        )
        for row in rows
    }
    unique_papers.discard(("", "", ""))
    pce_rows = [
        row for row in rows
        if row.get("JV_default_PCE") or row.get("JV_reverse_scan_PCE") or row.get("JV_forward_scan_PCE")
    ]
    stability_rows = [
        row for row in rows
        if row.get("Stability_PCE_T80")
        or row.get("Stability_PCE_T95")
        or row.get("Stability_PCE_after_1000_h")
        or row.get("Stability_PCE_end_of_experiment")
        or row.get("Stability_time_total_exposure")
    ]
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "out_work_dir": str(args.out_work_dir),
        "source_roots": [str(path) for path in args.source_root],
        "processed_registry": str(args.processed_registry) if args.processed_registry else "",
        "copied_per_paper_csvs": copied_csvs,
        "copied_sidecars": copied_sidecars,
        "aggregate_rows": len(rows),
        "unique_processed_papers": len(unique_papers),
        "duplicate_rows_removed": duplicate_rows,
        "pce_candidate_rows": len(pce_rows),
        "stability_candidate_rows": len(stability_rows),
        "all_records_csv": str(all_records),
        "all_records_from_manager_csv": str(all_records_from_manager),
        "manifest_csv": str(manifest_csv),
    }
    summary_json = manifest_dir / "consolidation_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
