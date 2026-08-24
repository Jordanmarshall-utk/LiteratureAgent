#!/usr/bin/env python
"""Build a verified, provenance-tracked union of expansion campaign records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


BLOCKED_PREFIXES = (
    "all_records", "paper_summaries", "accepted", "rejected", "audit", "duplicate"
)

PCE_FIELDS = [
    "JV_default_PCE", "JV_reverse_scan_PCE", "JV_forward_scan_PCE",
    "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",
]
STABILITY_FIELDS = [
    "Stability_PCE_T80", "Stability_PCE_T95", "Stability_PCE_after_1000_h",
    "Stability_PCE_end_of_experiment", "Stability_time_total_exposure",
    "Stability_temperature_range", "Stability_relative_humidity_range",
    "Stability_light_intensity", "Stability_atmosphere", "Stability_encapsulation",
]
MODEL_CRITICAL_FIELDS = [
    "Ref_DOI_number", "Ref_publication_date", "Ref_internal_sample_id",
    "Perovskite_composition_short_form", "Perovskite_composition_long_form",
    "Cell_stack_sequence", "Cell_architecture", "ETL_stack_sequence",
    "HTL_stack_sequence", "Backcontact_stack_sequence",
    "Perovskite_deposition_procedure", "Perovskite_deposition_solvent",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Cell_area_measured",
] + PCE_FIELDS + STABILITY_FIELDS


def clean(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "n/a", "na", "not reported", "unknown"}:
        return ""
    return text


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", clean(value).lower())


def norm_doi(value: object) -> str:
    text = norm(value)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip(" .;")


def iter_per_paper_csvs(root: Path):
    csv_dir = root / "csv"
    if not csv_dir.exists():
        return
    for path in sorted(csv_dir.glob("*.csv")):
        if path.name.lower().startswith(BLOCKED_PREFIXES):
            continue
        yield path


def read_paper_records(root: Path, csv_path: Path, prefer_json: bool):
    """Read the evidence-preserving per-paper JSON when available."""
    json_path = root / "json" / f"{csv_path.stem}.json"
    if prefer_json and json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get("records", [])
            records = [dict(record) for record in records if isinstance(record, dict)]
            if records:
                return records, json_path
        except Exception:
            pass
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)], csv_path


def record_key(row: dict[str, str]) -> tuple[str, ...]:
    doi = norm_doi(row.get("Ref_DOI_number"))
    sample = norm(row.get("Ref_internal_sample_id"))
    filename = norm(row.get("Ref_original_filename_data_upload"))
    stack = norm(row.get("Cell_stack_sequence"))
    pce = norm(row.get("JV_default_PCE") or row.get("JV_reverse_scan_PCE") or row.get("JV_forward_scan_PCE"))
    key = (doi, sample, filename, stack, pce)
    if any(key):
        return key
    payload = json.dumps({k: clean(v) for k, v in sorted(row.items()) if clean(v)}, sort_keys=True)
    return ("row_sha256", hashlib.sha256(payload.encode("utf-8")).hexdigest())


def paper_key(row: dict[str, str]) -> str:
    doi = norm_doi(row.get("Ref_DOI_number"))
    if doi:
        return f"doi:{doi}"
    sample = norm(row.get("Ref_internal_sample_id"))
    if sample:
        return f"sample:{sample}"
    title = norm(row.get("Ref_original_filename_data_upload"))
    if title:
        return f"title:{title}"
    return ""


def quality(row: dict[str, str]) -> tuple[int, int, int]:
    target_count = sum(bool(clean(row.get(field))) for field in PCE_FIELDS + STABILITY_FIELDS)
    critical_count = sum(bool(clean(row.get(field))) for field in MODEL_CRITICAL_FIELDS)
    total_count = sum(bool(clean(value)) for value in row.values())
    return target_count, critical_count, total_count


def plausible_number(value: object, low: float, high: float) -> bool:
    text = clean(value)
    if not text:
        return True
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return False
    number = float(match.group())
    return math.isfinite(number) and low <= number <= high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--out-work-dir", type=Path, required=True)
    parser.add_argument("--compare-aggregate", type=Path)
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Ignore per-paper JSON. By default JSON is preferred because it retains _evidence_* fields.",
    )
    args = parser.parse_args()

    out_csv_dir = args.out_work_dir / "csv"
    audit_dir = args.out_work_dir / "campaign_audit"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    seen_fields: set[str] = set()
    candidates: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    source_stats: list[dict] = []

    for root in args.source_root:
        files = list(iter_per_paper_csvs(root) or [])
        rows_read = 0
        for path in files:
            records, records_path = read_paper_records(root, path, prefer_json=not args.csv_only)
            for row_number, row in enumerate(records, start=1):
                for field in row:
                    if field not in seen_fields:
                        seen_fields.add(field)
                        fields.append(field)
                rows_read += 1
                candidates[record_key(row)].append({
                    "row": row,
                    "source_root": str(root),
                    "source_csv": str(records_path),
                    "source_row": row_number,
                    "quality": quality(row),
                    "modified": records_path.stat().st_mtime,
                })
        source_stats.append({
            "source_root": str(root),
            "exists": root.exists(),
            "per_paper_csvs": len(files),
            "rows_read": rows_read,
        })

    chosen: list[dict] = []
    duplicate_report: list[dict] = []
    for key, group in candidates.items():
        winner = max(group, key=lambda item: (item["quality"], item["modified"]))
        chosen.append(winner)
        for item in group:
            duplicate_report.append({
                "record_key": json.dumps(key, ensure_ascii=True),
                "candidate_count": len(group),
                "selected": item is winner,
                "target_fields_filled": item["quality"][0],
                "critical_fields_filled": item["quality"][1],
                "total_fields_filled": item["quality"][2],
                "source_root": item["source_root"],
                "source_csv": item["source_csv"],
                "source_row": item["source_row"],
            })

    chosen.sort(key=lambda item: (paper_key(item["row"]), record_key(item["row"])))
    all_records = out_csv_dir / "all_records.csv"
    with all_records.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(item["row"] for item in chosen)

    provenance_fields = [
        "output_row", "paper_key", "record_key", "source_root", "source_csv", "source_row",
        "duplicate_candidates", "target_fields_filled", "critical_fields_filled", "total_fields_filled",
    ]
    with (audit_dir / "selected_row_provenance.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=provenance_fields)
        writer.writeheader()
        for output_row, item in enumerate(chosen, start=2):
            key = record_key(item["row"])
            writer.writerow({
                "output_row": output_row,
                "paper_key": paper_key(item["row"]),
                "record_key": json.dumps(key, ensure_ascii=True),
                "source_root": item["source_root"],
                "source_csv": item["source_csv"],
                "source_row": item["source_row"],
                "duplicate_candidates": len(candidates[key]),
                "target_fields_filled": item["quality"][0],
                "critical_fields_filled": item["quality"][1],
                "total_fields_filled": item["quality"][2],
            })

    with (audit_dir / "duplicate_resolution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(duplicate_report[0]) if duplicate_report else ["record_key"])
        writer.writeheader()
        writer.writerows(row for row in duplicate_report if row.get("candidate_count", 1) > 1)

    with (audit_dir / "source_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_stats[0]) if source_stats else ["source_root"])
        writer.writeheader()
        writer.writerows(source_stats)

    rows = [item["row"] for item in chosen]
    paper_keys = {paper_key(row) for row in rows if paper_key(row)}
    pce_rows = [row for row in rows if any(clean(row.get(field)) for field in PCE_FIELDS)]
    stability_rows = [row for row in rows if any(clean(row.get(field)) for field in STABILITY_FIELDS)]
    range_flags = []
    for index, row in enumerate(rows, start=2):
        if not plausible_number(row.get("JV_default_PCE"), 0, 40):
            range_flags.append({"output_row": index, "field": "JV_default_PCE", "value": row.get("JV_default_PCE")})
        if not plausible_number(row.get("JV_default_Voc"), 0, 3):
            range_flags.append({"output_row": index, "field": "JV_default_Voc", "value": row.get("JV_default_Voc")})
        if not plausible_number(row.get("JV_default_FF"), 0, 100):
            range_flags.append({"output_row": index, "field": "JV_default_FF", "value": row.get("JV_default_FF")})
    with (audit_dir / "numeric_range_flags.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_row", "field", "value"])
        writer.writeheader()
        writer.writerows(range_flags)

    existing_keys: set[tuple[str, ...]] = set()
    if args.compare_aggregate and args.compare_aggregate.exists():
        with args.compare_aggregate.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_keys = {record_key(dict(row)) for row in csv.DictReader(handle)}
    verified_keys = set(candidates)
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_roots": [str(root) for root in args.source_root],
        "source_rows_read": sum(row["rows_read"] for row in source_stats),
        "verified_rows": len(rows),
        "unique_papers": len(paper_keys),
        "duplicate_candidate_copies_removed": sum(max(0, len(group) - 1) for group in candidates.values()),
        "duplicate_groups": sum(len(group) > 1 for group in candidates.values()),
        "pce_candidate_rows": len(pce_rows),
        "stability_candidate_rows": len(stability_rows),
        "numeric_range_flags": len(range_flags),
        "compared_aggregate": str(args.compare_aggregate or ""),
        "verified_keys_missing_from_compared_aggregate": len(verified_keys - existing_keys) if existing_keys else None,
        "compared_aggregate_keys_missing_from_verified_union": len(existing_keys - verified_keys) if existing_keys else None,
        "all_records_csv": str(all_records),
    }
    (audit_dir / "campaign_union_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (audit_dir / "README.md").write_text(
        "# Verified expansion campaign union\n\n"
        f"- Source rows read: {summary['source_rows_read']}\n"
        f"- Verified rows: {summary['verified_rows']}\n"
        f"- Unique papers: {summary['unique_papers']}\n"
        f"- Duplicate copies removed: {summary['duplicate_candidate_copies_removed']}\n"
        f"- PCE candidate rows: {summary['pce_candidate_rows']}\n"
        f"- Stability candidate rows: {summary['stability_candidate_rows']}\n"
        f"- Numeric range flags requiring review: {summary['numeric_range_flags']}\n\n"
        "Duplicate copies are resolved by target coverage, model-critical field coverage, total field coverage, then file modification time.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
