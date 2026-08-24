#!/usr/bin/env python
"""Overlay per-field JSON evidence onto a value-bearing aggregate CSV without changing values."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "n/a", "unknown"} else text


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", clean(value).lower()).strip()


def doi(value: object) -> str:
    text = norm(value)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    return re.sub(r"^doi:\s*", "", text).strip(" .;")


def keys(row: dict):
    row_doi = doi(row.get("Ref_DOI_number"))
    sample = norm(row.get("Ref_internal_sample_id"))
    title = norm(row.get("Ref_original_filename_data_upload"))
    out = []
    if row_doi and sample:
        out.append(("doi_sample", row_doi, sample))
    if title and sample:
        out.append(("title_sample", title, sample))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader], list(reader.fieldnames or [])

    records, fields = read(args.records_csv)
    evidence_rows, _ = read(args.evidence_csv)
    index = defaultdict(list)
    for evidence_row in evidence_rows:
        for key in keys(evidence_row):
            index[key].append(evidence_row)

    report = []
    matched_rows = 0
    for row_number, row in enumerate(records, start=2):
        candidates = []
        seen = set()
        for key in keys(row):
            for candidate in index.get(key, []):
                marker = id(candidate)
                if marker not in seen:
                    seen.add(marker)
                    candidates.append(candidate)
        copied = 0
        for candidate in candidates:
            for field, value in candidate.items():
                if not field.startswith("_evidence_") or not clean(value) or clean(row.get(field)):
                    continue
                row[field] = value
                if field not in fields:
                    fields.append(field)
                copied += 1
                report.append({
                    "record_row": row_number,
                    "Ref_DOI_number": clean(row.get("Ref_DOI_number")),
                    "Ref_internal_sample_id": clean(row.get("Ref_internal_sample_id")),
                    "evidence_field": field,
                    "match_candidates": len(candidates),
                })
        if copied:
            matched_rows += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_fields = ["record_row", "Ref_DOI_number", "Ref_internal_sample_id", "evidence_field", "match_candidates"]
    with args.report_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(report)
    print(f"Records: {len(records)}")
    print(f"Rows receiving evidence: {matched_rows}")
    print(f"Evidence fields copied: {len(report)}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()

