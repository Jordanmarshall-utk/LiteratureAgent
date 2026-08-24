#!/usr/bin/env python
"""Add field evidence only when a stored abstract explicitly supports the same target value."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TARGET_PATTERNS = {
    "JV_default_PCE": [
        r"\bPCE\s*(?:of|=|:|reached|reaches|up to|as high as)?\s*(\d{1,2}(?:\.\d+)?)\s*%",
        r"\bpower conversion efficienc(?:y|ies)\s*(?:of|=|:|reached|reaches|up to|as high as)?\s*(\d{1,2}(?:\.\d+)?)\s*%",
        r"\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:PCE|power conversion efficiency)\b",
    ],
    "JV_reverse_scan_PCE": [
        r"\b(?:reverse|backward)\s*(?:scan)?[^.;]{0,100}?\b(?:PCE|efficiency)\D{0,20}(\d{1,2}(?:\.\d+)?)\s*%",
    ],
    "JV_forward_scan_PCE": [
        r"\bforward\s*(?:scan)?[^.;]{0,100}?\b(?:PCE|efficiency)\D{0,20}(\d{1,2}(?:\.\d+)?)\s*%",
    ],
    "Stability_PCE_T80": [r"\bT\s*80\s*(?:=|:|of|over|exceeding)?\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b"],
    "Stability_PCE_T95": [r"\bT\s*95\s*(?:=|:|of|over|exceeding)?\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b"],
    "Stability_PCE_after_1000_h": [
        r"\b(?:after|over|for)\s*1000\s*(?:h|hours)[^.;]{0,120}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"\b(\d{1,3}(?:\.\d+)?)\s*%[^.;]{0,120}?\b(?:after|over|for)\s*1000\s*(?:h|hours)",
    ],
    "Stability_PCE_end_of_experiment": [
        r"\b(?:retained|retention(?:\s+of)?|maintained)\s*(\d{1,3}(?:\.\d+)?)\s*%",
        r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*(?:of\s+(?:the\s+)?)?(?:initial\s+)?(?:PCE\s+)?(?:retained|retention|remaining)",
    ],
    "Stability_time_total_exposure": [
        r"\b(?:retained|retention|maintained|stability|stable|aging|ageing)[^.;]{0,140}?\b(?:after|over|for)\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b",
        r"\b(?:after|over|for)\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b[^.;]{0,140}?\b(?:retained|retention|maintained|stability|stable|aging|ageing)\b",
    ],
}


def clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "n/a", "unknown"} else text


def number(value: object):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", clean(value).replace(",", ""))
    return float(match.group()) if match else None


def matching_evidence(abstract: str, target: float, patterns: list[str]):
    for pattern in patterns:
        for match in re.finditer(pattern, abstract, flags=re.IGNORECASE):
            candidate = float(match.group(1))
            tolerance = max(0.05, abs(target) * 0.002)
            if abs(candidate - target) > tolerance:
                continue
            start = max(0, match.start() - 180)
            end = min(len(abstract), match.end() + 180)
            return re.sub(r"\s+", " ", abstract[start:end]).strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, required=True)
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])

    report = []
    for row_number, row in enumerate(rows, start=2):
        abstract = clean(row.get("_source_abstract"))
        if not abstract:
            continue
        for field, patterns in TARGET_PATTERNS.items():
            target = number(row.get(field))
            evidence_field = f"_evidence_{field}"
            if target is None or clean(row.get(evidence_field)):
                continue
            evidence = matching_evidence(abstract, target, patterns)
            if not evidence:
                continue
            row[evidence_field] = evidence
            if evidence_field not in fields:
                fields.append(evidence_field)
            report.append({
                "input_row": row_number,
                "Ref_DOI_number": clean(row.get("Ref_DOI_number")),
                "field": field,
                "value": clean(row.get(field)),
                "evidence": evidence,
                "method": "exact_value_and_metric_phrase_in_source_abstract",
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        report_fields = ["input_row", "Ref_DOI_number", "field", "value", "evidence", "method"]
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(report)
    print(f"Rows read: {len(rows)}")
    print(f"Target evidence verified from abstracts: {len(report)}")
    for field in TARGET_PATTERNS:
        count = sum(item["field"] == field for item in report)
        if count:
            print(f"  {field}: {count}")
    print(f"Output: {args.output_csv}")
    print(f"Report: {args.report_csv}")


if __name__ == "__main__":
    main()

