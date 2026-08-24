#!/usr/bin/env python
"""Audit whether expansion extraction text matches each candidate paper."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


BLOCKED_PREFIXES = ("all_records", "paper_summaries", "accepted", "rejected", "audit", "duplicate")
STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "using", "via", "of", "in", "on", "to", "a", "an",
    "by", "toward", "towards", "study", "effect", "effects", "enhanced", "improved", "performance",
    "stability", "efficient", "high", "based", "cells", "cell", "solar",
}
PV_TERMS = (
    "photovoltaic", "power conversion efficiency", "pce", "open circuit voltage", "short circuit current",
    "fill factor", "electron transport", "hole transport", "device architecture", "j v", "stability",
)
FOREIGN_TERMS = (
    "dark matter", "cosmological", "astrophysical", "neutron star", "gravitational wave", "oral irrigation",
    "clinical trial", "patient cohort", "tumor", "protein expression",
)


def clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def normalize(value: object) -> str:
    text = re.sub(r"<[^>]+>", " ", clean(value))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(value: object) -> str:
    text = clean(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip(" .;")


def iter_paper_csvs(root: Path):
    csv_dir = root / "csv"
    for path in sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []:
        if not path.name.lower().startswith(BLOCKED_PREFIXES):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path)
    parser.add_argument("--out-filtered-csv", type=Path)
    parser.add_argument("--allowed-status", action="append", default=["pass"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paper_sources: dict[str, dict] = {}
    for root in args.source_root:
        for csv_path in iter_paper_csvs(root):
            slug = re.sub(r"__from_.*$", "", csv_path.stem)
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            candidate = {
                "slug": slug,
                "root": root,
                "csv_path": csv_path,
                "rows": rows,
                "modified": csv_path.stat().st_mtime,
            }
            if slug not in paper_sources or candidate["modified"] > paper_sources[slug]["modified"]:
                paper_sources[slug] = candidate

    report = []
    for slug, source in sorted(paper_sources.items()):
        root = source["root"]
        rows = source["rows"]
        title = next((clean(row.get("Ref_original_filename_data_upload")) for row in rows if clean(row.get("Ref_original_filename_data_upload"))), slug)
        doi = next((normalize_doi(row.get("Ref_DOI_number")) for row in rows if normalize_doi(row.get("Ref_DOI_number"))), "")
        text_files = sorted((root / "text").glob(f"{slug}_*ranked_chunk*.txt"))
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in text_files)
        body = normalize(text[:1000000])
        body_token_set = set(body.split())
        title_tokens = list(dict.fromkeys(
            token for token in normalize(title).split() if len(token) >= 4 and token not in STOPWORDS
        ))
        matched_tokens = [token for token in title_tokens if token in body_token_set]
        coverage = len(matched_tokens) / len(title_tokens) if title_tokens else 0.0
        doi_found = int(bool(doi and doi in text.lower().replace("https://doi.org/", "")))
        perovskite_count = sum(token.startswith("perovskit") for token in body.split())
        pv_term_count = sum(body.count(term) for term in PV_TERMS)
        foreign_hits = [term for term in FOREIGN_TERMS if term in body]
        identity_ok = bool(doi_found or (len(matched_tokens) >= (2 if len(title_tokens) <= 5 else 3) and coverage >= 0.40))
        domain_ok = perovskite_count >= 3 and pv_term_count >= 2
        if not text_files:
            status = "fail_no_ranked_source_text"
        elif foreign_hits and not domain_ok:
            status = "fail_foreign_domain"
        elif not domain_ok:
            status = "fail_target_domain"
        elif identity_ok:
            status = "pass"
        else:
            status = "review_identity_unconfirmed"
        report.append({
            "paper_slug": slug,
            "doi": doi,
            "title": title,
            "status": status,
            "source_root": str(root),
            "per_paper_csv": str(source["csv_path"]),
            "record_rows": len(rows),
            "ranked_text_files": len(text_files),
            "ranked_text_chars": len(text),
            "doi_found_in_text": doi_found,
            "title_tokens": len(title_tokens),
            "title_tokens_matched": len(matched_tokens),
            "title_token_coverage": round(coverage, 4),
            "matched_title_tokens": ";".join(matched_tokens),
            "perovskite_term_count": perovskite_count,
            "pv_term_count": pv_term_count,
            "foreign_domain_terms": ";".join(foreign_hits),
        })

    out_csv = args.out_dir / "expansion_source_alignment_audit.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]) if report else ["paper_slug"])
        writer.writeheader()
        writer.writerows(report)
    counts = Counter(row["status"] for row in report)
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "papers_audited": len(report),
        "status_counts": dict(sorted(counts.items())),
        "audit_csv": str(out_csv),
        "note": "Only pass rows have both target-domain and DOI/title identity support. Review rows require manual or renewed retrieval.",
    }
    if args.records_csv and args.out_filtered_csv:
        status_by_doi = {row["doi"]: row["status"] for row in report if row["doi"]}
        status_by_slug = {row["paper_slug"]: row["status"] for row in report}
        with args.records_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            record_fields = list(reader.fieldnames or [])
            record_rows = list(reader)
        allowed = set(args.allowed_status)
        kept_rows = []
        for row in record_rows:
            doi = normalize_doi(row.get("Ref_DOI_number"))
            sample = clean(row.get("Ref_internal_sample_id"))
            sample_slug = re.sub(r"_device_\d+$", "", sample)
            status = status_by_doi.get(doi) or status_by_slug.get(sample_slug) or "not_audited"
            if status in allowed:
                kept_rows.append(row)
        args.out_filtered_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_filtered_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=record_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept_rows)
        summary["records_csv"] = str(args.records_csv)
        summary["filtered_records_csv"] = str(args.out_filtered_csv)
        summary["allowed_status"] = sorted(allowed)
        summary["input_record_rows"] = len(record_rows)
        summary["filtered_record_rows"] = len(kept_rows)
    (args.out_dir / "expansion_source_alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
