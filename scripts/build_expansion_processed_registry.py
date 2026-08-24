#!/usr/bin/env python
"""Build a central processed-paper registry from previous LiteratureAgent outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


AGGREGATE_NAMES = {
    "all_records.csv",
    "all_records_from_manager.csv",
}


def normalize_doi(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip(" .;")


def normalize_title(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def iter_record_csvs(root: Path):
    if root.is_file() and root.suffix.lower() == ".csv":
        yield root
        return
    for csv_dir in [root / "csv", root / "combined_full_coverage" / "csv", root / "combined_full_coverage"]:
        if csv_dir.exists():
            for path in csv_dir.glob("*.csv"):
                name = path.name.lower()
                if name.startswith(("paper_summaries", "accepted", "rejected", "audit", "duplicate")):
                    continue
                yield path


def ingest_csv(path: Path, rows: dict[tuple[str, str, str], dict[str, str]], source_root: Path) -> int:
    added = 0
    now = datetime.now().isoformat(timespec="seconds")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                doi = normalize_doi(row.get("Ref_DOI_number") or row.get("doi") or row.get("DOI") or "")
                title = str(row.get("Ref_original_filename_data_upload") or row.get("title") or row.get("Title") or "").strip()
                title_key = normalize_title(title)
                if len(title_key) < 24:
                    title_key = ""
                paper_slug = str(row.get("Ref_internal_sample_id") or row.get("paper_slug") or "").strip().lower()
                key = (doi, title_key, paper_slug)
                if key == ("", "", "") or key in rows:
                    continue
                rows[key] = {
                    "doi": doi,
                    "title_key": title_key,
                    "title": title[:500],
                    "paper_slug": paper_slug,
                    "source_label": str(path),
                    "source_root": str(source_root),
                    "source_kind": "record_csv",
                    "status": "record_written",
                    "first_seen": now,
                    "last_seen": now,
                }
                added += 1
    except Exception as exc:
        print(f"[WARN] Could not ingest CSV {path}: {exc}")
    return added


def ingest_timing_log(path: Path, rows: dict[tuple[str, str, str], dict[str, str]], source_root: Path) -> int:
    added = 0
    now = datetime.now().isoformat(timespec="seconds")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("event") != "paper_complete":
                    continue
                doi = normalize_doi(event.get("doi") or "")
                title = str(event.get("title") or "").strip()
                title_key = normalize_title(title)
                if len(title_key) < 24:
                    title_key = ""
                paper_slug = str(event.get("paper_slug") or "").strip().lower()
                key = (doi, title_key, paper_slug)
                if key == ("", "", "") or key in rows:
                    continue
                rows[key] = {
                    "doi": doi,
                    "title_key": title_key,
                    "title": title[:500],
                    "paper_slug": paper_slug,
                    "source_label": str(path),
                    "source_root": str(source_root),
                    "source_kind": "timing_log",
                    "status": str(event.get("status") or ""),
                    "first_seen": now,
                    "last_seen": now,
                }
                added += 1
    except Exception as exc:
        print(f"[WARN] Could not ingest timing log {path}: {exc}")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True, help="LiteratureAgent output or campaign folder to scan. Repeatable.")
    parser.add_argument("--out-dir", type=Path, default=Path("data") / "expansion_processed_papers")
    parser.add_argument("--out-name", default="processed_papers_registry.csv")
    args = parser.parse_args()

    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    csv_count = 0
    timing_count = 0
    for root in args.root:
        if not root.exists():
            print(f"[WARN] Root not found: {root}")
            continue
        for path in iter_record_csvs(root):
            csv_count += 1
            ingest_csv(path, rows, root)
        for path in root.rglob("timing_logs/paper_timing.jsonl"):
            timing_count += 1
            ingest_timing_log(path, rows, root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / args.out_name
    fields = [
        "doi",
        "title_key",
        "title",
        "paper_slug",
        "source_label",
        "source_root",
        "source_kind",
        "status",
        "first_seen",
        "last_seen",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: (r.get("doi") or "", r.get("title_key") or "", r.get("paper_slug") or "")):
            writer.writerow(row)

    readme = args.out_dir / "README.md"
    readme.write_text(
        "# Expansion Processed Papers Registry\n\n"
        "This folder stores the central DOI/title/slug registry used to avoid reprocessing the same expansion papers across output folders.\n\n"
        f"- Registry: `{out_csv.name}`\n"
        f"- Unique paper keys: {len(rows)}\n"
        f"- CSV files scanned: {csv_count}\n"
        f"- Timing logs scanned: {timing_count}\n\n"
        "Future expansion runs should pass this file through `--processed-registry`.\n",
        encoding="utf-8",
    )
    print(f"Registry written: {out_csv}")
    print(f"Unique paper keys: {len(rows)}")
    print(f"CSV files scanned: {csv_count}")
    print(f"Timing logs scanned: {timing_count}")


if __name__ == "__main__":
    main()
