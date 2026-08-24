#!/usr/bin/env python
"""Triage LiteratureAgent Drive papers into device-focused and reference folders.

The script uses existing LiteratureAgent outputs instead of re-reading every PDF.
It writes an auditable manifest first, and can optionally copy PDFs into folders.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEVICE_TERMS = {
    "pce",
    "power conversion efficiency",
    "j-v",
    "jv",
    "voc",
    "jsc",
    "fill factor",
    "ff",
    "device architecture",
    "device stack",
    "solar cell",
    "champion device",
    "certified",
    "eqe",
    "mpp",
    "maximum power point",
}

STABILITY_TERMS = {
    "stability",
    "t80",
    "t95",
    "retention",
    "retained",
    "lifetime",
    "aging",
    "ageing",
    "1000 h",
    "1000h",
    "humidity",
    "thermal",
    "temperature",
    "illumination",
    "encapsulation",
    "operational stability",
}

REFERENCE_TERMS = {
    "review",
    "perspective",
    "rulebook",
    "roadmap",
    "overview",
    "computational",
    "dft",
    "theory",
    "first-principles",
    "first principles",
    "mechanistic",
    "photophysics",
    "single crystal",
}

PERFORMANCE_COLUMNS = [
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "Stabilised_performance_PCE",
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
]

STABILITY_COLUMNS = [
    "Stability_PCE_T80",
    "Stability_PCE_T95",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_after_1000_h",
    "Stability_time_total_exposure",
    "Outdoor_PCE_T80",
    "Outdoor_PCE_T95",
    "Outdoor_PCE_end_of_experiment",
    "Outdoor_PCE_after_1000_h",
    "Outdoor_time_total_exposure",
]

DESIGN_COLUMNS = [
    "Cell_stack_sequence",
    "Cell_architecture",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "Perovskite_composition_short_form",
    "Perovskite_composition_long_form",
    "Perovskite_deposition_procedure",
]


@dataclass
class PaperTriage:
    paper_slug: str
    recommended_folder: str
    confidence: str
    score_device: int
    score_stability: int
    score_reference: int
    paper_type: str
    doi_count: int
    record_rows: int
    performance_field_count: int
    stability_field_count: int
    design_field_count: int
    matched_pdf: str
    reason: str


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).lower()


def nonempty(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s not in {"", "nan", "none", "null", "unknown", "not reported", "n/a", "na", "-", "--"}


def count_nonempty(df: pd.DataFrame, columns: Iterable[str]) -> int:
    total = 0
    for col in columns:
        if col in df.columns:
            total += int(df[col].map(nonempty).sum())
    return total


def contains_count(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_summary_text(summary_dir: Path, slug: str) -> str:
    candidates = [
        summary_dir / f"{slug}_summary.txt",
        summary_dir / f"{slug}.txt",
    ]
    for path in candidates:
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""
    return ""


def find_pdf_for_slug(pdf_dirs: list[Path], slug: str) -> str:
    if not slug:
        return ""
    slug_low = slug.lower()
    compact_slug = re.sub(r"[^a-z0-9]+", "", slug_low)
    for folder in pdf_dirs:
        if not folder.exists():
            continue
        for pdf in folder.rglob("*.pdf"):
            stem = pdf.stem.lower()
            compact_stem = re.sub(r"[^a-z0-9]+", "", stem)
            if slug_low in stem or compact_slug[:30] in compact_stem:
                return str(pdf)
    return ""


def classify_one(
    slug: str,
    records: pd.DataFrame,
    gate_row: pd.Series | None,
    summary_text: str,
    matched_pdf: str,
) -> PaperTriage:
    text_bits = [summary_text]
    if not records.empty:
        for col in [
            "Ref_free_text_comment",
            "Ref_DOI_number",
            "Ref_lead_author",
            "Ref_journal",
            "_source_abstract",
        ]:
            if col in records.columns:
                text_bits.extend(records[col].dropna().astype(str).head(8).tolist())
    combined = norm_text("\n".join(text_bits))

    perf_count = count_nonempty(records, PERFORMANCE_COLUMNS) if not records.empty else 0
    stab_count = count_nonempty(records, STABILITY_COLUMNS) if not records.empty else 0
    design_count = count_nonempty(records, DESIGN_COLUMNS) if not records.empty else 0
    doi_count = int(records["Ref_DOI_number"].map(nonempty).sum()) if "Ref_DOI_number" in records.columns else 0

    paper_type = ""
    gate_device = gate_review = gate_comp = gate_materials = gate_metrics = 0
    if gate_row is not None:
        paper_type = str(gate_row.get("predicted_type", "") or "")
        gate_device = int(float(gate_row.get("device_score", 0) or 0))
        gate_review = int(float(gate_row.get("review_score", 0) or 0))
        gate_comp = int(float(gate_row.get("computational_score", 0) or 0))
        gate_materials = int(float(gate_row.get("materials_score", 0) or 0))
        gate_metrics = int(float(gate_row.get("metric_score", 0) or 0))

    term_device = contains_count(combined, DEVICE_TERMS)
    term_stability = contains_count(combined, STABILITY_TERMS)
    term_reference = contains_count(combined, REFERENCE_TERMS)

    score_device = (
        4 * perf_count
        + 2 * design_count
        + 2 * term_device
        + gate_device
        + 2 * gate_metrics
    )
    score_stability = 4 * stab_count + 2 * term_stability
    score_reference = 5 * term_reference + gate_review + gate_comp

    reasons = []
    if perf_count:
        reasons.append(f"performance_fields={perf_count}")
    if stab_count:
        reasons.append(f"stability_fields={stab_count}")
    if design_count:
        reasons.append(f"design_fields={design_count}")
    if paper_type:
        reasons.append(f"paper_type={paper_type}")
    if term_device:
        reasons.append(f"device_terms={term_device}")
    if term_stability:
        reasons.append(f"stability_terms={term_stability}")
    if term_reference:
        reasons.append(f"reference_terms={term_reference}")

    if perf_count > 0 or stab_count > 0:
        folder = "01_device_stability_extraction"
        confidence = "high"
    elif paper_type == "device_experimental" and (design_count > 0 or score_device >= 16):
        folder = "01_device_stability_extraction"
        confidence = "medium"
    elif score_device >= max(18, score_reference + 4) and gate_comp < 8:
        folder = "01_device_stability_extraction"
        confidence = "medium"
    elif score_stability >= 6 and score_device >= 8:
        folder = "01_device_stability_extraction"
        confidence = "medium"
    else:
        folder = "02_materials_reviews_reference"
        confidence = "medium" if score_reference or paper_type else "low"

    if paper_type in {"review_perspective", "computational_theory"} and not (perf_count or stab_count):
        folder = "02_materials_reviews_reference"
        confidence = "high"

    return PaperTriage(
        paper_slug=slug,
        recommended_folder=folder,
        confidence=confidence,
        score_device=score_device,
        score_stability=score_stability,
        score_reference=score_reference,
        paper_type=paper_type,
        doi_count=doi_count,
        record_rows=int(len(records)),
        performance_field_count=perf_count,
        stability_field_count=stab_count,
        design_field_count=design_count,
        matched_pdf=matched_pdf,
        reason="; ".join(reasons) if reasons else "limited metadata",
    )


def copy_triaged_pdfs(rows: list[PaperTriage], out_dir: Path) -> None:
    for row in rows:
        if not row.matched_pdf:
            continue
        src = Path(row.matched_pdf)
        if not src.exists():
            continue
        dest_dir = out_dir / row.recommended_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            continue
        shutil.copy2(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lit-work-dir", required=True, help="LiteratureAgent output folder from the Drive run.")
    parser.add_argument("--out-dir", required=True, help="Folder for triage manifest and optional copied PDFs.")
    parser.add_argument(
        "--pdf-dir",
        action="append",
        default=[],
        help="Optional source PDF folder. Can be supplied multiple times. Defaults to work-dir/pdf and work-dir/manual_google_drive_papers.",
    )
    parser.add_argument("--copy-pdfs", action="store_true", help="Copy matched PDFs into triaged folders.")
    args = parser.parse_args()

    work_dir = Path(args.lit_work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = read_csv_if_exists(work_dir / "csv" / "all_records.csv")
    gate = read_csv_if_exists(work_dir / "paper_type_gate_report.csv")
    summary_dir = work_dir / "paper_summaries_text"

    pdf_dirs = [Path(p) for p in args.pdf_dir]
    if not pdf_dirs:
        pdf_dirs = [work_dir / "pdf", work_dir / "manual_google_drive_papers"]

    slugs = set()
    if "paper_slug" in all_records.columns:
        slugs.update(str(x) for x in all_records["paper_slug"].dropna().unique())
    if "paper_slug" in gate.columns:
        slugs.update(str(x) for x in gate["paper_slug"].dropna().unique())
    for path in summary_dir.glob("*_summary.txt"):
        slugs.add(path.name.removesuffix("_summary.txt"))

    gate_by_slug = {}
    if "paper_slug" in gate.columns:
        for _, row in gate.iterrows():
            gate_by_slug[str(row.get("paper_slug", ""))] = row

    rows: list[PaperTriage] = []
    for slug in sorted(slugs):
        recs = all_records[all_records["paper_slug"].astype(str) == slug].copy() if "paper_slug" in all_records.columns else pd.DataFrame()
        summary_text = load_summary_text(summary_dir, slug)
        matched_pdf = find_pdf_for_slug(pdf_dirs, slug)
        rows.append(classify_one(slug, recs, gate_by_slug.get(slug), summary_text, matched_pdf))

    manifest_path = out_dir / "drive_paper_triage_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(PaperTriage.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    if args.copy_pdfs:
        copy_triaged_pdfs(rows, out_dir)

    counts = pd.Series([r.recommended_folder for r in rows]).value_counts()
    print("Drive paper triage complete")
    print(f"Manifest: {manifest_path}")
    print(counts.to_string())
    print(f"Matched PDFs: {sum(1 for r in rows if r.matched_pdf)} / {len(rows)}")
    if args.copy_pdfs:
        print(f"Copied triaged PDFs under: {out_dir}")


if __name__ == "__main__":
    main()
