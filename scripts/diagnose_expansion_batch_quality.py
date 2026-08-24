from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PCE_TEXT_RE = re.compile(
    r"(?:PCE|power conversion efficiency|efficiency)[^\n\r]{0,120}?\b\d{1,2}(?:\.\d+)?\s*%",
    re.IGNORECASE,
)
BAD_CONTEXT_RE = re.compile(
    r"brown dwarf|not applicable as this is an observational|not a device performance report",
    re.IGNORECASE,
)


KEY_FIELDS = [
    "Ref_DOI_number",
    "Ref_internal_sample_id",
    "Perovskite_composition_short_form",
    "Perovskite_composition_a_ions",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_c_ions",
    "Cell_stack_sequence",
    "Cell_architecture",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "JV_default_Voc",
    "JV_reverse_scan_Voc",
    "JV_default_Jsc",
    "JV_reverse_scan_Jsc",
    "JV_default_FF",
    "JV_reverse_scan_FF",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_T80",
    "Stability_PCE_T95",
]


def nonempty(value) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    return str(value).strip() not in {"", "nan", "None", "null"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--recent", type=int, default=25)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    csv_dir = work_dir / "csv"
    summary_json_dir = work_dir / "paper_summaries_json"
    summary_text_dir = work_dir / "paper_summaries_text"
    out_dir = Path(args.out_dir) if args.out_dir else work_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records_path = csv_dir / "all_records.csv"
    all_df = pd.read_csv(all_records_path) if all_records_path.exists() else pd.DataFrame()
    key_cols = [c for c in KEY_FIELDS if c in all_df.columns]

    recent_csvs = [
        p
        for p in sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not p.name.startswith("all_records") and not p.name.startswith("paper_summaries")
    ][-args.recent :]

    rows = []
    for csv_path in recent_csvs:
        slug = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            rows.append({"paper_slug": slug, "error": f"csv_read_failed: {exc}"})
            continue
        if df.empty:
            rows.append({"paper_slug": slug, "error": "empty_csv"})
            continue
        row = df.iloc[0]
        combined = " ".join(str(x) for x in row.values if nonempty(x))
        summary_json = summary_json_dir / f"{slug}_summary.json"
        summary_text = summary_text_dir / f"{slug}_summary.txt"
        summary_json_text = summary_json.read_text(encoding="utf-8", errors="ignore") if summary_json.exists() else ""
        summary_text_text = summary_text.read_text(encoding="utf-8", errors="ignore") if summary_text.exists() else ""

        mapped_pce = any(nonempty(row.get(c)) for c in ["JV_default_PCE", "JV_reverse_scan_PCE", "JV_forward_scan_PCE"])
        mapped_comp = any(
            nonempty(row.get(c))
            for c in [
                "Perovskite_composition_short_form",
                "Perovskite_composition_a_ions",
                "Perovskite_composition_b_ions",
                "Perovskite_composition_c_ions",
            ]
        )
        mapped_stability = any(
            nonempty(row.get(c))
            for c in ["Stability_PCE_end_of_experiment", "Stability_PCE_T80", "Stability_PCE_T95"]
        )
        pce_text_hit = PCE_TEXT_RE.search(combined) or PCE_TEXT_RE.search(summary_text_text) or PCE_TEXT_RE.search(summary_json_text)
        bad_context = BAD_CONTEXT_RE.search(summary_json_text) or BAD_CONTEXT_RE.search(summary_text_text)

        pce_example = ""
        if pce_text_hit:
            pce_example = " ".join(pce_text_hit.group(0).split())[:220]

        rows.append(
            {
                "paper_slug": slug,
                "doi": row.get("Ref_DOI_number", ""),
                "mapped_pce": mapped_pce,
                "mapped_composition": mapped_comp,
                "mapped_stability": mapped_stability,
                "has_pce_text_somewhere": bool(pce_text_hit),
                "bad_summary_context": bool(bad_context),
                "pce_text_example": pce_example,
                "record_confidence": row.get("_lit_record_confidence", ""),
                "flags": row.get("_lit_agent_flags", ""),
            }
        )

    detail = pd.DataFrame(rows)
    detail_path = out_dir / "recent_batch_mapping_diagnostic.csv"
    detail.to_csv(detail_path, index=False)

    summary = {
        "work_dir": str(work_dir),
        "all_records_rows": int(len(all_df)),
        "recent_csvs_checked": int(len(recent_csvs)),
        "mapped_pce_rows": int(detail.get("mapped_pce", pd.Series(dtype=bool)).sum()),
        "mapped_composition_rows": int(detail.get("mapped_composition", pd.Series(dtype=bool)).sum()),
        "mapped_stability_rows": int(detail.get("mapped_stability", pd.Series(dtype=bool)).sum()),
        "rows_with_pce_text_somewhere": int(detail.get("has_pce_text_somewhere", pd.Series(dtype=bool)).sum()),
        "bad_summary_context_rows": int(detail.get("bad_summary_context", pd.Series(dtype=bool)).sum()),
        "detail_csv": str(detail_path),
    }

    summary_path = out_dir / "recent_batch_mapping_diagnostic_summary.txt"
    lines = [f"{k}: {v}" for k, v in summary.items()]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    if key_cols and not all_df.empty:
        counts = all_df[key_cols].apply(lambda s: s.map(nonempty).sum()).sort_values(ascending=False)
        counts.to_csv(out_dir / "all_records_key_field_nonempty_counts.csv", header=["nonempty_rows"])
        print(f"key_field_counts: {out_dir / 'all_records_key_field_nonempty_counts.csv'}")


if __name__ == "__main__":
    main()
