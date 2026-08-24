#!/usr/bin/env python3
"""Check LiteratureAgent field coverage before/after deterministic enrichment.

This is a replay-style diagnostic for existing outputs. It only uses text already
present in CSV rows, so it is a lower-bound estimate of fresh-run enrichment,
where the controller has access to retrieved full text.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


KEY_FIELD_GROUPS = {
    "provenance": ["Ref_DOI_number", "Ref_internal_sample_id"],
    "pce": ["JV_default_PCE", "JV_reverse_scan_PCE", "JV_forward_scan_PCE", "Stabilised_performance_PCE"],
    "jv_submetrics": ["JV_default_Voc", "JV_default_Jsc", "JV_default_FF"],
    "composition": [
        "Perovskite_composition_short_form",
        "Perovskite_composition_long_form",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions",
    ],
    "device_stack": ["Cell_stack_sequence", "Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence"],
    "processing": [
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_solvents",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
    ],
    "stability": [
        "Stability_measured",
        "Stability_protocol",
        "Stability_time_total_exposure",
        "Stability_PCE_initial_value",
        "Stability_PCE_end_of_experiment",
        "Stability_PCE_T80",
        "Stability_PCE_T95",
        "Stability_PCE_after_1000_h",
    ],
}

TEXT_FIELDS = [
    "Ref_DOI_number",
    "Ref_lead_author",
    "Ref_journal",
    "Ref_original_filename_data_upload",
    "Ref_free_text_comment",
    "Cell_stack_sequence",
    "Cell_architecture",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "Perovskite_additives_compounds",
    "Perovskite_composition_short_form",
    "Perovskite_composition_long_form",
    "_source_title",
    "_source_abstract",
    "_source_landing_page",
    "_source_pdf_url",
    "_composition_json",
]


def present(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "[]", "{}", "unknown", "not reported", "n/a"}


def row_text(row: pd.Series) -> str:
    parts = []
    for col in TEXT_FIELDS:
        if col in row.index and present(row.get(col)):
            parts.append(str(row.get(col)))
    return " ".join(parts)


def load_controller(path: Path):
    spec = importlib.util.spec_from_file_location("literature_agent_controller_for_coverage", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load controller from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_group(df: pd.DataFrame, fields: list[str]) -> dict:
    existing = [c for c in fields if c in df.columns]
    if not existing:
        return {
            "fields_present_in_csv": 0,
            "rows_with_any": 0,
            "rows_with_all_existing": 0,
            "per_field": {},
        }
    per_field = {c: int(df[c].map(present).sum()) for c in existing}
    any_mask = df[existing].apply(lambda r: any(present(v) for v in r), axis=1)
    all_mask = df[existing].apply(lambda r: all(present(v) for v in r), axis=1)
    return {
        "fields_present_in_csv": len(existing),
        "rows_with_any": int(any_mask.sum()),
        "rows_with_all_existing": int(all_mask.sum()),
        "per_field": per_field,
    }


def summarize(df: pd.DataFrame) -> dict:
    out = {"rows": int(len(df)), "groups": {}}
    for group, fields in KEY_FIELD_GROUPS.items():
        out["groups"][group] = count_group(df, fields)
    key_model_fields = [
        "Ref_DOI_number",
        "JV_default_PCE",
        "Perovskite_composition_short_form",
        "Cell_stack_sequence",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
    ]
    existing = [c for c in key_model_fields if c in df.columns]
    if existing:
        out["rows_with_doi_and_pce"] = int(
            df.apply(lambda r: present(r.get("Ref_DOI_number")) and present(r.get("JV_default_PCE")), axis=1).sum()
        )
        out["rows_with_doi_pce_and_any_composition"] = int(
            df.apply(
                lambda r: present(r.get("Ref_DOI_number"))
                and present(r.get("JV_default_PCE"))
                and (
                    present(r.get("Perovskite_composition_short_form"))
                    or present(r.get("Perovskite_composition_long_form"))
                ),
                axis=1,
            ).sum()
        )
    return out


def replay_enrichment(df: pd.DataFrame, controller) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        meta = {
            "title": rec.get("_source_title") or rec.get("Ref_original_filename_data_upload") or "",
            "doi": rec.get("Ref_DOI_number") or rec.get("_source_landing_page") or "",
            "abstract": rec.get("_source_abstract") or "",
            "url": rec.get("_source_landing_page") or "",
            "pdf_url": rec.get("_source_pdf_url") or "",
        }
        text = row_text(row)
        enriched = controller._ctrl_v21_10_enrich_extraction_records(meta, text, text, [rec])[0]
        rows.append(enriched)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lit-csv", required=True)
    parser.add_argument("--controller", default="literature_agent_full_end_to_end_v21_3_english_sanitizer.py")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    lit_csv = Path(args.lit_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(lit_csv, low_memory=False)
    controller = load_controller(Path(args.controller))
    enriched = replay_enrichment(df, controller)

    before = summarize(df)
    after = summarize(enriched)
    summary = {
        "input_csv": str(lit_csv),
        "note": "Replay uses only text already present in all_records.csv, so this is a lower-bound estimate for fresh extraction with full text.",
        "before": before,
        "after_replay_enrichment": after,
    }

    enriched.to_csv(out_dir / "all_records_replayed_enrichment.csv", index=False)
    (out_dir / "field_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for group in KEY_FIELD_GROUPS:
        rows.append({
            "group": group,
            "before_rows_with_any": before["groups"][group]["rows_with_any"],
            "after_rows_with_any": after["groups"][group]["rows_with_any"],
            "before_rows_with_all_existing": before["groups"][group]["rows_with_all_existing"],
            "after_rows_with_all_existing": after["groups"][group]["rows_with_all_existing"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "field_coverage_by_group.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
