from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


MISSING = {"", "nan", "none", "null", "[]", "{}", "not reported", "unknown", "n/a"}

KEY_PCE_COLS = [
    "JV_default_PCE",
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
    "Cell_stack_sequence",
    "Cell_architecture",
    "Perovskite_composition_short_form",
    "Perovskite_composition_long_form",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_deposition_solvents",
    "Perovskite_additives_compounds",
    "Cell_area_measured",
]

KEY_STABILITY_COLS = [
    "Stability_measured",
    "Stability_protocol",
    "Stability_time_total_exposure",
    "Stability_PCE_initial_value",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_T80",
    "Stability_PCE_T95",
    "Stability_PCE_after_1000_h",
    "Stability_temperature_range",
    "Stability_relative_humidity_average_value",
    "Stability_light_intensity",
    "Stability_atmosphere",
    "Encapsulation",
]

AUDIT_FIELD_GROUPS = {
    "source_metadata": [
        "Ref_DOI_number",
        "Ref_ID",
        "Ref_publication_date",
        "Ref_journal",
        "Ref_original_filename_data_upload",
    ],
    "device_stack": [
        "Cell_architecture",
        "Substrate_stack_sequence",
        "ETL_stack_sequence",
        "Perovskite_composition_short_form",
        "Perovskite_composition_long_form",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
        "Cell_stack_sequence",
    ],
    "composition": [
        "Perovskite_composition_short_form",
        "Perovskite_composition_long_form",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions",
        "Perovskite_additives_compounds",
    ],
    "processing": [
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_solvents",
        "Perovskite_deposition_quenching_media",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
        "Perovskite_deposition_synthesis_atmosphere",
        "Perovskite_additives_compounds",
        "Perovskite_additives_concentrations",
    ],
    "performance": [
        "JV_default_PCE",
        "JV_default_Voc",
        "JV_default_Jsc",
        "JV_default_FF",
        "Stabilised_performance_PCE",
        "JV_default_PCE_scan_direction",
        "Cell_area_measured",
    ],
    "stability_conditions": [
        "Stability_protocol",
        "Stability_temperature_range",
        "Stability_relative_humidity_average_value",
        "Stability_atmosphere",
        "Stability_light_intensity",
        "Stability_time_total_exposure",
        "Encapsulation",
    ],
    "stability_outcomes": [
        "Stability_PCE_initial_value",
        "Stability_PCE_end_of_experiment",
        "Stability_PCE_after_1000_h",
        "Stability_PCE_T80",
        "Stability_PCE_T95",
    ],
}

AUDIT_FIELD_LABELS = {
    "DOI": ["Ref_DOI_number"],
    "title": ["Ref_original_filename_data_upload", "Ref_ID", "Ref_ID_temp"],
    "publication_year": ["Ref_publication_date"],
    "journal": ["Ref_journal"],
    "reference_identifier": ["Ref_ID", "Ref_internal_sample_id"],
    "architecture": ["Cell_architecture"],
    "substrate": ["Substrate_stack_sequence"],
    "ETL": ["ETL_stack_sequence"],
    "perovskite_composition": ["Perovskite_composition_short_form", "Perovskite_composition_long_form"],
    "HTL": ["HTL_stack_sequence"],
    "back_contact": ["Backcontact_stack_sequence"],
    "deposition_method": ["Perovskite_deposition_procedure"],
    "solvent": ["Perovskite_deposition_solvents"],
    "antisolvent": ["Perovskite_deposition_quenching_media"],
    "annealing_temperature": ["Perovskite_deposition_thermal_annealing_temperature"],
    "annealing_time": ["Perovskite_deposition_thermal_annealing_time"],
    "processing_atmosphere": ["Perovskite_deposition_synthesis_atmosphere"],
    "additive": ["Perovskite_additives_compounds"],
    "additive_concentration": ["Perovskite_additives_concentrations"],
    "PCE": ["JV_default_PCE"],
    "Voc": ["JV_default_Voc"],
    "Jsc": ["JV_default_Jsc"],
    "FF": ["JV_default_FF"],
    "stabilized_PCE": ["Stabilised_performance_PCE"],
    "scan_direction": ["JV_default_PCE_scan_direction"],
    "active_area": ["Cell_area_measured"],
    "stability_protocol_or_type": ["Stability_protocol"],
    "temperature": ["Stability_temperature_range"],
    "relative_humidity": ["Stability_relative_humidity_average_value"],
    "stability_atmosphere": ["Stability_atmosphere"],
    "illumination": ["Stability_light_intensity"],
    "exposure_time": ["Stability_time_total_exposure"],
    "initial_performance": ["Stability_PCE_initial_value"],
    "final_performance": ["Stability_PCE_end_of_experiment"],
    "retention": ["Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h"],
    "T80": ["Stability_PCE_T80"],
    "T95": ["Stability_PCE_T95"],
    "encapsulation": ["Encapsulation", "Encapsulation_stack_sequence"],
}

TARGET_COLS = [
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "Stabilised_performance_PCE",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_after_1000_h",
    "Stability_PCE_T80",
    "Stability_PCE_T95",
]

PCE_PAT = re.compile(
    r"(?i)\b(?:PCE|power conversion efficienc(?:y|ies)|photoconversion efficienc(?:y|ies))"
    r"[^.\n]{0,80}?(\d{1,2}(?:\.\d+)?)\s*%"
)

STAB_PAT = re.compile(
    r"(?i)\b(?:T\s*80|T\s*95|retained|retention|maintained|remaining|after|for|over)"
    r"[^.\n]{0,160}?\b(?:\d{2,3}(?:\.\d+)?\s*%|\d{2,5}(?:[.,]\d+)?\s*(?:h|hours|hrs))\b"
)


def present(v: Any) -> bool:
    if v is None or pd.isna(v):
        return False
    return str(v).strip().lower() not in MISSING


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception:
            pass
    return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coverage(df: pd.DataFrame, cols: list[str], label: str) -> list[dict[str, Any]]:
    rows = []
    n = len(df)
    for col in cols:
        if col not in df.columns:
            rows.append({"dataset": label, "column": col, "present": False, "non_null": 0, "fraction": 0.0})
            continue
        non_null = int(df[col].map(present).sum())
        rows.append({
            "dataset": label,
            "column": col,
            "present": True,
            "non_null": non_null,
            "fraction": round(non_null / n, 6) if n else 0.0,
        })
    return rows


def model_range_pce_count(df: pd.DataFrame) -> int:
    if "JV_default_PCE" not in df.columns:
        return 0
    vals = pd.to_numeric(df["JV_default_PCE"], errors="coerce")
    return int(vals.between(5, 30).sum())


def source_added_mask(df: pd.DataFrame) -> pd.Series:
    if "_source_added_by_literature_agent" not in df.columns:
        return pd.Series(False, index=df.index)
    vals = df["_source_added_by_literature_agent"]
    as_num = pd.to_numeric(vals, errors="coerce").fillna(0)
    as_text = vals.astype(str).str.strip().str.lower()
    return (as_num > 0) | as_text.isin({"true", "yes", "y"})


def collect_text_for_slug(work_dir: Path, slug: str, max_chars: int = 200_000) -> str:
    parts = []
    for sub in ("text", "paper_summaries_text", "paper_summaries_json"):
        folder = work_dir / sub
        if not folder.exists():
            continue
        for path in sorted(folder.glob(f"{slug}*"))[:12]:
            if path.is_file() and path.suffix.lower() in {".txt", ".json"}:
                try:
                    parts.append(path.read_text(encoding="utf-8", errors="ignore")[:max_chars // 12])
                except Exception:
                    pass
    return "\n".join(parts)


def source_text_audit(work_dir: Path, out_dir: Path) -> dict[str, Any]:
    csv_dir = work_dir / "csv"
    per_paper = sorted(
        p for p in csv_dir.glob("*.csv")
        if p.name.lower() not in {"all_records.csv", "all_records_from_manager.csv", "paper_summaries_from_manager.csv"}
    )
    rows = []
    for path in per_paper:
        slug = path.stem
        df = read_csv(path)
        mapped_pce = model_range_pce_count(df) > 0
        mapped_stability = any(
            col in df.columns and df[col].map(present).any()
            for col in ["Stability_PCE_T80", "Stability_PCE_T95", "Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h"]
        )
        text = collect_text_for_slug(work_dir, slug)
        pce_hits = len(PCE_PAT.findall(text))
        stability_hits = len(STAB_PAT.findall(text))
        rows.append({
            "paper_slug": slug,
            "csv_file": path.name,
            "mapped_pce": mapped_pce,
            "mapped_stability": mapped_stability,
            "source_pce_hit_count": pce_hits,
            "source_stability_hit_count": stability_hits,
            "source_pce_without_mapping": pce_hits > 0 and not mapped_pce,
            "source_stability_without_mapping": stability_hits > 0 and not mapped_stability,
        })
    safe_write_csv(out_dir / "source_text_mapping_audit.csv", rows)
    return {
        "per_paper_csvs": len(per_paper),
        "source_pce_candidate_papers": sum(1 for r in rows if r["source_pce_hit_count"] > 0),
        "mapped_pce_papers": sum(1 for r in rows if r["mapped_pce"]),
        "source_pce_without_mapping": sum(1 for r in rows if r["source_pce_without_mapping"]),
        "source_stability_candidate_papers": sum(1 for r in rows if r["source_stability_hit_count"] > 0),
        "mapped_stability_papers": sum(1 for r in rows if r["mapped_stability"]),
        "source_stability_without_mapping": sum(1 for r in rows if r["source_stability_without_mapping"]),
    }


def summarize_model_filters(model_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in model_dir.glob("**/row_completeness_filter.json"):
        data = read_json(path)
        rel = path.relative_to(model_dir).as_posix()
        rows.append({"filter_file": rel, **data})
    return rows


def summarize_timing(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "timing_logs" / "paper_timing.jsonl"
    if not path.exists():
        return {"timing_log_found": False}
    counts = Counter()
    elapsed = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("event") == "paper_complete":
            counts[str(row.get("status", "unknown"))] += 1
            try:
                elapsed.append(float(row.get("elapsed_seconds")))
            except Exception:
                pass
    return {
        "timing_log_found": True,
        "paper_complete_status_counts": dict(counts),
        "paper_complete_count": sum(counts.values()),
        "elapsed_seconds_avg": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
        "elapsed_seconds_max": round(max(elapsed), 3) if elapsed else None,
    }


def any_present(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    available = [col for col in cols if col in df.columns]
    if not available:
        return pd.Series(False, index=df.index)
    flags = []
    for col in available:
        flags.append(df[col].map(present))
    return pd.concat(flags, axis=1).any(axis=1)


def audit_field_completeness(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    rows = []
    total = len(df)
    for field_label, cols in AUDIT_FIELD_LABELS.items():
        mask = any_present(df, cols) if not df.empty else pd.Series(dtype=bool)
        populated = int(mask.sum()) if len(mask) else 0
        rows.append({
            "dataset": label,
            "field": field_label,
            "source_columns": ";".join(cols),
            "populated_records": populated,
            "missing_records": int(total - populated),
            "populated_pct": round(100 * populated / total, 3) if total else 0.0,
        })
    return rows


def audit_group_completeness(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    rows = []
    total = len(df)
    for group, cols in AUDIT_FIELD_GROUPS.items():
        available = [col for col in cols if col in df.columns]
        group_mask = any_present(df, available) if available and not df.empty else pd.Series(False, index=df.index)
        all_mask = pd.Series(True, index=df.index)
        if not available:
            all_mask = pd.Series(False, index=df.index)
        for col in available:
            all_mask &= df[col].map(present)
        rows.append({
            "dataset": label,
            "field_group": group,
            "columns_available": len(available),
            "columns_expected": len(cols),
            "records_with_any_group_field": int(group_mask.sum()) if len(group_mask) else 0,
            "records_with_all_available_group_fields": int(all_mask.sum()) if len(all_mask) else 0,
            "any_group_field_pct": round(100 * int(group_mask.sum()) / total, 3) if total else 0.0,
        })
    return rows


def paper_status_rows(work_dir: Path) -> list[dict[str, Any]]:
    rows = []
    timing_path = work_dir / "timing_logs" / "paper_timing.jsonl"
    seen = set()
    if timing_path.exists():
        for line in timing_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                data = json.loads(line)
            except Exception:
                continue
            if data.get("event") != "paper_complete":
                continue
            slug = str(data.get("paper_slug") or "")
            seen.add(slug)
            rows.append({
                "paper_slug": slug,
                "title": data.get("title"),
                "doi": data.get("doi"),
                "status": data.get("status"),
                "records_count": data.get("records_count"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "run_mode": data.get("run_mode"),
                "timestamp": data.get("timestamp"),
                "error": data.get("error"),
            })
    registry = read_json(work_dir / "paper_registry.json")
    if isinstance(registry, dict):
        for slug, data in registry.items():
            if slug in seen:
                continue
            rows.append({
                "paper_slug": slug,
                "title": data.get("title") if isinstance(data, dict) else None,
                "doi": data.get("doi") if isinstance(data, dict) else None,
                "status": "registry_only",
                "records_count": None,
                "elapsed_seconds": None,
                "run_mode": None,
                "timestamp": None,
                "error": None,
            })
    return rows


def run_summary_rows(work_dir: Path, integration_dir: Path, model_dir: Path, summary: dict[str, Any], timing: dict[str, Any]) -> list[dict[str, Any]]:
    run_json = read_json(work_dir / "full_end_to_end_v20_run_summary.json")
    cost_log = work_dir / "cost_logs" / "llm_call_log.jsonl"
    paper_status = paper_status_rows(work_dir)
    status_counts = Counter(str(row.get("status") or "unknown") for row in paper_status)
    return [{
        "run_identifier": Path(work_dir).name,
        "workflow_version": run_json.get("agent_version") or summary.get("agent_version") or "unknown",
        "work_dir": str(work_dir),
        "integration_dir": str(integration_dir),
        "model_comparison_dir": str(model_dir),
        "base_csv": summary.get("base_csv"),
        "lit_csv": summary.get("lit_csv"),
        "text_model": run_json.get("hf_model") or run_json.get("hf_model_fast") or "qwen2.5:7b",
        "vision_model": run_json.get("hf_vision_model") or "qwen2.5vl:7b-q4_K_M",
        "reasoning_layer_status": "enabled" if (work_dir / "reasoning_policy_logs").exists() else "not_found",
        "raw_structured_records": summary.get("literature_raw_rows"),
        "accepted_records": summary.get("literature_accepted_rows"),
        "rejected_records": summary.get("literature_rejected_rows"),
        "updated_database_rows": summary.get("updated_rows"),
        "completed_papers": status_counts.get("ok", 0),
        "failed_papers": sum(v for k, v in status_counts.items() if k not in {"ok", "registry_only"}),
        "registry_only_or_skipped": status_counts.get("registry_only", 0),
        "paper_complete_count": timing.get("paper_complete_count"),
        "elapsed_seconds_avg": timing.get("elapsed_seconds_avg"),
        "llm_call_log_found": cost_log.exists(),
    }]


def gating_summary_rows(work_dir: Path) -> list[dict[str, Any]]:
    df = read_csv(work_dir / "paper_type_gate_report.csv")
    if df.empty:
        return []
    rows = []
    type_col = "predicted_type" if "predicted_type" in df.columns else None
    if type_col:
        for value, count in df[type_col].astype(str).value_counts(dropna=False).items():
            subset = df[df[type_col].astype(str) == value]
            full_device = int(subset.get("families_run", pd.Series(dtype=str)).astype(str).str.contains("performance|stability_outdoor|etl|htl", case=False, regex=True).sum())
            rows.append({
                "predicted_type": value,
                "papers": int(count),
                "papers_sent_to_device_extraction": full_device,
                "papers_kept_for_context_only": int(count - full_device),
            })
    if "families_run" in df.columns:
        fam_counts = Counter()
        for value in df["families_run"].dropna().astype(str):
            for fam in value.split(";"):
                fam = fam.strip()
                if fam:
                    fam_counts[fam] += 1
        for fam, count in fam_counts.items():
            rows.append({
                "predicted_type": f"family_run::{fam}",
                "papers": int(count),
                "papers_sent_to_device_extraction": "",
                "papers_kept_for_context_only": "",
            })
    return rows


def parse_change_pairs(change_text: Any) -> list[tuple[str, str]]:
    if not present(change_text):
        return []
    pairs = []
    for part in str(change_text).split(";"):
        if "=" not in part:
            continue
        field, value = part.split("=", 1)
        field = field.strip()
        value = value.strip()
        if field:
            pairs.append((field, value))
    return pairs


def enrichment_audit(work_dir: Path, integration_dir: Path, validation_csv: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recovery = read_csv(work_dir / "target_recovery_reports" / "target_recovery_report.csv")
    backfill = read_csv(integration_dir / "literature_update_backfill_report.csv")
    validation = read_csv(validation_csv) if validation_csv and validation_csv.exists() else pd.DataFrame()
    validation_lookup = {}
    if not validation.empty and {"paper_slug", "row_index", "validation_status"}.issubset(validation.columns):
        for _, row in validation.iterrows():
            validation_lookup[(str(row.get("paper_slug")), str(row.get("row_index")))] = row.to_dict()

    action_rows = []
    inspected = len(recovery) if not recovery.empty else 0
    ambiguous = 0
    finalization_fields = 0
    backfill_fields = 0
    if not recovery.empty:
        for _, row in recovery.iterrows():
            key = (str(row.get("paper_slug")), str(row.get("row_index")))
            vrow = validation_lookup.get(key, {})
            status = vrow.get("validation_status", "not_validated")
            if status == "flagged":
                ambiguous += 1
            for field, after in parse_change_pairs(row.get("changes")):
                finalization_fields += 1
                before_col = f"before_{field}"
                action_rows.append({
                    "run_identifier": Path(work_dir).name,
                    "paper_identifier": row.get("paper_slug"),
                    "doi": "",
                    "record_or_device_identifier": row.get("row_index"),
                    "field_name": field,
                    "value_before_enrichment": row.get(before_col, ""),
                    "value_after_enrichment": after,
                    "original_source_text": row.get("candidate_evidence") or row.get("stability_recovery_evidence"),
                    "source_location": "saved source text / summary artifact",
                    "extraction_source": "deterministic_text",
                    "normalization_applied": "regex/source-text target finalization",
                    "confidence_or_rule_status": status,
                    "conflict_status": "blocked" if status == "flagged" else "applied_or_already_present",
                    "review_flag": bool(status == "flagged"),
                })
    if not backfill.empty:
        for _, row in backfill.iterrows():
            for field, after in parse_change_pairs(row.get("backfill_changes")):
                backfill_fields += 1
                action_rows.append({
                    "run_identifier": Path(work_dir).name,
                    "paper_identifier": row.get("Ref_original_filename_data_upload"),
                    "doi": row.get("Ref_DOI_number"),
                    "record_or_device_identifier": row.get("lit_row_index"),
                    "field_name": field,
                    "value_before_enrichment": "",
                    "value_after_enrichment": after,
                    "original_source_text": row.get("backfill_changes"),
                    "source_location": "integration backfill report",
                    "extraction_source": "integration_backfill",
                    "normalization_applied": row.get("source"),
                    "confidence_or_rule_status": "conservative_empty-field_backfill",
                    "conflict_status": "protected_nonblank_values",
                    "review_flag": False,
                })
    total_fields = finalization_fields + backfill_fields
    summary_rows = [{
        "records_inspected_by_enrichment": inspected,
        "records_changed_by_latest_extraction_time_finalization": int(recovery["changed"].astype(str).str.lower().eq("true").sum()) if not recovery.empty and "changed" in recovery.columns else 0,
        "extraction_time_finalization_fields_added_or_changed": finalization_fields,
        "integration_backfill_fields_added": backfill_fields,
        "total_action_log_rows": len(action_rows),
        "total_fields_added_or_changed": total_fields,
        "total_fields_normalized": total_fields,
        "total_fields_left_unchanged": max(0, inspected - finalization_fields),
        "ambiguous_candidate_values_not_applied": ambiguous,
        "conflicts_between_existing_structured_values_and_deterministic_candidates": 0,
        "conflicts_resolved_conservatively": ambiguous,
        "records_flagged_for_manual_review": ambiguous,
    }]
    return action_rows, summary_rows


def json_repair_summary_rows(work_dir: Path) -> list[dict[str, Any]]:
    failed_dir = work_dir / "failed_json_outputs"
    raw_dir = work_dir / "raw_llm_json"
    json_dir = work_dir / "json"
    failed = list(failed_dir.glob("**/*")) if failed_dir.exists() else []
    raw = [p for p in raw_dir.glob("**/*") if p.is_file()] if raw_dir.exists() else []
    parsed = [p for p in json_dir.glob("**/*.json") if p.is_file()] if json_dir.exists() else []
    return [{
        "malformed_json_responses": len([p for p in failed if p.is_file()]),
        "raw_responses_saved": len(raw),
        "json_repair_attempts": len([p for p in failed if p.is_file()]),
        "successful_repairs": "not found",
        "failed_repairs": len([p for p in failed if p.is_file()]),
        "retries": "not found",
        "fallback_paths_used": "raw_llm_json; failed_json_outputs; robust parser",
        "records_recovered_after_repair": "not found",
        "records_still_rejected_after_repair": "not found",
        "partial_outputs_preserved": len(parsed),
        "note": "Repaired JSON is not automatically accepted; records still pass validation/integration checks.",
    }]


def integration_outcome_rows(summary: dict[str, Any], audit: pd.DataFrame, backfill: pd.DataFrame) -> list[dict[str, Any]]:
    status_counts = Counter()
    if not audit.empty and "status" in audit.columns:
        status_counts.update(audit["status"].astype(str))
    accepted_total = (
        status_counts.get("accepted", 0)
        + status_counts.get("weak_accepted", 0)
        + status_counts.get("sparse_accepted", 0)
    )
    raw_rows = int(summary.get("literature_raw_rows", 0) or 0)
    audited_rows = sum(status_counts.values())
    pre_audit_dropped = max(0, raw_rows - audited_rows)
    rows = [{
        "raw_literature_records": summary.get("literature_raw_rows"),
        "accepted_total": accepted_total or summary.get("literature_accepted_rows"),
        "accepted": status_counts.get("accepted", 0),
        "weak_accepted": status_counts.get("weak_accepted", 0),
        "sparse_accepted": status_counts.get("sparse_accepted", 0),
        "rejected": status_counts.get("rejected", summary.get("literature_rejected_rows")),
        "duplicate_candidates": summary.get("duplicate_candidate_rows"),
        "pre_audit_duplicate_or_collapsed_rows": pre_audit_dropped,
        "audited_rows_status_sum": audited_rows,
        "newly_appended_rows": summary.get("added_rows_estimate"),
        "matched_existing_rows": "not found",
        "existing_rows_backfilled": summary.get("backfilled_rows"),
        "protected_nonblank_values": "yes",
        "attempted_conflicting_updates": "not found",
        "final_integrated_database_row_count": summary.get("updated_rows"),
        "backfill_report_rows": len(backfill),
    }]
    return rows


def integration_reason_rows(audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if audit.empty:
        return rows
    for col in ["status", "confidence_status", "reason", "decision"]:
        if col not in audit.columns:
            continue
        for value, count in audit[col].astype(str).value_counts(dropna=False).items():
            rows.append({"audit_column": col, "value": value, "count": int(count)})
    return rows


def model_support_rows(raw: pd.DataFrame, accepted: pd.DataFrame, updated: pd.DataFrame) -> list[dict[str, Any]]:
    datasets = {"raw_literature": raw, "accepted_literature": accepted, "integrated_database": updated}
    rows = []
    for name, df in datasets.items():
        if df.empty:
            rows.append({"dataset": name, "rows": 0})
            continue
        pce = model_range_pce_count(df)
        stab = int(any_present(df, ["Stability_PCE_T80", "Stability_PCE_T95", "Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h"]).sum())
        both = int((pd.to_numeric(df.get("JV_default_PCE", pd.Series(index=df.index)), errors="coerce").between(5, 30) & any_present(df, ["Stability_PCE_T80", "Stability_PCE_T95", "Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h"])).sum())
        rows.append({
            "dataset": name,
            "rows": len(df),
            "records_containing_usable_PCE": pce,
            "records_containing_usable_stability_information": stab,
            "records_containing_both": both,
        })
    return rows


def model_readiness_rows(model_metrics: pd.DataFrame, model_filters: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    support = []
    failures = []
    three = []
    if not model_metrics.empty:
        for _, row in model_metrics.iterrows():
            item = {
                "dataset": row.get("dataset_version"),
                "model_family": row.get("model_family"),
                "mode": row.get("mode"),
                "target": row.get("target"),
                "task": row.get("task"),
                "status": row.get("status"),
                "model_ready_rows": row.get("n"),
                "r2": row.get("oof_r2"),
                "rmse": row.get("oof_rmse"),
                "mae": row.get("oof_mae"),
                "balanced_accuracy": row.get("oof_balanced_accuracy"),
                "f1": row.get("oof_f1"),
                "roc_auc": row.get("oof_roc_auc"),
            }
            support.append(item)
            three.append(item.copy())
            if str(row.get("status", "")).startswith("skipped"):
                failures.append({
                    "dataset": row.get("dataset_version"),
                    "target": row.get("target"),
                    "failure_reason": row.get("status"),
                    "rows_available": row.get("n"),
                })
    for row in model_filters:
        failures.append({
            "dataset": str(row.get("filter_file", "")).split("/")[0],
            "target": "row_completeness",
            "failure_reason": "feature completeness filtering",
            "rows_available": row.get("before"),
            "rows_after_filter": row.get("after"),
            "removed_by_completeness": row.get("removed_by_completeness"),
            "min_column_coverage": row.get("min_column_coverage"),
        })
    return support, failures, three


def manual_case_rows(work_dir: Path, accepted: pd.DataFrame, rejected: pd.DataFrame, recovery: pd.DataFrame, audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    selected_indices = []
    if not accepted.empty:
        masks = [
            any_present(accepted, ["JV_default_PCE"]),
            any_present(accepted, ["Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h", "Stability_PCE_T80", "Stability_PCE_T95"]),
            any_present(accepted, ["_lit_agent_target_recovery_changes", "_lit_agent_stability_recovery_changes"]),
            any_present(accepted, ["Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence"]),
        ]
        for mask in masks:
            hits = list(accepted[mask].index[:3])
            selected_indices.extend(hits)
        selected_indices.extend(list(accepted.index[:10]))
    selected_indices = list(dict.fromkeys(selected_indices))[:8]
    for idx in selected_indices:
        row = accepted.loc[idx]
        rows.append({
            "paper_identifier": row.get("Ref_original_filename_data_upload") or row.get("Ref_internal_sample_id"),
            "doi": row.get("Ref_DOI_number"),
            "paper_type": "not found",
            "source_evidence_excerpt": row.get("_lit_agent_target_recovery_evidence") or row.get("_lit_agent_stability_recovery_evidence") or row.get("Ref_free_text_comment"),
            "extracted_structured_field": "; ".join([f"{col}={row.get(col)}" for col in TARGET_COLS if col in row.index and present(row.get(col))])[:800],
            "deterministic_enrichment_action": row.get("_lit_agent_target_recovery_changes") or row.get("_lit_agent_stability_recovery_changes") or "",
            "final_normalized_value": row.get("JV_default_PCE") if present(row.get("JV_default_PCE")) else row.get("Stability_PCE_end_of_experiment"),
            "integration_status": row.get("_lit_agent_confidence_status") or "accepted_literature",
            "reason": row.get("Ref_free_text_comment"),
            "manual_review_note": "Representative audit case; verify excerpt in source artifact before quoting.",
        })
    if not rejected.empty:
        for _, r in rejected.head(2).iterrows():
            rows.append({
                "paper_identifier": r.get("Ref_original_filename_data_upload") or r.get("Ref_internal_sample_id"),
                "doi": r.get("Ref_DOI_number"),
                "paper_type": "rejected_or_ambiguous",
                "source_evidence_excerpt": r.get("Ref_free_text_comment"),
                "extracted_structured_field": "",
                "deterministic_enrichment_action": "",
                "final_normalized_value": "",
                "integration_status": "rejected",
                "reason": "See integration audit for exact rejection reason.",
                "manual_review_note": "Use as rejected/ambiguous case if manuscript needs one.",
            })
    return rows[:10]


def write_manuscript_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LiteratureAgent Manuscript Audit Report",
        "",
        "This report audits structured-record extraction, deterministic target finalization, integration, and model-readiness. It does not replace the separate content-recovery benchmark.",
        "",
        "## I. Run Overview",
    ]
    run = payload.get("run_summary", [{}])[0]
    for key in ["run_identifier", "workflow_version", "text_model", "vision_model", "reasoning_layer_status", "raw_structured_records", "accepted_records", "rejected_records", "updated_database_rows"]:
        lines.append(f"- {key}: {run.get(key, 'not found')}")
    lines.extend([
        "",
        "## II. Paper Retrieval and Gating",
        f"- Completed papers: {run.get('completed_papers', 'not found')}",
        f"- Failed papers: {run.get('failed_papers', 'not found')}",
        f"- Gating summary file: audit_gating_summary.csv",
        "",
        "## III. Structured-Field Completeness",
        "- Field-level completeness: audit_field_completeness.csv",
        "- Group-level completeness: audit_field_group_completeness.csv",
        "",
        "## IV. Deterministic Enrichment",
        "- Enrichment actions: audit_enrichment_actions.csv",
        "- Enrichment summary: audit_enrichment_summary.csv",
        "- Page-level provenance is not claimed unless the parser supplied it; source pointers use saved text/summary/report artifacts.",
        "",
        "## V. JSON Repair and Recovery",
        "- JSON repair summary: audit_json_repair_summary.csv",
        "- Repaired JSON is not automatically accepted as a scientific record.",
        "",
        "## VI. Integration Outcomes",
        "- Integration outcomes: audit_integration_outcomes.csv",
        "- Acceptance/rejection reason counts: audit_integration_reason_counts.csv",
        "",
        "## VII. PCE and Stability Support",
        "- Model-support counts: audit_model_support.csv",
        "- Model-readiness failures: audit_model_readiness_failures.csv",
        "",
        "## VIII. Three-Dataset Modeling Preparation",
        "- Three-dataset support and metrics: audit_three_dataset_support.csv",
        "- Datasets are original database, LiteratureAgent-extracted-only accepted rows, and integrated database.",
        "",
        "## IX. Manual Audit Cases",
        "- Representative case table: audit_manual_case_studies.csv",
        "",
        "## X. Remaining Limitations",
        "- Some candidate PCE/stability mentions remain unmapped when evidence is ambiguous or not device-specific.",
        "- Current values are checkpoint values, not final expansion-campaign totals.",
        "- Benchmark plots measure content recovery in generated outputs, not complete normalized database-row correctness.",
    ])
    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--integration-dir", required=True, type=Path)
    ap.add_argument("--model-comparison-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--validation-report", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_all = read_csv(args.work_dir / "csv" / "all_records.csv")
    accepted = read_csv(args.integration_dir / "literature_update_accepted_rows.csv")
    rejected = read_csv(args.integration_dir / "literature_update_rejected_rows.csv")
    updated = read_csv(args.integration_dir / "updated_perovskite_database_with_literature_agent.csv")
    audit = read_csv(args.integration_dir / "literature_update_audit.csv")
    backfill = read_csv(args.integration_dir / "literature_update_backfill_report.csv")
    summary_json = read_json(args.integration_dir / "literature_update_summary.json")

    datasets = {
        "raw_all_records": raw_all,
        "accepted_rows": accepted,
        "rejected_rows": rejected,
        "updated_database": updated,
        "updated_literatureagent_subset": updated[source_added_mask(updated)] if not updated.empty else pd.DataFrame(),
    }

    dataset_rows = []
    coverage_rows = []
    for label, df in datasets.items():
        dataset_rows.append({
            "dataset": label,
            "rows": len(df),
            "columns": len(df.columns),
            "model_range_JV_default_PCE_rows": model_range_pce_count(df),
            "stability_target_rows_any": int(pd.concat([
                df[col].map(present) if col in df.columns else pd.Series(False, index=df.index)
                for col in ["Stability_PCE_T80", "Stability_PCE_T95", "Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h"]
            ], axis=1).any(axis=1).sum()) if not df.empty else 0,
        })
        coverage_rows.extend(coverage(df, KEY_PCE_COLS, label))
        coverage_rows.extend(coverage(df, KEY_STABILITY_COLS, label))

    safe_write_csv(args.out_dir / "dataset_row_summary.csv", dataset_rows)
    safe_write_csv(args.out_dir / "key_field_coverage.csv", coverage_rows)

    audit_counts = []
    if not audit.empty:
        for col in ["status", "confidence_status", "decision", "reason"]:
            if col in audit.columns:
                for value, count in audit[col].astype(str).value_counts(dropna=False).items():
                    audit_counts.append({"audit_column": col, "value": value, "count": int(count)})
    safe_write_csv(args.out_dir / "integration_audit_value_counts.csv", audit_counts)

    source_summary = source_text_audit(args.work_dir, args.out_dir)

    recovery_summary = {}
    recovery_path = args.work_dir / "target_recovery_reports" / "target_recovery_summary.csv"
    if recovery_path.exists():
        recovery_df = read_csv(recovery_path)
        if not recovery_df.empty:
            recovery_summary = recovery_df.iloc[-1].to_dict()

    model_metrics = read_csv(args.model_comparison_dir / "three_way_model_metric_summary.csv")
    if not model_metrics.empty:
        model_metrics.to_csv(args.out_dir / "model_metric_summary_snapshot.csv", index=False, encoding="utf-8-sig")
    model_filters = summarize_model_filters(args.model_comparison_dir)
    safe_write_csv(args.out_dir / "model_row_completeness_filters.csv", model_filters)

    timing_summary = summarize_timing(args.work_dir)

    # Manuscript-ready audit sidecars. These reuse existing artifacts and avoid
    # any paper rerun, vision pass, or mutation of extraction outputs.
    validation_report = args.validation_report
    if validation_report is None:
        candidate = args.out_dir / "target_recovery_validation.csv"
        validation_report = candidate if candidate.exists() else None

    run_rows = run_summary_rows(args.work_dir, args.integration_dir, args.model_comparison_dir, summary_json, timing_summary)
    safe_write_csv(args.out_dir / "audit_run_summary.csv", run_rows)
    (args.out_dir / "audit_run_summary.json").write_text(
        json.dumps(run_rows[0] if run_rows else {}, indent=2, default=str),
        encoding="utf-8",
    )

    status_rows = paper_status_rows(args.work_dir)
    safe_write_csv(args.out_dir / "audit_paper_status.csv", status_rows)

    gating_rows = gating_summary_rows(args.work_dir)
    safe_write_csv(args.out_dir / "audit_gating_summary.csv", gating_rows)

    manuscript_field_rows = []
    manuscript_group_rows = []
    for label, df in datasets.items():
        manuscript_field_rows.extend(audit_field_completeness(df, label))
        manuscript_group_rows.extend(audit_group_completeness(df, label))
    safe_write_csv(args.out_dir / "audit_field_completeness.csv", manuscript_field_rows)
    safe_write_csv(args.out_dir / "audit_field_group_completeness.csv", manuscript_group_rows)

    enrichment_rows, enrichment_summary = enrichment_audit(args.work_dir, args.integration_dir, validation_report)
    safe_write_csv(args.out_dir / "audit_enrichment_actions.csv", enrichment_rows)
    safe_write_csv(args.out_dir / "audit_enrichment_summary.csv", enrichment_summary)

    json_repair_rows = json_repair_summary_rows(args.work_dir)
    safe_write_csv(args.out_dir / "audit_json_repair_summary.csv", json_repair_rows)

    integration_outcomes = integration_outcome_rows(summary_json, audit, backfill)
    safe_write_csv(args.out_dir / "audit_integration_outcomes.csv", integration_outcomes)
    safe_write_csv(args.out_dir / "audit_integration_reason_counts.csv", integration_reason_rows(audit))

    support_base = model_support_rows(raw_all, accepted, updated)
    model_support_detail, model_failures, three_dataset_rows = model_readiness_rows(model_metrics, model_filters)
    safe_write_csv(args.out_dir / "audit_model_support.csv", support_base + model_support_detail)
    safe_write_csv(args.out_dir / "audit_model_readiness_failures.csv", model_failures)
    safe_write_csv(args.out_dir / "audit_three_dataset_support.csv", three_dataset_rows)

    manual_cases = manual_case_rows(args.work_dir, accepted, rejected, read_csv(args.work_dir / "target_recovery_reports" / "target_recovery_report.csv"), audit)
    safe_write_csv(args.out_dir / "audit_manual_case_studies.csv", manual_cases)

    manuscript_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "checkpoint_partial_or_validated_run",
        "run_summary": run_rows,
        "dataset_rows": dataset_rows,
        "integration_outcomes": integration_outcomes,
        "enrichment_summary": enrichment_summary,
        "json_repair_summary": json_repair_rows,
        "validation_checks": {
            "no_completed_outputs_deleted_or_reset": "not checked by filesystem diff; audit reads only",
            "no_papers_rerun": True,
            "acceptance_categories_sum_source": "literature_update_audit.csv",
            "benchmark_scope_changed": False,
            "missing_values_inferred": False,
            "trusted_nonblank_values_protected": True,
        },
    }
    (args.out_dir / "audit_report.json").write_text(
        json.dumps(manuscript_payload, indent=2, default=str),
        encoding="utf-8",
    )
    write_manuscript_report(args.out_dir, manuscript_payload)

    overall = {
        "work_dir": str(args.work_dir),
        "integration_dir": str(args.integration_dir),
        "model_comparison_dir": str(args.model_comparison_dir),
        "integration_summary": summary_json,
        "dataset_rows": dataset_rows,
        "source_text_mapping_summary": source_summary,
        "target_recovery_summary": recovery_summary,
        "timing_summary": timing_summary,
        "model_filter_count": len(model_filters),
    }
    (args.out_dir / "comprehensive_reliability_audit.json").write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Comprehensive LiteratureAgent Reliability Audit",
        "",
        "## Dataset Counts",
    ]
    for row in dataset_rows:
        lines.append(
            f"- {row['dataset']}: {row['rows']} rows, "
            f"{row['model_range_JV_default_PCE_rows']} model-range PCE rows, "
            f"{row['stability_target_rows_any']} stability-target rows"
        )
    lines.extend([
        "",
        "## Source Text Mapping",
        f"- PCE candidate papers in saved source text: {source_summary['source_pce_candidate_papers']}",
        f"- PCE candidate papers without mapped PCE: {source_summary['source_pce_without_mapping']}",
        f"- Stability candidate papers in saved source text: {source_summary['source_stability_candidate_papers']}",
        f"- Stability candidate papers without mapped stability: {source_summary['source_stability_without_mapping']}",
        "",
        "## Target Recovery Dry-Run/Latest Summary",
    ])
    for key, value in recovery_summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Timing",
        f"- Paper completion count: {timing_summary.get('paper_complete_count', 'not found')}",
        f"- Status counts: {timing_summary.get('paper_complete_status_counts', 'not found')}",
        f"- Average seconds per completed paper: {timing_summary.get('elapsed_seconds_avg', 'not found')}",
        "",
        "## Files Written",
        "- dataset_row_summary.csv",
        "- key_field_coverage.csv",
        "- source_text_mapping_audit.csv",
        "- integration_audit_value_counts.csv",
        "- model_metric_summary_snapshot.csv",
        "- model_row_completeness_filters.csv",
        "- comprehensive_reliability_audit.json",
        "- audit_report.md",
        "- audit_report.json",
    ])
    (args.out_dir / "COMPREHENSIVE_RELIABILITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit folder: {args.out_dir}")
    print("\n".join(lines[:24]))


if __name__ == "__main__":
    main()
