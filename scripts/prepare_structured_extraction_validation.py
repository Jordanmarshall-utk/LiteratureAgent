from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "data_provenance"
    / "cumulative_expansion"
)
DEFAULT_OUT = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "structured_extraction_validation_v1"
)

CORE_FIELDS = {
    "identity": [
        "Ref_DOI_number",
        "Ref_internal_sample_id",
    ],
    "device": [
        "Cell_stack_sequence",
        "Cell_architecture",
        "Cell_area_measured",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
    ],
    "composition": [
        "Perovskite_composition_short_form",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions",
        "Perovskite_additives_compounds",
    ],
    "processing": [
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_solvents",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
    ],
    "performance": [
        "JV_default_PCE",
        "JV_default_Voc",
        "JV_default_Jsc",
        "JV_default_FF",
    ],
    "stability": [
        "Stability_protocol",
        "Stability_time_total_exposure",
        "Stability_temperature_range",
        "Stability_relative_humidity_average_value",
        "Stability_light_intensity",
        "Stability_PCE_initial_value",
        "Stability_PCE_end_of_experiment",
        "Stability_PCE_T80",
        "Stability_PCE_T95",
    ],
}

PCE_FIELDS = [
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "Stabilised_performance_PCE",
]

STABILITY_TARGET_FIELDS = [
    "Stability_PCE_T80",
    "Stability_PCE_T95",
    "Stability_PCE_Ts80",
    "Stability_PCE_Ts95",
    "Stability_PCE_Te80",
    "Stability_PCE_Tse80",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_after_1000_h",
    "Outdoor_PCE_T80",
    "Outdoor_PCE_T95",
    "Outdoor_PCE_end_of_experiment",
    "Outdoor_PCE_after_1000_h",
]

EXCLUDED_POPULATED_PREFIXES = (
    "_",
    "Ref_ID",
    "Ref_name_of_person",
    "Ref_data_entered",
    "Ref_free_text_comment",
)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def numeric(value: object) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_pce_candidate(row: pd.Series) -> bool:
    for field in PCE_FIELDS:
        value = numeric(row.get(field))
        if value is not None and 1.0 <= value <= 35.0:
            return True
    return False


def is_stability_candidate(row: pd.Series) -> bool:
    return any(numeric(row.get(field)) is not None for field in STABILITY_TARGET_FIELDS)


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = pd.read_csv(
        PROVENANCE_DIR / "literature_update_accepted_rows.csv",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    rejected = pd.read_csv(
        PROVENANCE_DIR / "literature_update_rejected_rows.csv",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    accepted["integration_status"] = accepted.get(
        "_lit_agent_confidence_status", "accepted"
    ).replace("", "accepted")
    rejected["integration_status"] = "rejected"
    accepted["record_origin"] = "accepted_rows"
    rejected["record_origin"] = "rejected_rows"
    records = pd.concat([accepted, rejected], ignore_index=True, sort=False).fillna("")
    records["pce_candidate"] = records.apply(is_pce_candidate, axis=1)
    records["stability_candidate"] = records.apply(is_stability_candidate, axis=1)

    evidence = pd.read_csv(
        PROVENANCE_DIR / "literature_update_evidence_long.csv",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    ).fillna("")
    return records, evidence


def select_rows(
    records: pd.DataFrame,
    *,
    random_state: int,
    max_per_doi: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    doi_counts: defaultdict[str, int] = defaultdict(int)

    strata = [
        (
            "accepted_pce",
            20,
            records["integration_status"].eq("accepted")
            & records["pce_candidate"],
        ),
        (
            "accepted_stability",
            10,
            records["integration_status"].eq("accepted")
            & records["stability_candidate"],
        ),
        (
            "sparse_accepted",
            15,
            records["integration_status"].eq("sparse_accepted"),
        ),
        (
            "rejected",
            10,
            records["integration_status"].eq("rejected"),
        ),
        (
            "accepted_general",
            5,
            records["integration_status"].eq("accepted"),
        ),
    ]

    stratum_for_index: dict[int, str] = {}
    for stratum, target, mask in strata:
        candidates = [i for i in records.index[mask] if i not in selected]
        rng.shuffle(candidates)
        chosen = 0
        for index in candidates:
            doi = clean_text(records.at[index, "Ref_DOI_number"]).lower()
            group = doi or f"missing-doi-{index}"
            if doi_counts[group] >= max_per_doi:
                continue
            selected.append(index)
            stratum_for_index[index] = stratum
            doi_counts[group] += 1
            chosen += 1
            if chosen == target:
                break
        if chosen < target:
            raise RuntimeError(
                f"Could only select {chosen} of {target} records for {stratum}."
            )

    sample = records.loc[selected].copy().reset_index(drop=True)
    sample["validation_record_id"] = [f"VAL{i:03d}" for i in range(1, len(sample) + 1)]
    sample["validation_stratum"] = [stratum_for_index[i] for i in selected]
    return sample


def evidence_lookup(evidence: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for row in evidence.itertuples(index=False):
        key = (
            clean_text(getattr(row, "Ref_DOI_number", "")).lower(),
            clean_text(getattr(row, "Ref_internal_sample_id", "")),
            clean_text(getattr(row, "field", "")),
        )
        excerpt = clean_text(getattr(row, "evidence", ""))
        if key[0] and key[2] and excerpt and key not in lookup:
            lookup[key] = excerpt[:32000]
    return lookup


def build_annotation_tables(
    sample: pd.DataFrame,
    evidence: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    lookup = evidence_lookup(evidence)
    core_rows: list[dict[str, object]] = []
    populated_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []

    manifest_columns = [
        "validation_record_id",
        "validation_stratum",
        "integration_status",
        "pce_candidate",
        "stability_candidate",
        "Ref_DOI_number",
        "Ref_internal_sample_id",
        "Ref_lead_author",
        "Ref_publication_date",
        "Ref_journal",
        "_source_pdf_url",
        "_source_landing_page",
        "_full_text_source",
        "_lit_agent_row_hash",
        "_lit_agent_metric_consistency_warning",
        "_lit_agent_sample_identity_basis",
    ]
    for column in manifest_columns:
        if column not in sample.columns:
            sample[column] = ""
    manifest = sample[manifest_columns].copy()

    core_field_names = {field for fields in CORE_FIELDS.values() for field in fields}
    schema_columns = [
        column
        for column in sample.columns
        if column not in {
            "validation_record_id",
            "validation_stratum",
            "integration_status",
            "record_origin",
            "pce_candidate",
            "stability_candidate",
        }
        and not column.startswith(EXCLUDED_POPULATED_PREFIXES)
    ]

    for record in sample.to_dict(orient="records"):
        record_id = record["validation_record_id"]
        doi = clean_text(record.get("Ref_DOI_number")).lower()
        sample_id = clean_text(record.get("Ref_internal_sample_id"))
        base = {
            "validation_record_id": record_id,
            "validation_stratum": record["validation_stratum"],
            "integration_status": record["integration_status"],
            "Ref_DOI_number": record.get("Ref_DOI_number", ""),
            "Ref_internal_sample_id": sample_id,
        }
        for family, fields in CORE_FIELDS.items():
            for field in fields:
                core_rows.append(
                    {
                        **base,
                        "field_family": family,
                        "field": field,
                        "extracted_value": clean_text(record.get(field)),
                        "saved_evidence_excerpt": lookup.get((doi, sample_id, field), ""),
                        "r1_present_in_source_yes_no": "",
                        "r1_gold_value": "",
                        "r1_extracted_value_correct_yes_no_partial": "",
                        "r1_evidence_supports_value_yes_no_partial": "",
                        "r1_notes": "",
                        "r2_present_in_source_yes_no": "",
                        "r2_gold_value": "",
                        "r2_extracted_value_correct_yes_no_partial": "",
                        "r2_evidence_supports_value_yes_no_partial": "",
                        "r2_notes": "",
                        "adjudicated_present_in_source_yes_no": "",
                        "adjudicated_gold_value": "",
                        "adjudicated_extracted_value_correct_yes_no_partial": "",
                        "adjudicated_evidence_supports_value_yes_no_partial": "",
                    }
                )

        for field in schema_columns:
            value = clean_text(record.get(field))
            if not value or field in core_field_names:
                continue
            populated_rows.append(
                {
                    **base,
                    "field": field,
                    "extracted_value": value,
                    "saved_evidence_excerpt": lookup.get((doi, sample_id, field), ""),
                    "r1_correct_yes_no_partial": "",
                    "r1_supported_yes_no_partial": "",
                    "r1_notes": "",
                    "r2_correct_yes_no_partial": "",
                    "r2_supported_yes_no_partial": "",
                    "r2_notes": "",
                    "adjudicated_correct_yes_no_partial": "",
                    "adjudicated_supported_yes_no_partial": "",
                }
            )

        matched = evidence[
            evidence["Ref_DOI_number"].str.lower().eq(doi)
            & evidence["Ref_internal_sample_id"].eq(sample_id)
        ]
        for item in matched.to_dict(orient="records"):
            evidence_rows.append(
                {
                    **base,
                    "field": item.get("field", ""),
                    "evidence": clean_text(item.get("evidence"))[:32000],
                    "lit_row_index": item.get("lit_row_index", ""),
                }
            )

    record_review = manifest[
        [
            "validation_record_id",
            "validation_stratum",
            "integration_status",
            "Ref_DOI_number",
            "Ref_internal_sample_id",
        ]
    ].copy()
    for prefix in ("r1", "r2", "adjudicated"):
        record_review[f"{prefix}_source_recovered_yes_no"] = ""
        record_review[f"{prefix}_sample_identity_correct_yes_no"] = ""
        record_review[f"{prefix}_sample_count_correct_yes_no"] = ""
        record_review[f"{prefix}_complete_core_row_correct_yes_no"] = ""
        record_review[f"{prefix}_complete_populated_row_correct_yes_no"] = ""
        record_review[f"{prefix}_recommended_audit_status"] = ""
        record_review[f"{prefix}_notes"] = ""

    decision_rules = pd.DataFrame(
        [
            {
                "status": "accepted",
                "operational_rule": (
                    "DOI present; at least one plausible PCE target (1-35%) or a numeric "
                    "stability target; at least four important fields; not metadata-only."
                ),
            },
            {
                "status": "sparse_accepted",
                "operational_rule": (
                    "Scientifically useful performance/stability/device context is present, "
                    "but fewer than four important fields or no direct model target is present."
                ),
            },
            {
                "status": "weak_accepted",
                "operational_rule": (
                    "Otherwise accepted, but fewer than the configured minimum evidence fields."
                ),
            },
            {
                "status": "rejected",
                "operational_rule": (
                    "Required DOI missing or record is metadata-only/too sparse: no performance or "
                    "stability field and insufficient design context."
                ),
            },
        ]
    )

    instructions = pd.DataFrame(
        {
            "step": range(1, 10),
            "instruction": [
                "Review source papers without consulting the other reviewer's decisions.",
                "Confirm DOI and sample/device identity before scoring individual fields.",
                "Score every core field, including blank extracted values, to measure false negatives.",
                "Use yes/no/partial exactly; reserve partial for normalized values that preserve meaning.",
                "A populated value is supported only when the saved excerpt or paper directly supports it.",
                "Review every additional populated field for hallucination and value accuracy.",
                "Mark complete-row correctness only when identity and every assessed field are correct.",
                "Assign an independent accepted/sparse_accepted/weak_accepted/rejected decision.",
                "Adjudicate disagreements only after both reviewers finish their blinded columns.",
            ],
        }
    )

    metric_definitions = pd.DataFrame(
        [
            {"metric": "field_precision", "definition": "TP / (TP + FP) across core fields"},
            {"metric": "field_recall", "definition": "TP / (TP + FN) across core fields"},
            {"metric": "field_f1", "definition": "Harmonic mean of field precision and recall"},
            {"metric": "unsupported_value_rate", "definition": "Populated fields scored unsupported / populated fields assessed"},
            {"metric": "sample_linking_accuracy", "definition": "Records with correct sample identity / records assessed"},
            {"metric": "complete_core_row_accuracy", "definition": "Records with all core fields and identity correct / records assessed"},
            {"metric": "complete_populated_row_accuracy", "definition": "Records with every populated field and identity correct / records assessed"},
            {"metric": "audit_decision_accuracy", "definition": "System audit status agreeing with adjudicated status / records assessed"},
            {"metric": "reviewer_agreement", "definition": "Percent agreement and Cohen kappa on doubly reviewed fields"},
        ]
    )

    return {
        "instructions": instructions,
        "sample_manifest": manifest,
        "core_field_review": pd.DataFrame(core_rows),
        "populated_field_review": pd.DataFrame(populated_rows),
        "record_review": record_review,
        "evidence_lookup": pd.DataFrame(evidence_rows),
        "decision_rules": decision_rules,
        "metric_definitions": metric_definitions,
    }


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            safe_table = table.copy()
            for column in safe_table.select_dtypes(include="object").columns:
                safe_table[column] = safe_table[column].map(
                    lambda value: re.sub(
                        r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", clean_text(value)
                    )
                )
            safe_table.to_excel(writer, sheet_name=name[:31], index=False)
            worksheet = writer.book[name[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [clean_text(cell.value) for cell in column_cells[:80]]
                width = min(max([len(value) for value in values] + [10]) + 2, 55)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-per-doi", type=int, default=2)
    args = parser.parse_args()

    records, evidence = load_records()
    sample = select_rows(
        records,
        random_state=args.random_state,
        max_per_doi=args.max_per_doi,
    )
    tables = build_annotation_tables(sample, evidence)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    workbook = args.out_dir / "structured_extraction_gold_standard_annotation_v1.xlsx"
    write_workbook(tables, workbook)
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)

    manifest = {
        "records": int(len(sample)),
        "unique_dois": int(sample["Ref_DOI_number"].str.lower().nunique()),
        "random_state": args.random_state,
        "max_records_per_doi": args.max_per_doi,
        "strata": sample["validation_stratum"].value_counts().to_dict(),
        "core_field_annotations": int(len(tables["core_field_review"])),
        "additional_populated_field_annotations": int(
            len(tables["populated_field_review"])
        ),
        "workbook": str(workbook),
        "status": "awaiting_blinded_human_annotation",
    }
    (args.out_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
