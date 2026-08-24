#!/usr/bin/env python
"""Audit whether a LiteratureAgent expansion batch adds model-ready targets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PCE_TARGETS = ["JV_default_PCE", "JV_reverse_scan_PCE", "JV_forward_scan_PCE", "Stabilised_performance_PCE"]
COMPOSITION_FIELDS = ["Perovskite_composition_long_form", "Perovskite_composition_short_form"]
DEVICE_FIELDS = ["Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence"]
PROCESS_FIELDS = [
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_aggregation_state_of_reactants",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_solvents_mixing_ratios",
    "Perovskite_deposition_quenching_induced_crystallisation",
    "Perovskite_deposition_quenching_media",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_annealing_temperature",
]
STABILITY_TARGETS = [
    "Stability_PCE_T80",
    "Stability_PCE_T95",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_after_1000_h",
]
STABILITY_TIME_FIELDS = ["Stability_time_total_exposure", "Stability_PCE_T80", "Stability_PCE_T95"]
STABILITY_CONDITION_FIELDS = [
    "Stability_temperature_range",
    "Stability_relative_humidity_average_value",
    "Stability_light_intensity",
    "Stability_atmosphere",
    "Stability_encapsulation",
]


def _present(df: pd.DataFrame, names: list[str]) -> pd.Series:
    available = [name for name in names if name in df.columns]
    if not available:
        return pd.Series(False, index=df.index)
    values = df[available].copy()
    return values.apply(
        lambda col: col.notna() & col.astype(str).str.strip().ne("") & col.astype(str).str.lower().ne("nan")
    ).any(axis=1)


def _numeric_present(df: pd.DataFrame, names: list[str], low: float | None = None, high: float | None = None) -> pd.Series:
    available = [name for name in names if name in df.columns]
    if not available:
        return pd.Series(False, index=df.index)
    result = pd.Series(False, index=df.index)
    for name in available:
        values = pd.to_numeric(df[name], errors="coerce")
        valid = values.notna()
        if low is not None:
            valid &= values >= low
        if high is not None:
            valid &= values <= high
        result |= valid
    return result


def eligibility(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pce_target_available"] = _numeric_present(df, PCE_TARGETS, 0.01, 40)
    out["composition_available"] = _present(df, COMPOSITION_FIELDS)
    out["device_available"] = _present(df, DEVICE_FIELDS)
    out["processing_available"] = _present(df, PROCESS_FIELDS)
    out["pce_model_usable_target"] = out["pce_target_available"]
    out["pce_context_partial"] = out["pce_target_available"] & (
        out["composition_available"] | out["device_available"] | out["processing_available"]
    )
    out["pce_model_ready_minimal"] = out["pce_target_available"] & (
        out["composition_available"] | out["device_available"]
    )
    out["pce_model_ready_strict"] = (
        out["pce_target_available"]
        & out["composition_available"]
        & out["device_available"]
        & out["processing_available"]
    )

    out["stability_target_available"] = _numeric_present(df, STABILITY_TARGETS, 0)
    out["stability_time_available"] = _numeric_present(df, STABILITY_TIME_FIELDS, 0)
    out["stability_conditions_available"] = _present(df, STABILITY_CONDITION_FIELDS)
    out["stability_model_usable_target"] = out["stability_target_available"] & (
        out["stability_time_available"] | _numeric_present(df, ["Stability_PCE_after_1000_h"], 0)
    )
    out["stability_model_ready_minimal"] = (
        out["stability_model_usable_target"]
        & (out["composition_available"] | out["device_available"] | out["stability_conditions_available"])
    )
    out["stability_model_ready_strict"] = (
        out["stability_model_ready_minimal"] & out["device_available"] & out["stability_conditions_available"]
    )
    out["physical_stability_ready"] = (
        out["stability_target_available"]
        & _present(df, ["Stability_temperature_range"])
        & _present(df, ["Stability_relative_humidity_average_value"])
        & _present(df, ["Stability_light_intensity"])
    )
    return out


def _read(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _count_frame(name: str, df: pd.DataFrame) -> dict[str, int | str]:
    flags = eligibility(df) if not df.empty else pd.DataFrame()
    row: dict[str, int | str] = {"dataset": name, "rows": len(df)}
    for column in flags.columns:
        row[column] = int(flags[column].sum())
    return row


def identity_consistency_issues(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_index",
        "issue",
        "doi",
        "sample_id",
        "sample_id_pce",
        "record_pce",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    issues: list[dict[str, object]] = []
    non_scientific_columns = {
        "Ref_DOI_number",
        "Ref_internal_sample_id",
        "status",
        "row_tier",
        "pce_model_candidate",
        "stability_model_candidate",
        "has_design_context",
        "reasons",
    }
    scientific_columns = sorted(
        column
        for column in df.columns
        if column not in non_scientific_columns
        and not column.startswith(("_", "Ref_"))
        and "evidence" not in column.lower()
        and "confidence" not in column.lower()
    )

    def text(row: pd.Series, name: str) -> str:
        value = row.get(name, "")
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def first_pce(row: pd.Series) -> float | None:
        for field in PCE_TARGETS:
            value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
        return None

    final_identity_groups: dict[tuple[str, ...], list[int]] = {}
    for index, row in df.reset_index(drop=True).iterrows():
        sample_id = text(row, "Ref_internal_sample_id")
        record_pce = first_pce(row)
        match = re.search(
            r"_pce_([0-9]+(?:p[0-9]+)?)(?:_[0-9]+)?$",
            sample_id,
            flags=re.IGNORECASE,
        )
        if match and record_pce is not None:
            sample_id_pce = float(match.group(1).replace("p", "."))
            if abs(sample_id_pce - record_pce) > 0.02:
                issues.append({
                    "row_index": index,
                    "issue": "sample_id_pce_mismatch",
                    "doi": text(row, "Ref_DOI_number"),
                    "sample_id": sample_id,
                    "sample_id_pce": sample_id_pce,
                    "record_pce": record_pce,
                })

        base_sample_id = re.sub(
            r"_(?:control|target|device|certified|large_area)"
            r"(?:_pce_[0-9mp]+(?:_[0-9]+)?)?$",
            "",
            sample_id,
            flags=re.IGNORECASE,
        )
        key = (
            text(row, "Ref_DOI_number"),
            base_sample_id,
            *(text(row, field) for field in scientific_columns),
        )
        if key[0] and key[1] and any(key[2:]):
            final_identity_groups.setdefault(key, []).append(index)

    for indices in final_identity_groups.values():
        if len(indices) < 2:
            continue
        for index in indices:
            row = df.reset_index(drop=True).iloc[index]
            issues.append({
                "row_index": index,
                "issue": "duplicate_final_device_identity",
                "doi": text(row, "Ref_DOI_number"),
                "sample_id": text(row, "Ref_internal_sample_id"),
                "sample_id_pce": "",
                "record_pce": first_pce(row),
            })

    return pd.DataFrame(issues, columns=columns)


def _plot_support(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "pce_model_usable_target",
        "pce_model_ready_minimal",
        "pce_model_ready_strict",
        "stability_model_usable_target",
        "stability_model_ready_minimal",
        "stability_model_ready_strict",
        "physical_stability_ready",
    ]
    compare = summary[summary["dataset"].isin(["before", "after"])].set_index("dataset")
    if len(compare) != 2:
        return
    plot_df = compare[metrics].T
    plot_df.plot(kind="bar", figsize=(10, 6), color=["#587b8c", "#d07a52"])
    plt.title("Model-ready target support before and after expansion")
    plt.ylabel("Eligible rows")
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "target_support_before_after.png", dpi=220)
    plt.close()

    delta = plot_df["after"] - plot_df["before"]
    delta.plot(kind="barh", figsize=(9, 5), color="#4f8f6b")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Net model-ready target rows added")
    plt.xlabel("Change in eligible rows")
    plt.tight_layout()
    plt.savefig(out_dir / "target_support_delta.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--updated-csv", type=Path, required=True)
    parser.add_argument("--raw-literature-csv", type=Path)
    parser.add_argument("--accepted-csv", type=Path)
    parser.add_argument("--rejected-csv", type=Path)
    parser.add_argument("--paper-type-report", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "before": _read(args.base_csv),
        "after": _read(args.updated_csv),
        "raw_literature": _read(args.raw_literature_csv),
        "accepted_literature": _read(args.accepted_csv),
        "rejected_literature": _read(args.rejected_csv),
    }
    summary = pd.DataFrame([_count_frame(name, frame) for name, frame in datasets.items()]).fillna(0)
    summary.to_csv(args.out_dir / "target_support_summary.csv", index=False)
    _plot_support(summary, args.out_dir)

    accepted = datasets["accepted_literature"]
    identity_issues = identity_consistency_issues(accepted)
    identity_issues.to_csv(args.out_dir / "device_identity_consistency_issues.csv", index=False)
    if not accepted.empty:
        flags = eligibility(accepted)
        pd.concat([accepted.reset_index(drop=True), flags.reset_index(drop=True)], axis=1).to_csv(
            args.out_dir / "accepted_rows_with_model_eligibility.csv", index=False
        )

    paper_types = _read(args.paper_type_report)
    if not paper_types.empty:
        type_col = next((c for c in ["predicted_type", "paper_type"] if c in paper_types.columns), None)
        if type_col:
            paper_types[type_col].value_counts(dropna=False).rename_axis("paper_type").reset_index(
                name="papers"
            ).to_csv(args.out_dir / "paper_type_summary.csv", index=False)

    rows = summary.set_index("dataset")
    before = rows.loc["before"]
    after = rows.loc["after"]
    accepted_row = rows.loc["accepted_literature"]
    report = {
        "batch_name": args.batch_name,
        "raw_literature_rows": int(rows.loc["raw_literature", "rows"]),
        "accepted_rows": int(accepted_row["rows"]),
        "rejected_rows": int(rows.loc["rejected_literature", "rows"]),
        "accepted_pce_usable_target_rows": int(accepted_row.get("pce_model_usable_target", 0)),
        "accepted_pce_model_ready_minimal": int(accepted_row.get("pce_model_ready_minimal", 0)),
        "accepted_pce_model_ready_strict": int(accepted_row.get("pce_model_ready_strict", 0)),
        "accepted_stability_usable_target_rows": int(accepted_row.get("stability_model_usable_target", 0)),
        "accepted_stability_model_ready_minimal": int(accepted_row.get("stability_model_ready_minimal", 0)),
        "accepted_stability_model_ready_strict": int(accepted_row.get("stability_model_ready_strict", 0)),
        "accepted_physical_stability_ready": int(accepted_row.get("physical_stability_ready", 0)),
        "device_identity_consistency_issues": int(len(identity_issues)),
        "pce_usable_target_net_change": int(after.get("pce_model_usable_target", 0) - before.get("pce_model_usable_target", 0)),
        "stability_usable_target_net_change": int(after.get("stability_model_usable_target", 0) - before.get("stability_model_usable_target", 0)),
        "pce_model_ready_minimal_net_change": int(after.get("pce_model_ready_minimal", 0) - before.get("pce_model_ready_minimal", 0)),
        "stability_model_ready_minimal_net_change": int(after.get("stability_model_ready_minimal", 0) - before.get("stability_model_ready_minimal", 0)),
        "pce_model_ready_strict_net_change": int(after.get("pce_model_ready_strict", 0) - before.get("pce_model_ready_strict", 0)),
        "stability_model_ready_strict_net_change": int(after.get("stability_model_ready_strict", 0) - before.get("stability_model_ready_strict", 0)),
        "physical_stability_ready_net_change": int(after.get("physical_stability_ready", 0) - before.get("physical_stability_ready", 0)),
    }
    report["qa_pass"] = bool(
        (
            report["accepted_pce_usable_target_rows"] > 0
            or report["accepted_stability_usable_target_rows"] > 0
        )
        and report["pce_usable_target_net_change"] >= 0
        and report["stability_usable_target_net_change"] >= 0
        and report["device_identity_consistency_issues"] == 0
    )
    report["recommendation"] = (
        "Proceed to the next expansion batch, while continuing to improve strict composition/processing completeness."
        if report["qa_pass"]
        else "Hold scale-up. Improve source targeting or extraction completeness before the next batch."
    )
    (args.out_dir / "batch_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        f"# Expansion Batch QA: {args.batch_name}",
        "",
        f"- Raw LiteratureAgent rows: **{report['raw_literature_rows']}**",
        f"- Accepted rows: **{report['accepted_rows']}**",
        f"- Rejected rows: **{report['rejected_rows']}**",
        f"- Accepted usable PCE target rows: **{report['accepted_pce_usable_target_rows']}**",
        f"- Accepted minimal model-ready PCE rows: **{report['accepted_pce_model_ready_minimal']}**",
        f"- Accepted strict model-ready PCE rows: **{report['accepted_pce_model_ready_strict']}**",
        f"- Accepted usable stability target rows: **{report['accepted_stability_usable_target_rows']}**",
        f"- Accepted minimal model-ready stability rows: **{report['accepted_stability_model_ready_minimal']}**",
        f"- Accepted strict model-ready stability rows: **{report['accepted_stability_model_ready_strict']}**",
        f"- Accepted physical-stability-ready rows: **{report['accepted_physical_stability_ready']}**",
        f"- Net usable PCE target-support change after integration: **{report['pce_usable_target_net_change']}**",
        f"- Net usable stability target-support change after integration: **{report['stability_usable_target_net_change']}**",
        f"- Net minimal PCE feature-support change after integration: **{report['pce_model_ready_minimal_net_change']}**",
        f"- Net minimal stability feature-support change after integration: **{report['stability_model_ready_minimal_net_change']}**",
        f"- Net strict PCE feature-support change after integration: **{report['pce_model_ready_strict_net_change']}**",
        f"- Net strict stability feature-support change after integration: **{report['stability_model_ready_strict_net_change']}**",
        f"- Net physical-stability support change: **{report['physical_stability_ready_net_change']}**",
        f"- Device/sample identity consistency issues: **{report['device_identity_consistency_issues']}**",
        f"- QA decision: **{'PASS' if report['qa_pass'] else 'HOLD'}**",
        "",
        report["recommendation"],
    ]
    (args.out_dir / "BATCH_QA_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
