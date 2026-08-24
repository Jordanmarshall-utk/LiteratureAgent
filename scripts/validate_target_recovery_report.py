from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


def present(v: Any) -> bool:
    if v is None or pd.isna(v):
        return False
    return str(v).strip().lower() not in {"", "nan", "none", "null", "not reported", "unknown"}


def num(v: Any) -> float | None:
    try:
        out = float(str(v).replace(",", "").strip())
    except Exception:
        return None
    return out if out == out else None


BAD_STABILITY_CONTEXT = re.compile(
    r"\b(loss|lost|lose|loses|degrad(?:e|es|ed|ation) by|drops? by|dropped by|"
    r"decreases? by|decreased by|decays? by|decayed by|review|reported by others|"
    r"bibliography|references|copyright|permission)\b",
    flags=re.IGNORECASE,
)

GOOD_RETENTION_CONTEXT = re.compile(
    r"\b(retained|retains?|retention|maintained|remaining|preserved)\b",
    flags=re.IGNORECASE,
)

GOOD_DEVICE_CONTEXT = re.compile(
    r"\b(device|cell|PSC|solar cell|PCE|efficiency|performance|MPP|maximum power point|ISOS)\b",
    flags=re.IGNORECASE,
)

RH_NEAR_PERCENT = re.compile(
    r"\b(?:RH|relative humidity|humidity)\b.{0,30}\d{1,3}\s*%|\d{1,3}\s*%.{0,30}\b(?:RH|relative humidity|humidity)\b",
    flags=re.IGNORECASE,
)


def validate_pce(row: pd.Series) -> tuple[bool, list[str]]:
    issues = []
    pce = num(row.get("after_JV_default_PCE"))
    changed = str(row.get("pce_changed_to_model_range", "")).lower() == "true"
    if not changed:
        return True, issues
    evidence = str(row.get("candidate_evidence") or "")
    score = num(row.get("candidate_score")) or 0
    if pce is None or not (5 <= pce <= 30):
        issues.append("pce_outside_model_range")
    if score < 11:
        issues.append("pce_score_below_threshold")
    if not re.search(r"\b(PCE|power conversion|photoconversion|efficiency|device|cell|PSC|solar)\b", evidence, flags=re.IGNORECASE):
        issues.append("pce_evidence_lacks_device_context")
    if re.search(r"\b(PLQY|quantum yield|EQE|IPCE|absorbance|retention|humidity)\b", evidence, flags=re.IGNORECASE):
        issues.append("pce_evidence_possible_non_pce_metric")
    return not issues, issues


def validate_stability(row: pd.Series) -> tuple[bool, list[str]]:
    issues = []
    changed = str(row.get("stability_changed", "")).lower() == "true"
    if not changed:
        return True, issues
    evidence = str(row.get("stability_recovery_evidence") or "")
    t80 = num(row.get("after_Stability_PCE_T80"))
    t95 = num(row.get("after_Stability_PCE_T95"))
    retention = num(row.get("after_Stability_PCE_end_of_experiment"))
    after_1000h = num(row.get("after_Stability_PCE_after_1000_h"))
    exposure = num(row.get("after_Stability_time_total_exposure"))

    if BAD_STABILITY_CONTEXT.search(evidence):
        issues.append("bad_or_ambiguous_stability_context")
    if not GOOD_RETENTION_CONTEXT.search(evidence) and t80 is None and t95 is None:
        issues.append("stability_evidence_lacks_retention_language")
    if not GOOD_DEVICE_CONTEXT.search(evidence):
        issues.append("stability_evidence_lacks_device_context")
    if retention is not None and not (1 <= retention <= 120):
        issues.append("retention_outside_plausible_range")
    if after_1000h is not None:
        if not (50 <= after_1000h <= 120):
            issues.append("after_1000h_outside_conservative_range")
        if exposure is None or not (950 <= exposure <= 1050):
            issues.append("after_1000h_without_matching_1000h_exposure")
    if t80 is not None and not (1 <= t80 <= 100000):
        issues.append("t80_outside_plausible_range")
    if t95 is not None and not (1 <= t95 <= 100000):
        issues.append("t95_outside_plausible_range")
    if exposure is not None and not (1 <= exposure <= 100000):
        issues.append("exposure_outside_plausible_range")
    if retention is not None and RH_NEAR_PERCENT.search(evidence) and not GOOD_RETENTION_CONTEXT.search(evidence):
        issues.append("possible_humidity_percent_misread_as_retention")
    if retention is None and t80 is None and t95 is None and after_1000h is None:
        issues.append("no_target_value_recovered")
    return not issues, issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--summary-csv", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.report_csv, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
    rows = []
    for _, row in df.iterrows():
        pce_ok, pce_issues = validate_pce(row)
        stab_ok, stab_issues = validate_stability(row)
        changed = str(row.get("changed", "")).lower() == "true"
        validation_status = "not_changed"
        if changed:
            validation_status = "pass" if pce_ok and stab_ok else "flagged"
        out = row.to_dict()
        out["validation_status"] = validation_status
        out["validation_issues"] = "; ".join(pce_issues + stab_issues)
        out["pce_validation_pass"] = pce_ok
        out["stability_validation_pass"] = stab_ok
        rows.append(out)

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    summary = []
    for name, group in out_df.groupby("validation_status", dropna=False):
        summary.append({"metric": f"validation_status={name}", "count": int(len(group))})
    if "stability_changed" in out_df.columns:
        changed_stab = out_df[out_df["stability_changed"].astype(str).str.lower() == "true"]
        summary.append({"metric": "stability_changed_rows", "count": int(len(changed_stab))})
        summary.append({"metric": "stability_changed_validation_pass", "count": int((changed_stab["validation_status"] == "pass").sum())})
        summary.append({"metric": "stability_changed_flagged", "count": int((changed_stab["validation_status"] == "flagged").sum())})
    if "pce_changed_to_model_range" in out_df.columns:
        changed_pce = out_df[out_df["pce_changed_to_model_range"].astype(str).str.lower() == "true"]
        summary.append({"metric": "pce_changed_rows", "count": int(len(changed_pce))})
        summary.append({"metric": "pce_changed_validation_pass", "count": int((changed_pce["validation_status"] == "pass").sum())})
    pd.DataFrame(summary).to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
