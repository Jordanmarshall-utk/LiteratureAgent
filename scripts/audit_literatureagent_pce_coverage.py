from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PCE_COLS = [
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "Stabilised_performance_PCE",
]

PERF_COLS = [
    "JV_default_PCE",
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "Stabilised_performance_PCE",
]

CORE_FEATURE_COLS = [
    "Ref_DOI_number",
    "Ref_publication_date",
    "Cell_architecture",
    "Cell_stack_sequence",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "Perovskite_composition_short_form",
    "Perovskite_composition_long_form",
    "Perovskite_composition_a_ions",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_c_ions",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_additives_compounds",
    "JV_default_PCE",
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
    "Stability_measured",
    "Stability_time_total_exposure",
    "Stability_PCE_T80",
    "Stability_PCE_after_1000_h",
    "Stability_PCE_end_of_experiment",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def nonblank(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    return s.notna() & ~s.astype(str).str.strip().isin(["", "nan", "None", "null", "[]", "{}"])


def numeric_valid(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mask = x.notna()
    if lo is not None:
        mask &= x >= lo
    if hi is not None:
        mask &= x <= hi
    return mask


def has_any(df: pd.DataFrame, cols: list[str], numeric: bool = False, lo: float | None = None, hi: float | None = None) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        mask |= numeric_valid(df[col], lo, hi) if numeric else nonblank(df[col])
    return mask


def summarize_df(name: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"dataset": name, "rows": 0}
    pce_any = has_any(df, PCE_COLS, numeric=True, lo=0, hi=40)
    pce_model = has_any(df, ["JV_default_PCE"], numeric=True, lo=5, hi=30)
    perf_any = has_any(df, PERF_COLS)
    comp_any = has_any(df, [
        "Perovskite_composition_short_form",
        "Perovskite_composition_long_form",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions",
    ])
    stack_any = has_any(df, [
        "Cell_stack_sequence",
        "Cell_architecture",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
    ])
    stability_any = has_any(df, [
        "Stability_measured",
        "Stability_time_total_exposure",
        "Stability_PCE_T80",
        "Stability_PCE_after_1000_h",
        "Stability_PCE_end_of_experiment",
    ])
    core_cols = [c for c in CORE_FEATURE_COLS if c in df.columns]
    core_fill = pd.Series(0.0, index=df.index)
    if core_cols:
        core_fill = df[core_cols].apply(nonblank).mean(axis=1)
    out = {
        "dataset": name,
        "rows": int(len(df)),
        "rows_any_pce_col_numeric_0_40": int(pce_any.sum()),
        "rows_jv_default_pce_model_range_5_30": int(pce_model.sum()),
        "rows_any_performance_field": int(perf_any.sum()),
        "rows_any_composition_field": int(comp_any.sum()),
        "rows_any_stack_or_architecture": int(stack_any.sum()),
        "rows_any_stability_field": int(stability_any.sum()),
        "core_feature_cols_present": len(core_cols),
        "rows_core_fill_ge_20pct": int((core_fill >= 0.20).sum()),
        "rows_with_pce_and_core_fill_ge_20pct": int((pce_model & (core_fill >= 0.20)).sum()),
    }
    return out


def pce_row_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pce_mask = has_any(df, ["JV_default_PCE"], numeric=True, lo=5, hi=30)
    core_cols = [c for c in CORE_FEATURE_COLS if c in df.columns]
    core_fill = df[core_cols].apply(nonblank).mean(axis=1) if core_cols else pd.Series(0.0, index=df.index)
    cols = [
        "Ref_DOI_number",
        "Ref_publication_date",
        "Ref_free_text_comment",
        "Cell_architecture",
        "Cell_stack_sequence",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
        "Perovskite_composition_short_form",
        "Perovskite_deposition_procedure",
        "JV_default_PCE",
        "JV_default_Voc",
        "JV_default_Jsc",
        "JV_default_FF",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df.loc[pce_mask, cols].copy()
    out.insert(0, "source_dataset", name)
    out.insert(1, "row_index", out.index)
    out["core_fill_fraction_approx"] = core_fill.loc[out.index].round(3).to_numpy()
    out["model_core_fill_ge_20pct_approx"] = out["core_fill_fraction_approx"] >= 0.20
    return out


def audit_text_for_pce(work_dir: Path, slugs: list[str]) -> pd.DataFrame:
    rows = []
    text_dir = work_dir / "text"
    summary_dir = work_dir / "paper_summaries_text"
    for slug in slugs:
        blob = ""
        for path in list(text_dir.glob(f"{slug}*.txt"))[:12] + list(summary_dir.glob(f"{slug}*.txt"))[:2]:
            try:
                blob += "\n" + path.read_text(encoding="utf-8", errors="ignore")[:20000]
            except Exception:
                pass
        low = blob.lower()
        rows.append({
            "paper_slug": slug,
            "text_chars_sampled": len(blob),
            "mentions_pce": "pce" in low or "power conversion efficiency" in low,
            "mentions_voc": "voc" in low or "open-circuit" in low,
            "mentions_jsc": "jsc" in low or "short-circuit" in low,
            "mentions_fill_factor": "fill factor" in low or " ff " in low,
            "percent_sign_count": blob.count("%"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-records", required=True, type=Path)
    ap.add_argument("--accepted", required=True, type=Path)
    ap.add_argument("--rejected", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--integrated", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "raw_all_records": read_csv(args.raw_records),
        "accepted_rows": read_csv(args.accepted),
        "rejected_rows": read_csv(args.rejected),
        "integration_audit": read_csv(args.audit),
        "integrated_database": read_csv(args.integrated),
    }
    summary = pd.DataFrame([summarize_df(name, df) for name, df in datasets.items()])
    summary.to_csv(args.out_dir / "pce_coverage_summary.csv", index=False)

    pce_tables = []
    for name in ["raw_all_records", "accepted_rows", "rejected_rows"]:
        tab = pce_row_table(datasets[name], name)
        if not tab.empty:
            pce_tables.append(tab)
    pce_rows = pd.concat(pce_tables, ignore_index=True) if pce_tables else pd.DataFrame()
    pce_rows.to_csv(args.out_dir / "rows_with_jv_default_pce_model_range.csv", index=False)

    raw = datasets["raw_all_records"]
    slugs = []
    if "paper_slug" in raw.columns:
        slugs = raw["paper_slug"].dropna().astype(str).unique().tolist()
    elif "Ref_ID" in raw.columns:
        slugs = raw["Ref_ID"].dropna().astype(str).head(500).tolist()
    text_audit = audit_text_for_pce(args.work_dir, slugs)
    text_audit.to_csv(args.out_dir / "source_text_pce_keyword_audit.csv", index=False)

    report = {
        "summary_csv": str(args.out_dir / "pce_coverage_summary.csv"),
        "pce_rows_csv": str(args.out_dir / "rows_with_jv_default_pce_model_range.csv"),
        "text_keyword_audit_csv": str(args.out_dir / "source_text_pce_keyword_audit.csv"),
        "summary": summary.to_dict(orient="records"),
    }
    (args.out_dir / "pce_coverage_audit_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote audit to: {args.out_dir}")


if __name__ == "__main__":
    main()
