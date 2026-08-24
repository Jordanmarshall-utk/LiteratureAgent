#!/usr/bin/env python3
"""Map model SHAP outputs back to source database fields and audit coverage.

The model pipeline writes SHAP summaries for encoded model features. This
script maps those encoded features back to Perovskite Database source columns,
aggregates SHAP contribution magnitude by source column, and compares that
priority list against LiteratureAgent accepted-row field coverage.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MISSING = {"", "na", "n/a", "nan", "none", "null", "unknown", "not reported", "nr", "-"}


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)


def present_series(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.strip()
    return s.notna() & ~text.str.lower().isin(MISSING)


def coverage(df: pd.DataFrame, col: str) -> tuple[int, float]:
    if col not in df.columns or df.empty:
        return 0, 0.0
    n = int(present_series(df[col]).sum())
    return n, round(100.0 * n / len(df), 3)


def normalize_feature(feature: str) -> str:
    s = str(feature)
    for prefix in ("num__", "cat__", "remainder__"):
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def map_feature_to_source(feature: str, source_columns: list[str]) -> str | None:
    token = normalize_feature(feature)
    if token in source_columns:
        return token
    for col in sorted(source_columns, key=len, reverse=True):
        if token.startswith(col + "_"):
            return col
    return None


def infer_family(path: Path, label: str) -> str:
    text = str(path).lower() + " " + label.lower()
    if "pce_direct" in text:
        return "PCE"
    if "condition_normalized_hybrid" in text:
        return "stability_physical_normalized"
    if "physical_condition_features" in text:
        return "stability_physical_conditions"
    if "stability" in text:
        return "stability"
    return "unknown"


def infer_label(path: Path) -> str:
    name = path.name
    return re.sub(r"_shap_feature_summary\.csv$", "", name)


def collect_shap(shap_model_dir: Path, source_columns: list[str], top_n_per_model: int) -> pd.DataFrame:
    rows = []
    for path in shap_model_dir.glob("**/*_shap_feature_summary.csv"):
        df = read_csv(path)
        needed = {"feature", "mean_abs_shap", "mean_signed_shap"}
        if df.empty or not needed.issubset(df.columns):
            continue
        label = infer_label(path)
        family = infer_family(path, label)
        df = df.sort_values("mean_abs_shap", ascending=False).head(top_n_per_model)
        for _, row in df.iterrows():
            source_col = map_feature_to_source(row["feature"], source_columns)
            rows.append({
                "model_label": label,
                "model_family": family,
                "shap_file": str(path),
                "feature": row["feature"],
                "source_column": source_col or "",
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "mean_signed_shap": float(row["mean_signed_shap"]),
            })
    return pd.DataFrame(rows)


def aggregate_priority(shap_rows: pd.DataFrame, accepted: pd.DataFrame, integrated: pd.DataFrame) -> pd.DataFrame:
    if shap_rows.empty:
        return pd.DataFrame()
    mapped = shap_rows[shap_rows["source_column"].astype(str).str.len() > 0].copy()
    grouped = (
        mapped.groupby(["model_family", "source_column"], as_index=False)
        .agg(
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            mean_signed_shap=("mean_signed_shap", "mean"),
            appearances=("model_label", "nunique"),
            top_encoded_examples=("feature", lambda s: "; ".join(list(dict.fromkeys(map(str, s)))[:5])),
        )
        .sort_values(["model_family", "total_mean_abs_shap"], ascending=[True, False])
    )

    rows = []
    for _, item in grouped.iterrows():
        col = str(item["source_column"])
        acc_n, acc_pct = coverage(accepted, col)
        int_n, int_pct = coverage(integrated, col)
        row = item.to_dict()
        row.update({
            "accepted_present_n": acc_n,
            "accepted_present_pct": acc_pct,
            "integrated_literature_present_n": int_n,
            "integrated_literature_present_pct": int_pct,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["priority_gap"] = ""
    for family, sub_idx in out.groupby("model_family").groups.items():
        sub = out.loc[sub_idx]
        threshold = sub["total_mean_abs_shap"].quantile(0.75)
        mask = (out.index.isin(sub_idx)) & (out["total_mean_abs_shap"] >= threshold) & (out["accepted_present_pct"] < 50)
        out.loc[mask, "priority_gap"] = "high_shap_low_literatureagent_coverage"
    return out.sort_values(["model_family", "total_mean_abs_shap"], ascending=[True, False])


def plot_family(priority: pd.DataFrame, family: str, out_dir: Path, top_n: int) -> None:
    sub = priority[priority["model_family"].eq(family)].head(top_n).copy()
    if sub.empty:
        return
    y = np.arange(len(sub))
    fig, ax1 = plt.subplots(figsize=(10, max(5, 0.34 * len(sub))), dpi=200)
    ax1.barh(y, sub["total_mean_abs_shap"], color="#4f6f8f", alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(sub["source_column"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Aggregated mean |SHAP|")
    ax2 = ax1.twiny()
    ax2.plot(sub["accepted_present_pct"], y, color="#c44e52", marker="o", linewidth=1.4)
    ax2.set_xlabel("Accepted LiteratureAgent coverage (%)")
    ax1.set_title(f"{family}: SHAP-prioritized source fields vs extraction coverage")
    fig.tight_layout()
    fig.savefig(out_dir / f"{family}_shap_field_coverage.png", bbox_inches="tight")
    plt.close(fig)


def simple_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    show = df.head(max_rows).copy() if max_rows else df.copy()
    cols = list(show.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shap-model-dir", required=True)
    p.add_argument("--accepted-csv", required=True)
    p.add_argument("--integrated-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-n-per-model", type=int, default=80)
    p.add_argument("--plot-top-n", type=int, default=25)
    args = p.parse_args()

    shap_model_dir = Path(args.shap_model_dir)
    accepted = read_csv(Path(args.accepted_csv))
    integrated_all = read_csv(Path(args.integrated_csv))
    if "_source_added_by_literature_agent" in integrated_all.columns:
        integrated = integrated_all[integrated_all["_source_added_by_literature_agent"].astype(str).str.lower().isin(["1", "1.0", "true", "yes"])].copy()
    else:
        integrated = accepted.copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_columns = list(integrated_all.columns)
    shap_rows = collect_shap(shap_model_dir, source_columns, args.top_n_per_model)
    priority = aggregate_priority(shap_rows, accepted, integrated)

    shap_rows.to_csv(out_dir / "shap_encoded_features_mapped_to_source_columns.csv", index=False)
    priority.to_csv(out_dir / "shap_source_field_priority_coverage.csv", index=False)

    summary_rows = []
    for family in ["PCE", "stability", "stability_physical_conditions", "stability_physical_normalized"]:
        sub = priority[priority["model_family"].eq(family)]
        if sub.empty:
            continue
        plot_family(priority, family, out_dir, args.plot_top_n)
        summary_rows.append({
            "model_family": family,
            "top_source_columns_considered": int(min(args.plot_top_n, len(sub))),
            "median_accepted_coverage_pct_top_fields": float(sub.head(args.plot_top_n)["accepted_present_pct"].median()),
            "high_shap_low_coverage_fields": int(sub["priority_gap"].eq("high_shap_low_literatureagent_coverage").sum()),
            "top_12_columns": "; ".join(sub.head(12)["source_column"].astype(str).tolist()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "shap_field_coverage_summary.csv", index=False)

    priority_fields = priority[
        priority["priority_gap"].eq("high_shap_low_literatureagent_coverage")
    ][["model_family", "source_column", "total_mean_abs_shap", "accepted_present_pct", "top_encoded_examples"]]
    priority_fields.to_csv(out_dir / "extraction_bridge_shap_priority_fields.csv", index=False)

    md = [
        "# SHAP Field-Coverage Audit",
        "",
        "This audit uses SHAP contribution magnitudes from the fitted XGBoost PCE/stability models, maps encoded features back to source database columns, and compares those columns against accepted LiteratureAgent-row coverage.",
        "",
        "SHAP explains the fitted model's predictions; it does not prove causality.",
        "",
        f"- SHAP feature rows mapped: {len(shap_rows)}",
        f"- Accepted LiteratureAgent rows: {len(accepted)}",
        f"- Integrated LiteratureAgent rows: {len(integrated)}",
        "",
        "## Summary",
        "",
        simple_markdown_table(summary) if not summary.empty else "No SHAP source-field rows found.",
        "",
        "## Main Extraction-Bridge Priority",
        "",
        simple_markdown_table(priority_fields, max_rows=40) if not priority_fields.empty else "No high-SHAP / low-coverage fields found.",
        "",
        "## Files",
        "",
        "- shap_encoded_features_mapped_to_source_columns.csv",
        "- shap_source_field_priority_coverage.csv",
        "- extraction_bridge_shap_priority_fields.csv",
        "- *_shap_field_coverage.png",
    ]
    (out_dir / "SHAP_FIELD_COVERAGE_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "shap_field_coverage_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print((out_dir / "SHAP_FIELD_COVERAGE_AUDIT.md").resolve())
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
