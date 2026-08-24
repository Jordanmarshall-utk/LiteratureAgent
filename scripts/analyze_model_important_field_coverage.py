#!/usr/bin/env python
"""Compare model-important fields against LiteratureAgent extraction coverage.

The current model pipeline writes tree feature-importance files, not SHAP
values. This audit maps transformed model features back to source CSV columns
and asks: are LiteratureAgent accepted rows filling the fields the baseline
models actually use?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


MISSING = {"", "nan", "none", "null", "[]", "{}", "not reported", "unknown", "n/a"}


def present(v: Any) -> bool:
    if v is None or pd.isna(v):
        return False
    return str(v).strip().lower() not in MISSING


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.DataFrame()


def raw_col_from_feature(feature: str, columns: list[str]) -> str:
    f = str(feature)
    if f.startswith("num__"):
        f = f[5:]
    if f in columns:
        return f
    # One-hot categorical features are emitted as RawColumn_category. Match the
    # longest source-column prefix so columns with underscores map correctly.
    matches = [c for c in columns if f == c or f.startswith(c + "_")]
    if matches:
        return max(matches, key=len)
    return f


def target_family_from_path(path: Path) -> str:
    text = path.as_posix().lower()
    if "/pce/" in text:
        return "PCE"
    if "physical_condition_features" in text or "condition_normalized" in text:
        return "stability_physical"
    if "/stability/" in text:
        return "stability"
    return "other"


def collect_importance(model_dir: Path, source_columns: list[str], top_n_per_model: int) -> pd.DataFrame:
    rows = []
    for path in model_dir.glob("**/*feature_importance.csv"):
        df = read_csv(path)
        if df.empty or not {"feature", "importance"}.issubset(df.columns):
            continue
        df = df.sort_values("importance", ascending=False).head(top_n_per_model)
        rel = path.relative_to(model_dir).as_posix()
        family = target_family_from_path(path)
        for rank, row in enumerate(df.itertuples(index=False), start=1):
            raw_col = raw_col_from_feature(getattr(row, "feature"), source_columns)
            rows.append({
                "model_file": rel,
                "model_family": family,
                "rank_in_model": rank,
                "feature": getattr(row, "feature"),
                "source_column": raw_col,
                "importance": float(getattr(row, "importance")),
            })
    return pd.DataFrame(rows)


def coverage_for_columns(df: pd.DataFrame, columns: list[str], label: str) -> dict[str, dict[str, Any]]:
    out = {}
    n = len(df)
    for col in columns:
        if col not in df.columns:
            out[col] = {
                f"{label}_present_count": 0,
                f"{label}_present_pct": 0.0,
                f"{label}_column_exists": False,
            }
            continue
        count = int(df[col].map(present).sum())
        out[col] = {
            f"{label}_present_count": count,
            f"{label}_present_pct": round(100 * count / n, 3) if n else 0.0,
            f"{label}_column_exists": True,
        }
    return out


def make_priority_table(importances: pd.DataFrame, accepted: pd.DataFrame, integrated_la: pd.DataFrame) -> pd.DataFrame:
    if importances.empty:
        return pd.DataFrame()
    grouped = (
        importances.groupby(["model_family", "source_column"], as_index=False)
        .agg(
            total_importance=("importance", "sum"),
            mean_importance=("importance", "mean"),
            appearances=("model_file", "nunique"),
            best_rank=("rank_in_model", "min"),
            example_feature=("feature", "first"),
        )
        .sort_values(["model_family", "total_importance"], ascending=[True, False])
    )
    cols = grouped["source_column"].tolist()
    acc_cov = coverage_for_columns(accepted, cols, "accepted")
    la_cov = coverage_for_columns(integrated_la, cols, "integrated_literatureagent")
    rows = []
    for row in grouped.to_dict("records"):
        col = row["source_column"]
        item = dict(row)
        item.update(acc_cov.get(col, {}))
        item.update(la_cov.get(col, {}))
        item["priority_gap"] = (
            "high_importance_low_coverage"
            if float(item["total_importance"]) > grouped["total_importance"].quantile(0.75)
            and float(item.get("accepted_present_pct", 0) or 0) < 50
            else ""
        )
        rows.append(item)
    return pd.DataFrame(rows)


def plot_priority(priority: pd.DataFrame, out_dir: Path) -> None:
    if priority.empty:
        return
    for family in ["PCE", "stability", "stability_physical"]:
        sub = priority[priority["model_family"].eq(family)].head(18).copy()
        if sub.empty:
            continue
        labels = sub["source_column"].astype(str)
        fig, ax1 = plt.subplots(figsize=(10, max(5, 0.32 * len(sub))), dpi=220)
        y = range(len(sub))
        ax1.barh(y, sub["total_importance"], color="#557a95", alpha=0.85, label="total importance")
        ax1.set_yticks(list(y))
        ax1.set_yticklabels(labels)
        ax1.invert_yaxis()
        ax1.set_xlabel("Aggregated tree feature importance")
        ax2 = ax1.twiny()
        ax2.plot(sub["accepted_present_pct"], list(y), color="#c85f3d", marker="o", linewidth=1.5, label="accepted-row coverage")
        ax2.set_xlabel("LiteratureAgent accepted-row coverage (%)")
        ax2.set_xlim(0, 100)
        ax1.set_title(f"{family}: important model fields vs LiteratureAgent coverage")
        fig.tight_layout()
        fig.savefig(out_dir / f"{family}_important_field_coverage.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--accepted-csv", type=Path, required=True)
    ap.add_argument("--integrated-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-n-per-model", type=int, default=40)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    accepted = read_csv(args.accepted_csv)
    integrated = read_csv(args.integrated_csv)
    if "_source_added_by_literature_agent" in integrated.columns:
        marker = integrated["_source_added_by_literature_agent"].astype(str).str.lower()
        integrated_la = integrated[marker.isin({"1", "1.0", "true", "yes"})].copy()
    else:
        integrated_la = pd.DataFrame(columns=integrated.columns)

    source_columns = list(integrated.columns)
    importances = collect_importance(args.model_dir, source_columns, args.top_n_per_model)
    priority = make_priority_table(importances, accepted, integrated_la)

    importances.to_csv(args.out_dir / "model_feature_importance_mapped_to_source_columns.csv", index=False)
    priority.to_csv(args.out_dir / "model_important_field_coverage_priority.csv", index=False)
    plot_priority(priority, args.out_dir)

    summary_rows = []
    for family in ["PCE", "stability", "stability_physical"]:
        sub = priority[priority["model_family"].eq(family)].head(20)
        if sub.empty:
            continue
        summary_rows.append({
            "model_family": family,
            "top_source_columns_considered": len(sub),
            "median_accepted_coverage_pct": round(float(sub["accepted_present_pct"].median()), 3),
            "high_importance_low_coverage_fields": int(sub["priority_gap"].eq("high_importance_low_coverage").sum()),
            "top_10_columns": "; ".join(sub["source_column"].head(10).astype(str)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "model_important_field_coverage_summary.csv", index=False)

    lines = [
        "# Model-Important Field Coverage Audit",
        "",
        "The current model outputs tree feature importance, not SHAP. Bars indicate how much the model used a feature for reducing error; they do not prove causality.",
        "",
        f"- Accepted LiteratureAgent rows: {len(accepted)}",
        f"- Integrated LiteratureAgent rows: {len(integrated_la)}",
        f"- Feature-importance rows mapped: {len(importances)}",
        "",
        "## Summary",
    ]
    if not summary.empty:
        lines.append(summary.to_string(index=False))
    lines.extend([
        "",
        "## Outputs",
        "- model_feature_importance_mapped_to_source_columns.csv",
        "- model_important_field_coverage_priority.csv",
        "- model_important_field_coverage_summary.csv",
        "- PCE_important_field_coverage.png",
        "- stability_important_field_coverage.png",
        "- stability_physical_important_field_coverage.png",
    ])
    (args.out_dir / "MODEL_IMPORTANT_FIELD_COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
