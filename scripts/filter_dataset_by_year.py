#!/usr/bin/env python
"""Create a filtered Perovskite Database CSV by publication year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


YEAR_COLUMNS = [
    "Ref_publication_date",
    "Ref_publication_year",
    "Publication_year",
    "publication_year",
    "year",
]


def read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ["utf-8-sig", "utf-16", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def infer_years(df: pd.DataFrame) -> pd.Series:
    years = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for col in YEAR_COLUMNS:
        if col not in df.columns:
            continue
        extracted = df[col].astype(str).str.extract(r"((?:19|20)\d{2})", expand=False)
        parsed = pd.to_numeric(extracted, errors="coerce").astype("Float64")
        years = years.fillna(parsed)
    return years


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-year", type=int, default=2018, help="Keep rows from this year onward.")
    parser.add_argument(
        "--drop-unknown-year",
        action="store_true",
        help="Also remove rows where no publication year can be detected.",
    )
    args = parser.parse_args()

    df = read_csv(args.input)
    years = infer_years(df)
    keep = years >= args.min_year
    if not args.drop_unknown_year:
        keep = keep | years.isna()

    out = df.loc[keep.fillna(False)].copy()
    out["_publication_year_detected_for_filter"] = years.loc[out.index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    report = {
        "input_csv": str(args.input),
        "output_csv": str(args.output),
        "min_year_kept": args.min_year,
        "drop_unknown_year": bool(args.drop_unknown_year),
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "rows_with_detected_year": int(years.notna().sum()),
        "rows_without_detected_year": int(years.isna().sum()),
        "rows_removed_before_min_year": int(((years.notna()) & (years < args.min_year)).sum()),
        "rows_removed_unknown_year": int((years.isna() & ~keep.fillna(False)).sum()),
    }
    report_path = args.output.with_suffix(".filter_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
