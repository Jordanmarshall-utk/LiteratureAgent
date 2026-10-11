#!/usr/bin/env python3
"""Reproduce the four single-panel PCE SHAP interpretation figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from figure5_grouped_shap_plots import (
    DEFAULT_TOP_COUNT,
    render_figure_set,
    validate_canonical_shap_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_DATA = ROOT / "data" / "figure5"
PROJECT_DATA = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "publication_figures"
)


def default_data_dir() -> Path:
    return PORTABLE_DATA if PORTABLE_DATA.is_dir() else PROJECT_DATA


def default_validation(data_dir: Path) -> Path:
    return data_dir / "pce_grouped_shap_validation.json"


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance", type=Path, default=data_dir / "pce_grouped_shap_importance.csv")
    parser.add_argument("--values", type=Path, default=data_dir / "pce_grouped_shap_values.csv")
    parser.add_argument("--metadata", type=Path, default=data_dir / "pce_grouped_shap_row_metadata.csv")
    parser.add_argument(
        "--choice-values",
        type=Path,
        default=data_dir / "pce_choice_stratified_shap_values.csv",
    )
    parser.add_argument("--validation", type=Path, default=default_validation(data_dir))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figure5")
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_COUNT,
        help=f"Number of full-cohort grouped features to display (default: {DEFAULT_TOP_COUNT}).",
    )
    parser.add_argument("--local-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    importance = pd.read_csv(args.importance)
    values = pd.read_csv(args.values)
    metadata = pd.read_csv(args.metadata)
    choice_values = pd.read_csv(args.choice_values)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    validate_canonical_shap_contract(
        validation,
        explained_rows=len(values),
        top_count=args.top,
    )
    result = render_figure_set(
        importance=importance,
        values=values,
        metadata=metadata,
        choice_values=choice_values,
        validation=validation,
        out_dir=args.out_dir,
        top_count=args.top,
        local_seed=args.local_seed,
        metadata_source_path=args.metadata,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
