#!/usr/bin/env python
"""Create readable, specific plots from learned literature relationships."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clean(value: object) -> str:
    return " ".join(str(value).replace("_", " ").split())


def scientific_specificity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep relationships that read as reusable scientific findings."""
    if df.empty:
        return df
    subject = df["subject"].astype(str).str.strip()
    obj = df["object"].astype(str).str.strip()
    generic = subject.str.lower().str.startswith(
        ("review ", "evaluation ", "this study ", "the study ", "authors ", "the authors ")
    )
    return df[(~generic) & subject.str.len().between(3, 95) & obj.str.len().between(3, 125)].copy()


def plot_rows(df: pd.DataFrame, value: str, title: str, output: Path, limit: int = 16) -> None:
    if df.empty:
        return
    chosen = df.sort_values(value, ascending=False).head(limit).copy()
    chosen["label"] = chosen.apply(
        lambda row: textwrap.fill(
            f"{clean(row['subject'])}  -> {clean(row['relationship'])} ->  {clean(row['object'])}", 72
        ),
        axis=1,
    )
    chosen = chosen.sort_values(value)
    plt.figure(figsize=(14, max(7, 0.55 * len(chosen))))
    plt.barh(chosen["label"], chosen[value], color="#4f788f")
    plt.title(title)
    plt.xlabel(clean(value).title())
    plt.tight_layout()
    plt.savefig(output, dpi=240, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir or args.graph_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    patterns_path = args.graph_dir / "derived_views" / "cross_paper_relationship_patterns.csv"
    learned_path = args.graph_dir / "derived_views" / "learned_scientific_relationships.csv"
    patterns = pd.read_csv(patterns_path) if patterns_path.exists() else pd.DataFrame()
    learned = pd.read_csv(learned_path) if learned_path.exists() else pd.DataFrame()

    if not patterns.empty:
        value = "paper_count" if "paper_count" in patterns.columns else "claim_count"
        patterns = scientific_specificity_filter(patterns)
        repeated = patterns[patterns[value] > 1] if (patterns[value] > 1).any() else patterns
        plot_rows(
            repeated,
            value,
            "Repeated evidence-grounded scientific relationships across papers",
            out / "specific_repeated_scientific_relationships.png",
        )

    if not learned.empty:
        group_cols = ["subject", "relationship", "object"]
        specific = learned.groupby(group_cols, dropna=False).size().reset_index(name="evidence_count")
        specific = scientific_specificity_filter(specific)
        useful = specific[
            specific["relationship"].astype(str).str.upper().isin(
                ["IMPROVES", "PROMOTES", "REDUCES", "INHIBITS", "INCREASES"]
            )
        ]
        plot_rows(
            useful,
            "evidence_count",
            "Specific interventions, mechanisms, and outcomes learned from literature",
            out / "specific_intervention_outcome_relationships.png",
        )
    print(f"Wrote specific knowledge-graph plots to {out}")


if __name__ == "__main__":
    main()
