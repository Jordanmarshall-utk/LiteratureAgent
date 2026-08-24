#!/usr/bin/env python3
"""Plot reasoning-policy knowledge graph comparison outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


POLICY_LABELS = {
    "current": "Current",
    "socratic_uncertainty": "Socratic",
    "cartesian_decomposition": "Cartesian",
    "kantian_constraints": "Kantian",
    "humean_evidence": "Humean",
    "aristotelian_classification": "Aristotelian",
    "hegelian_conflict_resolution": "Hegelian",
    "platonic_abstraction": "Platonic",
}


COLORS = {
    "Current": "#6c757d",
    "Socratic": "#4c78a8",
    "Cartesian": "#72b7b2",
    "Kantian": "#f58518",
    "Humean": "#54a24b",
    "Aristotelian": "#b279a2",
    "Hegelian": "#e45756",
    "Platonic": "#9d755d",
}


def add_labels(ax) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + max(1, width * 0.01),
            patch.get_y() + patch.get_height() / 2,
            f"{int(width):,}",
            va="center",
            fontsize=9,
        )


def barh(df: pd.DataFrame, column: str, title: str, xlabel: str, output: Path) -> None:
    plot_df = df.sort_values(column).copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=220)
    labels = plot_df["label"].tolist()
    values = plot_df[column].astype(float).tolist()
    ax.barh(labels, values, color=[COLORS.get(label, "#577590") for label in labels])
    add_labels(ax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report plots from reasoning_policy_kg_comparison.csv")
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or args.comparison_csv.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.comparison_csv)
    df["label"] = df["policy"].map(POLICY_LABELS).fillna(df["policy"])
    current = float(df.loc[df["policy"] == "current", "relationships"].iloc[0])
    df["relationships_added_vs_current"] = df["relationships"].astype(float) - current

    barh(
        df,
        "relationships",
        "Knowledge graph relationships by reasoning policy",
        "Total graph relationships",
        out_dir / "kg_relationships_by_reasoning_policy.png",
    )
    barh(
        df[df["policy"] != "current"],
        "relationships_added_vs_current",
        "Reasoning-specific relationships added compared with current KG",
        "Additional relationships vs current",
        out_dir / "kg_reasoning_relationships_added_vs_current.png",
    )
    barh(
        df,
        "reasoning_findings",
        "Reasoning findings created by each policy",
        "Reasoning-specific findings",
        out_dir / "kg_reasoning_findings_by_policy.png",
    )

    summary = df[
        [
            "policy",
            "label",
            "nodes",
            "relationships",
            "relationships_added_vs_current",
            "scientific_claims",
            "learned_directional_relationships",
            "evidence_linked_claims",
            "summary_only_claims",
            "reasoning_findings",
        ]
    ].copy()
    summary.to_csv(out_dir / "reasoning_policy_kg_report_table.csv", index=False)
    print(f"Wrote reasoning-policy KG plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
