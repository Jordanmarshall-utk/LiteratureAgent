#!/usr/bin/env python3
"""Build the manuscript-facing LiteratureAgent knowledge-graph overview."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "literature_agent_pce_stability"
GRAPH_SUMMARY = (
    ARTIFACT
    / "source_data"
    / "knowledge_graph"
    / "all_policy_comparison"
    / "humean_evidence"
    / "reports"
    / "graph_summary.json"
)
REPRESENTATIVE_PATH = (
    ARTIFACT
    / "source_data"
    / "knowledge_graph"
    / "context_tables"
    / "representative_subgraph_source.json"
)
FIGURE_STEM = (
    ARTIFACT
    / "publication_figures"
    / "main"
    / "figure_literatureagent_knowledge_graph_overview"
)
SOURCE_DIR = ARTIFACT / "source_data" / "publication_figures"
CAPTION_PATH = (
    ARTIFACT
    / "publication_figures"
    / "KNOWLEDGE_GRAPH_OVERVIEW_CAPTION.md"
)


COLORS = {
    "ink": "#202A33",
    "muted": "#5D6670",
    "paper": "#FFFFFF",
    "blue": "#4C78A8",
    "teal": "#72B7B2",
    "gold": "#F2CF5B",
    "purple": "#B279A2",
    "green": "#59A14F",
    "coral": "#E45756",
    "gray": "#A7A7A7",
    "light": "#F5F6F7",
    "line": "#5B6168",
    "policy": "#D6E8D2",
}


def load_sources() -> tuple[dict, dict]:
    summary = json.loads(GRAPH_SUMMARY.read_text(encoding="utf-8"))
    representative = json.loads(REPRESENTATIVE_PATH.read_text(encoding="utf-8"))
    return summary, representative


def add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    facecolor: str,
    *,
    title_color: str = "white",
    detail_color: str | None = None,
    edgecolor: str | None = None,
    dashed: bool = False,
    title_size: float = 10.2,
    detail_size: float = 7.8,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor=facecolor,
        edgecolor=edgecolor or facecolor,
        linewidth=1.2,
        linestyle=(0, (4, 2)) if dashed else "solid",
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        linespacing=1.02,
        zorder=4,
    )
    axis.text(
        x + width / 2,
        y + height * 0.25,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=detail_color or title_color,
        linespacing=1.05,
        zorder=4,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    color: str = COLORS["line"],
    dashed: bool = False,
    curve: float = 0.0,
    label_offset: tuple[float, float] = (0.0, 0.0),
    size: float = 7.3,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        color=color,
        linestyle=(0, (4, 2)) if dashed else "solid",
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    axis.add_patch(arrow)
    midpoint = (
        (start[0] + end[0]) / 2 + label_offset[0],
        (start[1] + end[1]) / 2 + label_offset[1],
    )
    axis.text(
        midpoint[0],
        midpoint[1],
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.94},
        zorder=5,
    )


def draw_aggregate_graph(axis: plt.Axes, summary: dict) -> tuple[list[dict], list[dict]]:
    nodes = summary["node_counts_by_label"]
    relationships = summary["relationship_counts_by_type"]
    claims = summary["claim_support_status"]

    node_rows = [
        {"display_group": "Paper corpus", "count": nodes["Paper"], "source_labels": "Paper"},
        {
            "display_group": "Scientific context",
            "count": nodes["Device"] + nodes["MaterialSystem"],
            "source_labels": "Device + MaterialSystem",
        },
        {
            "display_group": "Structured observations",
            "count": nodes["StructuredObservation"],
            "source_labels": "StructuredObservation",
        },
        {
            "display_group": "Evidence nodes",
            "count": nodes["Evidence"],
            "source_labels": "Evidence",
        },
        {"display_group": "Scientific claims", "count": nodes["Claim"], "source_labels": "Claim"},
        {
            "display_group": "Reported directional relationships",
            "count": summary["learned_directional_relationships"],
            "source_labels": "IMPROVES + REDUCES + PROMOTES + INHIBITS + ATTRIBUTED_TO",
        },
        {
            "display_group": "Humean evidence-lens outputs",
            "count": claims["evidence_linked"] + claims["summary_only"],
            "source_labels": "prioritized evidence-linked + weak-evidence flags",
        },
    ]

    edge_rows = [
        {"source": "Paper corpus", "target": "Scientific context", "label": "reports/studies", "count": relationships["REPORTS_DEVICE"] + relationships["STUDIES_MATERIAL_SYSTEM"]},
        {"source": "Paper corpus", "target": "Structured observations", "label": "reports observations", "count": relationships["REPORTS_STRUCTURED_OBSERVATION"]},
        {"source": "Paper corpus", "target": "Evidence nodes", "label": "text + summary evidence", "count": relationships["HAS_TEXT_EVIDENCE"] + relationships["HAS_SUMMARY_EVIDENCE"]},
        {"source": "Paper corpus", "target": "Scientific claims", "label": "asserts claims", "count": relationships["ASSERTS_CLAIM"]},
        {"source": "Evidence nodes", "target": "Scientific claims", "label": "supports claims", "count": relationships["SUPPORTS_CLAIM"]},
        {"source": "Scientific claims", "target": "Reported directional relationships", "label": "reported directional edges", "count": summary["learned_directional_relationships"]},
        {"source": "Scientific claims", "target": "Humean evidence-lens outputs", "label": "policy evaluates", "count": nodes["Claim"]},
    ]

    axis.text(0.02, 0.955, "A", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.text(
        0.055,
        0.955,
        "Type-aggregated evidence graph",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    axis.text(
        0.055,
        0.912,
        "Controlled 36-paper manuscript snapshot",
        fontsize=8.5,
        color=COLORS["muted"],
        va="center",
    )

    total = FancyBboxPatch(
        (0.45, 0.875),
        0.285,
        0.085,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor=COLORS["light"],
        edgecolor="#D8DDE2",
        linewidth=0.9,
    )
    axis.add_patch(total)
    axis.text(
        0.592,
        0.918,
        f"{summary['nodes']:,} nodes  |  {summary['relationships']:,} relationships  |  0 dangling",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["ink"],
    )

    add_box(axis, 0.025, 0.43, 0.125, 0.16, "Papers", f"n={nodes['Paper']}", COLORS["blue"])
    add_box(
        axis,
        0.205,
        0.68,
        0.17,
        0.17,
        "Scientific context",
        f"devices n={nodes['Device']}\nmaterial systems n={nodes['MaterialSystem']}",
        COLORS["teal"],
        title_color=COLORS["ink"],
        detail_color=COLORS["ink"],
    )
    add_box(
        axis,
        0.205,
        0.39,
        0.17,
        0.17,
        "Structured\nobservations",
        f"n={nodes['StructuredObservation']:,}\nproperties n={nodes['ScientificProperty']}",
        COLORS["gold"],
        title_color=COLORS["ink"],
        detail_color=COLORS["ink"],
    )
    add_box(
        axis,
        0.205,
        0.10,
        0.17,
        0.17,
        "Evidence nodes",
        f"n={nodes['Evidence']:,}\nraw excerpts + summaries",
        COLORS["gray"],
    )
    add_box(
        axis,
        0.445,
        0.39,
        0.15,
        0.17,
        "Scientific claims",
        f"n={nodes['Claim']:,}\n7 claim classes",
        COLORS["purple"],
    )
    add_box(
        axis,
        0.64,
        0.39,
        0.135,
        0.17,
        "Reported directional\nrelationships",
        f"n={summary['learned_directional_relationships']}\nIMPROVES / REDUCES / ...",
        COLORS["coral"],
        title_size=9.0,
        detail_size=7.2,
    )
    add_box(
        axis,
        0.445,
        0.10,
        0.15,
        0.17,
        "Humean\nevidence lens",
        "claims checked against\nlinked evidence",
        COLORS["policy"],
        title_color=COLORS["ink"],
        detail_color=COLORS["ink"],
        edgecolor=COLORS["green"],
        dashed=True,
    )
    add_box(
        axis,
        0.64,
        0.10,
        0.135,
        0.17,
        "Policy outputs",
        f"{claims['evidence_linked']} prioritized\n{claims['summary_only']} weak-evidence flags",
        COLORS["green"],
        title_size=9.3,
        detail_size=7.2,
    )

    add_arrow(axis, (0.15, 0.55), (0.205, 0.745), "reports / studies\n(72)", curve=-0.12, label_offset=(-0.008, 0.035))
    add_arrow(axis, (0.15, 0.51), (0.205, 0.475), "reports observations\n(1,222)", label_offset=(0.02, -0.035))
    add_arrow(axis, (0.15, 0.46), (0.205, 0.185), "text + summary\nevidence (747)", curve=0.08, label_offset=(-0.01, -0.02))
    add_arrow(axis, (0.375, 0.475), (0.445, 0.475), "grounds\nclaim context", label_offset=(0.0, 0.045))
    add_arrow(axis, (0.375, 0.185), (0.445, 0.425), "supports claims\n(904)", label_offset=(0.005, -0.02))
    add_arrow(axis, (0.595, 0.475), (0.64, 0.475), "reported\nedges (93)", color=COLORS["coral"], label_offset=(0.0, 0.045))
    add_arrow(axis, (0.52, 0.39), (0.52, 0.27), "policy lens", dashed=True, color=COLORS["green"], label_offset=(0.045, 0.0))
    add_arrow(axis, (0.595, 0.185), (0.64, 0.185), "outputs", dashed=True, color=COLORS["green"], label_offset=(0.0, 0.035))

    axis.text(
        0.035,
        0.035,
        "Solid paths preserve source paper, source field, record identity, and evidence provenance. Dashed paths are reasoning-policy views.",
        fontsize=7.4,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )
    return node_rows, edge_rows


def draw_representative_path(axis: plt.Axes, representative: dict) -> None:
    panel = FancyBboxPatch(
        (0.805, 0.055),
        0.18,
        0.89,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor="#FAFAF8",
        edgecolor="#D8DDE2",
        linewidth=0.9,
        zorder=0,
    )
    axis.add_patch(panel)
    axis.text(0.82, 0.91, "B", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.text(
        0.855,
        0.91,
        "Representative\nreported path",
        fontsize=11.2,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
        linespacing=1.0,
    )
    axis.text(
        0.895,
        0.845,
        f"{representative['paper']}\nDOI {representative['doi']}",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["muted"],
    )

    add_box(
        axis,
        0.83,
        0.68,
        0.13,
        0.105,
        "Intervention",
        "NH4PF6-modified NiOx",
        COLORS["blue"],
        title_size=8.7,
        detail_size=7.2,
    )
    add_box(
        axis,
        0.83,
        0.48,
        0.13,
        0.105,
        "Reported mechanism",
        "reduced interface\nrecombination",
        COLORS["purple"],
        title_size=8.0,
        detail_size=7.0,
    )
    add_box(
        axis,
        0.83,
        0.28,
        0.13,
        0.105,
        "Reported outcome",
        "improved device\nperformance",
        COLORS["green"],
        title_size=8.2,
        detail_size=7.0,
    )
    add_arrow(
        axis,
        (0.895, 0.68),
        (0.895, 0.585),
        "REDUCES",
        color=COLORS["coral"],
        label_offset=(0.046, 0.0),
        size=7.0,
    )
    add_arrow(
        axis,
        (0.895, 0.48),
        (0.895, 0.385),
        "IMPROVES",
        color=COLORS["coral"],
        label_offset=(0.050, 0.0),
        size=7.0,
    )

    evidence = FancyBboxPatch(
        (0.822, 0.105),
        0.146,
        0.105,
        boxstyle="round,pad=0.007,rounding_size=0.008",
        facecolor="#EEF0F2",
        edgecolor=COLORS["gray"],
        linewidth=0.8,
    )
    axis.add_patch(evidence)
    axis.text(
        0.895,
        0.158,
        "Linked evidence\nPL/TR-PL + SEM excerpts",
        ha="center",
        va="center",
        fontsize=7.3,
        color=COLORS["ink"],
        fontweight="bold",
    )
    add_arrow(
        axis,
        (0.895, 0.21),
        (0.895, 0.28),
        "supports",
        color=COLORS["gray"],
        label_offset=(0.040, 0.0),
        size=6.8,
    )
    axis.text(
        0.895,
        0.075,
        "Reported claim, not causal proof",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["coral"],
        fontweight="bold",
    )


def save_source_tables(node_rows: list[dict], edge_rows: list[dict], summary: dict) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    node_path = SOURCE_DIR / "literatureagent_knowledge_graph_overview_nodes.csv"
    edge_path = SOURCE_DIR / "literatureagent_knowledge_graph_overview_edges.csv"
    with node_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["display_group", "count", "source_labels"])
        writer.writeheader()
        writer.writerows(node_rows)
    with edge_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "target", "label", "count"])
        writer.writeheader()
        writer.writerows(edge_rows)

    caption = (
        "**Evidence-grounded LiteratureAgent knowledge graph.** (A) Type-aggregated view of the controlled "
        f"{summary['node_counts_by_label']['Paper']}-paper manuscript snapshot under the Humean evidence lens, "
        f"containing {summary['nodes']:,} nodes, {summary['relationships']:,} relationships, "
        f"{summary['scientific_claims']:,} scientific claims, {summary['node_counts_by_label']['Evidence']:,} "
        f"evidence nodes, and {summary['learned_directional_relationships']:,} reported directional relationships. "
        "Solid paths retain paper, field, record, and evidence provenance; dashed paths indicate policy-derived views. "
        "(B) Representative evidence-linked path from NH4PF6-modified NiOx through reported reduced interface "
        "recombination to improved device performance. Directional edges encode paper-reported claims and provide "
        "a substrate for causal hypothesis generation; they do not establish experimentally validated causality."
    )
    CAPTION_PATH.write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    summary, representative = load_sources()
    figure, axis = plt.subplots(figsize=(14.0, 7.2), facecolor="white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0.79, 0.04), 0.002, 0.92, facecolor="#E0E3E6", edgecolor="none"))
    node_rows, edge_rows = draw_aggregate_graph(axis, summary)
    draw_representative_path(axis, representative)
    figure.subplots_adjust(left=0.01, right=0.995, bottom=0.015, top=0.995)
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    save_source_tables(node_rows, edge_rows, summary)
    print(FIGURE_STEM)
    print(CAPTION_PATH)


if __name__ == "__main__":
    main()
