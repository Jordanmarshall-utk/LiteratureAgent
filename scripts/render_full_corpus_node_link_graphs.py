from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "knowledge_graph"
    / "full_corpus_standardized_v1"
)
NODES_CSV = GRAPH_ROOT / "neo4j" / "nodes.csv"
RELATIONSHIPS_CSV = GRAPH_ROOT / "neo4j" / "relationships.csv"
SUMMARY_JSON = GRAPH_ROOT / "reports" / "graph_summary.json"
OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "publication_figures"
    / "exploratory"
)
FULL_NETWORK_PNG = OUTPUT_DIR / "figure_full_corpus_all_nodes_node_link_network.png"
TYPE_NETWORK_PNG = OUTPUT_DIR / "figure_full_corpus_type_level_node_link_network.png"
LAYOUT_REPORT = GRAPH_ROOT / "reports" / "full_corpus_node_link_rendering.json"


NODE_COLORS = {
    "DatasetSnapshot": "#173F5F",
    "ExtractionRun": "#20639B",
    "Paper": "#4C78A8",
    "Device": "#72B7B2",
    "StructuredObservation": "#F2CF5B",
    "Evidence": "#9E9E9E",
    "ScientificClaim": "#B279A2",
    "ScientificEntity": "#59A14F",
    "OntologyConcept": "#E07B39",
}

RING_RADII = {
    "DatasetSnapshot": 0.00,
    "ExtractionRun": 0.08,
    "OntologyConcept": 0.18,
    "ScientificEntity": 0.30,
    "ScientificClaim": 0.43,
    "Evidence": 0.56,
    "StructuredObservation": 0.70,
    "Device": 0.83,
    "Paper": 0.95,
}

RING_WIDTHS = {
    "DatasetSnapshot": 0.00,
    "ExtractionRun": 0.01,
    "OntologyConcept": 0.035,
    "ScientificEntity": 0.055,
    "ScientificClaim": 0.045,
    "Evidence": 0.055,
    "StructuredObservation": 0.055,
    "Device": 0.025,
    "Paper": 0.012,
}

NODE_SIZES = {
    "DatasetSnapshot": 52.0,
    "ExtractionRun": 34.0,
    "Paper": 2.8,
    "Device": 1.8,
    "StructuredObservation": 0.20,
    "Evidence": 0.22,
    "ScientificClaim": 0.85,
    "ScientificEntity": 0.48,
    "OntologyConcept": 4.2,
}


def stable_unit(value: str, salt: str) -> float:
    digest = hashlib.blake2b(f"{salt}|{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def relationship_family(relationship_type: str) -> str:
    if relationship_type == "DIRECTIONAL_RELATION":
        return "directional"
    if relationship_type in {"REPORTS_CLAIM", "SUPPORTED_BY", "HAS_SUBJECT", "HAS_OBJECT"}:
        return "claim"
    if relationship_type in {"HAS_DEVICE", "HAS_OBSERVATION", "HAS_FIELD_EVIDENCE", "MEASURES_PROPERTY"}:
        return "observation"
    return "provenance"


EDGE_STYLES = {
    "provenance": {"color": "#6C757D", "alpha": 0.018, "width": 0.10},
    "observation": {"color": "#2F6690", "alpha": 0.024, "width": 0.11},
    "claim": {"color": "#8F5C85", "alpha": 0.030, "width": 0.12},
    "directional": {"color": "#2A9D8F", "alpha": 0.070, "width": 0.16},
}


def load_graph() -> tuple[list[dict], list[tuple[str, str, str]], dict]:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    nodes: list[dict] = []
    with NODES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            properties = json.loads(row.get("properties_json") or "{}")
            doi = str(
                properties.get("doi")
                or properties.get("source_doi")
                or properties.get("paper_doi")
                or ""
            ).strip().lower()
            nodes.append(
                {
                    "id": row["id:ID"],
                    "label": row[":LABEL"],
                    "doi": doi,
                }
            )
    relationships: list[tuple[str, str, str]] = []
    with RELATIONSHIPS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relationships.append((row[":START_ID"], row[":TYPE"], row[":END_ID"]))
    expected_nodes = int(summary["graph"]["nodes"])
    expected_relationships = int(summary["graph"]["relationships"])
    if len(nodes) != expected_nodes or len(relationships) != expected_relationships:
        raise RuntimeError(
            f"Graph export mismatch: loaded {len(nodes):,}/{len(relationships):,}; "
            f"expected {expected_nodes:,}/{expected_relationships:,}."
        )
    return nodes, relationships, summary


def build_radial_positions(nodes: list[dict]) -> tuple[dict[str, tuple[float, float]], dict]:
    dois = sorted({node["doi"] for node in nodes if node["doi"]})
    doi_index = {doi: index for index, doi in enumerate(dois)}
    sector = 2 * math.pi / max(1, len(dois))
    positions: dict[str, tuple[float, float]] = {}
    for node in nodes:
        node_id = node["id"]
        label = node["label"]
        radius = RING_RADII.get(label, 0.50)
        width = RING_WIDTHS.get(label, 0.04)
        if label == "DatasetSnapshot":
            positions[node_id] = (0.0, 0.0)
            continue
        if node["doi"] in doi_index:
            base_angle = 2 * math.pi * doi_index[node["doi"]] / max(1, len(dois))
            angle = base_angle + (stable_unit(node_id, "angle") - 0.5) * sector * 0.82
        else:
            angle = 2 * math.pi * stable_unit(node_id, "angle")
        radius += (stable_unit(node_id, "radius") - 0.5) * width
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions, {"doi_count": len(dois), "layout": "doi-aligned concentric semantic rings"}


def render_full_network(nodes: list[dict], relationships: list[tuple[str, str, str]], summary: dict) -> dict:
    positions, layout_meta = build_radial_positions(nodes)
    id_to_label = {node["id"]: node["label"] for node in nodes}
    segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    relationship_counts = Counter()
    aggregate_type_edges = Counter()
    dangling = 0
    for start, relationship_type, end in relationships:
        if start not in positions or end not in positions:
            dangling += 1
            continue
        family = relationship_family(relationship_type)
        segments[family].append((positions[start], positions[end]))
        relationship_counts[relationship_type] += 1
        aggregate_type_edges[(id_to_label[start], relationship_type, id_to_label[end])] += 1

    figure = plt.figure(figsize=(16, 11.2), dpi=220, facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=[4.4, 1.25], wspace=0.02)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    axis.set_aspect("equal")
    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(-1.02, 1.02)
    axis.axis("off")
    legend_axis.axis("off")

    for label, radius in RING_RADII.items():
        if radius <= 0.0:
            continue
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                fill=False,
                edgecolor="#D8DADF",
                linewidth=0.45,
                alpha=0.38,
                zorder=0,
            )
        )

    for family in ("provenance", "observation", "claim", "directional"):
        style = EDGE_STYLES[family]
        collection = LineCollection(
            np.asarray(segments[family], dtype=np.float32),
            colors=style["color"],
            linewidths=style["width"],
            alpha=style["alpha"],
            rasterized=True,
            zorder=1,
        )
        axis.add_collection(collection)

    node_counts = Counter(node["label"] for node in nodes)
    for label in RING_RADII:
        selected = [node for node in nodes if node["label"] == label]
        if not selected:
            continue
        coordinates = np.asarray([positions[node["id"]] for node in selected], dtype=np.float32)
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=NODE_SIZES.get(label, 0.5),
            c=NODE_COLORS.get(label, "#333333"),
            alpha=0.88,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

    figure.suptitle(
        "LiteratureAgent full-corpus node-link rendering",
        x=0.05,
        y=0.975,
        ha="left",
        fontsize=25,
        fontweight="bold",
        color="#20242A",
    )
    figure.text(
        0.05,
        0.935,
        f"Every exported node and relationship | {len(nodes):,} nodes | {len(relationships):,} directed relationships",
        ha="left",
        fontsize=14.5,
        color="#555B63",
    )
    figure.text(
        0.05,
        0.035,
        "Concentric rings encode node type; DOI-linked nodes share angular neighborhoods. Arrowheads are suppressed only at this scale.",
        ha="left",
        fontsize=12.5,
        color="#555B63",
    )

    legend_axis.text(0.02, 0.94, "Node types", fontsize=16, fontweight="bold", color="#20242A")
    y = 0.89
    for label in reversed(list(RING_RADII)):
        count = node_counts.get(label, 0)
        if not count:
            continue
        legend_axis.scatter([0.05], [y], s=90, c=NODE_COLORS[label], edgecolors="#34383E", linewidths=0.6)
        display = label.replace("StructuredObservation", "Structured observation").replace("ScientificClaim", "Scientific claim").replace("ScientificEntity", "Scientific entity").replace("OntologyConcept", "Ontology concept").replace("DatasetSnapshot", "Dataset snapshot").replace("ExtractionRun", "Extraction run")
        legend_axis.text(0.13, y, f"{display}\n{count:,}", va="center", fontsize=11.5, color="#20242A")
        y -= 0.083

    legend_axis.text(0.02, y - 0.01, "Relationship families", fontsize=16, fontweight="bold", color="#20242A")
    y -= 0.065
    family_labels = {
        "provenance": "Provenance / snapshot",
        "observation": "Device / observation",
        "claim": "Claim / evidence",
        "directional": "Directional claim",
    }
    for family, display in family_labels.items():
        style = EDGE_STYLES[family]
        legend_axis.plot([0.02, 0.12], [y, y], color=style["color"], linewidth=2.4)
        legend_axis.text(0.15, y, f"{display}\n{len(segments[family]):,}", va="center", fontsize=11.3, color="#20242A")
        y -= 0.075
    legend_axis.set_xlim(0, 1)
    legend_axis.set_ylim(0, 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FULL_NETWORK_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        **layout_meta,
        "dangling_relationships": dangling,
        "node_counts": dict(node_counts),
        "relationship_counts": dict(relationship_counts),
        "aggregate_type_edges": [
            {"start_label": start, "type": relationship_type, "end_label": end, "count": count}
            for (start, relationship_type, end), count in sorted(aggregate_type_edges.items())
        ],
    }


def render_type_network(nodes: list[dict], relationships: list[tuple[str, str, str]]) -> dict:
    node_counts = Counter(node["label"] for node in nodes)
    id_to_label = {node["id"]: node["label"] for node in nodes}
    edge_counts = Counter()
    relationship_counts = Counter()
    for start, relationship_type, end in relationships:
        if start not in id_to_label or end not in id_to_label:
            continue
        edge_counts[(id_to_label[start], id_to_label[end])] += 1
        relationship_counts[relationship_type] += 1

    positions = {
        "DatasetSnapshot": (0.05, 0.90),
        "ExtractionRun": (-0.42, 0.90),
        "Paper": (-0.82, 0.22),
        "Device": (-0.48, 0.54),
        "StructuredObservation": (0.00, 0.55),
        "OntologyConcept": (0.55, 0.78),
        "ScientificClaim": (-0.35, -0.34),
        "Evidence": (0.18, -0.48),
        "ScientificEntity": (0.68, -0.22),
    }

    figure = plt.figure(figsize=(16, 9.2), dpi=220, facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=[4.2, 1.35], wspace=0.03)
    axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    axis.set_aspect("equal")
    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(-0.78, 1.04)
    axis.axis("off")
    table_axis.axis("off")

    pair_edges = sorted(edge_counts.items(), key=lambda item: item[1])
    for (start_label, end_label), count in pair_edges:
        start = positions[start_label]
        end = positions[end_label]
        width = 0.9 + 0.55 * math.log10(count + 1)
        if start_label == end_label:
            x, y = start
            arrow = FancyArrowPatch(
                (x - 0.055, y + 0.07),
                (x + 0.055, y + 0.07),
                connectionstyle="arc3,rad=-1.8",
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=width,
                color="#2A9D8F",
                alpha=0.80,
                zorder=1,
            )
        else:
            arrow = FancyArrowPatch(
                start,
                end,
                connectionstyle="arc3,rad=0.04",
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=width,
                color="#2F6690",
                alpha=0.72,
                shrinkA=28,
                shrinkB=28,
                zorder=1,
            )
        axis.add_patch(arrow)

    for label, (x, y) in positions.items():
        count = node_counts[label]
        radius = 0.072 + 0.011 * math.log10(count + 1)
        if label in {"DatasetSnapshot", "ExtractionRun"}:
            radius = max(radius, 0.088)
        axis.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=NODE_COLORS[label],
                edgecolor="#173042",
                linewidth=1.5,
                zorder=3,
            )
        )
        short = {
            "DatasetSnapshot": "Snapshot",
            "ExtractionRun": "Run",
            "StructuredObservation": "Observation",
            "ScientificClaim": "Claim",
            "ScientificEntity": "Entity",
            "OntologyConcept": "Ontology",
        }.get(label, label)
        text_color = "#20242A" if label in {"StructuredObservation", "OntologyConcept"} else "white"
        label_fontsize = 8.2 if label == "DatasetSnapshot" else 9.5
        axis.text(
            x,
            y,
            f"{short}\n{count:,}",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color=text_color,
            zorder=4,
        )

    figure.suptitle(
        "Full-corpus knowledge graph: conventional node-link view",
        x=0.05,
        y=0.965,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color="#20242A",
    )
    figure.text(
        0.05,
        0.918,
        "Node classes are aggregated for readability; arrow width scales with relationship count.",
        ha="left",
        fontsize=14,
        color="#555B63",
    )

    table_axis.text(0.02, 0.94, "Directed relationships", fontsize=16, fontweight="bold", color="#20242A")
    y = 0.89
    for relationship_type, count in relationship_counts.most_common():
        display = relationship_type.replace("_", " ").title()
        table_axis.text(0.02, y, display, fontsize=10.7, color="#20242A", va="center")
        table_axis.text(0.96, y, f"{count:,}", fontsize=10.7, color="#20242A", va="center", ha="right", fontweight="bold")
        y -= 0.052
    table_axis.text(
        0.02,
        0.08,
        "This is an aggregated view of the same\nfull production graph, not a separate graph.",
        fontsize=11.5,
        color="#555B63",
        va="bottom",
    )
    table_axis.set_xlim(0, 1)
    table_axis.set_ylim(0, 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(TYPE_NETWORK_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "node_counts": dict(node_counts),
        "aggregated_type_edges": [
            {"start_label": start, "end_label": end, "count": count}
            for (start, end), count in sorted(edge_counts.items())
        ],
    }


def main() -> None:
    for required in (NODES_CSV, RELATIONSHIPS_CSV, SUMMARY_JSON):
        if not required.exists():
            raise FileNotFoundError(required)
    nodes, relationships, summary = load_graph()
    full_report = render_full_network(nodes, relationships, summary)
    type_report = render_type_network(nodes, relationships)
    report = {
        "source_graph": summary.get("snapshot_id"),
        "graph_model": summary.get("graph_model"),
        "all_nodes_rendering": str(FULL_NETWORK_PNG),
        "type_level_rendering": str(TYPE_NETWORK_PNG),
        "full_network": full_report,
        "type_network": type_report,
    }
    LAYOUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"All-node rendering: {FULL_NETWORK_PNG}")
    print(f"Type-level rendering: {TYPE_NETWORK_PNG}")
    print(f"Nodes/relationships: {len(nodes):,}/{len(relationships):,}")


if __name__ == "__main__":
    main()
