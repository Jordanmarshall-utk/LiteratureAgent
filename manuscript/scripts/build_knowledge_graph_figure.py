"""Shared SVG drawing functions for the LiteratureAgent knowledge graph.

The supported Figure 6 entry point is ``build_fig6_integrated_graph_asset.py``.
The standalone functions below retain the preceding layout implementation and
are imported by the Origin drawing helpers.

The script validates the authoritative graph summary tables before drawing,
writes live-text SVG at exactly 6.5 inches wide, and renders a 3900-pixel PNG
with explicit 600-dpi metadata through the existing Chromium renderer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from render_svg_with_chrome import PNG_PIXELS_PER_METER_600_DPI, render as render_svg_png


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source"
SOURCE = PACKAGE / "graph_assets" / "fig6_integrated_graph"
EXPORT_ROOT = ROOT / "build"
INDIVIDUAL = EXPORT_ROOT / "figures_individual"
MULTIPANEL = EXPORT_ROOT / "figures_multipanel"
AUDIT_ROOT = EXPORT_ROOT / "svg_export_audits"
STEM = "Figure7_knowledge_graph_structure"

WIDTH_IN = 6.5
HEIGHT_IN = 5.0
WIDTH = 1300
HEIGHT = 1000
PNG_WIDTH = 3900

INK = "#080808"
SLATE = "#33546E"
SLATE_ALT = "#174766"
GRID = "#B2BCC9"
ORANGE = "#F28A00"
ORANGE_1 = "#FF8C00"
ORANGE_2 = "#FF9F25"
ORANGE_3 = "#FFB851"
ORANGE_4 = "#FFD082"
ORANGE_5 = "#FFDC9F"
PALE_ORANGE = "#FFF5E3"
PALE_SLATE = "#FCFCFD"
WHITE = "#FFFFFF"


def read_count_table(path: Path, key: str) -> dict[str, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return {str(row[key]): int(row["count"]) for row in rows}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sources() -> dict[str, object]:
    node_path = SOURCE / "node_counts.csv"
    relationship_path = SOURCE / "relationship_counts.csv"
    evidence_path = SOURCE / "TableS_KG_evidence_breakdown.csv"
    summary_path = SOURCE / "graph_summary.json"
    schema_path = SOURCE / "literatureagent_kg_schema_v2.json"
    validation_path = SOURCE / "validation_report.json"

    nodes = read_count_table(node_path, "node_type")
    relationships = read_count_table(relationship_path, "relationship_type")
    evidence = read_count_table(evidence_path, "quantity")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    expected_nodes = {
        "DatasetSnapshot": 1,
        "Device": 1843,
        "Evidence": 39770,
        "ExtractionRun": 1,
        "OntologyConcept": 266,
        "Paper": 1396,
        "ScientificClaim": 2694,
        "ScientificEntity": 5208,
        "StructuredObservation": 39844,
    }
    expected_relationships = {
        "DERIVED_FROM": 39770,
        "DIRECTIONAL_RELATION": 2694,
        "GENERATED_BY": 1,
        "HAS_DEVICE": 1843,
        "HAS_FIELD_EVIDENCE": 6689,
        "HAS_OBJECT": 2694,
        "HAS_OBSERVATION": 39844,
        "HAS_SUBJECT": 2694,
        "IN_SNAPSHOT": 1396,
        "MEASURES_PROPERTY": 39844,
        "REPORTS_CLAIM": 2694,
        "SUPPORTED_BY": 33668,
    }
    if nodes != expected_nodes:
        raise RuntimeError(f"Figure 7 node counts changed: {nodes}")
    if relationships != expected_relationships:
        raise RuntimeError(f"Figure 7 relationship counts changed: {relationships}")
    if sum(nodes.values()) != 91023 or sum(relationships.values()) != 173831:
        raise RuntimeError("Figure 7 graph totals do not match the validated manuscript snapshot")
    if evidence["Observation-to-evidence SUPPORTS relationships"] != 30974:
        raise RuntimeError("Observation support count changed")
    if evidence["Directional claims"] != 2694:
        raise RuntimeError("Claim support count changed")
    if 30974 + 2694 != relationships["SUPPORTED_BY"]:
        raise RuntimeError("SUPPORTED_BY split does not equal the validated total")
    if evidence["Total graph Evidence nodes"] != nodes["Evidence"]:
        raise RuntimeError("Evidence-node total mismatch")
    if not validation.get("validation_pass"):
        raise RuntimeError("The authoritative knowledge-graph validation report is not PASS")

    rel_types = schema["relationship_types"]
    for name in ("HAS_SUBJECT", "HAS_OBJECT"):
        if rel_types[name]["source"] != ["ScientificClaim"] or rel_types[name]["target"] != ["ScientificEntity"]:
            raise RuntimeError(f"{name} direction no longer matches Claim to Entity")
    if rel_types["DIRECTIONAL_RELATION"]["source"] != ["ScientificEntity"]:
        raise RuntimeError("Directional relation source changed")
    if summary["graph"]["nodes"] != 91023 or summary["graph"]["relationships"] != 173831:
        raise RuntimeError("graph_summary totals do not match the count tables")

    sources = [node_path, relationship_path, evidence_path, summary_path, schema_path, validation_path]
    return {
        "nodes": nodes,
        "relationships": relationships,
        "evidence": evidence,
        "summary": summary,
        "source_sha256": {str(path.resolve()): sha256(path) for path in sources},
    }


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text_element(
    x: float,
    y: float,
    lines: Iterable[str],
    *,
    size: float = 28,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "middle",
    line_height: float = 1.18,
    css_class: str = "",
) -> str:
    line_list = list(lines)
    klass = f' class="{esc(css_class)}"' if css_class else ""
    tspans = []
    for index, line in enumerate(line_list):
        dy = 0 if index == 0 else size * line_height
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}"{klass}>'
        + "".join(tspans)
        + "</text>"
    )


def rounded_rect(x: float, y: float, w: float, h: float, fill: str, *, stroke: str = INK, width: float = 4, radius: float = 18) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{radius:.1f}" ry="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def path_element(
    points: list[tuple[float, float]],
    *,
    color: str = INK,
    width: float = 4,
    dashed: bool = False,
    arrow: bool = True,
    css_class: str = "",
) -> str:
    if len(points) < 2:
        raise ValueError("A connector needs at least two points")
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    dash = ' stroke-dasharray="12 10"' if dashed else ""
    marker_id = "arrow-slate" if color == SLATE else "arrow"
    marker = f' marker-end="url(#{marker_id})"' if arrow else ""
    klass = f' class="{esc(css_class)}"' if css_class else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width:.1f}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash}{marker}{klass}/>'
    )


def label_chip(x: float, y: float, lines: list[str], *, width: float, height: float, size: float = 25, color: str = SLATE_ALT) -> str:
    top = y - height / 2
    line_shift = (len(lines) - 1) * size * 0.58
    return "".join(
        [
            rounded_rect(x - width / 2, top, width, height, WHITE, stroke=WHITE, width=2, radius=10),
            text_element(x, y - line_shift + size * 0.34, lines, size=size, weight=600, fill=color),
        ]
    )


def node_card(
    x: float,
    y: float,
    w: float,
    h: float,
    title_lines: list[str],
    count: int,
    *,
    fill: str,
    text_fill: str = INK,
) -> str:
    # SVG text coordinates use the baseline, not the glyph top.  Keep the
    # first baseline safely below the rounded border for both one- and
    # two-line node titles; the prior upward offset made two-line titles
    # visibly collide with the top border in the 600-dpi PNG.
    title_y = y + (39 if len(title_lines) == 1 else 30)
    return "".join(
        [
            rounded_rect(x, y, w, h, fill, stroke=INK, width=4.2, radius=18),
            text_element(
                x + w / 2,
                title_y,
                title_lines,
                size=27,
                weight=700,
                fill=text_fill,
                line_height=1.08,
            ),
            text_element(x + w / 2, y + h - 21, [f"n={count:,}"], size=25, weight=400, fill=text_fill),
        ]
    )


def build_svg(data: dict[str, object]) -> str:
    nodes: dict[str, int] = data["nodes"]  # type: ignore[assignment]
    relationships: dict[str, int] = data["relationships"]  # type: ignore[assignment]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_IN:.6f}in" height="{HEIGHT_IN:.6f}in" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        f'<marker id="arrow" markerWidth="8" markerHeight="8" refX="7.5" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="{INK}"/></marker>',
        f'<marker id="arrow-slate" markerWidth="8" markerHeight="8" refX="7.5" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="{SLATE}"/></marker>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{WHITE}"/>',
        '<g font-family="Times New Roman, Times, serif">',
        # Group cards follow Figure 1's rounded, black-outlined stage language.
        rounded_rect(40, 34, 1220, 196, "#F8F8F8", stroke=INK, width=4.2, radius=22),
        rounded_rect(40, 268, 1220, 304, PALE_ORANGE, stroke=INK, width=4.2, radius=22),
        rounded_rect(40, 612, 1220, 330, ORANGE_5, stroke=INK, width=4.2, radius=22),
        text_element(68, 76, ["PROVENANCE + SCOPE"], size=31, weight=700, anchor="start"),
        text_element(68, 310, ["STRUCTURED RECORDS"], size=31, weight=700, anchor="start"),
        text_element(68, 654, ["CLAIMS AND EVIDENCE"], size=31, weight=700, anchor="start"),
    ]

    # Draw connectors first so opaque nodes and white label chips mask endpoints.
    connectors = [
        # Provenance and primary structured-record flow.
        # Route the publication-to-snapshot and publication-to-claim edges
        # through the quiet left gutter so they do not cross section headers.
        path_element([(80, 416), (55, 416), (55, 242), (165, 242), (165, 212)], width=3.4),
        path_element([(240, 174), (365, 174)], width=3.4),
        path_element([(250, 442), (360, 442)]),
        path_element([(530, 442), (640, 442)]),
        path_element([(850, 442), (990, 442)]),
        # Paper reports claims.
        path_element([(80, 474), (55, 474), (55, 776), (340, 776)], width=3.4),
        # Claim subject/object and reported directional relation.
        path_element([(520, 776), (650, 776)], width=3.4),
        path_element([(740, 718), (740, 684), (880, 684), (880, 742), (830, 742)], width=2.8),
        # Exact observation and claim support routes.
        path_element([(745, 500), (745, 602), (1100, 602), (1100, 718)], width=3.2),
        path_element([(430, 834), (430, 890), (1100, 890), (1100, 834)], width=3.2),
        # The aggregate field-evidence bus is deliberately dashed and labeled once.
        path_element([(445, 500), (445, 580), (1218, 580), (1218, 760), (1190, 760)], color=SLATE, width=2.4, dashed=True),
        path_element([(165, 500), (165, 580), (445, 580)], color=SLATE, width=2.4, dashed=True, arrow=False),
        # Evidence provenance back to the publication/source identity.
        path_element([(1185, 834), (1185, 958), (28, 958), (28, 442), (80, 442)], color=SLATE, width=2.4),
    ]
    parts.extend(connectors)

    labels = [
        label_chip(185, 245, [f"in snapshot ({relationships['IN_SNAPSHOT']:,})"], width=255, height=38),
        text_element(303, 125, ["generated by", "(1)"], size=20, weight=600, fill=SLATE_ALT, line_height=1.05),
        text_element(305, 355, [f"has device ({relationships['HAS_DEVICE']:,})"], size=23, weight=600, fill=SLATE_ALT),
        text_element(585, 355, [f"has observation ({relationships['HAS_OBSERVATION']:,})"], size=23, weight=600, fill=SLATE_ALT),
        text_element(920, 355, [f"measures property ({relationships['MEASURES_PROPERTY']:,})"], size=23, weight=600, fill=SLATE_ALT),
        text_element(220, 692, [f"reports claim ({relationships['REPORTS_CLAIM']:,})"], size=23, weight=600, fill=SLATE_ALT),
        text_element(585, 692, ["subject + object: 5,388 links"], size=23, weight=600, fill=SLATE_ALT),
        label_chip(885, 675, ["reported directional relation", "2,694; non-causal"], width=260, height=64, size=23),
        text_element(930, 552, ["observation support: 30,974"], size=23, weight=600, fill=SLATE_ALT),
        label_chip(758, 875, ["claim evidence (2,694)"], width=250, height=42, size=23),
        label_chip(600, 580, [f"field evidence: {relationships['HAS_FIELD_EVIDENCE']:,} links total"], width=305, height=38, size=23, color=SLATE),
        label_chip(642, 940, [f"derived from publication/source ({relationships['DERIVED_FROM']:,})"], width=455, height=42, size=23, color=SLATE),
    ]
    parts.extend(labels)

    # Nodes use the exact Figure 1 orange progression and bold black outlines.
    parts.extend(
        [
            node_card(90, 93, 150, 116, ["Dataset", "snapshot"], nodes["DatasetSnapshot"], fill=PALE_SLATE),
            node_card(365, 93, 160, 116, ["Extraction", "run"], nodes["ExtractionRun"], fill=PALE_SLATE),
            node_card(80, 390, 170, 110, ["Publication /", "source"], nodes["Paper"], fill=ORANGE_1),
            node_card(360, 390, 170, 110, ["Device /", "sample"], nodes["Device"], fill=ORANGE_2),
            node_card(640, 390, 210, 110, ["Structured", "observation"], nodes["StructuredObservation"], fill=ORANGE_3),
            node_card(990, 390, 190, 110, ["Ontology", "concept"], nodes["OntologyConcept"], fill=ORANGE_4),
            node_card(340, 718, 180, 116, ["Scientific", "claim"], nodes["ScientificClaim"], fill=ORANGE_1),
            node_card(650, 718, 180, 116, ["Scientific", "entity"], nodes["ScientificEntity"], fill=ORANGE_3),
            node_card(1010, 718, 180, 116, ["Evidence"], nodes["Evidence"], fill=SLATE, text_fill=WHITE),
            rounded_rect(760, 86, 440, 122, WHITE, stroke=ORANGE, width=4.2, radius=18),
            text_element(980, 124, ["FINAL INTEGRATED GRAPH"], size=28, weight=700),
            text_element(980, 163, ["91,023 nodes  ·  173,831 relationships"], size=25, weight=600, fill=SLATE_ALT),
            text_element(980, 194, ["Validated final snapshot"], size=25, weight=400, fill=SLATE),
        ]
    )
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


def png_metadata(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG: {path}")
    width, height = struct.unpack(">II", data[16:24])
    offset = 8
    phys: list[tuple[int, int, int]] = []
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs":
            phys.append(struct.unpack(">IIB", chunk_data))
        offset += 12 + length
        if chunk_type == b"IEND":
            break
    return {"dimensions": [width, height], "phys": [list(item) for item in phys]}


def build_outputs() -> list[dict[str, object]]:
    data = validate_sources()
    svg_text = build_svg(data)
    ET.fromstring(svg_text)

    INDIVIDUAL.mkdir(parents=True, exist_ok=True)
    MULTIPANEL.mkdir(parents=True, exist_ok=True)
    individual_svg = INDIVIDUAL / f"{STEM}.svg"
    individual_png = INDIVIDUAL / f"{STEM}.png"
    individual_svg.write_text(svg_text, encoding="utf-8", newline="\n")
    rendered_width, rendered_height = render_svg_png(individual_svg, individual_png, PNG_WIDTH, None)
    metadata = png_metadata(individual_png)
    if rendered_width != PNG_WIDTH or metadata["dimensions"][0] != PNG_WIDTH:
        raise RuntimeError("Figure 7 PNG width is not 3900 pixels")
    if metadata["phys"] != [[PNG_PIXELS_PER_METER_600_DPI, PNG_PIXELS_PER_METER_600_DPI, 1]]:
        raise RuntimeError(f"Figure 7 PNG does not have exact 600-dpi pHYs metadata: {metadata['phys']}")

    multipanel_svg = MULTIPANEL / individual_svg.name
    multipanel_png = MULTIPANEL / individual_png.name
    shutil.copy2(individual_svg, multipanel_svg)
    shutil.copy2(individual_png, multipanel_png)

    svg_hash = sha256(individual_svg)
    png_hash = sha256(individual_png)
    audit_base = {
        "status": "PASS",
        "generator": str(Path(__file__).resolve()),
        "export_mode": "deterministic_python_svg_chrome_600dpi",
        "source_sha256": data["source_sha256"],
        "svg_width_inches": WIDTH_IN,
        "svg_height_inches": HEIGHT_IN,
        "svg_viewbox": [0, 0, WIDTH, HEIGHT],
        "png_dimensions": metadata["dimensions"],
        "png_phys": metadata["phys"],
        "svg_sha256": svg_hash,
        "png_sha256": png_hash,
        "minimum_font_points": 9.0,
        "node_total": 91023,
        "relationship_total": 173831,
        "supported_by_split": {"observation": 30974, "claim": 2694, "total": 33668},
        "direction_assertions": {
            "HAS_SUBJECT": "ScientificClaim -> ScientificEntity",
            "HAS_OBJECT": "ScientificClaim -> ScientificEntity",
            "DIRECTIONAL_RELATION": "ScientificEntity -> ScientificEntity; non-causal",
        },
        "removed_artifact_count": 0,
    }

    records: list[dict[str, object]] = []
    for directory, svg, png, copied_from in (
        (INDIVIDUAL, individual_svg, individual_png, None),
        (MULTIPANEL, multipanel_svg, multipanel_png, individual_svg),
    ):
        audit_dir = AUDIT_ROOT / directory.name
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{STEM}.json"
        audit = dict(audit_base)
        audit["destination"] = str(svg.resolve())
        if copied_from is not None:
            audit["identical_copy_of"] = str(copied_from.resolve())
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        record: dict[str, object] = {
            "graph": "F7_Python",
            "stem": STEM,
            "generator": str(Path(__file__).resolve()),
            "export_mode": "deterministic_python_svg_chrome_600dpi",
            "png": str(png),
            "png_bytes": png.stat().st_size,
            "png_dimensions": [rendered_width, rendered_height],
            "svg": str(svg),
            "svg_bytes": svg.stat().st_size,
            "svg_sanitizer_audit": str(audit_path),
            "removed_origin_text_artifacts": 0,
        }
        if copied_from is not None:
            record["identical_copy_of"] = str(copied_from)
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-records", action="store_true", help="Print manifest-ready export records")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit("Use scripts/build_fig6_integrated_graph_asset.py --aggregate-only for the released Figure 6 source tables.")


if __name__ == "__main__":
    main()
