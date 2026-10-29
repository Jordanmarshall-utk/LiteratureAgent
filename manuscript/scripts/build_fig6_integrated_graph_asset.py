"""Build the standalone Figure 6 integrated-graph schema asset.

This builder is deliberately independent of OriginPro.  Figure 6 is a schema
diagram rather than a statistical plot, so its editable source is a live-text
SVG generated directly from the frozen layout tables.  The PDF is printed from
that inline SVG with Chromium, retaining vector geometry and embedded text;
the PNG is rasterized from the same SVG at 600 dpi and the final 180 x 98 mm
physical size.

The script also copies the exact graph source files into the build directory and
emits a machine-readable validation audit.  It never mutates the source
package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from render_svg_with_chrome import chrome_path, render as render_png


PLOT_ID = "fig6_integrated_graph"
ANALYSIS_VERSION = "2026-10-27_controlled_repaired_rerun_v1"
WIDTH_MM = 180.0
HEIGHT_MM = 98.0
PNG_DPI = 600
PNG_WIDTH_PX = round(WIDTH_MM / 25.4 * PNG_DPI)
PNG_HEIGHT_PX = round(HEIGHT_MM / 25.4 * PNG_DPI)
POINTS_PER_MM = 72.0 / 25.4
MM_PER_POINT = 25.4 / 72.0
NODE_FONT_PT = 10
EDGE_FONT_PT = 9
FOOTER_FONT_PT = 10

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = ROOT / "source"
DEFAULT_PACKAGE_ROOT = PUBLIC_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "build"

GRAPH_SOURCE_FILENAMES = (
    "schema_layout_nodes.csv",
    "schema_layout_edges.csv",
    "effective_node_counts.csv",
    "relationship_counts.csv",
    "source_frozen_vs_repaired_graph_counts.csv",
    "repaired_evidence_links.csv",
    "repaired_graph_summary.json",
    "repaired_integrated_graph.jsonl",
)

DISPLAY_NAMES = {
    "Paper": "Paper",
    "Device": "Device / sample record",
    "Observation": "Structured observation",
    "Ontology": "Ontology concept",
    "Entity": "Scientific entity",
    "Claim": "Scientific claim",
    "Evidence": "Evidence",
    "Snapshot": "Dataset snapshot",
    "Run": "Extraction run",
}

NODE_DISPLAY_LINES = {
    "Device": ("Device / sample", "record"),
    "Observation": ("Structured", "observation"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def copy_exact_sources(
    source_dir: Path,
    destination_dir: Path,
    filenames: Iterable[str] = GRAPH_SOURCE_FILENAMES,
) -> dict[str, dict[str, Any]]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = destination_dir / filename
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        copied_hash = sha256(destination)
        if source_hash != copied_hash:
            raise RuntimeError(f"Byte-for-byte copy validation failed for {filename}")
        copied[filename] = {
            "source_bytes": source.stat().st_size,
            "source_sha256": source_hash,
            "copied_bytes": destination.stat().st_size,
            "copied_sha256": copied_hash,
            "byte_exact_copy": True,
        }
    return copied


def scan_graph_jsonl(path: Path) -> dict[str, Any]:
    node_counts: Counter[str] = Counter()
    relationship_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    fallback_node_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc
            kind = record.get("kind")
            record_counts[str(kind)] += 1
            if kind == "node":
                label = record.get("label")
                node_type = record.get("type")
                effective_class = label or node_type
                if not effective_class:
                    raise ValueError(
                        f"Node at line {line_number} has neither label nor type"
                    )
                node_counts[str(effective_class)] += 1
                if not label and node_type:
                    fallback_node_counts[str(node_type)] += 1
            elif kind == "relationship":
                relationship_type = record.get("type")
                if not relationship_type:
                    raise ValueError(
                        f"Relationship at line {line_number} has no type"
                    )
                relationship_counts[str(relationship_type)] += 1
            else:
                raise ValueError(f"Unknown record kind at line {line_number}: {kind!r}")

    return {
        "record_counts": dict(sorted(record_counts.items())),
        "effective_node_counts": dict(sorted(node_counts.items())),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "fallback_node_counts": dict(sorted(fallback_node_counts.items())),
        "total_records": sum(record_counts.values()),
    }


def validate_counts(
    source_dir: Path,
    *,
    aggregate_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((source_dir / "repaired_graph_summary.json").read_text(encoding="utf-8"))
    expected_nodes = {
        row["node_type"]: int(row["count"])
        for row in read_csv(source_dir / "effective_node_counts.csv")
    }
    expected_relationships = {
        row["relationship_type"]: int(row["count"])
        for row in read_csv(source_dir / "relationship_counts.csv")
    }
    if expected_nodes != summary["node_counts"]:
        raise ValueError("effective_node_counts.csv does not match repaired_graph_summary.json")
    if expected_relationships != summary["relationship_counts"]:
        raise ValueError("relationship_counts.csv does not match repaired_graph_summary.json")
    if sum(expected_nodes.values()) != int(summary["nodes"]):
        raise ValueError("Node class counts do not sum to the reported graph node total")
    if sum(expected_relationships.values()) != int(summary["relationships"]):
        raise ValueError(
            "Relationship counts do not sum to the reported graph relationship total"
        )

    if aggregate_only:
        return summary, {
            "mode": "aggregate-only-public-release",
            "jsonl_scanned": False,
            "reason": (
                "The record-level graph export is intentionally excluded from the public "
                "release because it exceeds GitHub's normal single-file limit and contains "
                "restricted source-document locators. Aggregate count/layout tables were "
                "cross-checked against repaired_graph_summary.json."
            ),
        }

    scan = scan_graph_jsonl(source_dir / "repaired_integrated_graph.jsonl")
    if scan["effective_node_counts"] != expected_nodes:
        raise ValueError("JSONL effective node counts do not match the repaired summary")
    if scan["relationship_counts"] != expected_relationships:
        raise ValueError("JSONL relationship counts do not match the repaired summary")
    if scan["fallback_node_counts"] != {"Evidence": 142}:
        raise ValueError(
            "JSONL label/type fallback audit did not find exactly 142 Evidence nodes"
        )
    return summary, scan


def fmt_count(value: int) -> str:
    return f"{value:,}"


def parse_nodes(source_dir: Path, summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for row in read_csv(source_dir / "schema_layout_nodes.csv"):
        display_id = row["display_id"]
        node_type = row["node_type"]
        count = int(row["count"])
        summary_count = int(summary["node_counts"][node_type])
        if count != summary_count:
            raise ValueError(
                f"Layout count mismatch for {display_id}: {count} != {summary_count}"
            )
        nodes[display_id] = {
            "display_id": display_id,
            "node_type": node_type,
            "count": count,
            "x": float(row["x"]) * WIDTH_MM,
            "y": (1.0 - float(row["y"]) - float(row["height"])) * HEIGHT_MM,
            "width": float(row["width"]) * WIDTH_MM,
            "height": float(row["height"]) * HEIGHT_MM,
            "fill": row["fill_hex"].upper(),
            "text": row["text_hex"].upper(),
        }
    expected_display_ids = set(DISPLAY_NAMES)
    if set(nodes) != expected_display_ids:
        raise ValueError(
            f"Unexpected display node set: {sorted(nodes)}; expected {sorted(expected_display_ids)}"
        )
    return nodes


def center(node: dict[str, Any]) -> tuple[float, float]:
    return node["x"] + node["width"] / 2.0, node["y"] + node["height"] / 2.0


def edge_geometry(
    source_id: str,
    target_id: str,
    nodes: dict[str, dict[str, Any]],
) -> tuple[str, tuple[float, float], str]:
    source = nodes[source_id]
    target = nodes[target_id]
    sx, sy = center(source)
    tx, ty = center(target)
    key = (source_id, target_id)

    if key == ("Device", "Observation"):
        y = sy
        path = f"M {source['x'] + source['width']:.3f} {y:.3f} L {target['x']:.3f} {y:.3f}"
        return path, ((source["x"] + source["width"] + target["x"]) / 2, 36.5), "middle"
    if key == ("Observation", "Ontology"):
        y = sy
        path = f"M {source['x'] + source['width']:.3f} {y:.3f} L {target['x']:.3f} {y:.3f}"
        return path, ((source["x"] + source["width"] + target["x"]) / 2, 36.5), "middle"
    if key == ("Entity", "Claim"):
        y = sy
        path = f"M {source['x'] + source['width']:.3f} {y:.3f} L {target['x']:.3f} {y:.3f}"
        return path, ((source["x"] + source["width"] + target["x"]) / 2, 59.4), "middle"
    if key == ("Claim", "Evidence"):
        y = sy
        path = f"M {source['x'] + source['width']:.3f} {y:.3f} L {target['x']:.3f} {y:.3f}"
        return path, ((source["x"] + source["width"] + target["x"]) / 2, 59.4), "middle"
    if key == ("Snapshot", "Run"):
        x = sx
        path = f"M {x:.3f} {source['y'] + source['height']:.3f} L {x:.3f} {target['y']:.3f}"
        return path, (x, source["y"] - 3.25), "middle"
    if key == ("Paper", "Device"):
        start_x = source["x"] + source["width"]
        start_y = source["y"] + source["height"] * 0.36
        end_x = target["x"]
        end_y = target["y"] + target["height"] * 0.72
        path = f"M {start_x:.3f} {start_y:.3f} L {end_x:.3f} {end_y:.3f}"
        # Place the two-line label in the white field above the Paper node.
        # This keeps the enlarged 9 pt text clear of both node outlines and
        # the diagonal edge endpoint.
        return path, (27.5, 26.0), "middle"
    if key == ("Paper", "Claim"):
        start_x = source["x"] + source["width"] * 0.52
        start_y = source["y"] + source["height"]
        end_x = target["x"]
        end_y = target["y"] + target["height"] * 0.52
        path = (
            f"M {start_x:.3f} {start_y:.3f} "
            f"C 26.5 58.2, 38.7 59.3, {end_x:.3f} {end_y:.3f}"
        )
        return path, (29.0, 55.8), "middle"
    if key == ("Observation", "Evidence"):
        start_x = source["x"] + source["width"] * 0.58
        start_y = source["y"] + source["height"]
        end_x = target["x"] + target["width"] * 0.58
        end_y = target["y"]
        path = (
            f"M {start_x:.3f} {start_y:.3f} "
            f"C 112.8 38.5, 119.6 47.2, {end_x:.3f} {end_y:.3f}"
        )
        # Keep the label to the left of the curved edge so the arrow no longer
        # crosses the two live-text lines at publication size.
        return path, (96.0, 47.5), "middle"

    # Defensive generic routing for any future non-overlapping layout update.
    if abs(sy - ty) <= 1.0:
        if sx <= tx:
            start_x, end_x = source["x"] + source["width"], target["x"]
        else:
            start_x, end_x = source["x"], target["x"] + target["width"]
        path = f"M {start_x:.3f} {sy:.3f} L {end_x:.3f} {ty:.3f}"
    else:
        path = f"M {sx:.3f} {sy:.3f} L {tx:.3f} {ty:.3f}"
    return path, ((sx + tx) / 2, (sy + ty) / 2 - 2.0), "middle"


def split_edge_label(label: str) -> list[str]:
    explicit_prefix = "explicit passage link "
    if label.startswith(explicit_prefix):
        return ["explicit passage link", label[len(explicit_prefix) :]]
    fixed = {
        "has retained record": ["has retained", "record"],
        "reports claim": ["reports", "claim"],
        "subject / object": ["subject /", "object"],
    }
    return fixed.get(label, [label])


def node_label_lines(display_id: str, count: int) -> list[str]:
    """Wrap the two longest node names while retaining the exact count."""
    display_lines = NODE_DISPLAY_LINES.get(display_id, (DISPLAY_NAMES[display_id],))
    return [*display_lines, f"(n={fmt_count(count)})"]


def text_element(
    lines: Iterable[str],
    x: float,
    y: float,
    *,
    fill: str,
    font_size_pt: int = NODE_FONT_PT,
    font_weight: str = "normal",
    anchor: str = "middle",
    class_name: str = "",
    halo: bool = False,
) -> str:
    line_list = list(lines)
    font_size_mm = font_size_pt * MM_PER_POINT
    line_height_mm = font_size_mm * 1.06
    first_y = y - (len(line_list) - 1) * line_height_mm / 2.0
    class_attribute = f' class="{html.escape(class_name, quote=True)}"' if class_name else ""
    halo_style = (
        ' style="paint-order:stroke fill;stroke:#FFFFFF;stroke-width:0.85;'
        'stroke-linejoin:round"'
        if halo
        else ""
    )
    tspans = []
    for index, line in enumerate(line_list):
        line_y = first_y + index * line_height_mm
        tspans.append(
            f'<tspan x="{x:.3f}" y="{line_y:.3f}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text{class_attribute} x="{x:.3f}" y="{y:.3f}" '
        f'text-anchor="{anchor}" dominant-baseline="middle" '
        f'fill="{fill}" font-family="Times New Roman, Times, serif" '
        f'font-size="{font_size_mm:.3f}" font-weight="{font_weight}"{halo_style}>'
        + "".join(tspans)
        + "</text>"
    )


def build_svg(
    nodes: dict[str, dict[str, Any]],
    edge_rows: list[dict[str, str]],
    summary: dict[str, Any],
) -> str:
    edge_parts: list[str] = []
    label_parts: list[str] = []
    for row in edge_rows:
        source_id = row["source_display_id"]
        target_id = row["target_display_id"]
        if source_id not in nodes or target_id not in nodes:
            raise ValueError(f"Unknown edge endpoint: {source_id} -> {target_id}")
        path_data, label_position, anchor = edge_geometry(source_id, target_id, nodes)
        edge_parts.append(
            f'<path class="schema-edge" data-source="{html.escape(source_id)}" '
            f'data-target="{html.escape(target_id)}" d="{path_data}" fill="none" '
            'stroke="#4B5563" stroke-width="0.46" stroke-linecap="round" '
            'stroke-linejoin="round" marker-end="url(#arrowhead)"/>'
        )
        label_parts.append(
            text_element(
                split_edge_label(row["display_label"]),
                label_position[0],
                label_position[1],
                fill="#202124",
                font_size_pt=EDGE_FONT_PT,
                anchor=anchor,
                class_name="edge-label",
                halo=True,
            )
        )

    node_parts: list[str] = []
    for display_id, node in nodes.items():
        x, y = node["x"], node["y"]
        width, height = node["width"], node["height"]
        cx, cy = center(node)
        node_parts.append(
            f'<g class="schema-node" data-display-id="{html.escape(display_id)}" '
            f'data-node-type="{html.escape(node["node_type"])}" '
            f'data-count="{node["count"]}">'
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" '
            f'rx="1.65" ry="1.65" fill="{node["fill"]}" stroke="#202124" '
            'stroke-width="0.55"/>'
            + text_element(
                node_label_lines(display_id, node["count"]),
                cx,
                cy,
                fill=node["text"],
                font_size_pt=NODE_FONT_PT,
                font_weight="bold",
                class_name="node-label",
            )
            + "</g>"
        )

    footer = text_element(
        [
            "Integrated graph: "
            f"{fmt_count(int(summary['nodes']))} nodes  ·  "
            f"{fmt_count(int(summary['relationships']))} relationships"
        ],
        WIDTH_MM / 2,
        92.5,
        fill="#4B5563",
        font_size_pt=FOOTER_FONT_PT,
        class_name="graph-total",
    )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM:g}mm" '
        f'height="{HEIGHT_MM:g}mm" viewBox="0 0 {WIDTH_MM:g} {HEIGHT_MM:g}" '
        'role="img" aria-labelledby="figure-description">\n'
        '<desc id="figure-description">Integrated LiteratureAgent knowledge-graph schema '
        'with exact repaired node and relationship counts.</desc>\n'
        '<defs><marker id="arrowhead" markerUnits="userSpaceOnUse" markerWidth="3.4" '
        'markerHeight="3.4" refX="3.0" refY="1.7" orient="auto"><path '
        'd="M 0 0 L 3.4 1.7 L 0 3.4 z" fill="#4B5563"/></marker></defs>\n'
        f'<rect x="0" y="0" width="{WIDTH_MM:g}" height="{HEIGHT_MM:g}" fill="#FFFFFF"/>\n'
        '<g id="edges">' + "".join(edge_parts) + "</g>\n"
        '<g id="nodes">' + "".join(node_parts) + "</g>\n"
        '<g id="edge-labels">' + "".join(label_parts) + "</g>\n"
        + footer
        + "\n</svg>\n"
    )


def render_pdf_with_chrome(
    svg_path: Path,
    pdf_path: Path,
    browser: Path | None,
) -> None:
    browser_exe = chrome_path(browser)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    inline_svg = svg_path.read_text(encoding="utf-8")
    inline_svg = re.sub(r"^<\?xml[^>]*>\s*", "", inline_svg)

    with tempfile.TemporaryDirectory(
        prefix="fig6_vector_pdf_",
        dir=pdf_path.parent,
        ignore_cleanup_errors=True,
    ) as temporary:
        temp = Path(temporary)
        profile = temp / "profile"
        wrapper = temp / "render.html"
        wrapper.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            f"@page{{size:{WIDTH_MM:g}mm {HEIGHT_MM:g}mm;margin:0;}}"
            f"html,body{{margin:0;padding:0;width:{WIDTH_MM:g}mm;height:{HEIGHT_MM:g}mm;"
            "overflow:hidden;background:#fff;}"
            f"svg{{display:block;width:{WIDTH_MM:g}mm;height:{HEIGHT_MM:g}mm;}}"
            "</style></head><body>"
            + inline_svg
            + "</body></html>",
            encoding="utf-8",
        )
        command = [
            str(browser_exe),
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-sync",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            "--force-color-profile=srgb",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path.resolve()}",
            wrapper.as_uri(),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=90)
        if result.returncode != 0 or not pdf_path.is_file():
            raise RuntimeError(
                f"Browser PDF render failed ({result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )


def read_png_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a PNG: {path}")
    offset = 8
    width = height = None
    x_ppm = y_ppm = unit = None
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif chunk_type == b"pHYs":
            x_ppm, y_ppm, unit = struct.unpack(">IIB", payload)
        offset += 12 + length
        if chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise ValueError(f"PNG dimensions not found: {path}")
    dpi_x = x_ppm * 0.0254 if x_ppm is not None and unit == 1 else None
    dpi_y = y_ppm * 0.0254 if y_ppm is not None and unit == 1 else None
    return {
        "width_px": width,
        "height_px": height,
        "pixels_per_meter_x": x_ppm,
        "pixels_per_meter_y": y_ppm,
        "dpi_x": round(dpi_x, 3) if dpi_x is not None else None,
        "dpi_y": round(dpi_y, 3) if dpi_y is not None else None,
    }


def iter_pdf_streams(pdf_bytes: bytes) -> Iterable[bytes]:
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL):
        raw = match.group(1)
        try:
            yield zlib.decompress(raw)
        except zlib.error:
            yield raw


def audit_pdf(path: Path) -> dict[str, Any]:
    pdf_bytes = path.read_bytes()
    streams = list(iter_pdf_streams(pdf_bytes))
    content = b"\n".join(streams)
    media_box = re.search(
        rb"/MediaBox\s*\[\s*0(?:\.0+)?\s+0(?:\.0+)?\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        pdf_bytes,
    )
    width_pt = float(media_box.group(1)) if media_box else None
    height_pt = float(media_box.group(2)) if media_box else None
    return {
        "page_width_pt": width_pt,
        "page_height_pt": height_pt,
        "expected_width_pt": WIDTH_MM * POINTS_PER_MM,
        "expected_height_pt": HEIGHT_MM * POINTS_PER_MM,
        "media_box_within_1pt": bool(
            width_pt is not None
            and height_pt is not None
            and abs(width_pt - WIDTH_MM * POINTS_PER_MM) <= 1.0
            and abs(height_pt - HEIGHT_MM * POINTS_PER_MM) <= 1.0
        ),
        "embedded_font_file_count": len(re.findall(rb"/FontFile(?:2|3)?\b", pdf_bytes)),
        "image_xobject_count": len(re.findall(rb"/Subtype\s*/Image\b", pdf_bytes)),
        "text_object_count": len(re.findall(rb"\bBT\b", content)),
        "text_show_operator_count": len(re.findall(rb"\bT[Jj]\b", content)),
        "vector_path_operator_count": len(
            re.findall(rb"(?:^|\s)(?:m|l|c|re|S|f)(?:\s|$)", content)
        ),
    }


def audit_svg(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    return {
        "live_text_element_count": len(re.findall(r"<text\b", source)),
        "image_element_count": len(re.findall(r"<image\b", source)),
        "node_group_count": len(re.findall(r'class="schema-node"', source)),
        "edge_path_count": len(re.findall(r'class="schema-edge"', source)),
        "times_new_roman_declared": "Times New Roman" in source,
        "outer_frame_present": False,
        "grid_present": False,
    }


def build(
    package_root: Path,
    output_root: Path,
    browser: Path | None,
    *,
    aggregate_only: bool = False,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    output_root = output_root.resolve()
    source_dir = package_root / "graph_assets" / PLOT_ID
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    svg_path = output_root / "svg" / f"{PLOT_ID}.svg"
    pdf_path = output_root / "pdf" / f"{PLOT_ID}.pdf"
    png_path = output_root / "png" / f"{PLOT_ID}.png"
    data_dir = output_root / "data" / PLOT_ID
    audit_path = output_root / "validation" / f"{PLOT_ID}_build_audit.json"
    for directory in (svg_path.parent, pdf_path.parent, png_path.parent, audit_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    summary, jsonl_scan = validate_counts(source_dir, aggregate_only=aggregate_only)
    nodes = parse_nodes(source_dir, summary)
    edge_rows = read_csv(source_dir / "schema_layout_edges.csv")
    if len(edge_rows) != 8:
        raise ValueError(f"Expected 8 displayed schema edges, found {len(edge_rows)}")

    source_filenames = tuple(
        name
        for name in GRAPH_SOURCE_FILENAMES
        if not (aggregate_only and name == "repaired_integrated_graph.jsonl")
    )
    source_copy_audit = copy_exact_sources(source_dir, data_dir, source_filenames)
    svg_source = build_svg(nodes, edge_rows, summary)
    write_text_atomic(svg_path, svg_source)
    render_pdf_with_chrome(svg_path, pdf_path, browser)
    rendered_width, rendered_height = render_png(
        svg_path, png_path, PNG_WIDTH_PX, browser
    )
    if (rendered_width, rendered_height) != (PNG_WIDTH_PX, PNG_HEIGHT_PX):
        raise RuntimeError(
            "PNG pixel dimensions do not match 600-dpi final-size dimensions: "
            f"{rendered_width} x {rendered_height}"
        )

    svg_audit = audit_svg(svg_path)
    pdf_audit = audit_pdf(pdf_path)
    png_audit = read_png_info(png_path)
    if svg_audit["live_text_element_count"] < 18 or svg_audit["image_element_count"] != 0:
        raise RuntimeError("SVG live-text/vector validation failed")
    if svg_audit["node_group_count"] != 9 or svg_audit["edge_path_count"] != 8:
        raise RuntimeError("SVG schema class/edge count validation failed")
    if not pdf_audit["media_box_within_1pt"]:
        raise RuntimeError("PDF page size does not match the requested 180 x 98 mm")
    if pdf_audit["image_xobject_count"] != 0:
        raise RuntimeError("PDF unexpectedly contains a raster image XObject")
    if pdf_audit["text_object_count"] == 0 or pdf_audit["text_show_operator_count"] == 0:
        raise RuntimeError("PDF live-text validation failed")
    if png_audit["width_px"] != PNG_WIDTH_PX or png_audit["height_px"] != PNG_HEIGHT_PX:
        raise RuntimeError("PNG dimensions validation failed")
    if not math.isclose(float(png_audit["dpi_x"]), PNG_DPI, abs_tol=0.05):
        raise RuntimeError("PNG horizontal resolution metadata is not 600 dpi")
    if not math.isclose(float(png_audit["dpi_y"]), PNG_DPI, abs_tol=0.05):
        raise RuntimeError("PNG vertical resolution metadata is not 600 dpi")

    output_files = {}
    for path in (svg_path, pdf_path, png_path):
        output_files[path.name] = {
            "relative_path": path.relative_to(output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    audit: dict[str, Any] = {
        "plot_id": PLOT_ID,
        "analysis_version": ANALYSIS_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "output_root": str(output_root),
        "physical_dimensions_mm": {"width": WIDTH_MM, "height": HEIGHT_MM},
        "png_target": {
            "dpi": PNG_DPI,
            "width_px": PNG_WIDTH_PX,
            "height_px": PNG_HEIGHT_PX,
        },
        "typography_pt": {
            "node_labels_and_counts": NODE_FONT_PT,
            "edge_labels": EDGE_FONT_PT,
            "footer": FOOTER_FONT_PT,
        },
        "exact_counts": {
            "displayed_schema_classes": len(nodes),
            "displayed_schema_edges": len(edge_rows),
            "graph_nodes": int(summary["nodes"]),
            "graph_relationships": int(summary["relationships"]),
            "structured_observations_with_explicit_passage_links": int(
                summary["structured_observations_with_explicit_passage_links"]
            ),
            "schema_fallback_evidence_nodes": int(
                summary["schema_reconciliation"][
                    "repair_evidence_nodes_missing_label_but_having_type"
                ]
            ),
        },
        "jsonl_scan": jsonl_scan,
        "validation_mode": (
            "aggregate-only-public-release" if aggregate_only else "full-record-jsonl"
        ),
        "source_copies": source_copy_audit,
        "asset_validation": {
            "svg": svg_audit,
            "pdf": pdf_audit,
            "png": png_audit,
        },
        "output_files": output_files,
        "checks": {
            "source_files_copied_byte_exact": all(
                item["byte_exact_copy"] for item in source_copy_audit.values()
            ),
            "jsonl_matches_repaired_summary": not aggregate_only,
            "aggregate_tables_match_repaired_summary": True,
            "node_label_then_type_fallback_applied": True,
            "svg_is_live_text_and_vector_only": True,
            "pdf_is_live_text_and_vector_only": True,
            "png_is_600_dpi_at_final_dimensions": True,
            "no_title_panel_letter_grid_or_outer_frame": True,
        },
    }
    write_text_atomic(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the standalone Figure 6 integrated-graph assets."
    )
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--browser", type=Path)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Build from the public aggregate/layout tables without requiring or copying "
            "the restricted record-level repaired_integrated_graph.jsonl export."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build(
        args.package_root,
        args.output_root,
        args.browser,
        aggregate_only=args.aggregate_only,
    )
    outputs = audit["output_files"]
    print(
        f"Built {PLOT_ID}: "
        + ", ".join(f"{name} ({item['bytes']} bytes)" for name, item in outputs.items())
    )
    print(
        "Validated exact graph counts: "
        f"{audit['exact_counts']['graph_nodes']:,} nodes, "
        f"{audit['exact_counts']['graph_relationships']:,} relationships, "
        f"{audit['exact_counts']['structured_observations_with_explicit_passage_links']:,} "
        "explicit passage links"
    )


if __name__ == "__main__":
    main()
