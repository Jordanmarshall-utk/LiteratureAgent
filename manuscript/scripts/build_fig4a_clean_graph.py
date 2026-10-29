"""Build the clean standalone Figure 4a evidence-graph panel.

Figure 4a is an editable diagram rather than a statistical chart, so it is
generated as live-text SVG from the authoritative five-paper source table.
The panel intentionally contains no panel letter, chart title, subtitle,
footer, outer card, or embedded legend; a separate legend is exported for
PowerPoint assembly.  Scientific identifiers and full evidence text remain in
the copied CSV while the artwork uses the approved shortened display labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from render_svg_with_chrome import chrome_path, render


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "2026-10-27_controlled_repaired_rerun"
PUBLIC_SOURCE_ROOT = ROOT / "source"
PACKAGE = PUBLIC_SOURCE_ROOT
SOURCE = PACKAGE / "graph_assets" / "fig4a_five_paper_graph" / "PanelA_source_paths.csv"
DEFAULT_OUTPUT = ROOT / "build"

# Match the tall left-panel geometry used by the approved manuscript deck.
# Keeping ten SVG units per millimetre makes the whole-number point sizes
# below physically meaningful while giving the relation labels enough room.
WIDTH_MM = 115.6
HEIGHT_MM = 184.7
VIEW_WIDTH = 1156
VIEW_HEIGHT = 1847
FONT = "Times New Roman"
HEADER_FONT_PT = 12
PAPER_FONT_PT = 9
CLAIM_FONT_PT = 10
SUBJECT_FONT_PT = 10
SUBJECT_COMPACT_FONT_PT = 9
OBJECT_FONT_PT = 10
RELATION_FONT_PT = 10
INK = "#202124"
NAVY = "#33546E"
ORANGE = "#F28A00"
NEUTRAL = "#6B7280"
PALE = "#FAF7F2"
WHITE = "#FFFFFF"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

REQUIRED_COLUMNS = (
    "paper_label",
    "publication_key",
    "doi",
    "year",
    "title",
    "paper_node_id",
    "claim_node_id",
    "claim_text",
    "subject_node_id",
    "subject_text",
    "predicate",
    "object_node_id",
    "object_text",
    "evidence_node_id",
    "evidence_text",
    "evidence_locator",
)

DISPLAY = {
    "P1": {
        "paper": ("P1", "2024", "L-EGT"),
        "subject": ("L-EGT", "passivation", "material"),
        "object": ("perovskite", "film quality"),
    },
    "P2": {
        "paper": ("P2", "2026", "Sn–Pb"),
        "subject": ("TLP surface", "polishing"),
        "object": ("charge", "extraction"),
    },
    "P3": {
        "paper": ("P3", "2021", "Al₂O₃"),
        "subject": ("electrochemical", "Al₂O₃", "incorporation"),
        "object": ("cell", "efficiency", "12.1–14.3%"),
    },
    "P4": {
        "paper": ("P4", "2023", "MDACl₂"),
        "subject": ("MDACl₂-modified", "high-quality", "films"),
        "object": ("crystallinity", "and reduced", "defect density"),
    },
    "P5": {
        "paper": ("P5 · 2021", "Nano-", "conf."),
        "subject": ("crystallization", "optimization"),
        "object": ("inorganic", "PSC", "performance"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source() -> list[dict[str, str]]:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = tuple(reader.fieldnames or ())
        if headers != REQUIRED_COLUMNS or len(headers) != len(set(headers)):
            raise RuntimeError(f"Unexpected or duplicate Figure 4a CSV headers: {headers}")
        rows = list(reader)
    if len(rows) != 5 or tuple(row["paper_label"] for row in rows) != tuple(DISPLAY):
        raise RuntimeError("Figure 4a requires the authoritative P1–P5 rows in order")
    for row in rows:
        if not all(row[column].strip() for column in REQUIRED_COLUMNS):
            raise RuntimeError(f"Figure 4a has an empty required field: {row['paper_label']}")
        if row["predicate"].strip().upper() not in {"IMPROVES", "PROMOTES", "INCREASES", "ENHANCES"}:
            raise RuntimeError(f"Unexpected predicate: {row['predicate']!r}")
    return rows


def svg_units_from_pt(points: int) -> float:
    """Convert whole-number physical points to this SVG's viewBox units."""
    return points * (VIEW_WIDTH / (WIDTH_MM / 25.4)) / 72.0


def text_block(x: float, y: float, lines: tuple[str, ...], *, size: float, weight: str = "normal") -> str:
    line_height = size * 1.05
    start = y - line_height * (len(lines) - 1) / 2
    spans = "".join(
        f'<tspan x="{x:g}" y="{start + index * line_height:g}">{html.escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="{size:g}" font-weight="{weight}" fill="{INK}">{spans}</text>'
    )


def svg_panel(rows: list[dict[str, str]]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}">',
        "<title>Five representative evidence-linked knowledge-graph paths</title>",
        "<desc>Five exact paper-to-claim-to-entity paths with evidence links and literature-reported relations. Full source text and identifiers are preserved in the accompanying CSV.</desc>",
        f'<rect width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" fill="{WHITE}"/>',
        "<defs>",
        f'<marker id="arrow-navy" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{NAVY}"/></marker>',
        f'<marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{ORANGE}"/></marker>',
        f'<marker id="arrow-gray" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{NEUTRAL}"/></marker>',
        f'<marker id="arrow-evidence" markerWidth="6" markerHeight="6" refX="5.4" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L6,3 z" fill="{NAVY}"/></marker>',
        "</defs>",
        text_block(105, 60, ("Paper",), size=svg_units_from_pt(HEADER_FONT_PT), weight="bold"),
        text_block(260, 60, ("Claim",), size=svg_units_from_pt(HEADER_FONT_PT), weight="bold"),
        text_block(480, 60, ("Subject",), size=svg_units_from_pt(HEADER_FONT_PT), weight="bold"),
        text_block(1005, 60, ("Object",), size=svg_units_from_pt(HEADER_FONT_PT), weight="bold"),
    ]

    row_centers = (230, 570, 910, 1250, 1590)
    for index, (row, center_y) in enumerate(zip(rows, row_centers), start=1):
        display = DISPLAY[row["paper_label"]]
        subject_size = svg_units_from_pt(
            SUBJECT_COMPACT_FONT_PT
            if row["paper_label"] in {"P3", "P4", "P5"}
            else SUBJECT_FONT_PT
        )
        if index > 1:
            parts.append(f'<line x1="25" y1="{center_y - 170}" x2="1131" y2="{center_y - 170}" stroke="#E5E7EB" stroke-width="2"/>')

        # Exact graph topology: paper→claim→subject, subject→object,
        # claim→object, and evidence→claim for every source row.
        parts.extend(
            [
                f'<line x1="195" y1="{center_y}" x2="220" y2="{center_y}" stroke="{NAVY}" stroke-width="4" marker-end="url(#arrow-navy)"/>',
                f'<line x1="300" y1="{center_y}" x2="330" y2="{center_y}" stroke="{NAVY}" stroke-width="4" marker-end="url(#arrow-navy)"/>',
                f'<line x1="650" y1="{center_y}" x2="870" y2="{center_y}" stroke="{ORANGE}" stroke-width="5" marker-end="url(#arrow-orange)"/>',
                f'<path d="M290 {center_y + 34} L340 {center_y + 112} L820 {center_y + 112} L870 {center_y + 56}" fill="none" stroke="{NEUTRAL}" stroke-width="3" marker-end="url(#arrow-gray)"/>',
                f'<line x1="260" y1="{center_y - 102}" x2="260" y2="{center_y - 43}" stroke="{NAVY}" stroke-width="3" stroke-dasharray="6 4" marker-end="url(#arrow-evidence)"/>',
                f'<rect x="15" y="{center_y - 105}" width="180" height="210" rx="18" fill="{PALE}" stroke="{NAVY}" stroke-width="4"/>',
                f'<circle cx="260" cy="{center_y}" r="40" fill="{PALE}" stroke="{ORANGE}" stroke-width="5"/>',
                f'<rect x="330" y="{center_y - 110}" width="320" height="220" rx="18" fill="{PALE}" stroke="{NAVY}" stroke-width="4"/>',
                f'<rect x="875" y="{center_y - 110}" width="260" height="220" rx="18" fill="{PALE}" stroke="{NAVY}" stroke-width="4"/>',
                f'<path d="M260 {center_y - 148} l24 24 -24 24 -24 -24 z" fill="{WHITE}" stroke="{NAVY}" stroke-width="4"/>',
                text_block(105, center_y + 1, display["paper"], size=svg_units_from_pt(PAPER_FONT_PT), weight="bold"),
                text_block(260, center_y + 9, (f"C{index}",), size=svg_units_from_pt(CLAIM_FONT_PT), weight="bold"),
                text_block(490, center_y + 2, display["subject"], size=subject_size, weight="bold"),
                text_block(1005, center_y + 2, display["object"], size=svg_units_from_pt(OBJECT_FONT_PT), weight="bold"),
                text_block(760, center_y - 26, (row["predicate"].strip().upper(),), size=svg_units_from_pt(RELATION_FONT_PT), weight="bold"),
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def svg_legend() -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="170mm" height="26mm" viewBox="0 0 1700 260">
  <rect width="1700" height="260" fill="{WHITE}"/>
  <defs>
    <marker id="n" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{NAVY}"/></marker>
    <marker id="o" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{ORANGE}"/></marker>
    <marker id="g" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{NEUTRAL}"/></marker>
  </defs>
  <path d="M55 65 l18 -18 18 18 -18 18 z" fill="{WHITE}" stroke="{NAVY}" stroke-width="5"/>
  <text x="110" y="78" font-family="{FONT}" font-size="{svg_units_from_pt(12):g}" fill="{INK}">evidence node</text>
  <line x1="430" y1="65" x2="535" y2="65" stroke="{NAVY}" stroke-width="4" stroke-dasharray="9 7" marker-end="url(#n)"/>
  <text x="565" y="78" font-family="{FONT}" font-size="{svg_units_from_pt(12):g}" fill="{INK}">evidence supports claim</text>
  <line x1="1120" y1="65" x2="1235" y2="65" stroke="{NAVY}" stroke-width="5" marker-end="url(#n)"/>
  <text x="1270" y="78" font-family="{FONT}" font-size="{svg_units_from_pt(12):g}" fill="{INK}">graph link</text>
  <line x1="250" y1="190" x2="385" y2="190" stroke="{ORANGE}" stroke-width="6" marker-end="url(#o)"/>
  <text x="420" y="203" font-family="{FONT}" font-size="{svg_units_from_pt(12):g}" fill="{INK}">reported relation</text>
  <line x1="960" y1="190" x2="1095" y2="190" stroke="{NEUTRAL}" stroke-width="5" marker-end="url(#g)"/>
  <text x="1130" y="203" font-family="{FONT}" font-size="{svg_units_from_pt(12):g}" fill="{INK}">claim-to-object link</text>
</svg>
'''


def print_svg_pdf(svg: Path, pdf: Path, width_mm: float, height_mm: float) -> None:
    browser = chrome_path(EDGE)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fig4a_pdf_", dir=pdf.parent, ignore_cleanup_errors=True) as temporary:
        temp = Path(temporary)
        wrapper = temp / "print.html"
        profile = temp / "profile"
        uri = svg.resolve().as_uri().replace("&", "&amp;").replace('"', "&quot;")
        wrapper.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            f"@page{{size:{width_mm}mm {height_mm}mm;margin:0}}"
            "html,body{margin:0;padding:0;width:100%;height:100%;}"
            "img{display:block;width:100%;height:100%;}"
            f"</style></head><body><img src=\"{uri}\"></body></html>",
            encoding="utf-8",
        )
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf.resolve()}",
            wrapper.as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0 or not pdf.is_file() or pdf.stat().st_size == 0:
            raise RuntimeError(f"Chrome PDF export failed ({result.returncode}): {result.stderr}")


def build(output_root: Path) -> dict[str, object]:
    rows = read_source()
    output_root = output_root.resolve()
    svg = output_root / "svg" / "fig4a_five_paper_graph.svg"
    png = output_root / "png" / "fig4a_five_paper_graph.png"
    pdf = output_root / "pdf" / "fig4a_five_paper_graph.pdf"
    data_copy = output_root / "data" / "fig4a_five_paper_graph" / SOURCE.name
    audit_path = output_root / "validation" / "fig4a_five_paper_graph_build_audit.json"
    legend_svg = output_root / "legends" / "svg" / "legend_fig4a_knowledge_graph.svg"
    legend_png = output_root / "legends" / "png" / "legend_fig4a_knowledge_graph.png"
    for path in (svg, png, pdf, data_copy, audit_path, legend_svg, legend_png):
        path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE, data_copy)
    if sha256(data_copy) != sha256(SOURCE):
        raise RuntimeError("Figure 4a source copy failed hash verification")
    svg.write_text(svg_panel(rows), encoding="utf-8", newline="\n")
    legend_svg.write_text(svg_legend(), encoding="utf-8", newline="\n")
    render(svg, png, round(WIDTH_MM / 25.4 * 600), EDGE)
    render(legend_svg, legend_png, round(170.0 / 25.4 * 600), EDGE)
    print_svg_pdf(svg, pdf, WIDTH_MM, HEIGHT_MM)

    outputs = {}
    for name, path in {
        "png": png,
        "svg": svg,
        "pdf": pdf,
        "source_csv": data_copy,
        "legend_png": legend_png,
        "legend_svg": legend_svg,
    }.items():
        outputs[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    report: dict[str, object] = {
        "status": "PASS_PENDING_VISUAL_REVIEW",
        "analysis_id": ANALYSIS_ID,
        "plot_id": "fig4a_five_paper_graph",
        "builder": str(Path(__file__).resolve()),
        "authoritative_source": str(SOURCE.resolve()),
        "authoritative_source_sha256": sha256(SOURCE),
        "source_rows": len(rows),
        "graph_topology": {
            "papers": 5,
            "claims": 5,
            "subjects": 5,
            "objects": 5,
            "evidence_nodes": 5,
            "relationships": 25,
        },
        "formatting": {
            "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
            "font": FONT,
            "font_sizes_pt": {
                "column_headers": HEADER_FONT_PT,
                "paper_labels": PAPER_FONT_PT,
                "object_labels": OBJECT_FONT_PT,
                "claim_labels": CLAIM_FONT_PT,
                "subject_labels": [SUBJECT_COMPACT_FONT_PT, SUBJECT_FONT_PT],
                "relation_labels": RELATION_FONT_PT,
                "separate_legend": 12,
            },
            "separate_legend_dimensions_mm": [170, 26],
            "background": WHITE,
            "embedded_panel_letter": False,
            "embedded_title": False,
            "embedded_footer": False,
            "embedded_legend": False,
            "gridlines": False,
            "palette": {"navy": NAVY, "orange": ORANGE, "neutral": NEUTRAL},
        },
        "scientific_integrity": "Full identifiers, claims, entity text, evidence passages, and locators are copied byte-exact in the CSV; only display labels are shortened.",
        "outputs": outputs,
    }
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = read_source()
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "origin_started": False,
            "source": str(SOURCE.resolve()),
            "source_sha256": sha256(SOURCE),
            "unique_headers": True,
            "rows": len(rows),
            "papers": [row["paper_label"] for row in rows],
            "planned_dimensions_mm": [WIDTH_MM, HEIGHT_MM],
            "planned_font": FONT,
        }, indent=2))
        return
    print(json.dumps(build(args.output_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
