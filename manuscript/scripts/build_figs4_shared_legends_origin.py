#!/usr/bin/env python3
"""Build the two Figure S4 shared legends in OriginPro.

The default output is the ``build/legends`` directory.  Each legend receives its own editable OPJU, vector SVG, and exact
600-dpi PNG.  ``--dry-run`` validates every input and never imports OriginPro.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = ROOT / "source"
DEFAULT_OUTPUT_ROOT = ROOT / "build"
STYLE_CSV = ROOT / "legend_inputs" / "figs4_shared_legend_styles.csv"
S4_BUILD_SCRIPT = ROOT / "scripts" / "build_figs4_origin_no_grid.py"
SOURCE_ROOT = PUBLIC_SOURCE_ROOT
SOURCE_SPEC_ROOT = SOURCE_ROOT / "specifications"
CURRENT_AUDIT_ROOT = DEFAULT_OUTPUT_ROOT / "validation"

FONT = "Times New Roman"
BLACK = "#000000"
BLUE = "#0072B2"
TEAL = "#009E73"
DPI = 600

REQUIRED_HEADERS = (
    "legend_id",
    "series_order",
    "label",
    "color_hex",
    "marker_shape",
    "origin_symbol_kind",
    "key_style",
    "symbol_size_pt",
)

LEGENDS: dict[str, dict[str, Any]] = {
    "figs4a": {
        "plot_id": "figs4a_normalized_stability",
        "stem": "legend_figs4a_normalized_stability",
        "graph": "LegendS4A",
        "long_name": "Figure S4a shared legend",
        "width_in": 4.5,
        "height_in": 0.36,
        "legend_text": r"\l(1) Sensitivity scenarios    \l(2) Baseline",
        "labels": ["Sensitivity scenarios", "Baseline"],
        "colors": [BLUE, TEAL],
        "markers": ["box", "box"],
        "symbol_kinds": [1, 1],
        "used_by": ["figs4a_normalized_stability"],
        "specification": SOURCE_SPEC_ROOT / "figs4a_normalized_stability.json",
        "current_audit": CURRENT_AUDIT_ROOT / "figs4a_normalized_stability_build_audit.json",
    },
    "figs4b": {
        "plot_id": "figs4b_normalization_sensitivity",
        "stem": "legend_figs4b_normalization_sensitivity",
        "graph": "LegendS4B",
        "long_name": "Figure S4b shared legend",
        "width_in": 6.4,
        "height_in": 0.36,
        "legend_text": r"\l(1) Spearman rank correlation    \l(2) Top-quartile overlap",
        "labels": ["Spearman rank correlation", "Top-quartile overlap"],
        "colors": [BLUE, TEAL],
        "markers": ["square", "circle"],
        "symbol_kinds": [1, 2],
        "used_by": ["figs4b_normalization_sensitivity"],
        "specification": SOURCE_SPEC_ROOT / "figs4b_normalization_sensitivity.json",
        "current_audit": CURRENT_AUDIT_ROOT / "figs4b_normalization_sensitivity_build_audit.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def read_style_rows() -> tuple[list[str], list[dict[str, str]]]:
    require_file(STYLE_CSV)
    with STYLE_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        if headers != list(REQUIRED_HEADERS) or len(headers) != len(set(headers)):
            raise RuntimeError(
                f"Legend style CSV headers must be unique and exactly {list(REQUIRED_HEADERS)}; got {headers}"
            )
        rows = [dict(row) for row in reader]
    if len(rows) != 4:
        raise RuntimeError(f"Expected four legend-style rows; got {len(rows)}")
    return headers, rows


def selected_keys(requested: str) -> list[str]:
    return list(LEGENDS) if requested == "all" else [requested]


def validate_reference_contract(
    key: str,
    rows: list[dict[str, str]],
    output_root: Path,
) -> dict[str, Any]:
    config = LEGENDS[key]
    plot_id = str(config["plot_id"])
    selected = sorted(
        (row for row in rows if row["legend_id"] == plot_id),
        key=lambda row: int(row["series_order"]),
    )
    if len(selected) != 2:
        raise RuntimeError(f"{plot_id}: expected two legend rows; got {len(selected)}")

    labels = [row["label"] for row in selected]
    colors = [row["color_hex"].upper() for row in selected]
    markers = [row["marker_shape"] for row in selected]
    symbol_kinds = [int(row["origin_symbol_kind"]) for row in selected]
    if labels != config["labels"]:
        raise RuntimeError(f"{plot_id}: label mismatch: {labels}")
    if colors != config["colors"]:
        raise RuntimeError(f"{plot_id}: color mismatch: {colors}")
    if markers != config["markers"]:
        raise RuntimeError(f"{plot_id}: marker mismatch: {markers}")
    if symbol_kinds != config["symbol_kinds"]:
        raise RuntimeError(f"{plot_id}: Origin symbol mismatch: {symbol_kinds}")
    if any(float(row["symbol_size_pt"]) <= 0 for row in selected):
        raise RuntimeError(f"{plot_id}: symbol sizes must be positive")

    specification = Path(config["specification"])
    current_audit = (
        output_root.resolve()
        / "validation"
        / f"{plot_id}_build_audit.json"
    )
    require_file(specification)
    spec = json.loads(specification.read_text(encoding="utf-8"))
    palette = spec.get("style", {}).get("palette", {})
    if palette.get("blue", "").upper() != BLUE or palette.get("teal", "").upper() != TEAL:
        raise RuntimeError(f"{plot_id}: specification palette does not match the legend source")
    if spec.get("style", {}).get("font_family") != FONT:
        raise RuntimeError(f"{plot_id}: Times New Roman contract is not satisfied")
    if key == "figs4a":
        categories = spec.get("category_order", [])
        if len(categories) != 9 or categories.count("Baseline") != 1:
            raise RuntimeError("S4a category grouping cannot distinguish Baseline from sensitivity scenarios")
    else:
        if spec.get("series_order") != config["labels"]:
            raise RuntimeError("S4b specification series order does not match legend order")

    # A fresh checkout can validate the encoded source contract before any
    # Origin project has been built. A build audit, when present, adds checks
    # for the exported project's font, legend strategy, and series styling.
    if current_audit.is_file():
        audit = json.loads(current_audit.read_text(encoding="utf-8"))
        if audit.get("font") != FONT:
            raise RuntimeError(f"{plot_id}: exported font does not match the legend")
        graph_audit = audit.get("graph_audit", {})
        if graph_audit.get("embedded_legend") is not False:
            raise RuntimeError(f"{plot_id}: current plot must use a separate legend")
        if graph_audit.get("legend_strategy") != "separate shared legend during multipanel assembly":
            raise RuntimeError(f"{plot_id}: separate legend strategy is not documented")
        if key == "figs4b":
            audited_series = graph_audit.get("series", [])
            compact = [(item.get("label"), str(item.get("color", "")).upper(), item.get("marker")) for item in audited_series]
            expected = list(zip(config["labels"], config["colors"], config["markers"]))
            if compact != expected:
                raise RuntimeError(f"S4b current-figure audit series mismatch: {compact}")

    return {
        "plot_id": plot_id,
        "style_rows": selected,
        "specification": str(specification.resolve()),
        "specification_sha256": sha256(specification),
        "current_plot_audit": str(current_audit.resolve()) if current_audit.is_file() else None,
        "current_plot_audit_sha256": sha256(current_audit) if current_audit.is_file() else None,
        "validation_scope": "source specification and build audit" if current_audit.is_file() else "static source specification only",
        "visual_review_performed": False,
        "reference_contract": "PASS",
    }


def planned_outputs(output_root: Path, key: str) -> dict[str, Any]:
    config = LEGENDS[key]
    legend_root = output_root.resolve() / "legends"
    stem = str(config["stem"])
    return {
        "legend_root": str(legend_root),
        "origin_project": str((legend_root / "origin_projects" / f"{stem}.opju").resolve()),
        "png": str((legend_root / "PNG_600dpi" / f"{stem}.png").resolve()),
        "svg": str((legend_root / "SVG" / f"{stem}.svg").resolve()),
        "build_audit": str((legend_root / "validation" / f"{stem}_build_audit.json").resolve()),
        "page_dimensions_in": [config["width_in"], config["height_in"]],
        "png_dimensions_px": [round(float(config["width_in"]) * DPI), round(float(config["height_in"]) * DPI)],
    }


def dry_run(requested: str, output_root: Path) -> dict[str, Any]:
    _, rows = read_style_rows()
    require_file(S4_BUILD_SCRIPT)
    plans = []
    for key in selected_keys(requested):
        plans.append(
            {
                "legend_key": key,
                "validation": validate_reference_contract(key, rows, output_root),
                "outputs": planned_outputs(output_root, key),
                "content_contract": {
                    "font": FONT,
                    "font_size_pt": 12,
                    "background": "white",
                    "title": None,
                    "gridlines": False,
                    "axes": False,
                    "panel_label": None,
                },
            }
        )
    return {
        "status": "PASS",
        "mode": "dry-run",
        "origin_imported": False,
        "style_csv": str(STYLE_CSV.resolve()),
        "style_csv_sha256": sha256(STYLE_CSV),
        "s4_origin_builder": str(S4_BUILD_SCRIPT.resolve()),
        "s4_origin_builder_sha256": sha256(S4_BUILD_SCRIPT),
        "plans": plans,
    }


def import_style_table(op: Any, base: Any, rows: list[dict[str, str]], plot_id: str) -> Any:
    selected = sorted(
        (row for row in rows if row["legend_id"] == plot_id),
        key=lambda row: int(row["series_order"]),
    )
    book = op.new_book("w", lname=f"{plot_id} shared-legend style data")
    if book is None:
        raise RuntimeError("Origin could not create the legend-data workbook")
    book.name = "LegendData"
    sheet = book[0]
    sheet.name = "Style"
    sheet.lname = "Exact shared-legend style source"
    for column, header in enumerate(REQUIRED_HEADERS):
        values: list[Any]
        if header in {"series_order", "origin_symbol_kind", "symbol_size_pt"}:
            values = [float(row[header]) for row in selected]
        else:
            values = [row[header] for row in selected]
        sheet.from_list(column, values, lname=header, axis="")
    return base.Table(
        key="LegendStyle",
        path=STYLE_CSV,
        sheet=sheet,
        headers=list(REQUIRED_HEADERS),
        columns={header: index for index, header in enumerate(REQUIRED_HEADERS)},
        rows=selected,
    )


def make_graph(op: Any, base: Any, key: str, table: Any) -> Any:
    config = LEGENDS[key]
    graph = base.new_graph(
        str(config["graph"]),
        str(config["long_name"]),
        float(config["width_in"]),
        float(config["height_in"]),
    )
    layer = graph[0]
    base.position_layer(layer, 1, 1, 98, 98)
    layer.axis("x").set_limits(0, 1, 1)
    layer.axis("y").set_limits(0, 1, 1)
    layer.lt_exec(
        "layer.x.showAxes=0; layer.y.showAxes=0; "
        "layer.x.label.show=0; layer.y.label.show=0; "
        "layer.x.majorTicks=0; layer.x.minorTicks=0; "
        "layer.y.majorTicks=0; layer.y.minorTicks=0; "
        "layer.x.grid=0; layer.y.grid=0; "
        "layer.x.grid.major=0; layer.x.grid.minor=0; "
        "layer.y.grid.major=0; layer.y.grid.minor=0; "
        "layer.sauto=0; label -r title; label -r legend;"
    )
    for index, row in enumerate(table.rows, start=1):
        plot = base.add_legend_key_plot(
            layer,
            table,
            f"legend_key_{index}",
            # Origin's legend sample inherits the symbol edge color.  Using a
            # black edge here caused both S4a box keys to render black even
            # though their fill colors were blue/teal.  The compact key must
            # privilege the scenario color; the panel itself retains its
            # black box/whisker outlines.
            color=row["color_hex"],
            kind=int(row["origin_symbol_kind"]),
            size=float(row["symbol_size_pt"]),
            interior=0,
            fill_color=row["color_hex"],
        )
    base.set_legend(layer, str(config["legend_text"]), 2, 18, 12, boxed=False)
    layer.lt_exec(
        f'legend.font=font("{FONT}"); legend.fsize=12; legend.bold=0; '
        "legend.italic=0; legend.underline=0; legend.whiteout=0; legend.background=0; "
        "label -r title; layer.x.grid=0; layer.y.grid=0;"
    )
    graph.lt_exec("page.nodark=1; page.speedmode=0; page.legendupdatemode=2;")
    return graph


def export_graph(op: Any, common: Any, sanitize_svg: Any, graph: Any, key: str, output_root: Path) -> dict[str, Any]:
    config = LEGENDS[key]
    legend_root = output_root.resolve() / "legends"
    stem = str(config["stem"])
    png = legend_root / "PNG_600dpi" / f"{stem}.png"
    svg = legend_root / "SVG" / f"{stem}.svg"
    raw_svg = legend_root / "validation" / "origin_raw_exports" / f"{stem}.origin.svg"
    sanitizer_audit = legend_root / "validation" / f"{stem}_svg_sanitizer.json"
    for path in (png, svg, raw_svg, sanitizer_audit):
        path.parent.mkdir(parents=True, exist_ok=True)
    graph.activate()
    graph.lt_exec("page.speedmode=0;")
    graph[0].lt_exec("layer.maxpts=0; page.speedmode=0;")
    op.wait()
    exported = graph.save_fig(str(raw_svg), type="svg", ratio=100)
    if not exported or not raw_svg.is_file() or raw_svg.stat().st_size == 0:
        raise RuntimeError(f"Origin failed to export SVG: {raw_svg}")
    sanitizer = sanitize_svg(raw_svg, svg, sanitizer_audit, float(config["width_in"]))
    common.force_svg_physical_size(svg, float(config["width_in"]), float(config["height_in"]))
    dimensions = (
        round(float(config["width_in"]) * DPI),
        round(float(config["height_in"]) * DPI),
    )
    rendered = common.render_svg_png_exact(svg, png, *dimensions)
    if rendered != dimensions:
        raise RuntimeError(f"Unexpected PNG dimensions for {stem}: {rendered} vs {dimensions}")
    for path in (png, svg):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty legend export: {path}")
    return {
        "png": str(png.resolve()),
        "png_bytes": png.stat().st_size,
        "png_sha256": sha256(png),
        "png_dimensions": list(dimensions),
        "svg": str(svg.resolve()),
        "svg_bytes": svg.stat().st_size,
        "svg_sha256": sha256(svg),
        "origin_raw_svg": str(raw_svg.resolve()),
        "svg_sanitizer": sanitizer,
    }


def build_one(
    op: Any,
    base: Any,
    common: Any,
    sanitize_svg: Any,
    key: str,
    rows: list[dict[str, str]],
    output_root: Path,
    close: bool,
) -> dict[str, Any]:
    config = LEGENDS[key]
    validation = validate_reference_contract(key, rows, output_root)
    legend_root = output_root.resolve() / "legends"
    project = legend_root / "origin_projects" / f"{config['stem']}.opju"
    delivered_csv = legend_root / "validation" / STYLE_CSV.name
    project.parent.mkdir(parents=True, exist_ok=True)
    delivered_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STYLE_CSV, delivered_csv)
    if getattr(op, "oext", False):
        op.set_show(not close)
    op.new(asksave=False)
    base.FONT_INDEX = int(op.lt_int(f'font("{FONT}")'))
    table = import_style_table(op, base, rows, str(config["plot_id"]))
    graph = make_graph(op, base, key, table)
    graph_pages = op.graph_list("p")
    if len(graph_pages) != 1 or graph_pages[0].name != config["graph"]:
        raise RuntimeError(f"Expected one graph page {config['graph']}; got {[page.name for page in graph_pages]}")
    if not op.save(str(project.resolve())):
        raise RuntimeError(f"Origin failed to save the editable legend project: {project}")
    exports = export_graph(op, common, sanitize_svg, graph, key, output_root)
    report = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "legend_key": key,
        "plot_id": config["plot_id"],
        "used_by": config["used_by"],
        "origin_project": str(project.resolve()),
        "origin_project_bytes": project.stat().st_size,
        "origin_project_sha256": sha256(project),
        "source_csv": str(delivered_csv.resolve()),
        "source_csv_sha256": sha256(delivered_csv),
        "source_csv_unique_headers": True,
        "font": FONT,
        "font_size_pt": 12,
        "page_dimensions_in": [config["width_in"], config["height_in"]],
        "title": None,
        "gridlines": False,
        "axes": False,
        "panel_label": None,
        "validation": validation,
        "exports": exports,
    }
    audit_path = legend_root / "validation" / f"{config['stem']}_build_audit.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["build_audit"] = str(audit_path.resolve())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legend_id", choices=[*LEGENDS, "all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--close", action="store_true", help="Close the external Origin session after the build.")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not import or launch OriginPro.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(args.legend_id, args.output_root), indent=2, ensure_ascii=False))
        return 0

    # Deliberately delayed: the dry-run path above cannot start or attach to Origin.
    import originpro as op
    import build_manuscript_plot as common
    import build_literatureagent_origin_figures as base
    from sanitize_origin_svg import sanitize as sanitize_svg
    common.op = base.require_origin()

    _, rows = read_style_rows()
    reports = []
    try:
        for key in selected_keys(args.legend_id):
            reports.append(build_one(op, base, common, sanitize_svg, key, rows, args.output_root, args.close))
        print(json.dumps({"status": "PASS", "reports": reports}, indent=2, ensure_ascii=False))
        return 0
    finally:
        if args.close and getattr(op, "oext", False):
            op.exit()


if __name__ == "__main__":
    raise SystemExit(main())
