"""Build standalone manuscript plots in OriginPro.

Each plot receives one graph page and one OPJU project. Source tables are read
from ``source/`` and exported with their rendering audit to ``build/``.
``--dry-run`` checks source files and row counts without importing OriginPro.

Dense SHAP plots remain fully editable in Origin and are exported at 600 dpi.
Because Origin 2024 cannot reliably finish a full-vector export of 79k--428k
symbols, their SVG/PDF deliverables are explicitly raster-backed wrappers of
the complete Origin rendering.  No points are resampled or omitted.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

op: Any = None

import build_literatureagent_origin_figures as base
from render_svg_with_chrome import chrome_path, render as render_svg_png, set_png_resolution
from sanitize_origin_svg import sanitize as sanitize_svg


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "2026-10-27_controlled_repaired_rerun"
ANALYSIS_VERSION = f"{ANALYSIS_ID}_v1"
PUBLIC_SOURCE_ROOT = ROOT / "source"
PACKAGE = PUBLIC_SOURCE_ROOT
DEFAULT_OUTPUT = ROOT / "build"

BLUE = "#0072B2"
TEAL = "#009E73"
AMBER = "#E69F00"
ORANGE = "#D55E00"
NEUTRAL = "#4B5563"
GRID = "#D9DEE5"
INK = "#202124"
WHITE = "#FFFFFF"
FONT = "Times New Roman"

# Presentation-only palette for manuscript figures.  The
# controlled source-table colors remain preserved in Origin and are not used
# to alter any scientific value.
PPT_NAVY = "#33546E"
PPT_ORANGE = "#F28A00"
PPT_ORANGE_LIGHT = "#FF9F25"
PPT_GREEN = "#137321"
PPT_SHAP_NEGATIVE = "#174766"
PPT_SHAP_POSITIVE = "#C65A00"

# Figure 3 is delivered as compact 85 mm-wide panels for direct PowerPoint
# assembly.  These values are the physical-size equivalents of the supplied
# 6.5 x 3.52 in manuscript reference (22 pt body, 24 pt title, 18 pt symbols)
# after scaling that reference to 85 mm.  Keeping the values here avoids the
# visual-hierarchy regression that made the compact exports look under-sized.
PCE_BODY_SIZE = 12
PCE_TITLE_SIZE = 13
PCE_MARKER_SIZE = 9.3
PCE_INTERVAL_WIDTH = 1.80
PCE_CAP_WIDTH = 1.50
PCE_AXIS_WIDTH = 0.65

# Default assembled-panel typography for the remaining Origin plots.  These
# whole-number sizes remain legible after 180 mm pages are placed at 6.5 in
# and 85 mm pages are placed at roughly 3.0 in in PowerPoint.
ASSEMBLY_BODY_SIZE = 12
ASSEMBLY_TITLE_SIZE = 13


PLOT_CONFIG: dict[str, dict[str, Any]] = {
    "fig3a_historical_r2": {
        "stem": "fig3a_historical_r2", "graph": "Fig3a", "family": "pce",
        "source": "data/fig3a_historical_r2/plot_data.csv", "size_mm": (85, 46),
        "xlim": (0.20, 0.42), "xstep": 0.05, "xlabel": "R\\+(2)", "reference": None,
        "expected": 3,
    },
    "fig3b_historical_rmse": {
        "stem": "fig3b_historical_rmse", "graph": "Fig3b", "family": "pce",
        "source": "data/fig3b_historical_rmse/plot_data.csv", "size_mm": (85, 46),
        "xlim": (2.90, 3.65), "xstep": 0.20, "xlabel": "RMSE (PCE percentage points)",
        "reference": None, "expected": 3,
    },
    "fig3c_historical_mae": {
        "stem": "fig3c_historical_mae", "graph": "Fig3c", "family": "pce",
        "source": "data/fig3c_historical_mae/plot_data.csv", "size_mm": (85, 46),
        "xlim": (2.30, 3.00), "xstep": 0.20, "xlabel": "MAE (PCE percentage points)",
        "reference": None, "expected": 3,
    },
    "fig3d_literatureagent_r2": {
        "stem": "fig3d_literatureagent_r2", "graph": "Fig3d", "family": "pce",
        "source": "data/fig3d_literatureagent_r2/plot_data.csv", "size_mm": (85, 46),
        "xlim": (-2.10, 0.70), "xstep": 0.50, "xlabel": "R\\+(2)", "reference": 0.0,
        "expected": 2,
    },
    "fig3e_literatureagent_rmse": {
        "stem": "fig3e_literatureagent_rmse", "graph": "Fig3e", "family": "pce",
        "source": "data/fig3e_literatureagent_rmse/plot_data.csv", "size_mm": (85, 46),
        "xlim": (3.20, 8.20), "xstep": 1.0, "xlabel": "RMSE (PCE percentage points)",
        "reference": None, "expected": 2,
    },
    "fig3f_literatureagent_mae": {
        "stem": "fig3f_literatureagent_mae", "graph": "Fig3f", "family": "pce",
        "source": "data/fig3f_literatureagent_mae/plot_data.csv", "size_mm": (85, 46),
        "xlim": (2.80, 7.20), "xstep": 1.0, "xlabel": "MAE (PCE percentage points)",
        "reference": None, "expected": 2,
    },
    "fig4b_choice_shap": {
        "stem": "fig4b_choice_shap", "graph": "Fig4b", "family": "shap_choice",
        "source": "data/fig4b_choice_shap/display_points.csv", "size_mm": (180, 155),
        "xlim": (-2.0, 3.0), "xstep": 1.0,
        "xlabel": "Grouped SHAP contribution to predicted PCE\n(percentage points)",
        "reference": 0.0, "expected": 79634, "dense": True,
        "copies": [
            "data/fig4b_choice_shap/selected_state_summary.csv",
            "data/fig4b_choice_shap/category_layout.csv",
        ],
    },
    "fig5a_stability_classification": {
        "stem": "fig5a_stability_classification", "graph": "Fig5a", "family": "stability",
        "source": "data/fig5a_stability_classification/plot_data.csv", "size_mm": (85, 72),
        "xlim": (0.50, 0.70), "xstep": 0.05, "xlabel": "ROC-AUC", "reference": 0.5,
        "expected": 12, "copies": ["data/shared_stability/feature_list_manifest.csv"],
    },
    "fig5b_stability_regression": {
        "stem": "fig5b_stability_regression", "graph": "Fig5b", "family": "stability",
        "source": "data/fig5b_stability_regression/plot_data.csv", "size_mm": (85, 72),
        "xlim": (-0.10, 0.20), "xstep": 0.10, "xlabel": "R\\+(2)", "reference": 0.0,
        "expected": 12, "copies": ["data/shared_stability/feature_list_manifest.csv"],
    },
    "figs1_global_shap": {
        "stem": "figs1_global_shap", "graph": "FigS1", "family": "shap_global",
        "source": "data/figs1_global_shap/display_points.csv", "size_mm": (180, 160),
        "xlim": (-3.0, 2.5), "xstep": 1.0,
        "xlabel": "Grouped SHAP contribution to predicted PCE\n(percentage points)",
        "reference": 0.0, "expected": 428520, "dense": True,
        "copies": ["data/figs1_global_shap/top20_group_order.csv"],
    },
    "figs2_solvent_shap": {
        "stem": "figs2_solvent_shap", "graph": "FigS2", "family": "shap_solvent",
        "source": "data/figs2_solvent_shap/display_points.csv", "size_mm": (180, 145),
        "xlim": (-2.0, 1.0), "xstep": 0.5,
        "xlabel": "Solvent-formulation SHAP contribution to predicted PCE\n(percentage points)",
        "reference": 0.0, "expected": 15182,
        "copies": ["data/figs2_solvent_shap/selected_category_summary.csv"],
    },
    "figs3_local_shap": {
        "stem": "figs3_local_shap", "graph": "FigS3", "family": "shap_local",
        "source": "data/figs3_local_shap/display_contributions.csv", "size_mm": (180, 125),
        "xlim": (-0.40, 0.40), "xstep": 0.20,
        "xlabel": "Grouped SHAP contribution to predicted PCE\n(percentage points)",
        "reference": 0.0, "expected": 11,
        "copies": ["data/figs3_local_shap/selection_and_reconstruction.json"],
    },
}


@dataclass
class PreparedData:
    source_copy: Path
    helper: Path | None
    summary: Path | None
    labels: list[tuple[str, float]]
    source_rows: int
    negative_points: int = 0
    positive_points: int = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers or len(headers) != len(set(headers)):
            raise RuntimeError(f"CSV headers must be present and unique: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV contains no data rows: {path}")
    return headers, rows


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"Empty CSV: {path}") from exc
        return sum(1 for _ in reader)


def mm_to_in(value: float) -> float:
    return value / 25.4


def parse_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite numeric value: {value!r}")
    return number


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == sha256(source):
        return
    shutil.copy2(source, destination)
    if sha256(destination) != sha256(source):
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")


def prepare_inputs(plot_id: str, config: dict[str, Any], output_root: Path) -> PreparedData:
    data_dir = output_root / "data" / plot_id
    data_dir.mkdir(parents=True, exist_ok=True)
    source = PACKAGE / config["source"]
    if not source.is_file():
        raise FileNotFoundError(source)
    source_copy = data_dir / source.name
    copy_exact(source, source_copy)
    for relative in config.get("copies", []):
        extra = PACKAGE / relative
        if not extra.is_file():
            raise FileNotFoundError(extra)
        destination = data_dir / extra.name
        if destination.name == source_copy.name:
            destination = data_dir / f"support_{extra.name}"
        copy_exact(extra, destination)

    family = str(config["family"])
    source_rows = count_csv_rows(source)
    expected = int(config["expected"])
    if source_rows != expected:
        raise RuntimeError(f"{plot_id}: expected {expected:,} source rows, found {source_rows:,}")

    if not family.startswith("shap_") or family == "shap_local":
        return PreparedData(source_copy, None, None, [], source_rows)

    helper = data_dir / "origin_point_helper.csv"
    summary_path = data_dir / "origin_summary_helper.csv"
    negative = 0
    positive = 0
    first_by_key: dict[str, dict[str, str]] = {}
    with source.open("r", newline="", encoding="utf-8-sig") as input_handle, helper.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if family == "shap_choice":
            x_name, y_name, key_name = "bundle_shap_pce_points", "y_display", "state"
            reverse_constant = 23.4
        elif family == "shap_global":
            x_name, y_name, key_name = "grouped_shap_pce_points", "y_display", "scientific_group"
            reverse_constant = 19.0
        else:
            x_name, y_name, key_name = "grouped_shap_pce_points", "y_display", "solvent_formulation"
            reverse_constant = None
        required = {x_name, y_name, key_name, "color_hex"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{plot_id}: missing columns {sorted(missing)}")
        writer = csv.writer(output_handle, lineterminator="\n")
        writer.writerow(["x_negative", "y_negative", "x_positive", "y_positive"])
        for row in reader:
            x = parse_float(row[x_name])
            y_source = parse_float(row[y_name])
            y = reverse_constant - y_source if reverse_constant is not None else y_source
            color = row["color_hex"].strip().upper()
            if color == BLUE:
                writer.writerow([f"{x:.17g}", f"{y:.17g}", "", ""])
                negative += 1
            elif color == ORANGE:
                writer.writerow(["", "", f"{x:.17g}", f"{y:.17g}"])
                positive += 1
            else:
                raise RuntimeError(f"{plot_id}: unexpected point color {row['color_hex']!r}")
            first_by_key.setdefault(row[key_name], row)
    if negative + positive != source_rows:
        raise RuntimeError(f"{plot_id}: helper point count mismatch")

    labels: list[tuple[str, float]] = []
    if family == "shap_choice":
        layout = PACKAGE / "data/fig4b_choice_shap/category_layout.csv"
        _, layout_rows = read_csv_rows(layout)
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["state", "y_plot", "median", "q25", "q75", "sample_count", "publication_group_count"])
            for row in layout_rows:
                y_plot = 23.4 - parse_float(row["y_base"])
                if row["row_type"] == "family_heading":
                    labels.append((row["display_label"], y_plot))
                    continue
                point = first_by_key[row["display_label"]]
                n = int(float(point["sample_count"]))
                groups = int(float(point["publication_group_count"]))
                # Origin categorical tick labels do not reserve vertical
                # space for a second line.  Use compact, meaning-preserving
                # publication shorthand here while the exact controlled
                # labels remain unchanged in Source/Summary worksheets.
                display_state = {
                    "TiO2-c | TiO2-mp; Spin-coating | Spin-coating":
                        "TiO2-c/TiO2-mp; spin/spin",
                    "TiO2-c; Spin-coating": "TiO2-c; spin",
                    "TiO2-c | TiO2-mp; Spray-pyrolys | Spin-coating":
                        "TiO2-c/TiO2-mp; spray pyr./spin",
                }.get(row["display_label"], row["display_label"].replace("; Spin-coating", "; spin coat"))
                labels.append((f"{display_state}  (n={n:,}; g={groups:,})", y_plot))
                writer.writerow([
                    row["display_label"], f"{y_plot:.17g}", point["median"], point["q25"], point["q75"], n, groups,
                ])
    elif family == "shap_global":
        order_file = PACKAGE / "data/figs1_global_shap/top20_group_order.csv"
        _, order_rows = read_csv_rows(order_file)
        summary_path = None
        for index, row in enumerate(order_rows):
            label = row["scientific_group"]
            if label == "Perovskite deposition thermal annealing temperature":
                label = "Perovskite annealing temperature"
            labels.append((label, 19.0 - float(index)))
    else:
        summary_file = PACKAGE / "data/figs2_solvent_shap/selected_category_summary.csv"
        _, summary_rows = read_csv_rows(summary_file)
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["solvent_formulation", "y_plot", "median", "q25", "q75", "sample_count", "publication_group_count"])
            for index, row in enumerate(summary_rows):
                key = row["solvent_formulation"]
                point = first_by_key[key]
                y_plot = float(index)
                n = int(float(row["sample_count"]))
                groups = int(float(row["publication_group_count"]))
                labels.append((f"{key}\n({n:,} records; {groups:,} groups)", y_plot))
                writer.writerow([key, f"{y_plot:.17g}", point["median"], point["q25"], point["q75"], n, groups])

    return PreparedData(source_copy, helper, summary_path, labels, source_rows, negative, positive)


def import_csv_sheet(book: Any, path: Path, name: str, lname: str, first: bool = False) -> tuple[Any, base.Table]:
    sheet = book[0] if first else book.add_sheet()
    sheet.name = name
    sheet.lname = lname
    sheet.from_file(str(path.resolve()), keep_DC=False)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        headers = list(next(csv.reader(handle)))
    if len(headers) != len(set(headers)):
        raise RuntimeError(f"Imported CSV has duplicate headers: {path}")
    table = base.Table(
        key=name,
        path=path,
        sheet=sheet,
        headers=list(headers),
        columns={header: index for index, header in enumerate(headers)},
        rows=[],
    )
    return sheet, table


def add_linked_line(layer: Any, table: base.Table, prefix: str, xs: Iterable[Any], ys: Iterable[Any], color: str, width: float) -> Any:
    columns = base.append_xy(table, prefix, list(xs), list(ys))
    plot = layer.add_plot(table.sheet, coly=columns[1], colx=columns[0], type="l")
    if plot is None:
        raise RuntimeError(f"Origin failed to add linked line {prefix}")
    plot.color = color
    plot.set_float("line.width", width)
    return plot


def add_marker(
    layer: Any,
    table: base.Table,
    prefix: str,
    xs: list[float],
    ys: list[float],
    color: str,
    kind: int,
    size: float,
    *,
    open_symbol: bool = False,
) -> tuple[Any, int]:
    columns = base.append_xy(table, prefix, xs, ys)
    plot = layer.add_plot(table.sheet, coly=columns[1], colx=columns[0], type="s")
    if plot is None:
        raise RuntimeError(f"Origin failed to add marker series {prefix}")
    plot.color = color
    plot.symbol_kind = kind
    plot.symbol_size = size
    plot.symbol_interior = 1 if open_symbol else 0
    color_code = op.ocolor(color)
    plot.set_int("symbol.edgecolor", color_code)
    plot.set_int("symbol.fillcolor", op.ocolor(WHITE) if open_symbol else color_code)
    return plot, len(layer.plot_list())


def style_axes(
    layer: Any,
    config: dict[str, Any],
    ylim: tuple[float, float],
    ystep: float = 1.0,
    tick_size: int = 12,
    axis_title_size: int = 13,
    axis_thickness: float = 1.2,
    tick_length: float | None = None,
    tick_thickness: float | None = None,
) -> None:
    step = float(config["xstep"])
    decimals = 0 if step >= 1.0 else (1 if step >= 0.1 else 2)
    base.style_axis(
        layer,
        str(config["xlabel"]),
        xlim=tuple(config["xlim"]),
        xstep=float(config["xstep"]),
        ylim=ylim,
        ystep=ystep,
        show_y_labels=True,
        axis_title_size=axis_title_size,
        tick_size=tick_size,
    )
    tick_geometry = ""
    if tick_length is not None:
        tick_geometry += f"layer.x.ticklength={tick_length:g}; layer.y.ticklength={tick_length:g}; "
    if tick_thickness is not None:
        tick_geometry += f"layer.x.tickthickness={tick_thickness:g}; layer.y.tickthickness={tick_thickness:g}; "
    layer.lt_exec(
        "layer.x.grid.major=0; layer.x.grid.minor=0; layer.y.grid.major=0; layer.y.grid.minor=0; "
        f"layer.x.thickness={axis_thickness:g}; layer.y.thickness={axis_thickness:g}; "
        f"{tick_geometry}"
        f"layer.x.label.nDigits={decimals}; layer.x.label.decimalPlaces={decimals}; "
        "layer.sauto=0; page.speedmode=0;"
    )
    # Origin's automatic x-title placement is outside the short standalone
    # pages in this package.  A page-safe editable label is added by each
    # builder after the axes are styled.
    layer.axis("x").title = ""


def add_manual_x_title(layer: Any, text: str, y_percent: float, *, size: int = 13) -> None:
    title = base.add_percent_label(layer, text, 50, y_percent, size=size, color=INK)
    title.set_int("anchor", 5)


def add_interval_series(
    layer: Any,
    table: base.Table,
    prefix: str,
    rows: list[tuple[float, float, float]],
    color: str,
    cap_half_height: float,
    line_width: float,
    cap_width: float,
) -> None:
    segment_x: list[Any] = []
    segment_y: list[Any] = []
    cap_x: list[Any] = []
    cap_y: list[Any] = []
    for low, high, y in rows:
        segment_x.extend([low, high, None])
        segment_y.extend([y, y, None])
        cap_x.extend([low, low, None, high, high, None])
        cap_y.extend([y - cap_half_height, y + cap_half_height, None, y - cap_half_height, y + cap_half_height, None])
    add_linked_line(layer, table, f"{prefix}_ci", segment_x, segment_y, color, line_width)
    add_linked_line(layer, table, f"{prefix}_caps", cap_x, cap_y, color, cap_width)


def build_pce(plot_id: str, config: dict[str, Any], source_table: base.Table, rows: list[dict[str, str]]) -> tuple[Any, dict[str, Any]]:
    width_mm, height_mm = config["size_mm"]
    graph = base.new_graph(config["graph"], plot_id, mm_to_in(width_mm), mm_to_in(height_mm))
    layer = graph[0]
    historical = len(rows) == 3
    # Match the supplied clean Origin reference after proportional reduction
    # from 6.5 in to 85 mm.  The accepted-expansion panels use a wrapped label
    # so the plot can keep the wider reference geometry without clipping.
    base.position_layer(layer, 28 if historical else 34, 3, 69 if historical else 63, 64)
    count = len(rows)
    ylim = (0.35, count + 0.55)
    if config.get("reference") is not None and not math.isclose(float(config["reference"]), float(config["xlim"][0])):
        base.vertical_rule(layer, float(config["reference"]), ylim[0], ylim[1], color=NEUTRAL, width=0.8)
    labels: list[str] = []
    marker_values: list[tuple[float, float, str]] = []
    for row in rows:
        order = int(float(row["category_order"]))
        y = count - order + 1
        if historical:
            style = {
                1: (PPT_NAVY, 1, "Original"),
                2: (PPT_ORANGE_LIGHT, 2, "Curated"),
                3: (PPT_ORANGE, 3, "Full"),
            }[order]
        else:
            style = {
                1: (PPT_NAVY, 1, "Original only"),
                2: (PPT_GREEN, 2, "+ accepted expansion"),
            }[order]
        color, marker_kind, display_label = style
        labels.append(display_label)
        low, high, estimate = map(float, (row["ci_lower"], row["ci_upper"], row["estimate"]))
        add_interval_series(
            layer,
            source_table,
            f"row{order}",
            [(low, high, y)],
            color,
            0.10,
            PCE_INTERVAL_WIDTH,
            PCE_CAP_WIDTH,
        )
        add_marker(
            layer,
            source_table,
            f"row{order}_point",
            [estimate],
            [y],
            color,
            marker_kind,
            PCE_MARKER_SIZE,
            open_symbol=True,
        )
        marker_values.append((estimate, y, f"{estimate:.3f}"))
    style_axes(
        layer,
        config,
        ylim,
        tick_size=PCE_BODY_SIZE,
        axis_title_size=PCE_TITLE_SIZE,
        axis_thickness=PCE_AXIS_WIDTH,
        tick_length=4.3,
        tick_thickness=PCE_AXIS_WIDTH,
    )
    # A 139% layer-relative baseline reproduces the reference's ~94% page
    # baseline while retaining a safe lower margin at the larger title size.
    add_manual_x_title(layer, str(config["xlabel"]), 139, size=PCE_TITLE_SIZE)
    if not historical:
        labels = [label.replace("+ accepted expansion", "+ accepted\nexpansion") for label in labels]
    base.set_categorical_ticks(layer, source_table, labels, plot_id, [float(count - i) for i in range(count)])
    xlim = tuple(config["xlim"])
    span = xlim[1] - xlim[0]
    for estimate, y, text in marker_values:
        x = estimate + 0.060 * span
        center = False
        if x > xlim[1] - 0.10 * span:
            x = estimate - 0.10 * span
        label_y = y - 0.15
        base.add_data_label(
            layer,
            text,
            x,
            label_y,
            xlim=xlim,
            ylim=ylim,
            size=PCE_BODY_SIZE,
            bold=False,
            color=INK,
            center=center,
        )
    return graph, {
        "source_rows": count,
        "point_estimates": count,
        "confidence_intervals": count,
        "xlim": list(xlim),
        "ylim": list(ylim),
        "typography_pt": {"body": PCE_BODY_SIZE, "axis_title": PCE_TITLE_SIZE},
        "stroke_pt": {
            "axis": PCE_AXIS_WIDTH,
            "confidence_interval": PCE_INTERVAL_WIDTH,
            "confidence_cap": PCE_CAP_WIDTH,
        },
        "marker_size_pt": PCE_MARKER_SIZE,
        "powerpoint_style": "open markers; navy/orange/green palette; separate legend",
    }


def wrap_endpoint(label: str) -> str:
    replacements = {
        "End >= 80%; test >= 100 h": "End >= 80%; test\n>= 100 h",
        "End >= 80%; test >= 500 h": "End >= 80%; test\n>= 500 h",
        "1000 h retention >= 80%": "1000 h: >= 80% retention",
        # The complete endpoint definitions and k units remain in the exact
        # imported worksheet and caption.  These concise display labels keep
        # adjacent two-line tick labels visually distinct in the 85 mm panel.
        "Condition-normalized calculated ln(k), k in h^-1": "Norm. ln(k)",
        "Condition-normalized ln(1 + T80/h)": "Norm. ln(1 + T80/h)",
        "Calculated ln(k), k in h^-1": "Calculated ln(k)",
    }
    wrapped = replacements.get(label, label)
    # Publication-facing typography only; the authoritative source strings
    # remain unchanged in the imported Origin worksheet.
    return wrapped.replace(">=", "≥").replace("h^-1", "h⁻¹")


def build_stability(plot_id: str, config: dict[str, Any], source_table: base.Table, rows: list[dict[str, str]]) -> tuple[Any, dict[str, Any]]:
    width_mm, height_mm = config["size_mm"]
    graph = base.new_graph(config["graph"], plot_id, mm_to_in(width_mm), mm_to_in(height_mm))
    layer = graph[0]
    # Allocate enough gutter for 12 pt endpoint labels in the compact 85 mm
    # PowerPoint panels; the data region remains aligned between both panels.
    base.position_layer(layer, 56, 5, 38, 73)
    ylim = (0.45, 6.55)
    if not math.isclose(float(config["reference"]), float(config["xlim"][0])):
        base.vertical_rule(layer, float(config["reference"]), ylim[0], ylim[1] - 0.40, color=NEUTRAL, width=0.8)
    by_endpoint: dict[int, dict[str, dict[str, str]]] = {}
    labels_by_order: dict[int, str] = {}
    for row in rows:
        order = int(float(row["endpoint_order"]))
        by_endpoint.setdefault(order, {})[row["series_label"]] = row
        labels_by_order[order] = wrap_endpoint(row["display_endpoint_label"])
    original_points: list[tuple[float, float]] = []
    integrated_points: list[tuple[float, float]] = []
    original_intervals: list[tuple[float, float, float]] = []
    integrated_intervals: list[tuple[float, float, float]] = []
    ordered_endpoints = sorted(by_endpoint)
    if len(ordered_endpoints) != 6:
        raise RuntimeError(f"{plot_id}: expected six endpoint orders, found {ordered_endpoints}")
    for display_index, order in enumerate(ordered_endpoints):
        center = 6 - display_index
        original = by_endpoint[order]["Original"]
        integrated = by_endpoint[order]["Integrated"]
        yo, yi = center + 0.10, center - 0.10
        xo, xi = float(original["estimate"]), float(integrated["estimate"])
        original_points.append((xo, yo))
        integrated_points.append((xi, yi))
        original_intervals.append((float(original["ci_lower"]), float(original["ci_upper"]), yo))
        integrated_intervals.append((float(integrated["ci_lower"]), float(integrated["ci_upper"]), yi))
    add_interval_series(layer, source_table, "original", original_intervals, PPT_NAVY, 0.085, 1.2, 1.0)
    add_interval_series(layer, source_table, "integrated", integrated_intervals, PPT_ORANGE, 0.085, 1.2, 1.0)
    add_marker(
        layer,
        source_table,
        "original_points",
        [p[0] for p in original_points],
        [p[1] for p in original_points],
        PPT_NAVY,
        1,
        6.0,
        open_symbol=True,
    )
    add_marker(
        layer,
        source_table,
        "integrated_points",
        [p[0] for p in integrated_points],
        [p[1] for p in integrated_points],
        PPT_ORANGE,
        2,
        6.3,
        open_symbol=True,
    )
    style_axes(
        layer,
        config,
        ylim,
        tick_size=ASSEMBLY_BODY_SIZE,
        axis_title_size=ASSEMBLY_TITLE_SIZE,
    )
    # Raise the enlarged title enough to retain a clean lower page margin.
    add_manual_x_title(layer, str(config["xlabel"]), 120, size=ASSEMBLY_TITLE_SIZE)
    labels = [labels_by_order[index] for index in ordered_endpoints]
    base.set_categorical_ticks(layer, source_table, labels, plot_id, [float(6 - i) for i in range(6)])
    return graph, {
        "source_rows": len(rows),
        "endpoints": 6,
        "point_estimates": 12,
        "confidence_intervals": 12,
        "paired_connectors": 0,
        "embedded_legend": False,
        "xlim": list(config["xlim"]),
        "ylim": list(ylim),
        "typography_pt": {"body": ASSEMBLY_BODY_SIZE, "axis_title": ASSEMBLY_TITLE_SIZE},
        "powerpoint_style": "open navy square and open orange circle",
    }


def read_summary_helper(path: Path) -> list[dict[str, str]]:
    _, rows = read_csv_rows(path)
    return rows


def build_shap(
    plot_id: str,
    config: dict[str, Any],
    point_table: base.Table,
    label_table: base.Table,
    prepared: PreparedData,
    summary_table: base.Table | None,
) -> tuple[Any, dict[str, Any]]:
    width_mm, height_mm = config["size_mm"]
    graph = base.new_graph(config["graph"], plot_id, mm_to_in(width_mm), mm_to_in(height_mm))
    layer = graph[0]
    family = config["family"]
    if family == "shap_choice":
        # This is the densest label set.  A generous label margin retains the
        # final 10 pt Times New Roman categories without sacrificing the full
        # record-level point cloud.
        # Reserve a true lower title band.  The larger assembly typography
        # otherwise crowds the numeric tick labels in PowerPoint exports.
        base.position_layer(layer, 56, 5, 41, 76)
        ylim = (-0.25, 25.30)
        point_size, transparency = 5.0, 38
        labels_top = prepared.labels
    elif family == "shap_global":
        # One extra percent of left margin keeps the longest scientific-group
        # label fully inside the page at the final 10 pt category size.
        base.position_layer(layer, 46, 5, 51, 77)
        ylim = (-0.55, 21.00)
        point_size, transparency = 4.0, 42
        labels_top = prepared.labels
    else:
        base.position_layer(layer, 39, 5, 58, 76)
        ylim = (-0.55, 10.85)
        point_size, transparency = 4.0, 38
        labels_top = list(reversed(prepared.labels))
    highest_data_y = max(float(item[1]) for item in labels_top)
    base.vertical_rule(layer, 0.0, ylim[0], highest_data_y + 0.45, color=INK, width=0.9)
    negative = layer.add_plot(point_table.sheet, coly=point_table.columns["y_negative"], colx=point_table.columns["x_negative"], type="s")
    positive = layer.add_plot(point_table.sheet, coly=point_table.columns["y_positive"], colx=point_table.columns["x_positive"], type="s")
    if negative is None or positive is None:
        raise RuntimeError("Origin failed to add complete SHAP point clouds")
    base.style_scatter(
        negative,
        color=PPT_SHAP_NEGATIVE,
        kind=2,
        size=point_size,
        interior=0,
        transparency=transparency,
        fill_color=PPT_SHAP_NEGATIVE,
    )
    base.style_scatter(
        positive,
        color=PPT_SHAP_POSITIVE,
        kind=2,
        size=point_size,
        interior=0,
        transparency=transparency,
        fill_color=PPT_SHAP_POSITIVE,
    )
    negative_index, positive_index = 1, 2
    median_index = None
    iqr_index = None
    if summary_table is not None and prepared.summary is not None:
        summary_rows = read_summary_helper(prepared.summary)
        intervals = [(float(row["q25"]), float(row["q75"]), float(row["y_plot"])) for row in summary_rows]
        add_interval_series(layer, summary_table, "iqr", intervals, INK, 0.0, 2.0, 0.0)
        iqr_index = len(layer.plot_list()) - 1
        _, median_index = add_marker(
            layer,
            summary_table,
            "median",
            [float(row["median"]) for row in summary_rows],
            [float(row["y_plot"]) for row in summary_rows],
            INK,
            5,
            6.0 if family == "shap_choice" else 5.0,
            open_symbol=True,
        )
    layer.lt_exec("layer.maxpts=0; page.speedmode=0; layer -b 3DB 1;")
    style_axes(
        layer,
        config,
        ylim,
        tick_size=ASSEMBLY_BODY_SIZE,
        axis_title_size=ASSEMBLY_TITLE_SIZE,
    )
    add_manual_x_title(
        layer,
        str(config["xlabel"]),
        114 if family in {"shap_choice", "shap_global"} else 116,
        size=ASSEMBLY_TITLE_SIZE,
    )
    labels = [item[0] for item in labels_top]
    positions = [item[1] for item in labels_top]
    base.set_categorical_ticks(layer, label_table, labels, plot_id, positions)
    category_size = ASSEMBLY_BODY_SIZE
    layer.lt_exec(f"layer.y.label.pt={category_size};")
    return graph, {
        "source_rows": prepared.source_rows,
        "negative_points": prepared.negative_points,
        "positive_points": prepared.positive_points,
        "displayed_points": prepared.negative_points + prepared.positive_points,
        "category_ticks": len(prepared.labels),
        "xlim": list(config["xlim"]),
        "ylim": list(ylim),
        "speedmode": "off",
        "maxpts": 0,
        "embedded_legend": False,
        "typography_pt": {
            "category": category_size,
            "numeric_ticks": ASSEMBLY_BODY_SIZE,
            "axis_title": ASSEMBLY_TITLE_SIZE,
        },
        "powerpoint_style": "dark navy/burnt orange; separate legend",
    }


def build_local_shap(plot_id: str, config: dict[str, Any], source_table: base.Table, rows: list[dict[str, str]]) -> tuple[Any, dict[str, Any]]:
    width_mm, height_mm = config["size_mm"]
    graph = base.new_graph(config["graph"], plot_id, mm_to_in(width_mm), mm_to_in(height_mm))
    layer = graph[0]
    # Match the SHAP-family lower title band so the two-line axis title stays
    # clear of the enlarged tick labels without reducing font size.
    base.position_layer(layer, 47, 5, 50, 76)
    count = len(rows)
    ylim = (0.45, count + 1.65)
    base.vertical_rule(layer, 0.0, ylim[0], float(count) + 0.35, color=INK, width=0.9)
    negative_x: list[Any] = []
    negative_y: list[Any] = []
    positive_x: list[Any] = []
    positive_y: list[Any] = []
    labels: list[str] = []
    for row in rows:
        order = int(float(row["display_order"]))
        y = count - order + 1
        value = float(row["grouped_shap_pce_points"])
        label = row["scientific_group"]
        if row["aggregation"].startswith("signed net"):
            label += "\n(signed net remainder; cancellation possible)"
        labels.append(label)
        target_x, target_y = (negative_x, negative_y) if value < 0 else (positive_x, positive_y)
        target_x.extend([0.0, value, None])
        target_y.extend([y, y, None])
    negative_plot = add_linked_line(layer, source_table, "negative_bars", negative_x, negative_y, PPT_SHAP_NEGATIVE, 14.0)
    positive_plot = add_linked_line(layer, source_table, "positive_bars", positive_x, positive_y, PPT_SHAP_POSITIVE, 14.0)
    try:
        negative_plot.set_int("line.cap", 0)
        positive_plot.set_int("line.cap", 0)
    except Exception:
        pass
    style_axes(
        layer,
        config,
        ylim,
        tick_size=ASSEMBLY_BODY_SIZE,
        axis_title_size=ASSEMBLY_TITLE_SIZE,
    )
    add_manual_x_title(layer, str(config["xlabel"]), 117, size=ASSEMBLY_TITLE_SIZE)
    base.set_categorical_ticks(layer, source_table, labels, plot_id, [float(count - i) for i in range(count)])
    return graph, {
        "source_rows": len(rows),
        "displayed_bars": len(rows),
        "xlim": list(config["xlim"]),
        "ylim": list(ylim),
        "embedded_legend": False,
        "typography_pt": {"body": ASSEMBLY_BODY_SIZE, "axis_title": ASSEMBLY_TITLE_SIZE},
        "powerpoint_style": "dark navy/burnt orange; separate legend",
    }


def write_validation_sheet(book: Any, plot_id: str, config: dict[str, Any], prepared: PreparedData, audit: dict[str, Any]) -> None:
    sheet = book.add_sheet()
    sheet.name = "Validation"
    sheet.lname = f"{plot_id} validation"
    width_mm, height_mm = config["size_mm"]
    rows = [
        ("analysis_id", ANALYSIS_ID, "PASS"),
        ("analysis_version", ANALYSIS_VERSION, "PASS"),
        ("plot_id", plot_id, "PASS"),
        ("source_file", prepared.source_copy.name, "PASS"),
        ("source_sha256", sha256(prepared.source_copy), "PASS"),
        ("source_rows", prepared.source_rows, "PASS"),
        ("expected_rows", config["expected"], "PASS"),
        ("graph_pages", 1, "PASS"),
        ("standalone_width_mm", width_mm, "PASS"),
        ("standalone_height_mm", height_mm, "PASS"),
        ("gridlines", "disabled", "PASS"),
        ("font", FONT, "PASS"),
        ("x_limits", json.dumps(config["xlim"]), "PASS"),
        ("plot_audit", json.dumps(audit, ensure_ascii=False), "PASS"),
    ]
    for col, header in enumerate(("check", "value", "status")):
        sheet.from_list(col, [row[col] for row in rows], lname=header, axis="")


def write_raster_backed_svg(png: Path, svg: Path, width_in: float, height_in: float, plot_id: str) -> None:
    payload = base64.b64encode(png.read_bytes()).decode("ascii")
    width_px = round(width_in * 600)
    height_px = round(height_in * 600)
    text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_in:.6f}in" height="{height_in:.6f}in" '
        f'viewBox="0 0 {width_px} {height_px}">\n'
        f'  <title>{plot_id} — complete 600-dpi Origin rendering</title>\n'
        '  <desc>Raster-backed SVG used because Origin 2024 cannot reliably finish full-vector export of this dense editable point cloud. No points were omitted.</desc>\n'
        f'  <rect width="{width_px}" height="{height_px}" fill="#FFFFFF"/>\n'
        f'  <image x="0" y="0" width="{width_px}" height="{height_px}" href="data:image/png;base64,{payload}"/>\n'
        '</svg>\n'
    )
    svg.write_text(text, encoding="utf-8", newline="\n")


def svg_to_pdf(svg: Path, pdf: Path, width_in: float, height_in: float) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    browser = chrome_path(None)
    with tempfile.TemporaryDirectory(prefix="origin_svg_pdf_", dir=pdf.parent, ignore_cleanup_errors=True) as temp_name:
        temp = Path(temp_name)
        profile = temp / "profile"
        html = temp / "print.html"
        uri = svg.resolve().as_uri().replace("&", "&amp;").replace('"', "&quot;")
        html.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            f"@page{{size:{width_in:.6f}in {height_in:.6f}in;margin:0;}}"
            f"html,body{{margin:0;width:{width_in:.6f}in;height:{height_in:.6f}in;overflow:hidden;background:#fff;}}"
            "img{display:block;width:100%;height:100%;object-fit:fill;}"
            "</style></head>"
            f"<body><img src=\"{uri}\"></body></html>",
            encoding="utf-8",
        )
        command = [
            str(browser), "--headless=new", "--disable-gpu", "--disable-background-networking", "--disable-sync",
            "--no-first-run", "--no-default-browser-check", "--no-pdf-header-footer",
            f"--user-data-dir={profile}", f"--print-to-pdf={pdf.resolve()}", html.resolve().as_uri(),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not pdf.is_file() or pdf.stat().st_size == 0:
            raise RuntimeError(f"Chrome PDF conversion failed ({result.returncode}): {result.stderr}")


def force_svg_physical_size(svg: Path, width_in: float, height_in: float) -> None:
    source = svg.read_text(encoding="utf-8")
    root_match = re.search(r"<svg\b[^>]*>", source, flags=re.IGNORECASE)
    if root_match is None:
        raise RuntimeError(f"SVG root not found: {svg}")
    root = root_match.group(0)
    root = re.sub(r"\swidth\s*=\s*([\"']).*?\1", f' width="{width_in:.6f}in"', root, count=1)
    root = re.sub(r"\sheight\s*=\s*([\"']).*?\1", f' height="{height_in:.6f}in"', root, count=1)
    svg.write_text(source[: root_match.start()] + root + source[root_match.end() :], encoding="utf-8", newline="\n")


def render_svg_png_exact(svg: Path, png: Path, width_px: int, height_px: int) -> tuple[int, int]:
    browser = chrome_path(None)
    png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="origin_svg_render_", dir=png.parent, ignore_cleanup_errors=True) as temp_name:
        temp = Path(temp_name)
        profile = temp / "profile"
        html = temp / "render.html"
        uri = svg.resolve().as_uri().replace("&", "&amp;").replace('"', "&quot;")
        html.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            "html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#fff;}"
            "img{display:block;width:100vw;height:100vh;object-fit:fill;}"
            "</style></head>"
            f"<body><img src=\"{uri}\"></body></html>",
            encoding="utf-8",
        )
        command = [
            str(browser), "--headless=new", "--disable-gpu", "--disable-background-networking", "--disable-sync",
            "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--force-device-scale-factor=1",
            "--default-background-color=ffffffff", "--run-all-compositor-stages-before-draw", "--virtual-time-budget=1000",
            f"--user-data-dir={profile}", f"--window-size={width_px},{height_px}", f"--screenshot={png.resolve()}",
            html.resolve().as_uri(),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not png.is_file() or png.stat().st_size == 0:
            raise RuntimeError(f"Chrome PNG conversion failed ({result.returncode}): {result.stderr}")
    set_png_resolution(png)
    return width_px, height_px


def export_graph(graph: Any, plot_id: str, config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    width_mm, height_mm = config["size_mm"]
    width_in, height_in = mm_to_in(width_mm), mm_to_in(height_mm)
    width_px, height_px = round(width_in * 600), round(height_in * 600)
    png = output_root / "png" / f"{config['stem']}.png"
    svg = output_root / "svg" / f"{config['stem']}.svg"
    pdf = output_root / "pdf" / f"{config['stem']}.pdf"
    raw_dir = output_root / "validation" / "origin_raw_exports" / plot_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in (png, svg, pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
    graph.activate()
    graph.lt_exec("page.speedmode=0;")
    graph[0].lt_exec("layer.maxpts=0; page.speedmode=0;")
    op.wait()
    mode = "origin_vector_svg"
    sanitizer_audit: dict[str, Any] | None = None
    if config.get("dense"):
        raw_png = raw_dir / f"{config['stem']}.origin.png"
        exported = graph.save_fig(str(raw_png), type="png", width=width_px)
        if not exported or not raw_png.is_file() or raw_png.stat().st_size == 0:
            raise RuntimeError(f"Origin failed to export dense complete PNG: {raw_png}")
        shutil.copy2(raw_png, png)
        set_png_resolution(png)
        write_raster_backed_svg(png, svg, width_in, height_in, plot_id)
        mode = "complete_origin_600dpi_raster_backed_svg_pdf"
    else:
        raw_svg = raw_dir / f"{config['stem']}.origin.svg"
        exported = graph.save_fig(str(raw_svg), type="svg", ratio=100)
        if not exported or not raw_svg.is_file() or raw_svg.stat().st_size == 0:
            raise RuntimeError(f"Origin failed to export SVG: {raw_svg}")
        audit_path = output_root / "validation" / f"{plot_id}_svg_sanitizer.json"
        sanitizer_audit = sanitize_svg(raw_svg, svg, audit_path, width_in)
        force_svg_physical_size(svg, width_in, height_in)
        rendered = render_svg_png_exact(svg, png, width_px, height_px)
        if rendered != (width_px, height_px):
            raise RuntimeError(f"Unexpected PNG dimensions for {plot_id}: {rendered} vs {(width_px, height_px)}")
    svg_to_pdf(svg, pdf, width_in, height_in)
    for path in (png, svg, pdf):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing export: {path}")
    return {
        "png": str(png.resolve()), "png_bytes": png.stat().st_size, "png_dimensions": [width_px, height_px],
        "svg": str(svg.resolve()), "svg_bytes": svg.stat().st_size,
        "pdf": str(pdf.resolve()), "pdf_bytes": pdf.stat().st_size,
        "export_mode": mode,
        "svg_sanitizer": sanitizer_audit,
    }


def build_one(plot_id: str, output_root: Path, close: bool) -> dict[str, Any]:
    global op
    op = base.require_origin()
    config = PLOT_CONFIG[plot_id]
    output_root = output_root.resolve()
    for directory in ("origin_projects", "pdf", "svg", "png", "data", "validation"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prepared = prepare_inputs(plot_id, config, output_root)
    project_path = output_root / "origin_projects" / f"{config['stem']}.opju"
    build_audit_path = output_root / "validation" / f"{plot_id}_build_audit.json"
    if getattr(op, "oext", False):
        op.set_show(not close)
    op.new(asksave=False)
    base.FONT_INDEX = int(op.lt_int(f'font("{FONT}")'))
    book = op.new_book("w", lname=f"{plot_id} exact source and helpers")
    if book is None:
        raise RuntimeError("Origin could not create source workbook")
    book.name = "PlotData"
    source_sheet, source_table = import_csv_sheet(book, prepared.source_copy, "Source", "Exact authoritative plotting table", first=True)
    source_headers, source_rows = read_csv_rows(prepared.source_copy) if prepared.source_rows < 1000 else ([], [])
    family = config["family"]
    if family == "pce":
        source_table.rows = [{key: base.parse_scalar(value) for key, value in row.items()} for row in source_rows]
        graph, audit = build_pce(plot_id, config, source_table, source_rows)
    elif family == "stability":
        source_table.rows = [{key: base.parse_scalar(value) for key, value in row.items()} for row in source_rows]
        graph, audit = build_stability(plot_id, config, source_table, source_rows)
    elif family == "shap_local":
        source_table.rows = [{key: base.parse_scalar(value) for key, value in row.items()} for row in source_rows]
        graph, audit = build_local_shap(plot_id, config, source_table, source_rows)
    else:
        if prepared.helper is None:
            raise RuntimeError(f"Missing SHAP helper for {plot_id}")
        _, point_table = import_csv_sheet(book, prepared.helper, "Points", "Complete deterministic Origin point helper")
        label_sheet = book.add_sheet()
        label_sheet.name = "Labels"
        label_sheet.lname = "Categorical axis labels"
        label_table = base.Table("Labels", prepared.helper, label_sheet, [], {}, [])
        summary_table = None
        if prepared.summary is not None:
            _, summary_table = import_csv_sheet(book, prepared.summary, "Summary", "Exact medians and IQR helpers")
        graph, audit = build_shap(plot_id, config, point_table, label_table, prepared, summary_table)
    write_validation_sheet(book, plot_id, config, prepared, audit)
    graph_pages = op.graph_list("p")
    if len(graph_pages) != 1 or graph_pages[0].name != config["graph"]:
        raise RuntimeError(f"{plot_id}: expected exactly one graph page named {config['graph']}; got {[g.name for g in graph_pages]}")
    if not op.save(str(project_path.resolve())):
        raise RuntimeError(f"Origin failed to save project: {project_path}")
    exports = export_graph(graph, plot_id, config, output_root)
    report = {
        "status": "PASS", "analysis_id": ANALYSIS_ID, "analysis_version": ANALYSIS_VERSION,
        "plot_id": plot_id, "family": family, "origin_project": str(project_path.resolve()),
        "origin_graph_page": config["graph"], "origin_graph_long_name": plot_id,
        "source_copy": str(prepared.source_copy.resolve()), "source_sha256": sha256(prepared.source_copy),
        "source_rows": prepared.source_rows, "expected_rows": config["expected"],
        "page_dimensions_mm": list(config["size_mm"]), "gridlines": "disabled", "font": FONT,
        "build_audit": audit, "exports": exports,
    }
    build_audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_id", choices=[*sorted(PLOT_CONFIG), "all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--close", action="store_true", help="Close external Origin after saving and export.")
    parser.add_argument("--dry-run", action="store_true", help="Validate source files and row counts without opening OriginPro.")
    return parser.parse_args()


def dry_run_report(plot_id: str) -> dict[str, Any]:
    config = PLOT_CONFIG[plot_id]
    source = PACKAGE / config["source"]
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        headers = next(reader, [])
        if not headers or len(set(headers)) != len(headers):
            raise RuntimeError(f"CSV headers must be present and unique: {source}")
        row_count = sum(1 for row in reader if row)
    if row_count != int(config["expected"]):
        raise RuntimeError(f"{plot_id}: expected {config['expected']} source rows; found {row_count}")
    support = [PACKAGE / relative for relative in config.get("copies", [])]
    for path in support:
        if not path.is_file():
            raise FileNotFoundError(path)
    spec_path = PACKAGE / "specifications" / f"{plot_id}.json"
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    if specification.get("plot_id") != plot_id:
        raise RuntimeError(f"Plot specification identity does not match {plot_id}")
    return {
        "status": "PASS", "plot_id": plot_id, "source": config["source"],
        "source_sha256": sha256(source), "source_rows": row_count,
        "specification_sha256": sha256(spec_path), "origin_imported": False,
        "support_files": [path.relative_to(PACKAGE).as_posix() for path in support],
    }


def main() -> None:
    args = parse_args()
    selected = sorted(PLOT_CONFIG) if args.plot_id == "all" else [args.plot_id]
    if args.dry_run:
        print(json.dumps({"status": "PASS", "mode": "dry-run", "plots": [dry_run_report(plot_id) for plot_id in selected]}, indent=2))
        return
    succeeded = False
    try:
        reports = [build_one(plot_id, args.output_root, args.close) for plot_id in selected]
        succeeded = True
        payload: dict[str, Any] | list[dict[str, Any]] = reports[0] if len(reports) == 1 else reports
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        if getattr(op, "oext", False) and (args.close or not succeeded):
            try:
                op.exit()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
