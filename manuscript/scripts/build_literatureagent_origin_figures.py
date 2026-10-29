"""Shared Origin drawing and export functions for the manuscript builders."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

op: Any = None

from render_svg_with_chrome import render as render_svg_png
from sanitize_origin_svg import sanitize as sanitize_svg
from build_knowledge_graph_figure import build_outputs as build_knowledge_graph_outputs


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source"
INPUTS = ROOT / "origin_ready_data"
EXPORT_ROOT = ROOT / "build"
INDIVIDUAL = EXPORT_ROOT / "figures_individual"
MULTIPANEL = EXPORT_ROOT / "figures_multipanel"
LEGENDS = EXPORT_ROOT / "legends"
PROJECT_DIR = EXPORT_ROOT / "origin_projects"
RAW_SVG_DIR = EXPORT_ROOT / "origin_raw_svg"
RAW_RASTER_DIR = EXPORT_ROOT / "origin_raw_raster"
SVG_AUDIT_DIR = EXPORT_ROOT / "svg_export_audits"
PROJECT_OUT = PROJECT_DIR / "LiteratureAgent_reference_plots.opju"
LOG = EXPORT_ROOT / "ORIGIN_BUILD_LOG.txt"
MANIFEST = EXPORT_ROOT / "ORIGIN_EXPORT_MANIFEST.json"

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
SHAP_NEGATIVE = "#174766"
SHAP_POSITIVE = "#C65A00"
ACCEPTED = "#137321"
SPARSE = "#AB4B00"
REJECTED = "#EC0611"
PALE_SLATE = "#E9EEF3"
WHITE = "#FFFFFF"
FONT = "Times New Roman"


PANEL_FILES = {
    "F2A": "figure2a_campaign_accounting.csv",
    "F2ACTX": "figure2a_context.csv",
    "F2B": "figure2b_model_readiness.csv",
    "F3A": "figure3a_phrase_recall.csv",
    "F3B": "figure3b_cosine_similarity.csv",
    "F3C": "figure3c_token_f1.csv",
    "F3D": "figure3d_verbosity_ratio.csv",
    "F4A": "figure4a_historical_r2.csv",
    "F4B": "figure4b_historical_rmse.csv",
    "F4C": "figure4c_historical_mae.csv",
    "F4D": "figure4d_accepted_literatureagent_r2.csv",
    "F4E": "figure4e_accepted_literatureagent_rmse.csv",
    "F4F": "figure4f_accepted_literatureagent_mae.csv",
    "F5PTS": "figure5_choice_shap_points.csv",
    "F5NEG": "figure5_choice_shap_negative_points.csv",
    "F5POS": "figure5_choice_shap_positive_points.csv",
    "F5SUM": "figure5_choice_shap_summary.csv",
    "F6A": "figure6a_classification.csv",
    "F6B": "figure6b_regression.csv",
    "F7E": "figure7_edges.csv",
    "F7N": "figure7_nodes.csv",
}


@dataclass
class Table:
    key: str
    path: Path
    sheet: Any
    headers: list[str]
    columns: dict[str, int]
    rows: list[dict[str, Any]]

    def append_column(self, name: str, values: list[Any], axis: str = "") -> int:
        if name in self.columns:
            index = self.columns[name]
            self.sheet.from_list(index, values, lname=name, axis=axis)
            return index
        index = self.sheet.cols
        self.sheet.from_list(index, values, lname=name, axis=axis)
        self.headers.append(name)
        self.columns[name] = index
        return index


TABLES: dict[str, Table] = {}
EXPORTS: list[dict[str, Any]] = []
FONT_INDEX = 0
_label_counter = 0


def require_origin() -> Any:
    """Load the Origin bridge only when an actual plot is requested."""
    global op
    if op is None:
        try:
            import originpro
        except ImportError as exc:
            raise RuntimeError("OriginPro plotting requires originpro and a licensed Windows Origin installation.") from exc
        op = originpro
    return op


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return number if math.isfinite(number) else None


def read_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers or len(headers) != len(set(headers)):
            raise RuntimeError(f"Invalid headers in {path}")
        rows = [{header: parse_scalar(row[header]) for header in headers} for row in reader]
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return headers, rows


FIGURE_TABLE_KEYS = {
    "figure2": {"F2A", "F2ACTX", "F2B"},
    "figure3": {"F3A", "F3B", "F3C", "F3D"},
    "figure4": {"F4A", "F4B", "F4C", "F4D", "F4E", "F4F"},
    "figure5": {"F5PTS", "F5NEG", "F5POS", "F5SUM"},
    "figure6": {"F6A", "F6B"},
    "figure7": {"F7E", "F7N"},
}


def import_tables(selected: set[str]) -> None:
    needed = set().union(*(FIGURE_TABLE_KEYS[name] for name in selected))
    book = op.new_book("w", lname="LiteratureAgent plot-ready data")
    if book is None:
        raise RuntimeError("Origin could not create the source-data workbook")
    book.name = "LAData"
    first = True
    for key, filename in PANEL_FILES.items():
        if key not in needed:
            continue
        path = INPUTS / filename
        headers, rows = read_rows(path)
        sheet = book[0] if first else book.add_sheet()
        first = False
        sheet.name = key
        sheet.lname = filename.removesuffix(".csv")
        for index, header in enumerate(headers):
            sheet.from_list(index, [row[header] for row in rows], lname=header, axis="")
        TABLES[key] = Table(
            key=key,
            path=path,
            sheet=sheet,
            headers=list(headers),
            columns={name: index for index, name in enumerate(headers)},
            rows=rows,
        )
        log(f"Imported {filename}: {len(rows):,} rows")


def set_page_size(graph: Any, width_in: float, height_in: float) -> None:
    graph.activate()
    width_dot = int(round(width_in * 600))
    height_dot = int(round(height_in * 600))
    graph.lt_exec(
        "page.updatetoprinter=0; page.kar=0; page.nodark=1; page.viewmode=1; "
        "page.legendupdatemode=2; "
        f"page.resx=600; page.resy=600; page.width={width_dot}; page.height={height_dot}; "
        "page.speedmode=0;"
    )


def new_graph(short_name: str, long_name: str, width_in: float, height_in: float) -> Any:
    existing = op.find_graph(short_name)
    if existing is not None:
        existing.destroy()
    graph = op.new_graph(lname=long_name)
    if graph is None:
        raise RuntimeError(f"Origin could not create graph {short_name}")
    graph.name = short_name
    set_page_size(graph, width_in, height_in)
    return graph


def position_layer(layer: Any, left: float, top: float, width: float, height: float) -> None:
    try:
        layer.set_int("unit", 1)
    except Exception:
        layer.lt_exec("layer.unit=1;")
    layer.set_float("left", left)
    layer.set_float("top", top)
    layer.set_float("width", width)
    layer.set_float("height", height)
    layer.lt_exec("label -r legend; label -r title;")


def style_label(label: Any, size: int = 9, bold: bool = False, color: str = INK) -> None:
    if label is None:
        return
    try:
        label.set_int("font", FONT_INDEX)
        label.set_int("fsize", int(size))
        label.set_int("bold", 1 if bold else 0)
        # Origin text objects can inherit character-decoration state from the
        # active template/session.  Reset it explicitly so unattended exports
        # do not acquire spurious rules above or through label text.
        label.set_int("italic", 0)
        label.set_int("underline", 0)
        label.set_int("background", 0)
        label.color = color
    except Exception:
        pass


def style_axis(
    layer: Any,
    x_title: str,
    *,
    xlim: tuple[float, float],
    xstep: float,
    ylim: tuple[float, float],
    ystep: float = 1,
    show_y_labels: bool = True,
    axis_title_size: int = 10,
    tick_size: int = 9,
) -> None:
    layer.axis("x").set_limits(xlim[0], xlim[1], xstep)
    layer.axis("y").set_limits(ylim[0], ylim[1], ystep)
    layer.axis("x").title = x_title
    layer.axis("y").title = ""
    style_label(layer.label("xb"), axis_title_size)
    style_label(layer.label("yl"), axis_title_size)
    show_y = 1 if show_y_labels else 0
    layer.lt_exec(
        f'layer.x.label.font=font("{FONT}"); layer.y.label.font=font("{FONT}"); '
        f"layer.x.label.pt={tick_size}; layer.y.label.pt={tick_size}; "
        f"layer.y.showlabel={show_y}; layer.x.thickness=1.2; layer.y.thickness=1.2; "
        "layer.x.ticks=2; layer.y.ticks=2; "
        "layer.x.grid.majorcolor=0; layer.x.grid.minorcolor=0; "
        "layer.y.grid.majorcolor=0; layer.y.grid.minorcolor=0; "
        f'xb.font=font("{FONT}"); xb.fsize={axis_title_size}; '
        f'yl.font=font("{FONT}"); yl.fsize={axis_title_size};'
    )
    layer.lt_exec("layer.sauto=0;")


_LINE_COUNTER = 0


class _LabTalkLine:
    """Line proxy for Origin versions whose embedded GLayer lacks add_line."""

    def __init__(self, layer: Any, name: str) -> None:
        self.layer = layer
        self.name = name

    def _set_numeric(self, name: str, value: float) -> None:
        self.layer.lt_exec(f"{self.name}.{name}={float(value):.15g};")

    def set_int(self, name: str, value: int) -> None:
        self._set_numeric(name, int(value))

    def set_float(self, name: str, value: float) -> None:
        self._set_numeric(name, value)

    @property
    def width(self) -> float:
        return 0.0

    @width.setter
    def width(self, value: float) -> None:
        self._set_numeric("lineWidth", value)

    @property
    def color(self) -> int:
        return 0

    @color.setter
    def color(self, value: Any) -> None:
        if isinstance(value, str):
            self.layer.lt_exec(f'{self.name}.color=color("{value}");')
        else:
            self._set_numeric("color", value)

    @property
    def type(self) -> int:
        return 0

    @type.setter
    def type(self, value: int) -> None:
        self._set_numeric("lineType", value)


def add_line_object(layer: Any, x1: float, y1: float, x2: float, y2: float) -> Any:
    """Add a data-attached line in external or Origin 2024 embedded Python."""
    add_line = getattr(layer, "add_line", None)
    if callable(add_line):
        try:
            return add_line(x1, y1, x2, y2)
        except AttributeError:
            pass
    global _LINE_COUNTER
    _LINE_COUNTER += 1
    name = f"laLine{_LINE_COUNTER}"
    coords = ",".join(f"{float(value):.15g}" for value in (x1, y1, x2, y2))
    layer.lt_exec(f"draw -n {name} -l {{{coords}}}; {name}.attach=2;")
    line = _LabTalkLine(layer, name)
    return line


def vertical_rule(layer: Any, x: float, ymin: float, ymax: float, *, color: str = GRID, width: float = 0.8, line_type: int = 1) -> None:
    line = add_line_object(layer, x, ymin, x, ymax)
    if line is not None:
        line.width = width
        line.color = color
        line.type = line_type


def add_data_label(
    layer: Any,
    text: str,
    x: float,
    y: float,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    size: int = 9,
    bold: bool = False,
    color: str = INK,
    center: bool = False,
    whiteout: bool = False,
) -> Any:
    x_percent = 100.0 * (x - xlim[0]) / (xlim[1] - xlim[0])
    y_percent = 100.0 * (ylim[1] - y) / (ylim[1] - ylim[0])
    label = add_percent_label(
        layer,
        text,
        x_percent,
        y_percent,
        size=size,
        bold=bold,
        color=color,
        whiteout=whiteout,
    )
    label.set_int("anchor", 5 if center else 1)
    return label


def add_boxed_data_label(
    layer: Any,
    text: str,
    x: float,
    y: float,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    border: str,
    fill: str,
    text_color: str = INK,
    size: int = 9,
) -> Any:
    """Add a centered, editable Origin text node with a colored box."""
    label = add_data_label(
        layer,
        text,
        x,
        y,
        xlim=xlim,
        ylim=ylim,
        size=size,
        bold=True,
        color=text_color,
        center=True,
    )
    name = str(label)
    layer.lt_exec(
        f'{name}.background=1; {name}.bordercolor=color("{border}"); '
        f'{name}.fillcolor=color("{fill}"); {name}.margin.unit=1; '
        f'{name}.margin.left=32; {name}.margin.right=32; '
        f'{name}.margin.top=20; {name}.margin.bottom=20;'
    )
    return label


def add_percent_label(
    layer: Any,
    text: str,
    x_percent: float,
    y_percent: float,
    *,
    size: int = 9,
    bold: bool = False,
    color: str = INK,
    whiteout: bool = False,
) -> Any:
    global _label_counter
    _label_counter += 1
    name = f"laLabel{_label_counter}"
    safe_text = str(text).replace('"', "'").replace("\r\n", "\\n").replace("\n", "\\n")
    layer.lt_exec(f'label -p {x_percent:g} {y_percent:g} -sa -n {name} "{safe_text}";')
    label = layer.label(name)
    if label is None:
        raise RuntimeError(f"Origin could not create percent label {name}")
    style_label(label, size, bold, color)
    layer.lt_exec(
        f'{name}.font=font("{FONT}"); {name}.fsize={int(size)}; '
        f'{name}.color=color("{color}"); {name}.whiteout={1 if whiteout else 0};'
    )
    return label


def set_legend(
    layer: Any,
    text: str,
    x_percent: float,
    y_percent: float,
    size: int = 9,
    *,
    boxed: bool = True,
) -> None:
    layer.lt_exec("label -r legend;")
    layer.lt_exec(f'label -p {x_percent:g} {y_percent:g} -n legend "legend";')
    legend = layer.label("legend")
    if legend is None:
        raise RuntimeError("Origin could not create a legend")
    legend.text = text
    if boxed:
        layer.lt_exec(
            f'legend.font=font("{FONT}"); legend.fsize={int(size)}; '
            f'legend.color=color("{INK}"); legend.whiteout=1; '
            f'legend.background=1; legend.bordercolor=color("{INK}"); '
            f'legend.fillcolor=color("{WHITE}"); legend.margin.unit=1; '
            f'legend.margin.left=12; legend.margin.right=12; '
            f'legend.margin.top=6; legend.margin.bottom=6;'
        )
    else:
        layer.lt_exec(
            f'legend.font=font("{FONT}"); legend.fsize={int(size)}; '
            f'legend.color=color("{INK}"); legend.bold=0; legend.italic=0; '
            f'legend.underline=0; legend.whiteout=0; legend.background=0;'
        )


def set_categorical_ticks(
    layer: Any,
    table: Table,
    labels_top_to_bottom: list[str],
    token: str,
    positions_top_to_bottom: list[float] | None = None,
) -> None:
    column_name = f"axis_labels_{token}"
    labels_bottom_to_top = list(reversed(labels_top_to_bottom))
    col = table.append_column(column_name, labels_bottom_to_top)
    dataset = f"{table.sheet.lt_range()}!col({col + 1})"
    layer.lt_exec(f'layer.y.label.type=2; layer.y.label.dataset$="{dataset}";')
    if positions_top_to_bottom is not None:
        if len(positions_top_to_bottom) != len(labels_top_to_bottom):
            raise RuntimeError(f"Mismatched categorical tick positions for {token}")
        positions = " ".join(f"{value:g}" for value in reversed(positions_top_to_bottom))
        layer.lt_exec(f'layer.y.ticksbydata$="{positions}";')


def append_xy(table: Table, prefix: str, xs: list[Any], ys: list[Any]) -> tuple[int, int]:
    if len(xs) != len(ys):
        raise RuntimeError(f"Mismatched helper XY lengths for {prefix}")
    clean_x = [math.nan if value is None else value for value in xs]
    clean_y = [math.nan if value is None else value for value in ys]
    x_col = table.append_column(f"{prefix}_x", clean_x, axis="X")
    y_col = table.append_column(f"{prefix}_y", clean_y, axis="Y")
    return x_col, y_col


def style_scatter(
    plot: Any,
    *,
    color: str,
    kind: int,
    size: float,
    interior: int = 1,
    transparency: int = 0,
    fill_color: str | None = None,
) -> None:
    plot.color = color
    plot.symbol_kind = kind
    plot.symbol_interior = interior
    plot.symbol_size = size
    if fill_color is not None:
        fill_code = op.ocolor(fill_color)
        plot.set_int("symbol.fillcolor", fill_code)
        plot.set_int("symbol.edgecolor", fill_code)
    try:
        plot.transparency = transparency
    except Exception:
        pass


def add_scatter(
    layer: Any,
    table: Table,
    x_col: int,
    y_col: int,
    *,
    color: str,
    kind: int,
    size: float,
    interior: int = 1,
    transparency: int = 0,
    fill_color: str | None = None,
) -> Any:
    plot = layer.add_plot(table.sheet, coly=y_col, colx=x_col, type="s")
    if plot is None:
        raise RuntimeError(f"Origin could not add scatter plot from {table.key}")
    style_scatter(
        plot,
        color=color,
        kind=kind,
        size=size,
        interior=interior,
        transparency=transparency,
        fill_color=fill_color,
    )
    return plot


def ci_segment(
    layer: Any,
    low: float,
    high: float,
    y: float,
    color: str,
    cap: float = 0.10,
    width_scale: float = 1.0,
) -> None:
    line = add_line_object(layer, low, y, high, y)
    if line is not None:
        line.width = 1.45 * width_scale
        line.color = color
    for x in (low, high):
        cap_line = add_line_object(layer, x, y - cap, x, y + cap)
        if cap_line is not None:
            cap_line.width = 1.20 * width_scale
            cap_line.color = color


def y_to_percent(y: float, ylim: tuple[float, float]) -> float:
    ymin, ymax = ylim
    return 100.0 * (ymax - y) / (ymax - ymin)


def build_figure2_panel(
    layer: Any,
    panel: str,
    *,
    show_letter: bool = True,
    font_scale: float = 1.0,
) -> None:
    data_size = max(1, int(round(9 * font_scale)))
    axis_size = max(1, int(round(10 * font_scale)))
    tick_size = max(1, int(round(9 * font_scale)))
    letter_size = max(1, int(round(11 * font_scale)))
    if panel == "a":
        table = TABLES["F2A"]
        xlim = (0.0, 1600.0)
        ylim = (0.5, 6.55)
        labels = [str(row["category"]) for row in table.rows]
        colors = {
            "neutral": SLATE,
            "retained": ORANGE,
            "accepted": ACCEPTED,
            "sparse_accepted": SPARSE,
            "rejected": REJECTED,
        }
        for index, row in enumerate(table.rows):
            y = 6.0 - index
            value = float(row["count"])
            line = add_line_object(layer, 0, y, value, y)
            if line is not None:
                line.width = 11
                line.color = colors[str(row["status_class"])]
            add_data_label(
                layer,
                f"{int(round(value)):,}",
                value + 12,
                y + 0.08,
                xlim=xlim,
                ylim=ylim,
                size=data_size,
                bold=True,
            )
        style_axis(
            layer,
            "Count",
            xlim=xlim,
            xstep=500,
            ylim=ylim,
            show_y_labels=True,
            axis_title_size=axis_size,
            tick_size=tick_size,
        )
        set_categorical_ticks(layer, table, labels, "f2a")
        if show_letter:
            add_data_label(layer, "(a)", 8, 6.42, xlim=xlim, ylim=ylim, size=letter_size, bold=True)
    else:
        table = TABLES["F2B"]
        xlim = (0.0, 960.0)
        ylim = (0.5, 6.55)
        labels = [str(row["category"]) for row in table.rows]
        colors = {
            "pce": SLATE,
            "stability": ORANGE_2,
            "physical": GRID,
            "completeness": ORANGE,
        }
        for index, row in enumerate(table.rows):
            y = 6.0 - index
            value = float(row["count"])
            if value > 0:
                line = add_line_object(layer, 0, y, value, y)
                if line is not None:
                    line.width = 11
                    line.color = colors[str(row["screen_group"])]
            value_label = str(row["value_label"]) if str(row["screen_group"]) == "completeness" else f"{int(round(value)):,}"
            add_data_label(
                layer,
                value_label,
                value + 18,
                y + 0.08,
                xlim=xlim,
                ylim=ylim,
                size=data_size,
                bold=True,
            )
        style_axis(
            layer,
            "Count passing screen",
            xlim=xlim,
            xstep=200,
            ylim=ylim,
            show_y_labels=True,
            axis_title_size=axis_size,
            tick_size=tick_size,
        )
        set_categorical_ticks(layer, table, labels, "f2b")
        if show_letter:
            add_data_label(layer, "(b)", 5, 6.42, xlim=xlim, ylim=ylim, size=letter_size, bold=True)


def build_figure2() -> None:
    for panel in ("a", "b"):
        graph = new_graph(f"F2{panel.upper()}_Ind", f"Figure 2{panel} individual", 6.5, 4.75)
        position_layer(graph[0], 46, 4, 51, 77)
        # Individual panels are assembled and lettered in PowerPoint.
        # The 2.05x source typography resolves to approximately 9–10 pt when
        # this 6.5-inch panel is scaled to a 3.15-inch two-column slot.
        build_figure2_panel(graph[0], panel, show_letter=False, font_scale=2.05)
        export_graph(graph, f"Figure2{panel}_individual", INDIVIDUAL, 6.5)

    graph = new_graph("F2_All", "Figure 2 manuscript composite", 6.5, 4.9)
    top = graph[0]
    bottom = graph.add_layer(0)
    position_layer(top, 29, 7, 67, 31)
    position_layer(bottom, 29, 54, 67, 31)
    build_figure2_panel(top, "a", font_scale=1.5)
    build_figure2_panel(bottom, "b", font_scale=1.5)
    export_graph(graph, "Figure2_campaign_and_model_readiness", MULTIPANEL, 6.5)


F3_CONFIG = {
    "a": ("F3A", "Ground-truth phrase recall (%)", (20.0, 105.0), 20.0),
    "b": ("F3B", "Cosine similarity (%)", (0.0, 40.0), 10.0),
    "c": ("F3C", "Ground-truth token F1 (%)", (0.0, 16.0), 4.0),
    "d": ("F3D", "Response/reference word ratio", (8.0, 80.0), 20.0),
}


def build_figure3_panel(
    layer: Any,
    panel: str,
    *,
    show_y_labels: bool,
    add_legend: bool = False,
    font_marker_scale: float = 1.0,
    legend_x_percent: float = 14.0,
    legend_y_percent: float = 2.0,
    shared_legend: bool = False,
    panel_label_y_percent: float = 1.0,
    show_letter: bool = True,
    tight_vertical_range: bool = False,
) -> None:
    key, x_title, xlim, xstep = F3_CONFIG[panel]
    table = TABLES[key]
    ylim = (0.5, 8.65) if tight_vertical_range else (0.5, 10.2)
    tick_size = max(1, round(9 * font_marker_scale))
    axis_title_size = max(1, round(10 * font_marker_scale))
    panel_label_size = max(1, round(11 * font_marker_scale))
    # The shared composite legend spans both columns; one source-point smaller
    # preserves the 9 pt final-size target without colliding with panel (b).
    legend_size = max(1, round((8 if shared_legend else 9) * font_marker_scale))
    seen: set[str] = set()
    comp_x: list[Any] = []
    comp_y: list[Any] = []
    la_x: list[Any] = []
    la_y: list[Any] = []
    comp_mean_x: list[Any] = []
    comp_mean_y: list[Any] = []
    la_mean_x: list[Any] = []
    la_mean_y: list[Any] = []
    labels_by_order: dict[int, str] = {}
    for row in table.rows:
        series = str(row["series_key"])
        point_x = float(row["point_value"])
        point_y = float(row["point_y"])
        is_la = series == "literatureagent"
        comp_x.append(None if is_la else point_x)
        comp_y.append(None if is_la else point_y)
        la_x.append(point_x if is_la else None)
        la_y.append(point_y if is_la else None)
        model = str(row["model_key"])
        is_first = model not in seen
        if is_first:
            seen.add(model)
            labels_by_order[int(float(row["model_order"]))] = str(row["model_label"])
        comp_mean_x.append(float(row["mean_value"]) if is_first and not is_la else None)
        comp_mean_y.append(float(row["summary_y"]) if is_first and not is_la else None)
        la_mean_x.append(float(row["mean_value"]) if is_first and is_la else None)
        la_mean_y.append(float(row["summary_y"]) if is_first and is_la else None)

    comp_cols = append_xy(table, f"{panel}_comparison_points", comp_x, comp_y)
    la_cols = append_xy(table, f"{panel}_la_points", la_x, la_y)
    comp_mean_cols = append_xy(table, f"{panel}_comparison_mean", comp_mean_x, comp_mean_y)
    la_mean_cols = append_xy(table, f"{panel}_la_mean", la_mean_x, la_mean_y)
    add_scatter(
        layer,
        table,
        *comp_cols,
        color=SLATE,
        kind=2,
        size=3.5 * font_marker_scale,
        transparency=25,
    )
    add_scatter(
        layer,
        table,
        *la_cols,
        color=ORANGE,
        kind=2,
        size=4.0 * font_marker_scale,
        transparency=12,
    )

    summary_rows = [row for row in table.rows if str(row["model_key"]) in seen]
    ci_seen: set[str] = set()
    for row in summary_rows:
        model = str(row["model_key"])
        if model in ci_seen:
            continue
        ci_seen.add(model)
        color = ORANGE if str(row["series_key"]) == "literatureagent" else INK
        ci_segment(
            layer,
            float(row["ci_low"]),
            float(row["ci_high"]),
            float(row["summary_y"]),
            color,
            0.12,
            font_marker_scale,
        )

    add_scatter(layer, table, *comp_mean_cols, color=INK, kind=1, size=7.5 * font_marker_scale)
    add_scatter(layer, table, *la_mean_cols, color=ORANGE, kind=1, size=8.0 * font_marker_scale)
    legend_mean_plot_index = 3
    if add_legend:
        # A legend-only, off-axis line+symbol plot makes the key faithfully
        # reproduce the plotted mean marker and its horizontal interval bar.
        # The scientific CI segments themselves remain the original editable
        # Origin line objects at their validated positions.
        legend_cols = append_xy(
            table,
            f"{panel}_mean_ci_legend_key",
            [xlim[0] - 200.0, xlim[0] - 190.0],
            [ylim[1] + 200.0, ylim[1] + 200.0],
        )
        legend_plot = layer.add_plot(
            table.sheet,
            coly=legend_cols[1],
            colx=legend_cols[0],
            type="y",
        )
        if legend_plot is None:
            raise RuntimeError("Origin could not create the Figure 3 CI legend key")
        style_scatter(
            legend_plot,
            color=INK,
            kind=1,
            size=7.5 * font_marker_scale,
            interior=1,
        )
        legend_plot.set_float("line.width", 1.45 * font_marker_scale)
        legend_mean_plot_index = len(layer.plot_list())
    style_axis(
        layer,
        x_title,
        xlim=xlim,
        xstep=xstep,
        ylim=ylim,
        show_y_labels=show_y_labels,
        axis_title_size=axis_title_size,
        tick_size=tick_size,
    )
    labels = [labels_by_order[index] for index in range(8)]
    set_categorical_ticks(layer, table, labels, f"f3{panel}", [float(8 - index) for index in range(8)])
    if show_letter:
        add_percent_label(
            layer,
            f"({panel})",
            1,
            panel_label_y_percent,
            size=panel_label_size,
            bold=True,
        )
    if add_legend:
        legend_text = (
            f"\\l({legend_mean_plot_index}) Mean (95% CI)   \\l(1) Other systems\r\n\\l(2) LiteratureAgent"
            if shared_legend
            else f"\\l(1) Other systems   \\l(2) LiteratureAgent\r\n\\l({legend_mean_plot_index}) Mean (95% CI)"
        )
        set_legend(
            layer,
            legend_text,
            legend_x_percent,
            legend_y_percent,
            legend_size,
            boxed=not shared_legend,
        )


def build_figure3() -> None:
    for panel in "abcd":
        graph = new_graph(f"F3{panel.upper()}_Ind", f"Figure 3{panel} individual", 6.5, 4.85)
        position_layer(graph[0], 42, 4, 55, 75)
        build_figure3_panel(
            graph[0],
            panel,
            show_y_labels=True,
            add_legend=False,
            font_marker_scale=2.05,
            show_letter=False,
            tight_vertical_range=True,
        )
        export_graph(graph, f"Figure3{panel}_individual", INDIVIDUAL, 6.5)

    graph = new_graph("F3_All", "Figure 3 manuscript composite", 6.5, 5.4)
    layers = [graph[0], graph.add_layer(0), graph.add_layer(0), graph.add_layer(0)]
    positions = [(22, 7, 29, 38), (63, 7, 34, 38), (22, 55, 29, 30), (63, 55, 34, 30)]
    for layer, position in zip(layers, positions):
        position_layer(layer, *position)
    for index, panel in enumerate("abcd"):
        build_figure3_panel(
            layers[index],
            panel,
            show_y_labels=panel in "ac",
            add_legend=(panel == "a"),
            font_marker_scale=2.3,
            # Keep the shared legend visually separate from panel (b)'s
            # left axis and panel labels at final manuscript size.
            legend_x_percent=8.0,
            legend_y_percent=2.0,
            shared_legend=True,
            panel_label_y_percent=10.0 if panel in "ab" else 1.0,
        )
    export_graph(graph, "Figure3_literature_system_benchmark", MULTIPANEL, 6.5)


F4_CONFIG = {
    "a": ("F4A", "R²", (0.20, 0.42), 0.05),
    "b": ("F4B", "RMSE (PCE points)", (2.90, 3.65), 0.20),
    "c": ("F4C", "MAE (PCE points)", (2.30, 3.00), 0.20),
    "d": ("F4D", "R²", (-2.10, 0.70), 0.50),
    "e": ("F4E", "RMSE (PCE points)", (3.20, 8.20), 1.00),
    "f": ("F4F", "MAE (PCE points)", (2.80, 7.20), 1.00),
}

F4_STYLE = {
    "original": (SLATE, 1),
    "curated": (ORANGE_2, 2),
    "full": (ORANGE, 3),
    "accepted_expansion": (ACCEPTED, 2),
}


def build_figure4_panel(
    layer: Any,
    panel: str,
    *,
    show_y_labels: bool,
    add_legend: bool = False,
    scale: float = 1.0,
    show_letter: bool = True,
    tight_vertical_range: bool = False,
) -> None:
    key, x_title, xlim, xstep = F4_CONFIG[panel]
    table = TABLES[key]
    n = len(table.rows)
    ylim = (0.35, n + 0.55) if tight_vertical_range else (0.5, n + 1.6)
    value_label_size = max(1, int(round(9 * scale)))
    axis_title_size = max(1, int(round(10 * scale)))
    tick_size = max(1, int(round(9 * scale)))
    panel_label_size = max(1, int(round(11 * scale)))
    legend_size = max(1, int(round(9 * scale)))
    marker_size = 7.5 * scale

    if panel == "d":
        # This panel crosses R² = 0; retain one quiet scientific reference.
        vertical_rule(layer, 0, ylim[0], ylim[1], color=GRID, width=0.7)

    plot_indices: dict[str, int] = {}
    for row_index, row in enumerate(table.rows):
        style_key = str(row["series_key"])
        color, kind = F4_STYLE[style_key]
        x_values = [None] * n
        y_values = [None] * n
        x_values[row_index] = float(row["estimate"])
        y_values[row_index] = float(row["y_position"])
        cols = append_xy(table, f"{panel}_{style_key}_{row_index}", x_values, y_values)
        plot = add_scatter(layer, table, *cols, color=color, kind=kind, size=marker_size)
        plot_indices[style_key] = len(layer.plot_list())
        ci_segment(
            layer,
            float(row["ci_low"]),
            float(row["ci_high"]),
            float(row["y_position"]),
            color,
            0.10,
            scale,
        )
        value = float(row["estimate"])
        label_x = value + 0.025 * scale * (xlim[1] - xlim[0])
        if label_x > xlim[1] - 0.04 * scale * (xlim[1] - xlim[0]):
            label_x = value - 0.18 * scale * (xlim[1] - xlim[0])
        add_data_label(
            layer,
            f"{value:.3f}",
            label_x,
            float(row["y_position"]) - 0.15,
            xlim=xlim,
            ylim=ylim,
            size=value_label_size,
            bold=False,
        )

    style_axis(
        layer,
        x_title,
        xlim=xlim,
        xstep=xstep,
        ylim=ylim,
        show_y_labels=show_y_labels,
        axis_title_size=axis_title_size,
        tick_size=tick_size,
    )
    condition_labels = [str(row["condition_label"]) for row in table.rows]
    if panel in "def":
        condition_labels = [
            label.replace("+ accepted expansion", "+ accepted\nexpansion")
            for label in condition_labels
        ]
    set_categorical_ticks(
        layer,
        table,
        condition_labels,
        f"f4{panel}",
        [float(row["y_position"]) for row in table.rows],
    )
    if show_letter:
        add_percent_label(layer, f"({panel})", 1, 1, size=panel_label_size, bold=True)
    if add_legend:
        if panel in "abc":
            legend_text = "\\l(1) Original   \\l(2) Curated   \\l(3) Full"
            if scale > 1:
                legend_text = "\\l(1) Original   \\l(2) Curated\r\n\\l(3) Full"
            set_legend(layer, legend_text, 14, 2, legend_size)
        else:
            set_legend(
                layer,
                "\\l(1) Original only\r\n\\l(2) + accepted expansion",
                14,
                2,
                legend_size,
            )


def build_figure4() -> None:
    for panel in "abcdef":
        graph = new_graph(f"F4{panel.upper()}_Ind", f"Figure 4{panel} individual", 6.5, 3.52)
        if panel in "abc":
            position_layer(graph[0], 28, 3, 69, 64)
        else:
            position_layer(graph[0], 34, 3, 63, 64)
        build_figure4_panel(
            graph[0],
            panel,
            show_y_labels=True,
            add_legend=False,
            scale=2.40,
            show_letter=False,
            tight_vertical_range=True,
        )
        export_graph(graph, f"Figure4{panel}_individual", INDIVIDUAL, 6.5)

    graph = new_graph("F4_All", "Figure 4 manuscript composite", 6.5, 6.4)
    layers = [graph[0]] + [graph.add_layer(0) for _ in range(5)]
    positions = [
        (15, 4, 31, 24),
        (66, 4, 31, 24),
        (15, 36, 31, 24),
        (66, 36, 31, 24),
        (15, 68, 31, 20),
        (66, 68, 31, 20),
    ]
    for layer, position in zip(layers, positions):
        position_layer(layer, *position)
    panel_order = ("a", "d", "b", "e", "c", "f")
    for layer, panel in zip(layers, panel_order):
        build_figure4_panel(
            layer,
            panel,
            show_y_labels=True,
            add_legend=panel in "ad",
            scale=2.8,
        )
    export_graph(graph, "Figure4_PCE_domain_utility", MULTIPANEL, 6.5)


def build_figure5_panel(
    layer: Any,
    *,
    add_legend: bool = True,
    font_scale: float = 1.0,
) -> None:
    negative_points = TABLES["F5NEG"]
    positive_points = TABLES["F5POS"]
    summary = TABLES["F5SUM"]
    y_values = [float(row["y_position"]) for row in summary.rows]
    ylim = (min(y_values) - 0.8, max(y_values) + 0.8)
    xlim = (-1.8, 3.05)
    # Retain only the scientifically meaningful zero reference; decorative
    # background gridlines are intentionally omitted throughout the package.
    vertical_rule(layer, 0, ylim[0], ylim[1], color=INK, width=0.9)

    negative = add_scatter(
        layer,
        negative_points,
        negative_points.columns["shap_value"],
        negative_points.columns["display_y"],
        color=SHAP_NEGATIVE,
        kind=2,
        size=2.3,
        interior=0,
        transparency=28,
        fill_color=SHAP_NEGATIVE,
    )
    positive = add_scatter(
        layer,
        positive_points,
        positive_points.columns["shap_value"],
        positive_points.columns["display_y"],
        color=SHAP_POSITIVE,
        kind=2,
        size=2.3,
        interior=0,
        transparency=28,
        fill_color=SHAP_POSITIVE,
    )
    for row in summary.rows:
        ci_segment(
            layer,
            float(row["q25"]),
            float(row["q75"]),
            float(row["y_position"]),
            INK,
            0.10,
            font_scale,
        )
    median_cols = append_xy(
        summary,
        "median",
        [float(row["median"]) for row in summary.rows],
        [float(row["y_position"]) for row in summary.rows],
    )
    add_scatter(layer, summary, *median_cols, color=INK, kind=5, size=8.0 * font_scale)
    layer.lt_exec("layer.maxpts=0; page.speedmode=0;")
    style_axis(
        layer,
        "Choice-bundle TreeSHAP contribution to fitted PCE (percentage points)",
        xlim=xlim,
        xstep=1,
        ylim=ylim,
        show_y_labels=True,
        axis_title_size=max(1, int(round(10 * font_scale))),
        tick_size=max(1, int(round(9 * font_scale))),
    )
    labels: list[str] = []
    seen_families: set[str] = set()
    for row in summary.rows:
        family = str(row["family_label"])
        state = str(row["state_label"])
        if state == "compact/mesoporous TiO₂; spray pyrolysis | spin coating":
            state = "c-TiO₂ / m-TiO₂; spray / spin coated"
        if family not in seen_families:
            labels.append(f"{family.upper()} — {state}")
            seen_families.add(family)
        else:
            labels.append(state)
    set_categorical_ticks(layer, summary, labels, "f5", y_values)
    if add_legend:
        set_legend(
            layer,
            "\\l(1) Lowers prediction   \\l(2) Raises prediction\r\n\\l(3) Median (IQR)",
            2,
            2,
            max(1, int(round(9 * font_scale))),
        )


def build_figure5() -> None:
    graph = new_graph("F5_Main", "Figure 5 SHAP interpretation", 6.5, 6.3)
    position_layer(graph[0], 46, 3, 51, 84)
    build_figure5_panel(graph[0], add_legend=False, font_scale=1.2)
    export_figure5_hybrid(
        graph,
        "Figure5_choice_stratified_SHAP",
        INDIVIDUAL,
        6.5,
        include_legend=False,
    )
    duplicate_export(graph, "Figure5_choice_stratified_SHAP", INDIVIDUAL, MULTIPANEL)


F6_CONFIG = {
    "a": ("F6A", "ROC–AUC", (0.49, 0.70), 0.05, "Classification"),
    "b": ("F6B", "R²", (-0.07, 0.18), 0.05, "Regression"),
}


def build_figure6_panel(
    layer: Any,
    panel: str,
    *,
    add_legend: bool = False,
    scale: float = 1.0,
    show_letter: bool = True,
    tight_vertical_range: bool = False,
    show_heading: bool = True,
) -> None:
    key, x_title, xlim, xstep, heading = F6_CONFIG[panel]
    table = TABLES[key]
    targets: list[str] = []
    labels: dict[str, str] = {}
    for row in table.rows:
        target = str(row["target"])
        if target not in targets:
            targets.append(target)
            label = str(row["target_label"])
            # The two longest regression labels are wrapped so that 18--20 pt
            # source text stays readable when the panel is reduced to a
            # half-width PowerPoint slot without creating a vast left gutter.
            if label.startswith("Condition-normalized "):
                label = label.replace("Condition-normalized ", "Condition-normalized\n", 1)
            labels[target] = label
    if tight_vertical_range:
        ylim = (0.4, len(targets) + (0.8 if panel == "b" else 0.7))
    else:
        ylim = (0.5, len(targets) + 1.6)
    if panel == "b":
        # R² = 0 is a scientific reference threshold, not a decorative grid.
        # Use a regular data plot instead of a foreground drawing object so
        # its upper endpoint is honored and it stays behind the heading.
        rule_top = ylim[1] - (0.48 if show_heading else 0.0)
        rule_cols = append_xy(
            table,
            "f6b_zero_reference",
            [0.0, 0.0],
            [ylim[0], rule_top],
        )
        rule_plot = layer.add_plot(table.sheet, coly=rule_cols[1], colx=rule_cols[0], type="l")
        if rule_plot is None:
            raise RuntimeError("Origin could not add the Figure 6b zero-reference line")
        rule_plot.color = GRID
        rule_plot.set_float("line.width", 0.7)

    baseline_x: list[Any] = []
    baseline_y: list[Any] = []
    integrated_x: list[Any] = []
    integrated_y: list[Any] = []
    for row in table.rows:
        dataset = str(row["dataset"])
        baseline_x.append(float(row["estimate"]) if dataset == "baseline" else None)
        baseline_y.append(float(row["y_position"]) if dataset == "baseline" else None)
        integrated_x.append(float(row["estimate"]) if dataset == "integrated" else None)
        integrated_y.append(float(row["y_position"]) if dataset == "integrated" else None)
        color = SLATE if dataset == "baseline" else ORANGE
        ci_segment(
            layer,
            float(row["ci_low"]),
            float(row["ci_high"]),
            float(row["y_position"]),
            color,
            0.09,
            scale,
        )
    baseline_cols = append_xy(table, f"f6{panel}_baseline", baseline_x, baseline_y)
    integrated_cols = append_xy(table, f"f6{panel}_integrated", integrated_x, integrated_y)
    add_scatter(layer, table, *baseline_cols, color=SLATE, kind=1, size=7.5 * scale)
    add_scatter(layer, table, *integrated_cols, color=ORANGE, kind=2, size=8.0 * scale)
    style_axis(
        layer,
        x_title,
        xlim=xlim,
        xstep=xstep,
        ylim=ylim,
        show_y_labels=True,
        axis_title_size=max(1, int(round(10 * scale))),
        tick_size=max(1, int(round(9 * scale))),
    )
    set_categorical_ticks(
        layer,
        table,
        [labels[target] for target in targets],
        f"f6{panel}",
        [float(len(targets) - index) for index in range(len(targets))],
    )
    if show_heading:
        heading_y = 0 if panel == "b" else 1
        heading_label = add_percent_label(
            layer,
            f"({panel}) {heading}" if show_letter else heading,
            1,
            heading_y,
            size=max(1, int(round(11 * scale))),
            bold=True,
            whiteout=True,
        )
        heading_name = str(heading_label)
        layer.lt_exec(
            f'{heading_name}.background=1; {heading_name}.bordercolor=color("{WHITE}"); '
            f'{heading_name}.fillcolor=color("{WHITE}"); {heading_name}.margin.unit=1; '
            f'{heading_name}.margin.left=4; {heading_name}.margin.right=4; '
            f'{heading_name}.margin.top=2; {heading_name}.margin.bottom=2;'
        )
    if add_legend:
        set_legend(
            layer,
            "\\l(1) Original   \\l(2) Integrated",
            40,
            2,
            max(1, int(round(9 * scale))),
        )


def build_figure6() -> None:
    for panel in "ab":
        graph = new_graph(f"F6{panel.upper()}_Ind", f"Figure 6{panel} individual", 6.5, 5.25)
        if panel == "a":
            position_layer(graph[0], 42, 3, 54, 80)
        else:
            position_layer(graph[0], 48, 3, 49, 80)
        build_figure6_panel(
            graph[0],
            panel,
            add_legend=False,
            scale=2.05,
            show_letter=False,
            tight_vertical_range=True,
            show_heading=True,
        )
        export_graph(graph, f"Figure6{panel}_individual", INDIVIDUAL, 6.5)

    graph = new_graph("F6_All", "Figure 6 manuscript composite", 6.5, 5.5)
    top = graph[0]
    bottom = graph.add_layer(0)
    position_layer(top, 34, 7, 62, 34)
    position_layer(bottom, 34, 54, 62, 32)
    build_figure6_panel(top, "a", add_legend=True, scale=1.6)
    build_figure6_panel(bottom, "b", add_legend=False, scale=1.6)
    export_graph(graph, "Figure6_stability_model_performance", MULTIPANEL, 6.5)


def new_legend_graph(short_name: str, long_name: str, width_in: float) -> tuple[Any, Any]:
    """Create a compact, axis-free Origin page for a shared PowerPoint legend."""
    graph = new_graph(short_name, long_name, width_in, 0.36)
    layer = graph[0]
    position_layer(layer, 1, 1, 98, 98)
    layer.axis("x").set_limits(0, 1, 1)
    layer.axis("y").set_limits(0, 1, 1)
    layer.lt_exec(
        "layer.x.showAxes=0; layer.y.showAxes=0; "
        "layer.x.label.show=0; layer.y.label.show=0; "
        "layer.x.majorTicks=0; layer.y.majorTicks=0; layer.sauto=0;"
    )
    return graph, layer


def add_legend_key_plot(
    layer: Any,
    table: Table,
    token: str,
    *,
    color: str,
    kind: int,
    size: float,
    line: bool = False,
    interior: int = 1,
    fill_color: str | None = None,
) -> Any:
    """Add an off-axis dummy plot whose style is rendered by Origin's legend."""
    columns = append_xy(table, token, [-20.0, -19.0], [-20.0, -20.0])
    if line:
        plot = layer.add_plot(table.sheet, coly=columns[1], colx=columns[0], type="y")
        if plot is None:
            raise RuntimeError(f"Origin could not create legend key {token}")
        style_scatter(
            plot,
            color=color,
            kind=kind,
            size=size,
            interior=interior,
            fill_color=fill_color,
        )
        plot.set_float("line.width", 1.45)
        return plot
    return add_scatter(
        layer,
        table,
        *columns,
        color=color,
        kind=kind,
        size=size,
        interior=interior,
        fill_color=fill_color,
    )


def build_shared_legends(selected: set[str]) -> None:
    """Export one compact legend PNG for each distinct manuscript encoding."""
    if "figure3" in selected:
        graph, layer = new_legend_graph("L3_Shared", "Figure 3 shared legend", 6.5)
        table = TABLES["F3A"]
        add_legend_key_plot(layer, table, "l3_other", color=SLATE, kind=2, size=4.0)
        add_legend_key_plot(layer, table, "l3_la", color=ORANGE, kind=2, size=4.2)
        add_legend_key_plot(
            layer,
            table,
            "l3_mean_ci",
            color=INK,
            kind=1,
            size=7.5,
            line=True,
            interior=1,
        )
        set_legend(
            layer,
            "\\l(1) Other systems    \\l(2) LiteratureAgent (Qwen)    \\l(3) Mean (95% CI)",
            2,
            18,
            11,
            boxed=False,
        )
        export_graph(graph, "Legend_Figure3_system_benchmark", LEGENDS, 6.5)

    if "figure4" in selected:
        table = TABLES["F4A"]
        graph, layer = new_legend_graph("L4_Hist", "Figure 4 historical-model legend", 3.8)
        add_legend_key_plot(layer, table, "l4_original", color=SLATE, kind=1, size=7.5)
        add_legend_key_plot(layer, table, "l4_curated", color=ORANGE_2, kind=2, size=7.5)
        add_legend_key_plot(layer, table, "l4_full", color=ORANGE, kind=3, size=7.5)
        set_legend(
            layer,
            "\\l(1) Original    \\l(2) Curated    \\l(3) Full",
            2,
            18,
            11,
            boxed=False,
        )
        export_graph(graph, "Legend_Figure4_historical_models", LEGENDS, 3.8)

        table = TABLES["F4D"]
        graph, layer = new_legend_graph("L4_Expand", "Figure 4 accepted-expansion legend", 4.3)
        add_legend_key_plot(layer, table, "l4_original_only", color=SLATE, kind=1, size=7.5)
        add_legend_key_plot(layer, table, "l4_accepted_expansion", color=ACCEPTED, kind=2, size=7.5)
        set_legend(
            layer,
            "\\l(1) Original only    \\l(2) + accepted expansion",
            2,
            18,
            11,
            boxed=False,
        )
        export_graph(graph, "Legend_Figure4_accepted_expansion", LEGENDS, 4.3)

    if "figure5" in selected:
        graph, layer = new_legend_graph("L5_Shared", "Figure 5 SHAP legend", 6.4)
        table = TABLES["F5SUM"]
        add_legend_key_plot(
            layer,
            table,
            "l5_lower",
            color=SHAP_NEGATIVE,
            kind=2,
            size=5.0,
            interior=0,
            fill_color=SHAP_NEGATIVE,
        )
        add_legend_key_plot(
            layer,
            table,
            "l5_raise",
            color=SHAP_POSITIVE,
            kind=2,
            size=5.0,
            interior=0,
            fill_color=SHAP_POSITIVE,
        )
        add_legend_key_plot(
            layer,
            table,
            "l5_median_iqr",
            color=INK,
            kind=5,
            size=8.0,
            line=True,
        )
        set_legend(
            layer,
            "\\l(1) Lowers prediction    \\l(2) Raises prediction    \\l(3) Median (IQR)",
            2,
            18,
            11,
            boxed=False,
        )
        export_graph(graph, "Legend_Figure5_SHAP_direction", LEGENDS, 6.4)

    if "figure6" in selected:
        graph, layer = new_legend_graph("L6_Shared", "Figure 6 stability-model legend", 3.5)
        table = TABLES["F6A"]
        add_legend_key_plot(layer, table, "l6_original", color=SLATE, kind=1, size=7.5)
        add_legend_key_plot(layer, table, "l6_integrated", color=ORANGE, kind=2, size=8.0)
        set_legend(
            layer,
            "\\l(1) Original    \\l(2) Integrated",
            2,
            18,
            11,
            boxed=False,
        )
        export_graph(graph, "Legend_Figure6_stability_models", LEGENDS, 3.5)


def arrow_segment(
    layer: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    arrow: bool = True,
    dashed: bool = False,
    color: str = SLATE,
    width: float = 0.8,
) -> None:
    line = add_line_object(layer, x1, y1, x2, y2)
    if line is None:
        return
    line.width = width
    line.color = color
    line.type = 2 if dashed else 1
    if arrow:
        line.set_int("arrowendshape", 2)
        line.set_float("arrowendlength", 6)
        line.set_float("arrowendwidth", 4)


def build_figure7_panel(layer: Any) -> None:
    nodes = TABLES["F7N"]
    xlim = (0.0, 100.0)
    ylim = (0.0, 100.0)
    layer.axis("x").set_limits(*xlim, 20)
    layer.axis("y").set_limits(*ylim, 20)
    layer.axis("x").title = ""
    layer.axis("y").title = ""
    layer.lt_exec(
        "layer.x.showAxes=0; layer.y.showAxes=0; layer.x.label.show=0; layer.y.label.show=0; "
        "layer.x.majorTicks=0; layer.y.majorTicks=0; layer.sauto=0;"
    )

    # Direct connectors terminate just outside the boxed text nodes.  Nodes are
    # drawn last so their opaque fills cleanly mask the connector endpoints.
    arrow_segment(layer, 12, 69, 12, 81)
    arrow_segment(layer, 18.5, 88, 28.5, 88)
    arrow_segment(layer, 19.5, 62, 27.5, 62)
    arrow_segment(layer, 42.5, 62, 52.5, 62)
    arrow_segment(layer, 67.5, 62, 77.5, 62)
    arrow_segment(layer, 19.5, 58, 23.5, 58, arrow=False)
    arrow_segment(layer, 23.5, 58, 23.5, 27, arrow=False)
    arrow_segment(layer, 23.5, 27, 27.5, 27)
    arrow_segment(layer, 42.5, 27, 52.5, 27)
    arrow_segment(layer, 64.3, 56, 80.7, 33)

    # Claim support is routed below the claim/entity row, avoiding the entity node.
    arrow_segment(layer, 35, 21, 35, 15, arrow=False)
    arrow_segment(layer, 35, 15, 85, 15, arrow=False)
    arrow_segment(layer, 85, 15, 85, 21)

    # Field-evidence routes are thin and dashed because they are secondary paths.
    arrow_segment(layer, 40, 68, 47, 68, arrow=False, dashed=True, width=0.65)
    arrow_segment(layer, 47, 68, 47, 75, arrow=False, dashed=True, width=0.65)
    arrow_segment(layer, 47, 75, 96, 75, arrow=False, dashed=True, width=0.65)
    arrow_segment(layer, 96, 75, 96, 27, arrow=False, dashed=True, width=0.65)
    arrow_segment(layer, 96, 27, 92.5, 27, dashed=True, width=0.65)
    arrow_segment(layer, 12, 56, 12, 9, arrow=False, dashed=True, width=0.65)
    arrow_segment(layer, 12, 9, 94, 9, arrow=False, dashed=True, width=0.65)
    arrow_segment(layer, 94, 9, 94, 24, arrow=False, dashed=True, width=0.65)
    arrow_segment(layer, 94, 24, 92.5, 24, dashed=True, width=0.65)

    # Evidence derives from the publication/source identity.
    arrow_segment(layer, 85, 21, 85, 3, arrow=False)
    arrow_segment(layer, 85, 3, 1.5, 3, arrow=False)
    arrow_segment(layer, 1.5, 3, 1.5, 62, arrow=False)
    arrow_segment(layer, 1.5, 62, 4.5, 62)

    # Type-level entity-to-entity directional relation; this is not a literal
    # self-edge on one entity instance and does not imply causality.
    arrow_segment(layer, 57, 33, 57, 40, arrow=False)
    arrow_segment(layer, 57, 40, 48, 40, arrow=False)
    arrow_segment(layer, 48, 40, 48, 31, arrow=False)
    arrow_segment(layer, 48, 31, 52.5, 31)

    edge_labels = [
        (22, 75, "in snapshot"),
        (23.5, 84, "generated\nby"),
        (23.5, 65, "has\ndevice"),
        (47.5, 65, "has\nobservation"),
        (72.5, 65, "measures\nproperty"),
        (22, 43, "reports\nclaim"),
        (47.5, 23, "has subject /\nobject"),
        (73, 53, "supported by:\n30,974"),
        (72, 17.5, "supported by: 2,694"),
        (72, 78, "field evidence"),
        (41, 11.5, "field evidence"),
        (60, 5.5, "derived from"),
        (52.5, 44.5, "directional relation"),
    ]
    for x, y, text in edge_labels:
        add_data_label(
            layer,
            text,
            x,
            y,
            xlim=xlim,
            ylim=ylim,
            size=7,
            color=SLATE,
            center=True,
            whiteout=True,
        )

    node_styles = {
        "orange_strong": (ORANGE, "#FFF4E5", INK),
        "slate_pale": (SLATE, PALE_SLATE, INK),
        "orange_1": (ORANGE_1, "#FFF4E5", INK),
        "orange_2": (ORANGE_3, "#FFF7EC", INK),
        "orange_3": (ORANGE_4, "#FFF8ED", INK),
        "orange_4": (ORANGE_5, "#FFF9F1", INK),
        "slate": (SLATE, SLATE, WHITE),
    }
    node_positions = {
        "Paper": (12.0, 62.0),
        "DatasetSnapshot": (12.0, 88.0),
        "ExtractionRun": (35.0, 88.0),
        "Device": (35.0, 62.0),
        "StructuredObservation": (60.0, 62.0),
        "OntologyConcept": (85.0, 62.0),
        "ScientificClaim": (35.0, 27.0),
        "ScientificEntity": (60.0, 27.0),
        "Evidence": (85.0, 27.0),
    }
    node_labels = {
        "Paper": "Publication /\nsource\n(n=1,396)",
        "DatasetSnapshot": "Dataset\nsnapshot\n(n=1)",
        "ExtractionRun": "Extraction\nrun\n(n=1)",
        "Device": "Device /\nsample\n(n=1,843)",
        "StructuredObservation": "Structured\nobservation\n(n=39,844)",
        "OntologyConcept": "Ontology\nconcept\n(n=266)",
        "ScientificClaim": "Scientific\nclaim\n(n=2,694)",
        "ScientificEntity": "Scientific\nentity\n(n=5,208)",
        "Evidence": "Evidence\n(n=39,770)",
    }
    for row in nodes.rows:
        node_type = str(row["node_type"])
        border, fill, text_color = node_styles[str(row["style_key"])]
        node_x, node_y = node_positions[node_type]
        add_boxed_data_label(
            layer,
            node_labels[node_type],
            node_x,
            node_y,
            xlim=xlim,
            ylim=ylim,
            border=border,
            fill=fill,
            text_color=text_color,
            size=8,
        )
    add_percent_label(layer, "Provenance", 1, 1, size=9, bold=True, color=SLATE)
    add_percent_label(layer, "Structured records", 29, 22, size=8, bold=True, color=SLATE)
    add_percent_label(layer, "Claims and evidence", 29, 49, size=8, bold=True, color=SLATE)


def build_figure7() -> None:
    graph = new_graph("F7_Main", "Figure 7 knowledge graph structure", 6.5, 5.0)
    position_layer(graph[0], 3, 4, 94, 92)
    build_figure7_panel(graph[0])
    export_graph(graph, "Figure7_knowledge_graph_structure", INDIVIDUAL, 6.5)
    duplicate_export(graph, "Figure7_knowledge_graph_structure", INDIVIDUAL, MULTIPANEL)


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def write_figure5_hybrid_svg(
    origin_png: Path,
    svg: Path,
    width_in: float,
    *,
    include_legend: bool = True,
) -> tuple[int, int]:
    """Embed the full-resolution Origin raster and add crisp vector text.

    Origin 2024's SVG exporter does not complete for the 79,640-point SHAP
    cloud.  The editable Origin graph retains every point; the publication SVG
    uses Origin's 600-dpi, speed-mode-off raster for the dense layer and keeps
    the affected axis-title region as live vector text.  The legend overlay is
    optional because PowerPoint assembly uses a separate shared legend asset.
    """
    width_px, height_px = png_dimensions(origin_png)
    image_data = base64.b64encode(origin_png.read_bytes()).decode("ascii")
    height_in = width_in * height_px / width_px
    card_x = width_px * 0.485
    card_y = height_px * 0.048
    card_w = width_px * 0.400
    card_h = height_px * 0.098
    first_row_y = height_px * 0.078
    second_row_y = height_px * 0.122
    left_symbol_x = width_px * 0.510
    right_symbol_x = width_px * 0.705
    second_symbol_x = width_px * 0.610
    symbol_text_gap = 42.0
    interval_text_gap = 72.0
    axis_x = width_px * 0.715
    axis_y_1 = height_px * 0.952
    axis_y_2 = height_px * 0.985
    legend_overlay = ""
    if include_legend:
        legend_overlay = f'''
  <rect x="{width_px * 0.470:.1f}" y="{height_px * 0.032:.1f}" width="{width_px * 0.430:.1f}" height="{height_px * 0.125:.1f}" fill="#FFFFFF"/>
  <rect x="{card_x:.1f}" y="{card_y:.1f}" width="{card_w:.1f}" height="{card_h:.1f}" rx="28" ry="28" fill="#FFFFFF" stroke="{INK}" stroke-width="5"/>
  <g font-family="Times New Roman, Times, serif" font-size="91.667" fill="{INK}">
    <circle cx="{left_symbol_x:.1f}" cy="{first_row_y:.1f}" r="14" fill="{SHAP_NEGATIVE}" stroke="{SHAP_NEGATIVE}" stroke-width="4" opacity="0.88"/>
    <text x="{left_symbol_x + symbol_text_gap:.1f}" y="{first_row_y + 23:.1f}">Lowers prediction</text>
    <circle cx="{right_symbol_x:.1f}" cy="{first_row_y:.1f}" r="14" fill="{SHAP_POSITIVE}" stroke="{SHAP_POSITIVE}" stroke-width="4" opacity="0.88"/>
    <text x="{right_symbol_x + symbol_text_gap:.1f}" y="{first_row_y + 23:.1f}">Raises prediction</text>
    <line x1="{second_symbol_x - 42:.1f}" y1="{second_row_y:.1f}" x2="{second_symbol_x + 42:.1f}" y2="{second_row_y:.1f}" stroke="{INK}" stroke-width="7"/>
    <line x1="{second_symbol_x - 42:.1f}" y1="{second_row_y - 15:.1f}" x2="{second_symbol_x - 42:.1f}" y2="{second_row_y + 15:.1f}" stroke="{INK}" stroke-width="6"/>
    <line x1="{second_symbol_x + 42:.1f}" y1="{second_row_y - 15:.1f}" x2="{second_symbol_x + 42:.1f}" y2="{second_row_y + 15:.1f}" stroke="{INK}" stroke-width="6"/>
    <polygon points="{second_symbol_x:.1f},{second_row_y - 18:.1f} {second_symbol_x + 18:.1f},{second_row_y:.1f} {second_symbol_x:.1f},{second_row_y + 18:.1f} {second_symbol_x - 18:.1f},{second_row_y:.1f}" fill="#FFFFFF" stroke="{INK}" stroke-width="6"/>
    <text x="{second_symbol_x + interval_text_gap:.1f}" y="{second_row_y + 23:.1f}">Median (IQR)</text>
  </g>'''
    svg_text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width_in:.6f}in" height="{height_in:.6f}in" viewBox="0 0 {width_px} {height_px}">
  <rect width="{width_px}" height="{height_px}" fill="#FFFFFF"/>
  <image x="0" y="0" width="{width_px}" height="{height_px}" href="data:image/png;base64,{image_data}"/>
  <!-- The whiteout covers only Origin 2024's x-axis-title render artifact. -->
  <rect x="0" y="{height_px * 0.928:.1f}" width="{width_px}" height="{height_px * 0.072:.1f}" fill="#FFFFFF"/>
{legend_overlay}
  <g text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="100" fill="{INK}">
    <text x="{axis_x:.1f}" y="{axis_y_1:.1f}">Choice-bundle TreeSHAP contribution to fitted PCE</text>
    <text x="{axis_x:.1f}" y="{axis_y_2:.1f}">(percentage points)</text>
  </g>
</svg>
'''
    svg.parent.mkdir(parents=True, exist_ok=True)
    svg.write_text(svg_text, encoding="utf-8", newline="\n")
    return width_px, height_px


def export_figure5_hybrid(
    graph: Any,
    stem: str,
    directory: Path,
    width_in: float,
    *,
    include_legend: bool = True,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    raw_png = RAW_RASTER_DIR / directory.name / f"{stem}.origin.png"
    raw_png.parent.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    svg = directory / f"{stem}.svg"
    audit_path = SVG_AUDIT_DIR / directory.name / f"{stem}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    graph.activate()
    layer = graph[0]
    layer.lt_exec("page.speedmode=0; layer.maxpts=0; layer -b 3DB 1;")
    op.wait()
    exported_png = graph.save_fig(str(raw_png), type="png", width=int(round(width_in * 600)))
    if not exported_png or not raw_png.is_file() or raw_png.stat().st_size == 0:
        raise RuntimeError(f"Origin failed to export dense Figure 5 raster: {raw_png}")
    source_width, source_height = write_figure5_hybrid_svg(
        raw_png,
        svg,
        width_in,
        include_legend=include_legend,
    )
    rendered_width, rendered_height = render_svg_png(svg, png, int(round(width_in * 600)), None)
    audit = {
        "source": str(raw_png.resolve()),
        "destination": str(svg.resolve()),
        "export_mode": "origin_600dpi_raster_with_vector_text_overlay",
        "reason": "Origin 2024 full-vector SVG export did not complete for 79,640 editable points",
        "origin_speed_mode": "off",
        "dense_point_rows": len(TABLES["F5PTS"].rows),
        "negative_point_rows": len(TABLES["F5NEG"].rows),
        "positive_point_rows": len(TABLES["F5POS"].rows),
        "origin_png_dimensions": [source_width, source_height],
        "origin_png_bytes": raw_png.stat().st_size,
        "origin_png_sha256": hashlib.sha256(raw_png.read_bytes()).hexdigest(),
        "removed_artifact_count": 0,
        "masked_origin_2024_text_artifact_regions": ["x_axis_title"],
        "vector_overlay_objects": (["three-entry legend"] if include_legend else []) + ["x-axis title"],
        "shared_legend_external": not include_legend,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    EXPORTS.append(
        {
            "graph": graph.name,
            "stem": stem,
            "origin_raw_png": str(raw_png),
            "origin_raw_png_bytes": raw_png.stat().st_size,
            "png": str(png),
            "png_bytes": png.stat().st_size,
            "png_dimensions": [rendered_width, rendered_height],
            "svg": str(svg),
            "svg_bytes": svg.stat().st_size,
            "svg_sanitizer_audit": str(audit_path),
            "removed_origin_text_artifacts": 0,
            "export_mode": audit["export_mode"],
        }
    )
    log(f"Exported {graph.name} as 600-dpi Origin raster-backed SVG/PNG: {png.name}, {svg.name}")


def export_graph(graph: Any, stem: str, directory: Path, width_in: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    graph.activate()
    graph.lt_exec("page.speedmode=0;")
    op.wait()
    png = directory / f"{stem}.png"
    svg = directory / f"{stem}.svg"
    raw_svg = RAW_SVG_DIR / directory.name / f"{stem}.origin.svg"
    audit_path = SVG_AUDIT_DIR / directory.name / f"{stem}.json"
    raw_svg.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    exported_svg = graph.save_fig(str(raw_svg), type="svg", ratio=100)
    if not exported_svg:
        raise RuntimeError(f"Origin failed to export {raw_svg}")
    if not raw_svg.is_file() or raw_svg.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty Origin SVG export: {raw_svg}")
    audit = sanitize_svg(raw_svg, svg, audit_path, width_in)
    rendered_width, rendered_height = render_svg_png(svg, png, int(round(width_in * 600)), None)
    for path in (png, svg, audit_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty corrected export: {path}")
    EXPORTS.append(
        {
            "graph": graph.name,
            "stem": stem,
            "origin_raw_svg": str(raw_svg),
            "origin_raw_svg_bytes": raw_svg.stat().st_size,
            "png": str(png),
            "png_bytes": png.stat().st_size,
            "png_dimensions": [rendered_width, rendered_height],
            "svg": str(svg),
            "svg_bytes": svg.stat().st_size,
            "svg_sanitizer_audit": str(audit_path),
            "removed_origin_text_artifacts": audit["removed_artifact_count"],
        }
    )
    log(
        f"Exported {graph.name}: {png.name}, {svg.name}; "
        f"removed {audit['removed_artifact_count']} Origin text-render paths"
    )


def duplicate_export(graph: Any, stem: str, source_directory: Path, destination_directory: Path) -> None:
    """Copy an identical single-panel export into the multipanel deliverable set."""
    destination_directory.mkdir(parents=True, exist_ok=True)
    source_record = next(
        record
        for record in reversed(EXPORTS)
        if record["stem"] == stem and Path(record["png"]).parent == source_directory
    )
    source_png = source_directory / f"{stem}.png"
    source_svg = source_directory / f"{stem}.svg"
    source_audit = SVG_AUDIT_DIR / source_directory.name / f"{stem}.json"
    destination_png = destination_directory / source_png.name
    destination_svg = destination_directory / source_svg.name
    if "origin_raw_svg" in source_record:
        raw_key, raw_bytes_key = "origin_raw_svg", "origin_raw_svg_bytes"
        source_raw = Path(source_record[raw_key])
        destination_raw = RAW_SVG_DIR / destination_directory.name / source_raw.name
    else:
        raw_key, raw_bytes_key = "origin_raw_png", "origin_raw_png_bytes"
        source_raw = Path(source_record[raw_key])
        destination_raw = RAW_RASTER_DIR / destination_directory.name / source_raw.name
    destination_audit = SVG_AUDIT_DIR / destination_directory.name / source_audit.name
    destination_raw.parent.mkdir(parents=True, exist_ok=True)
    destination_audit.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (source_png, destination_png),
        (source_svg, destination_svg),
        (source_raw, destination_raw),
    ):
        if not source.is_file():
            raise RuntimeError(f"Cannot duplicate missing export: {source}")
        shutil.copy2(source, destination)
    audit = json.loads(source_audit.read_text(encoding="utf-8"))
    audit["source"] = str(destination_raw.resolve())
    audit["destination"] = str(destination_svg.resolve())
    audit["copied_from_audit"] = str(source_audit.resolve())
    destination_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    record = {
        "graph": graph.name,
        "stem": stem,
        raw_key: str(destination_raw),
        raw_bytes_key: destination_raw.stat().st_size,
        "png": str(destination_png),
        "png_bytes": destination_png.stat().st_size,
        "png_dimensions": source_record["png_dimensions"],
        "svg": str(destination_svg),
        "svg_bytes": destination_svg.stat().st_size,
        "svg_sanitizer_audit": str(destination_audit),
        "removed_origin_text_artifacts": audit["removed_artifact_count"],
        "identical_copy_of": str(source_svg),
    }
    if "export_mode" in source_record:
        record["export_mode"] = source_record["export_mode"]
    EXPORTS.append(record)
    log(f"Copied identical {graph.name} export into {destination_directory.name}")


def build(selected: set[str]) -> None:
    if "figure2" in selected:
        build_figure2()
    if "figure3" in selected:
        build_figure3()
    if "figure4" in selected:
        build_figure4()
    if "figure5" in selected:
        build_figure5()
    if "figure6" in selected:
        build_figure6()
    if "figure7" in selected:
        # Figure 7 is a deterministic Python-generated explanatory diagram.
        # Its validated source worksheets remain in this OPJU, while the
        # publication SVG/PNG are built outside Origin for clean live text,
        # rounded cards, and explicit orthogonal routing.
        figure7_exports = build_knowledge_graph_outputs()
        EXPORTS.extend(figure7_exports)
        log(
            "Generated F7_Python as deterministic Python SVG/PNG in "
            "figures_individual and figures_multipanel"
        )
    build_shared_legends(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["figure2", "figure3", "figure4", "figure5", "figure6", "figure7"],
        help="Build only selected figures; default builds Figures 2-7.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="Optional output .opju path; defaults to the individual-panel revision project.",
    )
    parser.add_argument("--close", action="store_true", help="Close the external Origin session after saving.")
    args = parser.parse_args()
    # Origin's `run -pyf` command can execute this builder inside Origin 2024,
    # but paths containing spaces are not preserved reliably as Python argv.
    # Environment fallbacks keep the same builder usable through that fully
    # supported embedded-Python route without hard-coding output locations.
    env_only = os.environ.get("LITERATURE_ORIGIN_ONLY", "").strip()
    env_project = os.environ.get("LITERATURE_ORIGIN_PROJECT", "").strip()
    env_close = os.environ.get("LITERATURE_ORIGIN_CLOSE", "").strip().lower()
    if args.only is None and env_only:
        selected = [value.strip() for value in env_only.split(",") if value.strip()]
        invalid = sorted(set(selected) - {"figure2", "figure3", "figure4", "figure5", "figure6", "figure7"})
        if invalid:
            parser.error(f"invalid LITERATURE_ORIGIN_ONLY value(s): {', '.join(invalid)}")
        args.only = selected
    if args.project is None and env_project:
        args.project = Path(env_project)
    if not args.close and env_close in {"1", "true", "yes", "on"}:
        args.close = True
    return args


def main() -> None:
    parse_args()
    raise SystemExit("Use scripts/build_manuscript_plot.py, build_fig2_campaign_origin.py, or the Figure S4/S6 builders for the released source tables.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write("FAILED\n")
            handle.write(traceback.format_exc())
        if getattr(op, "oext", False):
            try:
                op.exit()
            except Exception:
                pass
        raise
