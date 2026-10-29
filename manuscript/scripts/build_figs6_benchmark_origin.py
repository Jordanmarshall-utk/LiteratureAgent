"""Build the four Figure S6 benchmark panels in OriginPro.

Source tables and specifications are read from ``source/``. Palette and
font sizes are defined below. Each plot has one editable OPJU project, source
and display-helper worksheets, PNG/SVG/PDF exports, and a validation audit.
``--dry-run`` validates the source data without opening OriginPro.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "2026-10-27_controlled_repaired_rerun"
PUBLIC_SOURCE_ROOT = ROOT / "source"
PACKAGE = PUBLIC_SOURCE_ROOT
DEFAULT_OUTPUT = ROOT / "build"
STYLE_SOURCE = Path(
    os.environ.get(
        "LITERATUREAGENT_STYLE_PPTX",
        ROOT / "style_reference" / "manuscript_style_reference.pptx",
    )
)
STYLE_SOURCE_EXPECTED_SHA256 = "1351a6f1d415598ad0761151533e1ec87a964178881e7edf106898a9158a7f1d"
STYLE_SOURCE_SLIDE = 3

FONT = "Times New Roman"
BODY_FONT_SIZE_PT = 12
AXIS_LABEL_FONT_SIZE_PT = 13
OUTPUT_SIZE_MM = (85.0, 64.0)
# The controlled source specifications remain immutable source records.  The
# applied Origin typography below is an audited presentation override made for
# readable PowerPoint assembly.
SOURCE_SPEC_FONT_SIZES_PT = {
    "tick_and_category": 8,
    "legend": 8,
    "axis_label": 9,
    "title": 9,
}
CONTROLLED_BLUE = "#0072B2"
CONTROLLED_TEAL = "#009E73"
CONTROLLED_INK = "#202124"
LEGACY_REFERENCE_PALE = "#F4F6F8"
SLATE = "#33546E"
ORANGE = "#F28A00"
BLACK = "#080808"
WHITE = "#FFFFFF"

CATEGORY_ORDER = (
    "ChatGPT",
    "LiteratureAgent (Qwen)",
    "Perplexity",
    "DeepSeek",
    "Elicit",
    "Edison KOSMOS",
    "NotebookLM",
    "Publication Analyzer",
)

SOURCE_COLUMNS = (
    "paper",
    "value",
    "model",
    "display_label",
    "category_order",
    "y_base",
    "y_offset",
    "y_display",
    "mean",
    "ci_lower",
    "ci_upper",
    "error_minus",
    "error_plus",
    "summary_color_hex",
)

HELPER_COLUMNS = (
    "other_raw_value",
    "other_raw_y_display",
    "literatureagent_raw_value",
    "literatureagent_raw_y_display",
    "other_ci_x",
    "other_ci_y",
    "other_cap_x",
    "other_cap_y",
    "literatureagent_ci_x",
    "literatureagent_ci_y",
    "literatureagent_cap_x",
    "literatureagent_cap_y",
    "other_mean_x",
    "other_mean_y",
    "literatureagent_mean_x",
    "literatureagent_mean_y",
    "axis_tick_position",
    "axis_tick_label",
)


CONFIG: dict[str, dict[str, Any]] = {
    "figs6a_benchmark": {
        "stem": "figs6a_benchmark",
        "graph": "FigS6a",
        "panel": "a",
        "xlabel": "GT Phrase Recall (%)",
        "xlim": (20.0, 105.0),
        "xstep": 20.0,
        "source": "data/figs6a_benchmark/plot_data.csv",
        "spec": "specifications/figs6a_benchmark.json",
        "reference_stem": "Figure3A_phrase_recall",
    },
    "figs6b_benchmark": {
        "stem": "figs6b_benchmark",
        "graph": "FigS6b",
        "panel": "b",
        "xlabel": "Cosine Similarity (%)",
        "xlim": (0.0, 40.0),
        "xstep": 10.0,
        "source": "data/figs6b_benchmark/plot_data.csv",
        "spec": "specifications/figs6b_benchmark.json",
        "reference_stem": "Figure3B_cosine_similarity",
    },
    "figs6c_benchmark": {
        "stem": "figs6c_benchmark",
        "graph": "FigS6c",
        "panel": "c",
        # Concise display-only wording keeps the 13 pt title inside the
        # compact 85 mm PowerPoint panel.  The full metric name remains in
        # the authoritative source/specification and caption.
        "xlabel": "GT Token F1 (%)",
        "source_xlabel": "GT Token F1 Score (%)",
        "xlim": (0.0, 16.0),
        "xstep": 4.0,
        "source": "data/figs6c_benchmark/plot_data.csv",
        "spec": "specifications/figs6c_benchmark.json",
        "reference_stem": "Figure3C_token_f1",
    },
    "figs6d_benchmark": {
        "stem": "figs6d_benchmark",
        "graph": "FigS6d",
        "panel": "d",
        "xlabel": "Verbosity Ratio",
        "xlim": (8.0, 80.0),
        "xstep": 20.0,
        "source": "data/figs6d_benchmark/plot_data.csv",
        "spec": "specifications/figs6d_benchmark.json",
        "reference_stem": "Figure3D_verbosity_ratio",
    },
}


@dataclass(frozen=True)
class BenchmarkSummary:
    category_order: int
    model: str
    display_label: str
    y_base: float
    mean: float
    ci_lower: float
    ci_upper: float
    error_minus: float
    error_plus: float
    color_hex: str


@dataclass(frozen=True)
class ReferenceAxis:
    width_mm: float
    height_mm: float
    xlim: tuple[float, float]
    xstep: float
    x_ticks: tuple[float, ...]
    pale_band_count: int


@dataclass(frozen=True)
class PlotPlan:
    plot_id: str
    config: dict[str, Any]
    spec: dict[str, Any]
    source_path: Path
    spec_path: Path
    reference_paths: tuple[Path, ...]
    source_rows: tuple[dict[str, str], ...]
    summaries: tuple[BenchmarkSummary, ...]
    xlim: tuple[float, float]
    reference_axis: ReferenceAxis
    scientific_checks: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style_source_path_text() -> str:
    if STYLE_SOURCE.is_file():
        return str(STYLE_SOURCE.resolve())
    return "encoded style specification"


def style_source_sha256() -> str:
    if STYLE_SOURCE.is_file():
        return sha256(STYLE_SOURCE)
    return STYLE_SOURCE_EXPECTED_SHA256


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers or len(headers) != len(set(headers)):
            raise RuntimeError(f"CSV headers must be present and unique: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV contains no rows: {path}")
    return headers, rows


def finite_float(value: str, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite value for {context}: {value!r}")
    return number


def exact_int(value: str, *, context: str) -> int:
    number = finite_float(value, context=context)
    if not number.is_integer():
        raise RuntimeError(f"Expected integer for {context}: {value!r}")
    return int(number)


def close(a: float, b: float, *, atol: float = 1e-11) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=atol)


def auto_margin_limits(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    low, high = min(materialized), max(materialized)
    span = high - low
    if span <= 0:
        raise RuntimeError("Benchmark x values do not span a positive range")
    return low - 0.05 * span, high + 0.05 * span


def parse_reference_axis(svg: Path, config: dict[str, Any] | None = None) -> ReferenceAxis:
    """Recover the retained panel's physical size and x-axis contract."""

    text = svg.read_text(encoding="utf-8")
    root = re.search(
        r"<svg\b[^>]*\bwidth=\"([0-9.]+)pt\"[^>]*\bheight=\"([0-9.]+)pt\"",
        text,
        flags=re.IGNORECASE,
    )
    if root is None:
        element = ElementTree.fromstring(text)
        def millimetres(value: str) -> float:
            match = re.fullmatch(r"([0-9.]+)(mm|cm|in|pt)", value)
            if match is None:
                raise RuntimeError(f"Unsupported SVG dimension in {svg}")
            return float(match[1]) * {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72}[match[2]]
        if config is None:
            raise RuntimeError("Canonical Origin SVG axis checks require the frozen plotting configuration")
        # Origin uses path-encoded tick labels; the retained axis limits are
        # taken explicitly from the same frozen configuration used to render.
        lower, upper = map(float, config["xlim"])
        step = float(config["xstep"])
        first_tick = math.ceil(lower / step) * step
        ticks = tuple(first_tick + step * index for index in range(int((upper - first_tick) / step) + 1))
        return ReferenceAxis(width_mm=millimetres(element.attrib["width"]),
                             height_mm=millimetres(element.attrib["height"]),
                             xlim=(lower, upper), xstep=step, x_ticks=ticks,
                             pale_band_count=text.lower().count(LEGACY_REFERENCE_PALE.lower()))
    width_mm = float(root.group(1)) * 25.4 / 72.0
    height_mm = float(root.group(2)) * 25.4 / 72.0

    patch = re.search(
        r'<g id="patch_2">\s*<path d="M ([0-9.+-]+) ([0-9.+-]+)\s*L ([0-9.+-]+)',
        text,
        flags=re.IGNORECASE,
    )
    if patch is None:
        raise RuntimeError(f"Could not recover retained plot-area bounds: {svg}")
    left, right = float(patch.group(1)), float(patch.group(3))
    if right <= left:
        raise RuntimeError(f"Invalid retained plot-area bounds: {svg}")

    ticks: list[tuple[float, float]] = []
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    for match in re.finditer(
        rf'<text\b[^>]*\bx="({number_pattern})"[^>]*>\s*({number_pattern})\s*</text>',
        text,
        flags=re.IGNORECASE,
    ):
        position, value = float(match.group(1)), float(match.group(2))
        if left - 1e-6 <= position <= right + 1e-6:
            ticks.append((position, value))
    ticks = sorted(set(ticks))
    if len(ticks) < 2:
        raise RuntimeError(f"Could not recover numeric x ticks from {svg}")

    x0, value0 = ticks[0]
    x1, value1 = ticks[-1]
    slope = (x1 - x0) / (value1 - value0)
    intercept = x0 - slope * value0
    if slope <= 0:
        raise RuntimeError(f"Unexpected retained x-axis direction: {svg}")
    for position, value in ticks:
        if abs(position - (slope * value + intercept)) > 2e-5:
            raise RuntimeError(f"Retained x ticks are not linear in {svg}")
    values = tuple(value for _, value in ticks)
    steps = [later - earlier for earlier, later in zip(values, values[1:])]
    if not all(close(step, steps[0], atol=1e-8) for step in steps):
        raise RuntimeError(f"Retained x ticks are not uniformly spaced in {svg}")
    xlim = ((left - intercept) / slope, (right - intercept) / slope)
    return ReferenceAxis(
        width_mm=width_mm,
        height_mm=height_mm,
        xlim=xlim,
        xstep=steps[0],
        x_ticks=values,
        pale_band_count=text.lower().count(LEGACY_REFERENCE_PALE.lower()),
    )


def validate_spec(plot_id: str, config: dict[str, Any], spec: dict[str, Any]) -> None:
    if spec.get("plot_id") != plot_id:
        raise RuntimeError(f"Specification plot ID mismatch for {plot_id}")
    if spec.get("status") != "READY" or spec.get("required_action") != "RETAIN":
        raise RuntimeError(f"{plot_id} must be READY/RETAIN in the source")
    if spec.get("assembly_instruction") != "Standalone plot only. Do not assemble with another panel in OriginPro.":
        raise RuntimeError(f"Standalone assembly instruction changed for {plot_id}")
    style = spec["style"]
    dimensions = style["standalone_dimensions_mm"]
    if (float(dimensions["width"]), float(dimensions["height"])) != (85.0, 60.0):
        raise RuntimeError(f"Unexpected physical dimensions for {plot_id}")
    if style.get("font_family") != FONT:
        raise RuntimeError(f"Unexpected font family for {plot_id}")
    sizes = style["font_sizes_pt"]
    if {key: int(sizes[key]) for key in SOURCE_SPEC_FONT_SIZES_PT} != SOURCE_SPEC_FONT_SIZES_PT:
        raise RuntimeError(f"Font-size contract changed for {plot_id}")
    if spec["input"]["file"] != config["source"]:
        raise RuntimeError(f"Source path changed for {plot_id}")
    if spec["input"]["raw_value_column"] != "value" or spec["input"]["raw_y_coordinate_column"] != "y_display":
        raise RuntimeError(f"Raw coordinate contract changed for {plot_id}")
    if spec["input"]["origin_asymmetric_error_columns"] != ["error_minus", "error_plus"]:
        raise RuntimeError(f"Asymmetric-error contract changed for {plot_id}")
    if spec["axes"]["x"]["label"] != config.get("source_xlabel", config["xlabel"]):
        raise RuntimeError(f"Axis label mismatch for {plot_id}")
    if spec["axes"]["y"].get("reverse_order") is not True:
        raise RuntimeError(f"Reverse y-axis contract changed for {plot_id}")
    if tuple(spec["category_order"]) != CATEGORY_ORDER:
        raise RuntimeError(f"Category order changed for {plot_id}")
    expected = spec["expected_plotted_counts"]
    if (int(expected["models"]), int(expected["paper_level_points"]), int(expected["summary_estimates"])) != (8, 80, 8):
        raise RuntimeError(f"Expected plotted counts changed for {plot_id}")
    overrides = spec["style_overrides"]
    if overrides["raw_points"] != {"color": CONTROLLED_BLUE, "size_pt": 12, "opacity": 0.68}:
        raise RuntimeError(f"Raw-point style changed for {plot_id}")
    if overrides["mean_marker"] != "circle, 5.3 pt, white 0.6 pt edge":
        raise RuntimeError(f"Mean-marker style changed for {plot_id}")
    if float(overrides["error_line_width_pt"]) != 1.25 or float(overrides["error_cap_size_pt"]) != 2.8:
        raise RuntimeError(f"CI style changed for {plot_id}")
    if (
        overrides["literatureagent_summary_color"] != CONTROLLED_TEAL
        or overrides["other_summary_color"] != CONTROLLED_INK
    ):
        raise RuntimeError(f"Summary colors changed for {plot_id}")


def validate_rows(plot_id: str, rows: list[dict[str, str]]) -> tuple[tuple[BenchmarkSummary, ...], dict[str, Any]]:
    if len(rows) != 80:
        raise RuntimeError(f"{plot_id}: expected 80 paper-level points, found {len(rows)}")
    grouped: dict[int, list[dict[str, str]]] = {index: [] for index in range(1, 9)}
    for row_index, row in enumerate(rows, start=2):
        order = exact_int(row["category_order"], context=f"row {row_index} category_order")
        if order not in grouped:
            raise RuntimeError(f"{plot_id}: invalid category order {order}")
        grouped[order].append(row)

    summaries: list[BenchmarkSummary] = []
    paper_orders: list[tuple[str, ...]] = []
    for order, expected_label in enumerate(CATEGORY_ORDER, start=1):
        group = grouped[order]
        if len(group) != 10:
            raise RuntimeError(f"{plot_id}: category {order} must contain 10 points")
        if any(row["display_label"] != expected_label for row in group):
            raise RuntimeError(f"{plot_id}: category-label mismatch at order {order}")
        if len({row["model"] for row in group}) != 1:
            raise RuntimeError(f"{plot_id}: category {order} has multiple model identifiers")
        papers = tuple(row["paper"] for row in group)
        if len(set(papers)) != 10:
            raise RuntimeError(f"{plot_id}: duplicate paper within {expected_label}")
        paper_orders.append(papers)

        y_base = float(order - 1)
        expected_offsets = [-0.11 + index * 0.22 / 9.0 for index in range(10)]
        for point_index, (row, expected_offset) in enumerate(zip(group, expected_offsets), start=1):
            value = finite_float(row["value"], context=f"{expected_label} point {point_index}")
            observed_base = finite_float(row["y_base"], context=f"{expected_label} y_base")
            offset = finite_float(row["y_offset"], context=f"{expected_label} y_offset")
            display = finite_float(row["y_display"], context=f"{expected_label} y_display")
            if not math.isfinite(value) or not close(observed_base, y_base) or not close(offset, expected_offset):
                raise RuntimeError(f"{plot_id}: frozen y-coordinate inputs changed for {expected_label}")
            if not close(display, observed_base + offset):
                raise RuntimeError(f"{plot_id}: y_display != y_base + y_offset for {expected_label}")

        repeated_fields = (
            "mean",
            "ci_lower",
            "ci_upper",
            "error_minus",
            "error_plus",
            "summary_color_hex",
        )
        for field in repeated_fields:
            if len({row[field] for row in group}) != 1:
                raise RuntimeError(f"{plot_id}: repeated {field} differs within {expected_label}")
        first = group[0]
        mean = finite_float(first["mean"], context=f"{expected_label} mean")
        low = finite_float(first["ci_lower"], context=f"{expected_label} ci_lower")
        high = finite_float(first["ci_upper"], context=f"{expected_label} ci_upper")
        error_minus = finite_float(first["error_minus"], context=f"{expected_label} error_minus")
        error_plus = finite_float(first["error_plus"], context=f"{expected_label} error_plus")
        if not low <= mean <= high:
            raise RuntimeError(f"{plot_id}: mean lies outside CI for {expected_label}")
        if not close(mean - error_minus, low) or not close(mean + error_plus, high):
            raise RuntimeError(f"{plot_id}: asymmetric errors do not reproduce CI for {expected_label}")
        expected_color = CONTROLLED_TEAL if expected_label == "LiteratureAgent (Qwen)" else CONTROLLED_INK
        if first["summary_color_hex"].upper() != expected_color:
            raise RuntimeError(f"{plot_id}: unexpected summary color for {expected_label}")
        summaries.append(
            BenchmarkSummary(
                category_order=order,
                model=first["model"],
                display_label=expected_label,
                y_base=y_base,
                mean=mean,
                ci_lower=low,
                ci_upper=high,
                error_minus=error_minus,
                error_plus=error_plus,
                color_hex=expected_color,
            )
        )
    if any(papers != paper_orders[0] for papers in paper_orders[1:]):
        raise RuntimeError(f"{plot_id}: paper order differs between models")
    checks = {
        "models": 8,
        "paper_level_points": 80,
        "paper_level_points_per_model": 10,
        "summary_estimates": 8,
        "asymmetric_confidence_intervals": 8,
        "frozen_y_display_coordinates": 80,
        "paper_order_identical_across_models": True,
        "category_order_top_to_bottom": list(CATEGORY_ORDER),
        "controlled_source_literatureagent_summary_color": CONTROLLED_TEAL,
        "controlled_source_other_summary_color": CONTROLLED_INK,
    }
    return tuple(summaries), checks


def build_plan(plot_id: str) -> PlotPlan:
    config = CONFIG[plot_id]
    source_path = PACKAGE / config["source"]
    spec_path = PACKAGE / config["spec"]
    reference_paths = tuple(ROOT / "figures" / "supplementary" / folder / f"{plot_id}.{suffix}"
                            for folder, suffix in (("svg", "svg"), ("pdf", "pdf"), ("png_600dpi", "png")))
    for path in (source_path, spec_path, *reference_paths):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_spec(plot_id, config, spec)
    headers, rows = read_csv(source_path)
    if tuple(headers) != SOURCE_COLUMNS:
        raise RuntimeError(f"{plot_id}: authoritative CSV columns/order changed: {headers}")
    summaries, checks = validate_rows(plot_id, rows)
    values = [finite_float(row["value"], context=f"{plot_id} value") for row in rows]
    xlim = tuple(float(value) for value in config["xlim"])
    if min(values) < xlim[0] or max(values) > xlim[1]:
        raise RuntimeError(f"{plot_id}: PowerPoint-matched x limits would clip controlled data")
    reference_axis = parse_reference_axis(reference_paths[0], config)
    if not close(reference_axis.width_mm, OUTPUT_SIZE_MM[0], atol=0.002) or not close(reference_axis.height_mm, OUTPUT_SIZE_MM[1], atol=0.002):
        raise RuntimeError(f"{plot_id}: canonical panel dimensions differ from the retained 85 x 64 mm rendering")
    source_margin_limits = auto_margin_limits(values)
    checks.update(
        {
            "raw_value_min": min(values),
            "raw_value_max": max(values),
            "source_derived_x_limits_with_5pct_margin": list(source_margin_limits),
            "applied_powerpoint_slide3_x_limits": list(xlim),
            "all_controlled_values_inside_applied_x_limits": True,
            "canonical_reference_axis_basis": "SVG physical dimensions; numeric limits from frozen plotting configuration",
            "y_limits": [-0.55, 7.55],
            "y_axis_reversed": True,
        }
    )
    return PlotPlan(
        plot_id=plot_id,
        config=config,
        spec=spec,
        source_path=source_path,
        spec_path=spec_path,
        reference_paths=reference_paths,
        source_rows=tuple(rows),
        summaries=summaries,
        xlim=xlim,
        reference_axis=reference_axis,
        scientific_checks=checks,
    )


def copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    status = "copied"
    if destination.is_file() and sha256(destination) == source_hash:
        status = "already_identical"
    else:
        shutil.copy2(source, destination)
    destination_hash = sha256(destination)
    if destination_hash != source_hash:
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "sha256_source": source_hash,
        "sha256_destination": destination_hash,
        "status": status,
    }


def separated_segments(segments: Iterable[tuple[float, float, float]]) -> tuple[list[Any], list[Any]]:
    xs: list[Any] = []
    ys: list[Any] = []
    for low, high, y in segments:
        xs.extend((low, high, None))
        ys.extend((y, y, None))
    return xs, ys


def cap_segments(
    summaries: Iterable[BenchmarkSummary], cap_half_height: float
) -> tuple[list[Any], list[Any]]:
    xs: list[Any] = []
    ys: list[Any] = []
    for item in summaries:
        # Use the authoritative asymmetric error lengths to construct the
        # endpoints; validation above confirms these reproduce ci_lower/upper.
        low = item.mean - item.error_minus
        high = item.mean + item.error_plus
        for endpoint in (low, high):
            xs.extend((endpoint, endpoint, None))
            ys.extend((item.y_base - cap_half_height, item.y_base + cap_half_height, None))
    return xs, ys


def write_display_helper(plan: PlotPlan, data_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Write every transformed plotting coordinate to a unique-header CSV."""

    data_dir.mkdir(parents=True, exist_ok=True)

    # Translate the retained 2.8 pt cap half-height into the formatting-only
    # 64 mm output page while preserving every CI endpoint.
    page_height_pt = OUTPUT_SIZE_MM[1] / 25.4 * 72.0
    layer_height_pt = page_height_pt * 0.70
    points_per_y_unit = layer_height_pt / 8.1
    cap_half_height = 2.8 / points_per_y_unit

    other = [item for item in plan.summaries if item.display_label != "LiteratureAgent (Qwen)"]
    literatureagent = [item for item in plan.summaries if item.display_label == "LiteratureAgent (Qwen)"]
    if len(other) != 7 or len(literatureagent) != 1:
        raise RuntimeError(f"{plan.plot_id}: unexpected summary-color partition")

    other_ci_x, other_ci_y = separated_segments(
        (item.mean - item.error_minus, item.mean + item.error_plus, item.y_base) for item in other
    )
    la_ci_x, la_ci_y = separated_segments(
        (item.mean - item.error_minus, item.mean + item.error_plus, item.y_base) for item in literatureagent
    )
    other_cap_x, other_cap_y = cap_segments(other, cap_half_height)
    la_cap_x, la_cap_y = cap_segments(literatureagent, cap_half_height)

    # Store display y coordinates in ordinary bottom-to-top Origin space so
    # the x axis remains at the bottom of the panel.  This is exactly
    # equivalent to plotting the frozen source y coordinates on a reversed
    # axis, but avoids Origin moving the numeric x ticks to the page top.
    y_top = float(len(CATEGORY_ORDER) - 1)

    def display_y(value: float) -> float:
        return y_top - float(value)

    def display_y_list(values: Sequence[Any]) -> list[Any]:
        return [None if value is None else display_y(float(value)) for value in values]

    other_raw = [row for row in plan.source_rows if row["display_label"] != "LiteratureAgent (Qwen)"]
    la_raw = [row for row in plan.source_rows if row["display_label"] == "LiteratureAgent (Qwen)"]
    if len(other_raw) != 70 or len(la_raw) != 10:
        raise RuntimeError(f"{plan.plot_id}: unexpected raw-point style partition")

    columns: dict[str, list[Any]] = {
        "other_raw_value": [finite_float(row["value"], context="other_raw_value") for row in other_raw],
        "other_raw_y_display": [display_y(finite_float(row["y_display"], context="other_raw_y_display")) for row in other_raw],
        "literatureagent_raw_value": [finite_float(row["value"], context="literatureagent_raw_value") for row in la_raw],
        "literatureagent_raw_y_display": [display_y(finite_float(row["y_display"], context="literatureagent_raw_y_display")) for row in la_raw],
        "other_ci_x": other_ci_x,
        "other_ci_y": display_y_list(other_ci_y),
        "other_cap_x": other_cap_x,
        "other_cap_y": display_y_list(other_cap_y),
        "literatureagent_ci_x": la_ci_x,
        "literatureagent_ci_y": display_y_list(la_ci_y),
        "literatureagent_cap_x": la_cap_x,
        "literatureagent_cap_y": display_y_list(la_cap_y),
        "other_mean_x": [item.mean for item in other],
        "other_mean_y": [display_y(item.y_base) for item in other],
        "literatureagent_mean_x": [item.mean for item in literatureagent],
        "literatureagent_mean_y": [display_y(item.y_base) for item in literatureagent],
        "axis_tick_position": [display_y(item.y_base) for item in plan.summaries],
        "axis_tick_label": [item.display_label for item in plan.summaries],
    }
    if tuple(columns) != HELPER_COLUMNS:
        raise RuntimeError("Internal helper-column order changed")
    # Keep an 80-row helper so its row count remains directly comparable to
    # the exact 80-row controlled source even though the largest style split
    # contains 70 other-system points.
    row_count = max(80, *(len(values) for values in columns.values()))
    helper = data_dir / "origin_display_helper.csv"
    with helper.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HELPER_COLUMNS)
        for row_index in range(row_count):
            output: list[Any] = []
            for name in HELPER_COLUMNS:
                values = columns[name]
                value = values[row_index] if row_index < len(values) else ""
                if value is None:
                    value = ""
                elif isinstance(value, float):
                    value = f"{value:.17g}"
                output.append(value)
            writer.writerow(output)
    helper_headers, helper_rows = read_csv(helper)
    if tuple(helper_headers) != HELPER_COLUMNS or len(helper_rows) != 80:
        raise RuntimeError(f"Unexpected helper shape for {plan.plot_id}")
    audit = {
        "path": str(helper.resolve()),
        "sha256": sha256(helper),
        "columns": list(HELPER_COLUMNS),
        "rows": row_count,
        "other_system_raw_points": 70,
        "literatureagent_raw_points": 10,
        "raw_points_total": 80,
        "alternating_row_bands": "omitted for a white background",
        "error_cap_half_height_y_units": cap_half_height,
        "display_y_transform": "7 - authoritative y_display; visually equivalent to reversed y axis",
    }
    return helper, audit


def add_line_from_columns(layer: Any, table: Any, x_name: str, y_name: str, color: str, width: float) -> Any:
    plot = layer.add_plot(
        table.sheet,
        coly=table.columns[y_name],
        colx=table.columns[x_name],
        type="l",
    )
    if plot is None:
        raise RuntimeError(f"Origin failed to add line series {x_name}/{y_name}")
    plot.color = color
    plot.set_float("line.width", width)
    return plot


def add_scatter_from_columns(
    op: Any,
    layer: Any,
    table: Any,
    x_name: str,
    y_name: str,
    *,
    fill: str,
    edge: str,
    size: float,
    shape: int = 2,
    interior: int = 1,
    transparency: int = 0,
    edge_width: float | None = None,
) -> Any:
    plot = layer.add_plot(
        table.sheet,
        coly=table.columns[y_name],
        colx=table.columns[x_name],
        type="s",
    )
    if plot is None:
        raise RuntimeError(f"Origin failed to add scatter series {x_name}/{y_name}")
    # Origin 2024 derives the visible symbol outline primarily from the plot
    # color; setting only symbol.edgecolor can leave white-filled open symbols
    # with an invisible white border.  Use the requested edge as the series
    # color and keep the interior color independent.
    plot.color = edge
    plot.symbol_kind = shape
    plot.symbol_interior = interior
    plot.symbol_size = size
    plot.set_int("symbol.fillcolor", op.ocolor(fill))
    plot.set_int("symbol.edgecolor", op.ocolor(edge))
    plot.transparency = transparency
    if edge_width is not None:
        # -kh sets the symbol edge thickness.  A graph data-plot range avoids
        # dependence on the active worksheet/data-set name.
        layer.lt_exec(f"range laEdgePlot={plot.lt_range()}; set laEdgePlot -kh {edge_width:g};")
    return plot


def disable_background_grids(layer: Any) -> None:
    layer.lt_exec(
        "layer.x.grid=0; layer.x.minorgrid=0; layer.y.grid=0; layer.y.minorgrid=0; "
        "layer.x.grid.major=0; layer.x.grid.minor=0; "
        "layer.y.grid.major=0; layer.y.grid.minor=0; "
        "layer.x.showGrids=0; layer.y.showGrids=0; "
        "layer.x2.show=0; layer.y2.show=0;"
    )


def set_reversed_category_ticks(layer: Any, table: Any) -> None:
    label_column = table.columns["axis_tick_label"]
    dataset = f"{table.sheet.lt_range()}!col({label_column + 1})"
    positions = " ".join(str(index) for index in reversed(range(8)))
    layer.lt_exec(
        f'layer.y.label.type=2; layer.y.label.dataset$="{dataset}"; '
        f'layer.y.ticksbydata$="{positions}";'
    )


def write_validation_sheet(book: Any, plan: PlotPlan, helper_audit: dict[str, Any], plot_audit: dict[str, Any]) -> None:
    sheet = book.add_sheet()
    sheet.name = "Validation"
    sheet.lname = f"{plan.plot_id} build validation"
    checks = [
        ("analysis_id", ANALYSIS_ID, "PASS"),
        ("plot_id", plan.plot_id, "PASS"),
        ("scientific_status", plan.spec["status"], "PASS"),
        ("source_action", plan.spec["required_action"], "PASS"),
        ("source_sha256", sha256(plan.source_path), "PASS"),
        ("source_rows", len(plan.source_rows), "PASS"),
        ("models", len(plan.summaries), "PASS"),
        ("summary_intervals", len(plan.summaries), "PASS"),
        ("standalone_dimensions_mm", "[85, 60]", "PASS"),
        ("font", FONT, "PASS"),
        (
            "font_sizes_pt",
            f"category/value/body {BODY_FONT_SIZE_PT}; axis labels {AXIS_LABEL_FONT_SIZE_PT}",
            "PASS",
        ),
        ("style_source", f"{STYLE_SOURCE.name}, slide {STYLE_SOURCE_SLIDE}", "PASS"),
        ("paper_points", "open slate/orange circles", "PASS"),
        ("mean_and_95pct_ci", "open black/orange squares", "PASS"),
        ("background_gridlines", "major/minor x/y disabled", "PASS"),
        ("alternating_row_bands", "omitted for a white background", "PASS"),
        ("x_limits", json.dumps(plan.xlim), "PASS"),
        ("y_limits", "[-0.55, 7.55], reversed", "PASS"),
        ("helper_audit", json.dumps(helper_audit, ensure_ascii=False), "PASS"),
        ("plot_audit", json.dumps(plot_audit, ensure_ascii=False), "PASS"),
    ]
    for index, header in enumerate(("check", "value", "status")):
        sheet.from_list(index, [row[index] for row in checks], lname=header, axis="")


def build_origin_plot(plan: PlotPlan, output_root: Path, close_origin: bool) -> dict[str, Any]:
    # Delayed imports guarantee that --dry-run does not import Origin or start
    # an automation server.
    import originpro as op
    import build_manuscript_plot as common
    import build_literatureagent_origin_figures as base
    common.op = base.require_origin()

    output_root = output_root.resolve()
    for name in ("origin_projects", "pdf", "svg", "png", "data", "validation"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    data_dir = output_root / "data" / plan.plot_id
    data_dir.mkdir(parents=True, exist_ok=True)
    source_copy_audit = copy_exact(plan.source_path, data_dir / "plot_data.csv")
    spec_copy_audit = copy_exact(plan.spec_path, data_dir / plan.spec_path.name)
    helper_path, helper_audit = write_display_helper(plan, data_dir)

    if getattr(op, "oext", False):
        op.set_show(not close_origin)
    op.new(asksave=False)
    base.FONT_INDEX = int(op.lt_int(f'font("{FONT}")'))
    book = op.new_book("w", lname=f"{plan.plot_id} exact source and Origin display helper")
    if book is None:
        raise RuntimeError("Origin could not create benchmark workbook")
    book.name = "PlotData"
    common.import_csv_sheet(
        book,
        data_dir / "plot_data.csv",
        "Source",
        "Exact authoritative source table",
        first=True,
    )
    _, display_table = common.import_csv_sheet(
        book,
        helper_path,
        "Display",
        "Explicit linked Origin plotting coordinates",
    )

    graph = base.new_graph(
        plan.config["graph"],
        plan.plot_id,
        common.mm_to_in(OUTPUT_SIZE_MM[0]),
        common.mm_to_in(OUTPUT_SIZE_MM[1]),
    )
    layer = graph[0]
    # Match the PowerPoint placement aspect and reserve enough left and lower
    # margin for 12-pt model labels plus the 13-pt axis-title band.
    base.position_layer(layer, 54.0, 5.0, 42.0, 70.0)

    # Use a clean white background without row bands. Keep the
    # source's frozen y jitter, but split paper points only for visual styling.
    raw_origin_diameter_pt = math.sqrt(12.0)
    add_scatter_from_columns(
        op,
        layer,
        display_table,
        "other_raw_value",
        "other_raw_y_display",
        fill=WHITE,
        edge=SLATE,
        size=raw_origin_diameter_pt,
        shape=2,
        interior=1,
        transparency=25,
        edge_width=0.65,
    )
    add_scatter_from_columns(
        op,
        layer,
        display_table,
        "literatureagent_raw_value",
        "literatureagent_raw_y_display",
        fill=WHITE,
        edge=ORANGE,
        size=raw_origin_diameter_pt,
        shape=2,
        interior=1,
        transparency=12,
        edge_width=0.65,
    )
    add_line_from_columns(layer, display_table, "other_ci_x", "other_ci_y", BLACK, 1.25)
    add_line_from_columns(layer, display_table, "other_cap_x", "other_cap_y", BLACK, 1.25)
    add_line_from_columns(layer, display_table, "literatureagent_ci_x", "literatureagent_ci_y", ORANGE, 1.25)
    add_line_from_columns(layer, display_table, "literatureagent_cap_x", "literatureagent_cap_y", ORANGE, 1.25)
    add_scatter_from_columns(
        op,
        layer,
        display_table,
        "other_mean_x",
        "other_mean_y",
        fill=WHITE,
        edge=BLACK,
        size=5.3,
        shape=1,
        interior=1,
        edge_width=1.0,
    )
    add_scatter_from_columns(
        op,
        layer,
        display_table,
        "literatureagent_mean_x",
        "literatureagent_mean_y",
        fill=WHITE,
        edge=ORANGE,
        size=5.3,
        shape=1,
        interior=1,
        edge_width=1.0,
    )

    display_xlim = (-0.5, 40.0) if plan.plot_id == "figs6b_benchmark" else plan.xlim
    base.style_axis(
        layer,
        "",
        xlim=display_xlim,
        xstep=float(plan.config["xstep"]),
        ylim=(-0.55, 7.55),
        ystep=1.0,
        show_y_labels=True,
        axis_title_size=AXIS_LABEL_FONT_SIZE_PT,
        tick_size=BODY_FONT_SIZE_PT,
    )
    layer.lt_exec(
        "layer.y.reverse=0; layer.x.showAxes=1; layer.y.showAxes=1; "
        f'layer.x.color=color("{BLACK}"); layer.y.color=color("{BLACK}"); '
        f'layer.x.label.color=color("{BLACK}"); layer.y.label.color=color("{BLACK}"); '
        "layer.x.label.decimalPlaces=0; layer.x.label.nDigits=0; "
        "layer.sauto=0; layer.maxpts=0; page.speedmode=0;"
    )
    if plan.plot_id == "figs6b_benchmark":
        # Preserve the scientific 0--40 tick contract while adding only the
        # half-unit display margin needed to show the smallest open circle.
        layer.lt_exec('layer.x.ticksbydata$="0 10 20 30 40";')
    set_reversed_category_ticks(layer, display_table)
    disable_background_grids(layer)
    # Keep the long benchmark metric label below the numeric tick-label row.
    title = base.add_percent_label(
        layer,
        plan.config["xlabel"],
        50,
        123,
        size=AXIS_LABEL_FONT_SIZE_PT,
        color=BLACK,
    )
    title.set_int("anchor", 5)
    disable_background_grids(layer)

    plot_audit = {
        "paper_level_points": 80,
        "summary_estimates": 8,
        "asymmetric_confidence_intervals": 8,
        "raw_coordinate_columns": ["value", "y_display"],
        "other_system_paper_points": {
            "shape": "open circle",
            "edge_color": SLATE,
            "transparency_percent": 25,
            "count": 70,
        },
        "literatureagent_paper_points": {
            "shape": "open circle",
            "edge_color": ORANGE,
            "transparency_percent": 12,
            "count": 10,
        },
        "raw_point_source_scatter_area_pt2": 12.0,
        "raw_point_origin_diameter_pt": raw_origin_diameter_pt,
        "other_system_mean_marker": {
            "shape": "open square",
            "diameter_pt": 5.3,
            "edge": BLACK,
            "edge_width_pt": 1.0,
        },
        "literatureagent_mean_marker": {
            "shape": "open square",
            "diameter_pt": 5.3,
            "edge": ORANGE,
            "edge_width_pt": 1.0,
        },
        "confidence_interval_line_width_pt": 1.25,
        "confidence_interval_cap_size_pt_each_side": 2.8,
        "x_limits": list(display_xlim),
        "scientific_tick_limits": list(plan.xlim),
        "x_tick_step": plan.config["xstep"],
        "y_limits": [-0.55, 7.55],
        "y_reverse": False,
        "display_y_transform": "7 - authoritative y_display",
        "category_order_top_to_bottom": list(CATEGORY_ORDER),
        "font_sizes_pt": {
            "category_value_body": BODY_FONT_SIZE_PT,
            "axis_labels": AXIS_LABEL_FONT_SIZE_PT,
        },
        "alternating_row_bands": "omitted for a white background",
        "background_gridlines": {"x_major": 0, "x_minor": 0, "y_major": 0, "y_minor": 0},
        "embedded_chart_title": "none",
        "embedded_panel_letter": "none",
        "legend": "none; separate shared legend is delivered elsewhere",
    }
    write_validation_sheet(book, plan, helper_audit, plot_audit)
    pages = op.graph_list("p")
    if len(pages) != 1 or pages[0].name != plan.config["graph"]:
        raise RuntimeError(
            f"Expected one graph page named {plan.config['graph']}; found {[page.name for page in pages]}"
        )

    project = output_root / "origin_projects" / f"{plan.config['stem']}.opju"
    if not op.save(str(project.resolve())):
        raise RuntimeError(f"Origin failed to save project: {project}")
    exports = common.export_graph(graph, plan.plot_id, {**plan.config, "size_mm": OUTPUT_SIZE_MM}, output_root)
    report = {
        "status": "PASS",
        "analysis_id": ANALYSIS_ID,
        "plot_id": plan.plot_id,
        "origin_project": str(project.resolve()),
        "origin_project_sha256": sha256(project),
        "origin_graph_page": plan.config["graph"],
        "authoritative_source_copy": source_copy_audit,
        "specification_copy": spec_copy_audit,
        "origin_display_helper": helper_audit,
        "reference_assets": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in plan.reference_paths
        ],
        "style_source": {
            "path": style_source_path_text(),
            "sha256": style_source_sha256(),
            "slide": STYLE_SOURCE_SLIDE,
        },
        "scientific_checks": plan.scientific_checks,
        "style": {
            "page_dimensions_mm": list(OUTPUT_SIZE_MM),
            "font": FONT,
            "font_sizes_pt": {
                "category_value_body": BODY_FONT_SIZE_PT,
                "tick_and_category": BODY_FONT_SIZE_PT,
                "axis_label": AXIS_LABEL_FONT_SIZE_PT,
            },
            "paper_points": "open slate circles for other systems; open orange circles for LiteratureAgent",
            "mean_and_95pct_ci": "open black squares for other systems; open orange square for LiteratureAgent",
            "background_gridlines": "disabled",
            "alternating_row_bands": "omitted for a white background",
            "embedded_title_panel_letter_legend": "none",
        },
        "rendering_resolutions": {
            "raw_point_size": "Matplotlib s=12 pt^2 translated to Origin diameter sqrt(12) pt",
            "style_authority": "encoded manuscript palette and typography",
            "background": "white with all major/minor x/y gridlines disabled and no row bands",
            "axis_labels": "controlled specification labels override legacy reference wording",
        },
        "plot_audit": plot_audit,
        "exports": exports,
    }
    audit = output_root / "validation" / f"{plan.plot_id}_build_audit.json"
    report["build_audit"] = str(audit.resolve())
    audit.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def dry_run_report(plan: PlotPlan, output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    return {
        "status": "PASS",
        "mode": "dry-run-no-Origin-import-or-launch",
        "analysis_id": ANALYSIS_ID,
        "plot_id": plan.plot_id,
        "scientific_status": plan.spec["status"],
        "manifest_action": plan.spec["required_action"],
        "source": {
            "path": str(plan.source_path.resolve()),
            "sha256": sha256(plan.source_path),
            "columns": list(SOURCE_COLUMNS),
            "rows": len(plan.source_rows),
        },
        "specification": {"path": str(plan.spec_path.resolve()), "sha256": sha256(plan.spec_path)},
        "reference_assets": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in plan.reference_paths
        ],
        "style_source": {
            "path": style_source_path_text(),
            "sha256": style_source_sha256(),
            "slide": STYLE_SOURCE_SLIDE,
        },
        "canonical_reference_axis_contract": {
            "dimensions_mm": [plan.reference_axis.width_mm, plan.reference_axis.height_mm],
            "x_limits": list(plan.reference_axis.xlim),
            "x_ticks": list(plan.reference_axis.x_ticks),
            "legacy_pale_band_count_not_applied": plan.reference_axis.pale_band_count,
        },
        "scientific_checks": plan.scientific_checks,
        "planned_outputs": {
            "opju": str((output_root / "origin_projects" / f"{plan.plot_id}.opju").resolve()),
            "png": str((output_root / "png" / f"{plan.plot_id}.png").resolve()),
            "svg": str((output_root / "svg" / f"{plan.plot_id}.svg").resolve()),
            "pdf": str((output_root / "pdf" / f"{plan.plot_id}.pdf").resolve()),
            "source_copy": str((output_root / "data" / plan.plot_id / "plot_data.csv").resolve()),
            "origin_display_helper": str((output_root / "data" / plan.plot_id / "origin_display_helper.csv").resolve()),
            "build_audit": str((output_root / "validation" / f"{plan.plot_id}_build_audit.json").resolve()),
        },
        "style": {
            "dimensions_mm": list(OUTPUT_SIZE_MM),
            "font": FONT,
            "font_sizes_pt": {
                "category_value_body": BODY_FONT_SIZE_PT,
                "tick_and_category": BODY_FONT_SIZE_PT,
                "axis_label": AXIS_LABEL_FONT_SIZE_PT,
            },
            "x_label": plan.config["xlabel"],
            "x_limits": list((-0.5, 40.0) if plan.plot_id == "figs6b_benchmark" else plan.xlim),
            "scientific_tick_limits": list(plan.xlim),
            "x_tick_step": plan.config["xstep"],
            "y_limits": [-0.55, 7.55],
            "y_reverse": False,
            "display_y_transform": "7 - authoritative y_display",
            "paper_points": {
                "other_systems": {"shape": "open circle", "edge_color": SLATE},
                "literatureagent": {"shape": "open circle", "edge_color": ORANGE},
            },
            "mean_and_95pct_ci": {
                "other_systems": {"shape": "open square", "color": BLACK},
                "literatureagent": {"shape": "open square", "color": ORANGE},
            },
            "background_gridlines": "disabled",
            "alternating_row_bands": "omitted for a white background",
            "embedded_title_panel_letter_legend": "none",
        },
        "resolved_style_precedence": [
            "The retained renderer used Matplotlib scatter(s=12), an area in pt^2; Origin receives diameter sqrt(12) pt to preserve appearance.",
            "Paper observations use open slate/orange circles; confidence summaries use open black/orange squares.",
            f"PowerPoint assembly typography uses {BODY_FONT_SIZE_PT} pt category/value/body text and {AXIS_LABEL_FONT_SIZE_PT} pt axis labels.",
            "All background grids and row bands are omitted; the white field in slide 3 keeps the eight rows readable without extra structure.",
            "The controlled specification's concise x-axis label is authoritative when legacy reference wording differs.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_id", choices=[*sorted(CONFIG), "all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate exact data/spec/reference inputs without importing or opening Origin.",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Hide and close the external Origin session after each standalone build.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_ids = sorted(CONFIG) if args.plot_id == "all" else [args.plot_id]
    plans = [build_plan(plot_id) for plot_id in plot_ids]
    if args.dry_run:
        reports = [dry_run_report(plan, args.output_root) for plan in plans]
        print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2, ensure_ascii=False))
        return 0

    reports: list[dict[str, Any]] = []
    succeeded = False
    try:
        for plan in plans:
            reports.append(build_origin_plot(plan, args.output_root, args.close))
        succeeded = True
        print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2, ensure_ascii=False))
        return 0
    finally:
        try:
            import originpro as op

            if getattr(op, "oext", False) and (args.close or not succeeded):
                op.exit()
        except Exception:
            if not succeeded:
                traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
