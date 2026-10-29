"""Build Figure S4 normalized-stability plots in OriginPro.

The builder preserves the source values, scenario order, boxplot statistics,
and paired agreement estimates. It uses a white background without gridlines,
chart titles, panel letters, or embedded legends. Shared legends are exported
separately. ``--dry-run`` validates the source tables and specifications using
only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "2026-10-27_controlled_repaired_rerun"
PUBLIC_SOURCE_ROOT = ROOT / "source"
PACKAGE_ROOT = PUBLIC_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "build"
STYLE_SOURCE = Path(
    os.environ.get(
        "LITERATUREAGENT_STYLE_PPTX",
        ROOT / "style_reference" / "manuscript_style_reference.pptx",
    )
)
STYLE_SOURCE_EXPECTED_SHA256 = "1351a6f1d415598ad0761151533e1ec87a964178881e7edf106898a9158a7f1d"
STYLE_SOURCE_SLIDE = 8

FONT = "Times New Roman"
BODY_FONT_SIZE_PT = 12
AXIS_LABEL_FONT_SIZE_PT = 13
POWERPOINT_ASSEMBLY_WIDTH_IN = 6.5
BLUE = "#0072B2"
TEAL = "#009E73"
NEUTRAL = "#4B5563"
INK = "#202124"
AXIS_BLACK = "#000000"
WHITE = "#FFFFFF"

SCENARIO_ORDER = (
    "Ea 0.30 eV",
    "Ea 0.45 eV",
    "Baseline",
    "Ea 0.75 eV",
    "Ea 0.90 eV",
    "RH exponent 0.5",
    "RH exponent 2.0",
    "Light exponent 0.5",
    "Light exponent 1.0",
)

PLOT_CONFIG: dict[str, dict[str, Any]] = {
    "figs4a_normalized_stability": {
        "stem": "figs4a_normalized_stability",
        "graph": "FigS4A",
        "family": "boxplot",
        "data": "data/figs4a_normalized_stability/plot_data.csv",
        "spec": "specifications/figs4a_normalized_stability.json",
        "reference_dir": "references/current_individual_exports/figs4a_normalized_stability",
        "size_mm": (180.0, 85.0),
        "xlim": (-0.30, 5.60),
        "xstep": 1.0,
        "xlabel": r"log\-(10) condition-normalized \i(T)\-(80) (h)",
        "expected_rows": 6075,
        "helper_name": "origin_box_summary.csv",
    },
    "figs4b_normalization_sensitivity": {
        "stem": "figs4b_normalization_sensitivity",
        "graph": "FigS4B",
        "family": "paired",
        "data": "data/figs4b_normalization_sensitivity/plot_data.csv",
        "spec": "specifications/figs4b_normalization_sensitivity.json",
        "reference_dir": "references/current_individual_exports/figs4b_normalization_sensitivity",
        # The controlled spec records the original 180 x 78 mm asset.  The
        # formatting-only export uses the same 85 mm height as S4a so both
        # panels assemble at the equal height shown in the manuscript deck.
        "size_mm": (180.0, 85.0),
        "spec_size_mm": (180.0, 78.0),
        "xlim": (0.87, 1.01),
        "xstep": 0.02,
        "xlabel": "Agreement with baseline scenario",
        "expected_rows": 9,
        "helper_name": "origin_pair_helper.csv",
        "reference": 1.0,
    },
}


@dataclass(frozen=True)
class ValidationPlan:
    plot_id: str
    config: dict[str, Any]
    spec: dict[str, Any]
    headers: tuple[str, ...]
    source_rows: tuple[dict[str, str], ...]
    helper_headers: tuple[str, ...]
    helper_rows: tuple[dict[str, Any], ...]
    source_sha256: str
    spec_sha256: str
    reference_assets: tuple[dict[str, Any], ...]
    validation: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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


def typography_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Document native and expected PowerPoint-assembly typography."""
    native_width_in = float(config["size_mm"][0]) / 25.4
    placement_scale = POWERPOINT_ASSEMBLY_WIDTH_IN / native_width_in
    font_sizes = {
        "tick_and_category": BODY_FONT_SIZE_PT,
        "value_and_annotation": BODY_FONT_SIZE_PT,
        "axis_label": AXIS_LABEL_FONT_SIZE_PT,
    }
    return {
        "font_sizes_pt": font_sizes,
        "powerpoint_assembly": {
            "target_width_in": POWERPOINT_ASSEMBLY_WIDTH_IN,
            "native_width_in": round(native_width_in, 6),
            "placement_scale": round(placement_scale, 6),
            "effective_font_sizes_pt": {
                name: round(size * placement_scale, 2)
                for name, size in font_sizes.items()
            },
        },
    }


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        if not headers or len(headers) != len(set(headers)):
            raise RuntimeError(f"CSV headers must be present and unique: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV contains no data rows: {path}")
    return headers, rows


def finite_float(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Non-numeric {field!r} at data row {row_number}: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise RuntimeError(
            f"Non-finite {field!r} at data row {row_number}: {value!r}"
        )
    return result


def quantile_type7(sorted_values: Sequence[float], probability: float) -> float:
    """Return the linear/type-7 sample quantile used by NumPy/Matplotlib."""
    if not sorted_values:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(probability)
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def scenario_origin_label(scenario: str) -> str:
    if scenario.startswith("Ea "):
        value = scenario.removeprefix("Ea ")
        return rf"\i(E)\-(a) = {value}"
    if scenario.startswith("RH exponent "):
        return "RH exponent = " + scenario.removeprefix("RH exponent ")
    if scenario.startswith("Light exponent "):
        return "Light exponent = " + scenario.removeprefix("Light exponent ")
    return scenario


def build_box_helper(rows: Sequence[dict[str, str]]) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    required = {"scenario", "log10_T80_ref_h"}
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("S4a source is missing scenario or log10_T80_ref_h")
    grouped: dict[str, list[float]] = {scenario: [] for scenario in SCENARIO_ORDER}
    for index, row in enumerate(rows, start=2):
        scenario = row["scenario"]
        if scenario not in grouped:
            raise RuntimeError(f"Unexpected S4a scenario at row {index}: {scenario!r}")
        grouped[scenario].append(
            finite_float(row["log10_T80_ref_h"], "log10_T80_ref_h", index)
        )

    counts = {scenario: len(values) for scenario, values in grouped.items()}
    if set(grouped) != set(SCENARIO_ORDER) or any(count != 675 for count in counts.values()):
        raise RuntimeError(f"S4a expected 675 values in each of nine scenarios; found {counts}")

    headers = [
        "scenario",
        "scenario_origin_label",
        "category_order",
        "y_plot",
        "n",
        "q1_type7",
        "median_type7",
        "q3_type7",
        "iqr",
        "lower_fence",
        "upper_fence",
        "whisker_low",
        "whisker_high",
        "outlier_count_hidden",
        "fill_hex",
        "fill_opacity",
    ]
    helper_rows: list[dict[str, Any]] = []
    total_outliers = 0
    for order, scenario in enumerate(SCENARIO_ORDER, start=1):
        values = sorted(grouped[scenario])
        q1 = quantile_type7(values, 0.25)
        median = quantile_type7(values, 0.50)
        q3 = quantile_type7(values, 0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        retained = [value for value in values if lower_fence <= value <= upper_fence]
        whisker_low = retained[0]
        whisker_high = retained[-1]
        outlier_count = len(values) - len(retained)
        total_outliers += outlier_count
        helper_rows.append(
            {
                "scenario": scenario,
                "scenario_origin_label": scenario_origin_label(scenario),
                "category_order": order,
                "y_plot": len(SCENARIO_ORDER) - order + 1,
                "n": len(values),
                "q1_type7": format(q1, ".17g"),
                "median_type7": format(median, ".17g"),
                "q3_type7": format(q3, ".17g"),
                "iqr": format(iqr, ".17g"),
                "lower_fence": format(lower_fence, ".17g"),
                "upper_fence": format(upper_fence, ".17g"),
                "whisker_low": format(whisker_low, ".17g"),
                "whisker_high": format(whisker_high, ".17g"),
                "outlier_count_hidden": outlier_count,
                "fill_hex": TEAL if scenario == "Baseline" else BLUE,
                "fill_opacity": 0.82,
            }
        )
    audit = {
        "scenarios": len(helper_rows),
        "underlying_values": sum(counts.values()),
        "values_per_scenario": counts,
        "quartile_method": "linear/type-7: h=(n-1)p",
        "whisker_method": "most extreme observations within Q1-1.5*IQR and Q3+1.5*IQR",
        "showfliers": False,
        "hidden_outlier_count": total_outliers,
    }
    return headers, helper_rows, audit


def build_pair_helper(rows: Sequence[dict[str, str]]) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    required = {"scenario", "spearman_vs_baseline", "top_quartile_overlap_vs_baseline"}
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("S4b source is missing one or more required columns")
    by_scenario: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        scenario = row["scenario"]
        if scenario not in SCENARIO_ORDER:
            raise RuntimeError(f"Unexpected S4b scenario at row {index}: {scenario!r}")
        if scenario in by_scenario:
            raise RuntimeError(f"Duplicate S4b scenario: {scenario}")
        by_scenario[scenario] = row
    if set(by_scenario) != set(SCENARIO_ORDER):
        raise RuntimeError(
            f"S4b scenario set mismatch: {sorted(by_scenario)}"
        )

    headers = [
        "scenario",
        "scenario_origin_label",
        "category_order",
        "y_plot",
        "spearman_vs_baseline",
        "top_quartile_overlap_vs_baseline",
        "connector_low",
        "connector_high",
    ]
    helper_rows: list[dict[str, Any]] = []
    for order, scenario in enumerate(SCENARIO_ORDER, start=1):
        source = by_scenario[scenario]
        spearman = finite_float(source["spearman_vs_baseline"], "spearman_vs_baseline", order + 1)
        overlap = finite_float(
            source["top_quartile_overlap_vs_baseline"],
            "top_quartile_overlap_vs_baseline",
            order + 1,
        )
        if not 0.0 <= spearman <= 1.0 or not 0.0 <= overlap <= 1.0:
            raise RuntimeError(
                f"S4b agreement values must be in [0,1]: {scenario}: {spearman}, {overlap}"
            )
        helper_rows.append(
            {
                "scenario": scenario,
                "scenario_origin_label": scenario_origin_label(scenario),
                "category_order": order,
                "y_plot": len(SCENARIO_ORDER) - order + 1,
                "spearman_vs_baseline": format(spearman, ".17g"),
                "top_quartile_overlap_vs_baseline": format(overlap, ".17g"),
                "connector_low": format(min(spearman, overlap), ".17g"),
                "connector_high": format(max(spearman, overlap), ".17g"),
            }
        )
    audit = {
        "scenarios": len(helper_rows),
        "point_estimates": len(helper_rows) * 2,
        "paired_connectors": len(helper_rows),
        "reference_line": 1.0,
        "reference_line_style": "dashed",
    }
    return headers, helper_rows, audit


def collect_reference_assets(plot_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    reference_dir = PACKAGE_ROOT / config["reference_dir"]
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    references: list[dict[str, Any]] = []
    for suffix in (".png", ".svg", ".pdf"):
        matches = sorted(reference_dir.glob(f"*{suffix}"))
        if len(matches) != 1:
            raise RuntimeError(
                f"{plot_id}: expected one retained {suffix} reference, found {matches}"
            )
        path = matches[0]
        references.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return references


def validate_plot(plot_id: str) -> ValidationPlan:
    config = PLOT_CONFIG[plot_id]
    source_path = PACKAGE_ROOT / config["data"]
    spec_path = PACKAGE_ROOT / config["spec"]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    headers, source_rows = read_csv_rows(source_path)
    if len(source_rows) != int(config["expected_rows"]):
        raise RuntimeError(
            f"{plot_id}: expected {config['expected_rows']:,} rows, found {len(source_rows):,}"
        )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("plot_id") != plot_id or spec.get("status") != "READY":
        raise RuntimeError(f"Unexpected specification identity/status for {plot_id}")
    if tuple(spec.get("category_order", [])) != SCENARIO_ORDER:
        raise RuntimeError(f"Specification category order mismatch for {plot_id}")
    spec_size = spec["style"]["standalone_dimensions_mm"]
    expected_spec_size = tuple(config.get("spec_size_mm", config["size_mm"]))
    if (float(spec_size["width"]), float(spec_size["height"])) != expected_spec_size:
        raise RuntimeError(f"Specification dimensions mismatch for {plot_id}")

    if config["family"] == "boxplot":
        helper_headers, helper_rows, validation = build_box_helper(source_rows)
        expected = spec["expected_plotted_counts"]
        if validation["scenarios"] != int(expected["scenarios"]):
            raise RuntimeError("S4a scenario count does not match specification")
        if validation["underlying_values"] != int(expected["underlying_values"]):
            raise RuntimeError("S4a underlying value count does not match specification")
    else:
        helper_headers, helper_rows, validation = build_pair_helper(source_rows)
        expected = spec["expected_plotted_counts"]
        if validation["scenarios"] != int(expected["scenarios"]):
            raise RuntimeError("S4b scenario count does not match specification")
        if validation["point_estimates"] != int(expected["point_estimates"]):
            raise RuntimeError("S4b point count does not match specification")

    validation.update(
        {
            "source_rows": len(source_rows),
            "unique_headers": len(headers) == len(set(headers)),
            "gridlines": "disabled on x major/minor and y major/minor",
            "font": FONT,
            "typography": typography_metadata(config),
            "axis_color": AXIS_BLACK,
            "embedded_legend": False,
            "legend_strategy": "separate shared legend during multipanel assembly",
            "style_source": style_source_path_text(),
            "style_source_slide": STYLE_SOURCE_SLIDE,
            "style_source_sha256": style_source_sha256(),
            "dimensions_mm": list(config["size_mm"]),
            "formatting_only_override": (
                "Retained numerical source; publication styling is applied by this "
                "presentation-only grid and typography rebuild without numerical changes."
            ),
        }
    )
    return ValidationPlan(
        plot_id=plot_id,
        config=config,
        spec=spec,
        headers=tuple(headers),
        source_rows=tuple(source_rows),
        helper_headers=tuple(helper_headers),
        helper_rows=tuple(helper_rows),
        source_sha256=sha256(source_path),
        spec_sha256=sha256(spec_path),
        reference_assets=tuple(collect_reference_assets(plot_id, config)),
        validation=validation,
    )


def dry_run_report(plan: ValidationPlan) -> dict[str, Any]:
    config = plan.config
    return {
        "status": "DRY_RUN_PASS",
        "origin_started": False,
        "plot_id": plan.plot_id,
        "source": str((PACKAGE_ROOT / config["data"]).resolve()),
        "source_sha256": plan.source_sha256,
        "source_rows": len(plan.source_rows),
        "specification": str((PACKAGE_ROOT / config["spec"]).resolve()),
        "specification_sha256": plan.spec_sha256,
        "style_source": {
            "path": style_source_path_text(),
            "slide": STYLE_SOURCE_SLIDE,
            "sha256": plan.validation["style_source_sha256"],
        },
        "reference_assets": list(plan.reference_assets),
        "planned_outputs": {
            "opju": f"origin_projects/{plan.plot_id}.opju",
            "pdf": f"pdf/{plan.plot_id}.pdf",
            "svg": f"svg/{plan.plot_id}.svg",
            "png": f"png/{plan.plot_id}.png",
            "audit": f"validation/{plan.plot_id}_build_audit.json",
        },
        "validation": plan.validation,
        "scientific_marks": (
            "nine horizontal Tukey box-and-whisker distributions; hidden fliers"
            if config["family"] == "boxplot"
            else "nine paired connectors, 18 estimates, and the dashed x=1 reference"
        ),
        "style": {
            "font": FONT,
            "typography": typography_metadata(config),
            "palette": {
                "blue": BLUE,
                "teal": TEAL,
                "neutral": NEUTRAL,
                "ink": INK,
                "axes": AXIS_BLACK,
            },
            "background": "white",
            "gridlines": "none",
            "title": None,
            "panel_letter": None,
            "embedded_legend": False,
            "legend_strategy": "separate shared legend during multipanel assembly",
            "standalone_dimensions_mm": list(config["size_mm"]),
        },
    }


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != source_hash:
            raise RuntimeError(
                f"Refusing to overwrite non-identical existing file: {destination}"
            )
        return
    shutil.copy2(source, destination)
    if sha256(destination) != source_hash:
        raise RuntimeError(f"Byte-exact copy verification failed: {destination}")


def write_csv_atomic(path: Path, headers: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    if not headers or len(headers) != len(set(headers)):
        raise RuntimeError(f"Generated CSV headers are not unique: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(headers), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def prepare_output_inputs(plan: ValidationPlan, output_root: Path) -> dict[str, Any]:
    config = plan.config
    data_dir = output_root / "data" / plan.plot_id
    source_path = PACKAGE_ROOT / config["data"]
    spec_path = PACKAGE_ROOT / config["spec"]
    source_copy = data_dir / "plot_data.csv"
    spec_copy = data_dir / "plot_specification.json"
    helper_path = data_dir / config["helper_name"]
    copy_exact(source_path, source_copy)
    copy_exact(spec_path, spec_copy)
    write_csv_atomic(helper_path, plan.helper_headers, plan.helper_rows)

    retained_dir = output_root / "retained_reference_assets" / plan.plot_id
    reference_copies: list[dict[str, Any]] = []
    for item in plan.reference_assets:
        source = Path(item["path"])
        destination = retained_dir / source.name
        copy_exact(source, destination)
        reference_copies.append(
            {
                "path": str(destination.resolve()),
                "sha256": sha256(destination),
                "byte_exact": sha256(destination) == item["sha256"],
            }
        )
    return {
        "source_copy": source_copy,
        "spec_copy": spec_copy,
        "helper_path": helper_path,
        "reference_copies": reference_copies,
    }


def append_xy(base: Any, table: Any, prefix: str, xs: Iterable[Any], ys: Iterable[Any]) -> tuple[int, int]:
    x_values = list(xs)
    y_values = list(ys)
    if len(x_values) != len(y_values):
        raise RuntimeError(f"Mismatched linked series lengths: {prefix}")
    return base.append_xy(table, prefix, x_values, y_values)


def add_linked_line(
    base: Any,
    layer: Any,
    table: Any,
    prefix: str,
    xs: Iterable[Any],
    ys: Iterable[Any],
    color: str,
    width: float,
    *,
    transparency: int = 0,
    flat_cap: bool = False,
) -> Any:
    x_column, y_column = append_xy(base, table, prefix, xs, ys)
    plot = layer.add_plot(table.sheet, coly=y_column, colx=x_column, type="l")
    if plot is None:
        raise RuntimeError(f"Origin failed to add linked series: {prefix}")
    plot.color = color
    plot.set_float("line.width", float(width))
    if transparency:
        try:
            plot.transparency = int(transparency)
        except Exception:
            pass
    if flat_cap:
        try:
            plot.set_int("line.cap", 0)
        except Exception:
            pass
    return plot


def add_marker_series(
    op: Any,
    base: Any,
    layer: Any,
    table: Any,
    prefix: str,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    color: str,
    symbol_kind: int,
    size: float,
    edge_color: str = WHITE,
    fill_color: str | None = None,
    transparency: int = 0,
) -> tuple[Any, int]:
    x_column, y_column = append_xy(base, table, prefix, xs, ys)
    plot = layer.add_plot(table.sheet, coly=y_column, colx=x_column, type="s")
    if plot is None:
        raise RuntimeError(f"Origin failed to add marker series: {prefix}")
    plot.color = color
    plot.symbol_kind = symbol_kind
    plot.symbol_size = size
    plot.symbol_interior = 0
    plot.set_int("symbol.fillcolor", op.ocolor(fill_color or color))
    plot.set_int("symbol.edgecolor", op.ocolor(edge_color))
    if transparency:
        try:
            plot.transparency = int(transparency)
        except Exception:
            pass
    return plot, len(layer.plot_list())


def disable_all_grids(layer: Any) -> None:
    layer.lt_exec(
        "layer.x.grid.major=0; layer.x.grid.minor=0; "
        "layer.y.grid.major=0; layer.y.grid.minor=0; "
        "layer.x.grid.majorcolor=0; layer.x.grid.minorcolor=0; "
        "layer.y.grid.majorcolor=0; layer.y.grid.minorcolor=0; "
        "layer.sauto=0; layer.maxpts=0; page.speedmode=0;"
    )


def style_axes(base: Any, layer: Any, config: dict[str, Any], ylim: tuple[float, float]) -> None:
    base.style_axis(
        layer,
        "",
        xlim=tuple(config["xlim"]),
        xstep=float(config["xstep"]),
        ylim=ylim,
        ystep=1.0,
        show_y_labels=True,
        axis_title_size=AXIS_LABEL_FONT_SIZE_PT,
        tick_size=BODY_FONT_SIZE_PT,
    )
    layer.axis("x").title = ""
    layer.lt_exec(
        "layer.x2.show=0; layer.y2.show=0; "
        f'layer.x.color=color("{AXIS_BLACK}"); layer.y.color=color("{AXIS_BLACK}"); '
        f'layer.x.label.color=color("{AXIS_BLACK}"); '
        f'layer.y.label.color=color("{AXIS_BLACK}");'
    )
    disable_all_grids(layer)


def add_manual_x_title(base: Any, layer: Any, text: str, y_percent: float) -> None:
    label = base.add_percent_label(
        layer,
        text,
        50,
        y_percent,
        size=AXIS_LABEL_FONT_SIZE_PT,
        color=AXIS_BLACK,
    )
    label.set_int("anchor", 5)


def build_s4a_graph(
    op: Any,
    base: Any,
    helper_table: Any,
    plan: ValidationPlan,
) -> tuple[Any, dict[str, Any]]:
    config = plan.config
    width_mm, height_mm = config["size_mm"]
    graph = base.new_graph(
        config["graph"], plan.plot_id, width_mm / 25.4, height_mm / 25.4
    )
    layer = graph[0]
    # Reserve additional left and lower-page space for the normalized 12/13
    # pt assembly typography while retaining the same 180 x 85 mm canvas.
    base.position_layer(layer, 27, 5, 70, 73)
    ylim = (0.45, 9.55)

    whisker_x: list[Any] = []
    whisker_y: list[Any] = []
    cap_x: list[Any] = []
    cap_y: list[Any] = []
    blue_fill_x: list[Any] = []
    blue_fill_y: list[Any] = []
    teal_fill_x: list[Any] = []
    teal_fill_y: list[Any] = []
    blue_cap_x: list[float] = []
    blue_cap_y: list[float] = []
    teal_cap_x: list[float] = []
    teal_cap_y: list[float] = []
    border_x: list[Any] = []
    border_y: list[Any] = []
    median_x: list[Any] = []
    median_y: list[Any] = []
    box_half_height = 0.29
    cap_half_height = 0.17
    fill_width_pt = 12.0
    # Origin 2024 renders thick line ends as round even when line.cap is set
    # through the Python bridge.  Shorten each fill segment by one physical
    # half-width and cover the ends with same-size square symbols.  The black
    # outline is then drawn last, yielding a true rectangular box with no
    # colored "pill" protruding beyond Q1/Q3.
    layer_width_mm = float(width_mm) * 0.70
    fill_half_width_data = (
        (fill_width_pt * 25.4 / 72.0 / 2.0)
        / layer_width_mm
        * (float(config["xlim"][1]) - float(config["xlim"][0]))
    )

    for row in plan.helper_rows:
        y = float(row["y_plot"])
        q1 = float(row["q1_type7"])
        median = float(row["median_type7"])
        q3 = float(row["q3_type7"])
        whisker_low = float(row["whisker_low"])
        whisker_high = float(row["whisker_high"])
        whisker_x.extend([whisker_low, whisker_high, None])
        whisker_y.extend([y, y, None])
        cap_x.extend(
            [whisker_low, whisker_low, None, whisker_high, whisker_high, None]
        )
        cap_y.extend(
            [
                y - cap_half_height,
                y + cap_half_height,
                None,
                y - cap_half_height,
                y + cap_half_height,
                None,
            ]
        )
        fill_x, fill_y, cap_marker_x, cap_marker_y = (
            (teal_fill_x, teal_fill_y, teal_cap_x, teal_cap_y)
            if row["scenario"] == "Baseline"
            else (blue_fill_x, blue_fill_y, blue_cap_x, blue_cap_y)
        )
        fill_left = q1 + fill_half_width_data
        fill_right = q3 - fill_half_width_data
        fill_x.extend([fill_left, fill_right, None])
        fill_y.extend([y, y, None])
        cap_marker_x.extend([fill_left, fill_right])
        cap_marker_y.extend([y, y])
        border_x.extend([q1, q3, q3, q1, q1, None])
        border_y.extend(
            [
                y - box_half_height,
                y - box_half_height,
                y + box_half_height,
                y + box_half_height,
                y - box_half_height,
                None,
            ]
        )
        median_x.extend([median, median, None])
        median_y.extend([y - box_half_height, y + box_half_height, None])

    add_linked_line(
        base, layer, helper_table, "whiskers", whisker_x, whisker_y, AXIS_BLACK, 1.1
    )
    add_linked_line(
        base, layer, helper_table, "whisker_caps", cap_x, cap_y, AXIS_BLACK, 1.1
    )
    add_linked_line(
        base,
        layer,
        helper_table,
        "blue_box_fill",
        blue_fill_x,
        blue_fill_y,
        BLUE,
        fill_width_pt,
        transparency=0,
        flat_cap=True,
    )
    add_linked_line(
        base,
        layer,
        helper_table,
        "baseline_box_fill",
        teal_fill_x,
        teal_fill_y,
        TEAL,
        fill_width_pt,
        transparency=0,
        flat_cap=True,
    )
    add_marker_series(
        op,
        base,
        layer,
        helper_table,
        "blue_box_square_ends",
        blue_cap_x,
        blue_cap_y,
        color=BLUE,
        symbol_kind=1,
        size=fill_width_pt,
        edge_color=BLUE,
        fill_color=BLUE,
        transparency=0,
    )
    add_marker_series(
        op,
        base,
        layer,
        helper_table,
        "baseline_box_square_ends",
        teal_cap_x,
        teal_cap_y,
        color=TEAL,
        symbol_kind=1,
        size=fill_width_pt,
        edge_color=TEAL,
        fill_color=TEAL,
        transparency=0,
    )
    add_linked_line(
        base, layer, helper_table, "box_borders", border_x, border_y, AXIS_BLACK, 0.8
    )
    add_linked_line(
        base, layer, helper_table, "medians", median_x, median_y, AXIS_BLACK, 1.2
    )

    style_axes(base, layer, config, ylim)
    labels = [str(row["scenario_origin_label"]) for row in plan.helper_rows]
    positions = [float(row["y_plot"]) for row in plan.helper_rows]
    base.set_categorical_ticks(layer, helper_table, labels, plan.plot_id, positions)
    add_manual_x_title(base, layer, str(config["xlabel"]), 119)
    disable_all_grids(layer)
    layer.lt_exec("label -r legend; label -r title;")
    return graph, {
        **plan.validation,
        "boxplots": 9,
        "data_linkage": "raw source worksheet plus linked summary geometry worksheet",
        "fill_opacity": 1.0,
        "box_outline_color": AXIS_BLACK,
        "embedded_legend": False,
        "legend_strategy": "separate shared legend during multipanel assembly",
        "xlim": list(config["xlim"]),
        "ylim": list(ylim),
        "scientific_reference_lines": [],
    }


def build_s4b_graph(
    op: Any,
    base: Any,
    helper_table: Any,
    plan: ValidationPlan,
) -> tuple[Any, dict[str, Any]]:
    config = plan.config
    width_mm, height_mm = config["size_mm"]
    graph = base.new_graph(
        config["graph"], plan.plot_id, width_mm / 25.4, height_mm / 25.4
    )
    layer = graph[0]
    # Match S4a's plotting geometry so both panels align when assembled.
    base.position_layer(layer, 27, 5, 70, 73)
    ylim = (0.45, 9.55)

    connector_x: list[Any] = []
    connector_y: list[Any] = []
    spearman_x: list[float] = []
    overlap_x: list[float] = []
    y_values: list[float] = []
    for row in plan.helper_rows:
        y = float(row["y_plot"])
        low = float(row["connector_low"])
        high = float(row["connector_high"])
        connector_x.extend([low, high, None])
        connector_y.extend([y, y, None])
        spearman_x.append(float(row["spearman_vs_baseline"]))
        overlap_x.append(float(row["top_quartile_overlap_vs_baseline"]))
        y_values.append(y)

    add_linked_line(
        base,
        layer,
        helper_table,
        "paired_connectors",
        connector_x,
        connector_y,
        NEUTRAL,
        1.1,
        flat_cap=True,
    )
    _, spearman_index = add_marker_series(
        op,
        base,
        layer,
        helper_table,
        "spearman_points",
        spearman_x,
        y_values,
        color=BLUE,
        symbol_kind=1,
        size=5.5,
    )
    _, overlap_index = add_marker_series(
        op,
        base,
        layer,
        helper_table,
        "overlap_points",
        overlap_x,
        y_values,
        color=TEAL,
        symbol_kind=2,
        size=5.5,
    )

    # This is the sole vertical line retained: a scientific agreement target,
    # not a decorative gridline.
    base.vertical_rule(
        layer,
        float(config["reference"]),
        ylim[0],
        ylim[1],
        color=NEUTRAL,
        width=0.8,
        line_type=2,
    )
    style_axes(base, layer, config, ylim)
    layer.lt_exec("layer.x.label.nDigits=2; layer.x.label.decimalPlaces=2;")
    labels = [str(row["scenario_origin_label"]) for row in plan.helper_rows]
    positions = [float(row["y_plot"]) for row in plan.helper_rows]
    base.set_categorical_ticks(layer, helper_table, labels, plan.plot_id, positions)
    add_manual_x_title(base, layer, str(config["xlabel"]), 119)
    disable_all_grids(layer)
    layer.lt_exec("label -r legend; label -r title;")
    return graph, {
        **plan.validation,
        "data_linkage": "exact source worksheet plus linked paired-geometry worksheet",
        "series": [
            {
                "label": "Spearman rank correlation",
                "color": BLUE,
                "marker": "square",
                "plot_index": spearman_index,
            },
            {
                "label": "Top-quartile overlap",
                "color": TEAL,
                "marker": "circle",
                "plot_index": overlap_index,
            },
        ],
        "connector_color": NEUTRAL,
        "embedded_legend": False,
        "legend_strategy": "separate shared legend during multipanel assembly",
        "xlim": list(config["xlim"]),
        "ylim": list(ylim),
        "scientific_reference_lines": [
            {"axis": "x", "value": 1.0, "style": "dashed", "purpose": "perfect agreement"}
        ],
    }


def write_validation_sheet(
    book: Any,
    plan: ValidationPlan,
    prepared: dict[str, Any],
    graph_audit: dict[str, Any],
) -> None:
    sheet = book.add_sheet()
    sheet.name = "Validation"
    sheet.lname = f"{plan.plot_id} build validation"
    checks = [
        ("plot_id", plan.plot_id, "PASS"),
        ("source_sha256", plan.source_sha256, "PASS"),
        ("specification_sha256", plan.spec_sha256, "PASS"),
        ("source_rows", len(plan.source_rows), "PASS"),
        ("expected_rows", plan.config["expected_rows"], "PASS"),
        ("standalone_dimensions_mm", json.dumps(plan.config["size_mm"]), "PASS"),
        ("font", FONT, "PASS"),
        (
            "font_sizes_pt",
            json.dumps(typography_metadata(plan.config)["font_sizes_pt"]),
            "PASS",
        ),
        (
            "powerpoint_assembly_typography",
            json.dumps(typography_metadata(plan.config)["powerpoint_assembly"]),
            "PASS",
        ),
        ("axis_color", AXIS_BLACK, "PASS"),
        ("style_source", style_source_path_text(), "PASS"),
        ("style_source_slide", STYLE_SOURCE_SLIDE, "PASS"),
        ("x_major_grid", "disabled", "PASS"),
        ("x_minor_grid", "disabled", "PASS"),
        ("y_major_grid", "disabled", "PASS"),
        ("y_minor_grid", "disabled", "PASS"),
        ("panel_letter", "absent", "PASS"),
        ("chart_title", "absent", "PASS"),
        ("embedded_legend", "absent; shared legend assembled separately", "PASS"),
        ("reference_asset_copies", json.dumps(prepared["reference_copies"]), "PASS"),
        ("graph_audit", json.dumps(graph_audit, ensure_ascii=False), "PASS"),
        ("visual_review", "PENDING after PNG export", "PENDING"),
    ]
    for column, header in enumerate(("check", "value", "status")):
        sheet.from_list(column, [row[column] for row in checks], lname=header, axis="")


def output_file_audit(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected output is missing or empty: {path}")
        result[path.name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


def build_origin_plot(
    plan: ValidationPlan,
    output_root: Path,
    *,
    hidden: bool,
    close: bool,
) -> dict[str, Any]:
    # Origin imports are deliberately deferred so --dry-run cannot launch or
    # initialize the application bridge.
    import originpro as op  # type: ignore
    import build_manuscript_plot as common  # type: ignore
    import build_literatureagent_origin_figures as base  # type: ignore
    common.op = base.require_origin()

    output_root = output_root.resolve()
    for name in (
        "origin_projects",
        "pdf",
        "svg",
        "png",
        "data",
        "validation",
        "retained_reference_assets",
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    prepared = prepare_output_inputs(plan, output_root)
    project_path = output_root / "origin_projects" / f"{plan.plot_id}.opju"
    audit_path = output_root / "validation" / f"{plan.plot_id}_build_audit.json"
    succeeded = False
    try:
        if getattr(op, "oext", False):
            op.set_show(not hidden)
        op.new(asksave=False)
        base.FONT_INDEX = int(op.lt_int(f'font("{FONT}")'))
        book = op.new_book("w", lname=f"{plan.plot_id} exact data and linked geometry")
        if book is None:
            raise RuntimeError("Origin could not create the source workbook")
        book.name = "PlotData"
        _, source_table = common.import_csv_sheet(
            book,
            prepared["source_copy"],
            "Source",
            "Exact authoritative source data",
            first=True,
        )
        _, helper_table = common.import_csv_sheet(
            book,
            prepared["helper_path"],
            "Geometry",
            "Deterministic linked plotting geometry",
        )
        if plan.config["family"] == "boxplot":
            graph, graph_audit = build_s4a_graph(op, base, helper_table, plan)
        else:
            graph, graph_audit = build_s4b_graph(op, base, helper_table, plan)
        write_validation_sheet(book, plan, prepared, graph_audit)
        graph_pages = op.graph_list("p")
        if len(graph_pages) != 1 or graph_pages[0].name != plan.config["graph"]:
            raise RuntimeError(
                f"Expected one graph page {plan.config['graph']}; found "
                f"{[page.name for page in graph_pages]}"
            )
        if not op.save(str(project_path.resolve())):
            raise RuntimeError(f"Origin failed to save project: {project_path}")
        exports = common.export_graph(graph, plan.plot_id, plan.config, output_root)
        output_paths = [
            project_path,
            output_root / "pdf" / f"{plan.plot_id}.pdf",
            output_root / "svg" / f"{plan.plot_id}.svg",
            output_root / "png" / f"{plan.plot_id}.png",
        ]
        report = {
            "status": "PASS_PENDING_VISUAL_REVIEW",
            "analysis_id": ANALYSIS_ID,
            "plot_id": plan.plot_id,
            "formatting_only_revision": True,
            "origin_project": str(project_path.resolve()),
            "origin_graph_page": plan.config["graph"],
            "graph_pages": 1,
            "source_copy": str(prepared["source_copy"].resolve()),
            "source_sha256": sha256(prepared["source_copy"]),
            "source_matches_source": sha256(prepared["source_copy"]) == plan.source_sha256,
            "specification_copy": str(prepared["spec_copy"].resolve()),
            "specification_sha256": sha256(prepared["spec_copy"]),
            "helper_csv": str(prepared["helper_path"].resolve()),
            "helper_sha256": sha256(prepared["helper_path"]),
            "reference_assets": prepared["reference_copies"],
            "page_dimensions_mm": list(plan.config["size_mm"]),
            "font": FONT,
            "typography": typography_metadata(plan.config),
            "style_source": {
                "path": style_source_path_text(),
                "slide": STYLE_SOURCE_SLIDE,
                "sha256": plan.validation["style_source_sha256"],
            },
            "axis_color": AXIS_BLACK,
            "embedded_legend": False,
            "gridlines": {
                "x_major": False,
                "x_minor": False,
                "y_major": False,
                "y_minor": False,
            },
            "scientific_reference_lines": graph_audit["scientific_reference_lines"],
            "graph_audit": graph_audit,
            "exports": exports,
            "output_files": output_file_audit(output_paths),
            "visual_review": {
                "status": "PENDING",
                "required_checks": [
                    "no visible background gridlines",
                    "S4b dashed x=1 scientific reference retained",
                    "all box/whisker or paired marks present",
                    "Times New Roman and final-size readability",
                    "no clipping, title, panel letter, embedded legend, or outer frame",
                ],
            },
        }
        write_json_atomic(audit_path, report)
        succeeded = True
        return report
    finally:
        if getattr(op, "oext", False) and (close or not succeeded):
            try:
                op.exit()
            except Exception:
                pass


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_id", choices=[*sorted(PLOT_CONFIG), "all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate exact inputs and print the build plan without importing or starting Origin.",
    )
    parser.add_argument(
        "--hidden",
        action="store_true",
        help="Keep Origin hidden during unattended build/export.",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close the external Origin process after saving and exporting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    plot_ids = sorted(PLOT_CONFIG) if args.plot_id == "all" else [args.plot_id]
    plans = [validate_plot(plot_id) for plot_id in plot_ids]
    if args.dry_run:
        reports = [dry_run_report(plan) for plan in plans]
        payload: dict[str, Any]
        if len(reports) == 1:
            payload = reports[0]
        else:
            payload = {"status": "PASS", "mode": "dry-run", "reports": reports}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    reports = [
        build_origin_plot(
            plan,
            args.output_root,
            hidden=args.hidden,
            close=args.close,
        )
        for plan in plans
    ]
    payload = reports[0] if len(reports) == 1 else {"status": "PASS", "reports": reports}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
