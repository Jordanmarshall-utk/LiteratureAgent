#!/usr/bin/env python3
"""Build standalone OriginPro Figure 2a/2b panels without background grids.

Source tables and specifications are read from ``source/``. Each invocation
creates one independent OPJU containing the exact source tables, a linked
display-helper worksheet, a single editable graph page, and a validation
worksheet.  Publication exports are written with plot-ID filenames.

``--dry-run`` performs all source/spec/reference validation without importing
``originpro`` or opening Origin.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "2026-10-27_controlled_repaired_rerun"
PUBLIC_SOURCE_ROOT = ROOT / "source"
PACKAGE = PUBLIC_SOURCE_ROOT
DEFAULT_OUTPUT = ROOT / "build"
POWERPOINT_STYLE_SOURCE = Path(
    os.environ.get(
        "LITERATUREAGENT_STYLE_PPTX",
        ROOT / "style_reference" / "manuscript_style_reference.pptx",
    )
)
POWERPOINT_STYLE_SHA256 = "1351a6f1d415598ad0761151533e1ec87a964178881e7edf106898a9158a7f1d"
POWERPOINT_ERA_BUILDER = ROOT / "scripts" / "build_literatureagent_origin_figures.py"

FONT = "Times New Roman"
# Manuscript palette shared with the Origin drawing helpers.
INK = "#080808"
SLATE = "#33546E"
ORANGE = "#F28A00"
ORANGE_2 = "#FF9F25"
ACCEPTED = "#137321"
SPARSE = "#AB4B00"
REJECTED = "#EC0611"
PALE_RULE = "#B2BCC9"
WHITE = "#FFFFFF"

# Whole-number publication typography.  Figure 2 is assembled with other
# standalone panels in PowerPoint, so these source sizes must remain legible
# after the panel is reduced in the composite.
BODY_FONT_PT = 12
AXIS_LABEL_FONT_PT = 13

# Manual display-only wrapping keeps the larger 12 pt category labels inside
# the 85 mm panel without changing any authoritative category text in the
# linked worksheet.
CATEGORY_DISPLAY_TEXT: dict[str, dict[str, str]] = {
    "fig2a_campaign_accounting": {
        "Retained representations": "Retained\nrepresentations",
    },
    "fig2b_campaign_readiness": {},
}


CONFIG: dict[str, dict[str, Any]] = {
    "fig2a_campaign_accounting": {
        "stem": "fig2a_campaign_accounting",
        "graph": "Fig2a",
        "size_mm": (85.0, 72.0),
        "xlabel": "Count",
        "xlim": (0.0, 1600.0),
        "xstep": 500.0,
        # The plot area is inset at right so the enlarged 12-pt value labels
        # remain on the page.  At this layer width, 118 data units clear the
        # rounded 11-pt bar cap by about 0.75 mm; the longest value (1,349)
        # then retains roughly 0.5 mm of right-page clearance.
        "label_offset": 118.0,
        # The shorter layer leaves a safe lower title band for the 13-pt axis
        # label.  One percent is moved from the plot width to the right margin
        # so the enlarged terminal value label remains fully on the page.
        "layer": (49.0, 3.0, 42.0, 76.0),
        "primary": "data/fig2a_campaign_accounting/plot_data.csv",
        "sources": [
            "data/fig2a_campaign_accounting/plot_data.csv",
            "data/fig2a_campaign_accounting/Figure2_campaign_yield_source.csv",
            "data/fig2a_campaign_accounting/Figure3_retention_status_source.csv",
        ],
        "spec": "specifications/fig2a_campaign_accounting.json",
        "references": [
            "references/current_individual_exports/fig2a_campaign_accounting/Figure2A_campaign_accounting.svg",
            "references/current_individual_exports/fig2a_campaign_accounting/Figure2A_campaign_accounting.pdf",
            "references/current_individual_exports/fig2a_campaign_accounting/Figure2A_campaign_accounting.png",
        ],
    },
    "fig2b_campaign_readiness": {
        "stem": "fig2b_campaign_readiness",
        "graph": "Fig2b",
        "size_mm": (85.0, 72.0),
        "xlabel": "Count passing screen",
        "xlim": (0.0, 960.0),
        # Three major labels remain legible after PowerPoint assembly at the
        # larger 12 pt tick size.
        "xstep": 400.0,
        # The narrower layer leaves room for the enlarged gate-ratio label.
        # 73 data units preserve about 0.75 mm between the 11-pt rounded cap
        # and the 12-pt label.  A zero-value row has no cap and retains the
        # compact offset below.
        "label_offset": 73.0,
        # A 45-unit offset moves the left edge of the zero-value annotation
        # about 1.7 mm beyond the y-axis at this layer width, eliminating the
        # visible axis/glyph collision while keeping the label compact.
        "zero_label_offset": 45.0,
        # The larger left margin fits all six 12-pt categories on one line,
        # avoiding crowded two-line tick labels.  The 17% right margin still
        # keeps the longest gate-ratio annotation fully on the page.
        "layer": (58.0, 3.0, 25.0, 76.0),
        "primary": "data/fig2b_campaign_readiness/plot_data.csv",
        "sources": [
            "data/fig2b_campaign_readiness/plot_data.csv",
            "data/fig2b_campaign_readiness/campaign_shortcut_scope_audit.json",
            "data/fig2b_campaign_readiness/row_completeness_filter.json",
        ],
        "spec": "specifications/fig2b_campaign_readiness.json",
        "references": [
            "references/current_individual_exports/fig2b_campaign_readiness/Figure2B_task_specific_support_and_filtering.svg",
            "references/current_individual_exports/fig2b_campaign_readiness/Figure2B_task_specific_support_and_filtering.pdf",
            "references/current_individual_exports/fig2b_campaign_readiness/Figure2B_task_specific_support_and_filtering.png",
        ],
    },
}


@dataclass(frozen=True)
class DisplayRow:
    category_order: int
    display_label: str
    value: int
    display_group: str
    color_hex: str
    value_label: str
    label_inside_bar: bool = False


@dataclass(frozen=True)
class PlotPlan:
    plot_id: str
    config: dict[str, Any]
    spec: dict[str, Any]
    rows: tuple[DisplayRow, ...]
    source_paths: tuple[Path, ...]
    reference_paths: tuple[Path, ...]
    validations: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style_sources_metadata() -> list[dict[str, str | None]]:
    """Describe the encoded style contract from the encoded palette and typography."""

    if POWERPOINT_STYLE_SOURCE.is_file():
        style_record: dict[str, str | None] = {
            "path": str(POWERPOINT_STYLE_SOURCE.resolve()),
            "sha256": sha256(POWERPOINT_STYLE_SOURCE),
            "status": "local reference available",
        }
    else:
        style_record = {
            "path": None,
            "sha256": POWERPOINT_STYLE_SHA256,
            "status": "palette and typography are defined in this script",
        }
    return [
        style_record,
        {
            "path": str(POWERPOINT_ERA_BUILDER.resolve()),
            "sha256": sha256(POWERPOINT_ERA_BUILDER),
            "status": "public builder source",
        },
    ]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers or len(headers) != len(set(headers)):
            raise RuntimeError(f"CSV headers must be present and unique: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV contains no data rows: {path}")
    return headers, rows


def exact_int(value: str, *, context: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise RuntimeError(f"Expected an integer for {context}: {value!r}")
    return int(number)


def require_columns(headers: list[str], required: set[str], path: Path) -> None:
    missing = required.difference(headers)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")


def validate_fig2a(config: dict[str, Any], spec: dict[str, Any]) -> tuple[tuple[DisplayRow, ...], dict[str, Any]]:
    primary = PACKAGE / config["primary"]
    headers, status_rows = read_csv(primary)
    require_columns(headers, {"status", "value", "category_order"}, primary)
    status_rows.sort(key=lambda row: exact_int(row["category_order"], context="fig2a category_order"))
    expected_statuses = ["Direct retained", "Sparse retained", "Rejected"]
    if [row["status"] for row in status_rows] != expected_statuses:
        raise RuntimeError("Figure 2a status order does not match the controlled specification")
    status = {row["status"]: exact_int(row["value"], context=row["status"]) for row in status_rows}

    campaign_path = PACKAGE / "data/fig2a_campaign_accounting/Figure2_campaign_yield_source.csv"
    campaign_headers, campaign_rows = read_csv(campaign_path)
    require_columns(campaign_headers, {"quantity", "value"}, campaign_path)
    campaign = {row["quantity"]: exact_int(row["value"], context=row["quantity"]) for row in campaign_rows}
    required_quantities = {
        "Raw extracted rows",
        "Audited after coalescing",
        "Accepted and appended",
    }
    if not required_quantities.issubset(campaign):
        raise RuntimeError("Figure 2a campaign source is incomplete")

    audited = campaign["Audited after coalescing"]
    retained = campaign["Accepted and appended"]
    if sum(status.values()) != audited:
        raise RuntimeError("Figure 2a status counts do not reconcile to audited rows")
    if status["Direct retained"] + status["Sparse retained"] != retained:
        raise RuntimeError("Figure 2a retained counts do not reconcile")
    if int(spec["expected_plotted_counts"]["accounted_rows"]) != audited:
        raise RuntimeError("Figure 2a specification accounted-row total changed")

    # The retained reference shows the three audit context rows followed by
    # the three authoritative status rows.  The spec's primary plot_data.csv
    # remains the authority for the status breakdown; no values are inferred.
    rows = (
        DisplayRow(1, "Raw extracted rows", campaign["Raw extracted rows"], "audit", SLATE, f"{campaign['Raw extracted rows']:,}"),
        DisplayRow(2, "Audited/coalesced", audited, "audit", SLATE, f"{audited:,}"),
        DisplayRow(3, "Retained representations", retained, "retained", ORANGE, f"{retained:,}"),
        DisplayRow(4, "Accepted", status["Direct retained"], "status", ACCEPTED, f"{status['Direct retained']:,}"),
        DisplayRow(5, "Sparse accepted", status["Sparse retained"], "status", SPARSE, f"{status['Sparse retained']:,}"),
        DisplayRow(6, "Rejected", status["Rejected"], "status", REJECTED, f"{status['Rejected']:,}"),
    )
    validations = {
        "authoritative_status_rows": len(status_rows),
        "display_rows": len(rows),
        "audited_rows": audited,
        "retained_rows": retained,
        "status_sum": sum(status.values()),
        "reference_context_rows_added_from_exact_campaign_source": 3,
        "group_boundary_after_display_order": 3,
        "drawn_group_separator": False,
    }
    return rows, validations


def validate_fig2b(config: dict[str, Any], spec: dict[str, Any]) -> tuple[tuple[DisplayRow, ...], dict[str, Any]]:
    primary = PACKAGE / config["primary"]
    headers, source_rows = read_csv(primary)
    require_columns(
        headers,
        {"display_label", "section", "value", "category_order", "eligible_before_gate"},
        primary,
    )
    source_rows.sort(key=lambda row: exact_int(row["category_order"], context="fig2b category_order"))
    expected_order = list(spec["category_order"])
    if [row["display_label"] for row in source_rows] != expected_order:
        raise RuntimeError("Figure 2b category order does not match the controlled specification")
    if len(source_rows) != int(spec["expected_plotted_counts"]["categories"]):
        raise RuntimeError("Figure 2b category count changed")

    completeness_path = PACKAGE / "data/fig2b_campaign_readiness/row_completeness_filter.json"
    completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
    gate_before = int(completeness["literature_agent_rows_before"])
    gate_after = int(completeness["literature_agent_rows_after"])
    if f"{gate_after}/{gate_before}" != str(spec["expected_plotted_counts"]["completeness_gate"]):
        raise RuntimeError("Figure 2b completeness-gate source changed")

    shortcut_path = PACKAGE / "data/fig2b_campaign_readiness/campaign_shortcut_scope_audit.json"
    shortcut = json.loads(shortcut_path.read_text(encoding="utf-8"))
    observed_shortcut = {
        "minimal_pce_ready": int(shortcut["current_campaign_shortcut_minimal_pce_ready"]),
        "strict_pce_ready": int(shortcut["current_campaign_shortcut_strict_pce_ready"]),
        "minimal_stability_ready": int(shortcut["current_campaign_shortcut_minimal_stability_ready"]),
        "strict_stability_ready": int(shortcut["current_campaign_shortcut_strict_stability_ready"]),
    }
    if observed_shortcut != {key: int(value) for key, value in spec["shortcut_scope_check"].items()}:
        raise RuntimeError("Figure 2b shortcut-scope audit changed")

    labels = {
        "PCE minimal": "Minimal PCE-ready",
        "PCE strict": "Strict PCE-ready",
        "Stability minimal": "Minimal stability-ready",
        "Stability strict": "Strict stability-ready",
        "Physical stability": "Physical stability-ready",
        "PCE completeness gate": "PCE completeness gate",
    }
    colors = [SLATE, SLATE, ORANGE_2, ORANGE_2, PALE_RULE, ORANGE]
    rows: list[DisplayRow] = []
    for index, (row, color) in enumerate(zip(source_rows, colors), start=1):
        value = exact_int(row["value"], context=row["display_label"])
        is_gate = row["section"] == "completeness_gate"
        if is_gate:
            eligible = exact_int(row["eligible_before_gate"], context="eligible_before_gate")
            if (value, eligible) != (gate_after, gate_before):
                raise RuntimeError("Figure 2b plotted gate values do not match the exact audit JSON")
            value_label = f"{value:,} / {eligible:,}"
        else:
            value_label = "0 records" if value == 0 else f"{value:,}"
        rows.append(
            DisplayRow(
                index,
                labels[row["display_label"]],
                value,
                row["section"],
                color,
                value_label,
                # The PowerPoint-era/old-script figure places every value
                # label outside the bar in black, including the gate ratio.
                label_inside_bar=False,
            )
        )
    validations = {
        "authoritative_rows": len(source_rows),
        "display_rows": len(rows),
        "completeness_gate_passed": gate_after,
        "completeness_gate_eligible": gate_before,
        "shortcut_scope_check": observed_shortcut,
        "group_boundary_after_display_order": 5,
        "drawn_group_separator": False,
    }
    return tuple(rows), validations


def build_plan(plot_id: str) -> PlotPlan:
    config = CONFIG[plot_id]
    if not POWERPOINT_ERA_BUILDER.is_file() or POWERPOINT_ERA_BUILDER.stat().st_size == 0:
        raise FileNotFoundError(POWERPOINT_ERA_BUILDER)
    spec_path = PACKAGE / config["spec"]
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("plot_id") != plot_id or spec.get("status") != "READY":
        raise RuntimeError(f"Controlled specification is not READY for {plot_id}")
    if spec.get("style", {}).get("font_family") != FONT:
        raise RuntimeError(f"Unexpected font in {spec_path}")
    dimensions = spec["style"]["standalone_dimensions_mm"]
    if (float(dimensions["width"]), float(dimensions["height"])) != tuple(config["size_mm"]):
        raise RuntimeError(f"Standalone dimensions changed for {plot_id}")

    source_paths = tuple(PACKAGE / relative for relative in config["sources"])
    reference_paths = tuple(PACKAGE / relative for relative in config["references"])
    for path in (*source_paths, *reference_paths):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    if plot_id == "fig2a_campaign_accounting":
        rows, validations = validate_fig2a(config, spec)
    else:
        rows, validations = validate_fig2b(config, spec)
    if [row.category_order for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"Non-consecutive display order for {plot_id}")
    return PlotPlan(plot_id, config, spec, rows, source_paths, reference_paths, validations)


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


def write_display_helper(plan: PlotPlan, data_dir: Path) -> Path:
    helper = data_dir / "origin_display_helper.csv"
    with helper.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category_order",
                "display_label",
                "value",
                "display_group",
                "color_hex",
                "value_label",
                "label_inside_bar",
                "y_position",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        count = len(plan.rows)
        for row in plan.rows:
            writer.writerow(
                {
                    "category_order": row.category_order,
                    "display_label": row.display_label,
                    "value": row.value,
                    "display_group": row.display_group,
                    "color_hex": row.color_hex,
                    "value_label": row.value_label,
                    "label_inside_bar": int(row.label_inside_bar),
                    "y_position": count - row.category_order + 1,
                }
            )
    return helper


def disable_background_grids(layer: Any) -> None:
    """Disable major/minor grids explicitly while leaving separators intact."""

    layer.lt_exec(
        "layer.x.grid=0; layer.x.minorgrid=0; layer.y.grid=0; layer.y.minorgrid=0; "
        "layer.x.grid.major=0; layer.x.grid.minor=0; "
        "layer.y.grid.major=0; layer.y.grid.minor=0; "
        "layer.x2.show=0; layer.y2.show=0; "
        f'layer.x.color=color("{INK}"); layer.y.color=color("{INK}"); '
        f'layer.x.label.color=color("{INK}"); layer.y.label.color=color("{INK}");'
    )


def write_validation_sheet(book: Any, plan: PlotPlan, plot_audit: dict[str, Any]) -> None:
    sheet = book.add_sheet()
    sheet.name = "Validation"
    sheet.lname = f"{plan.plot_id} build validation"
    checks = [
        ("analysis_id", ANALYSIS_ID, "PASS"),
        ("plot_id", plan.plot_id, "PASS"),
        ("scientific_status", plan.spec["status"], "PASS"),
        ("source_action", plan.spec["required_action"], "PASS"),
        ("standalone_dimensions_mm", json.dumps(plan.config["size_mm"]), "PASS"),
        ("font_family", FONT, "PASS"),
        (
            "font_sizes_pt",
            f"tick/category/value/body {BODY_FONT_PT}; axis {AXIS_LABEL_FONT_PT}",
            "PASS",
        ),
        ("background_gridlines", "major/minor x/y disabled", "PASS"),
        ("group_separator", "not drawn; PowerPoint-era source style", "PASS"),
        ("display_rows", len(plan.rows), "PASS"),
        ("plot_audit", json.dumps(plot_audit, ensure_ascii=False), "PASS"),
    ]
    for index, header in enumerate(("check", "value", "status")):
        sheet.from_list(index, [row[index] for row in checks], lname=header, axis="")


def build_origin_plot(plan: PlotPlan, output_root: Path, close: bool) -> dict[str, Any]:
    # Delayed imports guarantee that --dry-run never imports Origin or starts
    # an automation session.
    import originpro as op
    import build_manuscript_plot as common
    import build_literatureagent_origin_figures as base
    common.op = base.require_origin()

    output_root = output_root.resolve()
    for name in ("origin_projects", "pdf", "svg", "png", "data", "validation"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    data_dir = output_root / "data" / plan.plot_id
    data_dir.mkdir(parents=True, exist_ok=True)
    copy_audit = [copy_exact(path, data_dir / path.name) for path in plan.source_paths]
    spec_path = PACKAGE / plan.config["spec"]
    copy_audit.append(copy_exact(spec_path, data_dir / spec_path.name))
    helper_path = write_display_helper(plan, data_dir)

    if getattr(op, "oext", False):
        op.set_show(not close)
    op.new(asksave=False)
    base.FONT_INDEX = int(op.lt_int(f'font("{FONT}")'))
    book = op.new_book("w", lname=f"{plan.plot_id} exact source and linked display data")
    if book is None:
        raise RuntimeError("Origin could not create the source workbook")
    book.name = "PlotData"
    _, display_table = common.import_csv_sheet(
        book,
        helper_path,
        "Display",
        "Linked publication display data",
        first=True,
    )
    for index, source in enumerate(plan.source_paths, start=1):
        copied = data_dir / source.name
        if copied.suffix.lower() == ".csv":
            common.import_csv_sheet(
                book,
                copied,
                "Source" if index == 1 else f"Source{index}",
                f"Exact source: {source.name}",
            )
        else:
            payload = json.loads(copied.read_text(encoding="utf-8"))
            sheet = book.add_sheet()
            sheet.name = "Source" if index == 1 else f"Source{index}"
            sheet.lname = f"Exact source: {source.name}"
            keys = list(payload)
            sheet.from_list(0, keys, lname="json_key", axis="")
            sheet.from_list(
                1,
                [json.dumps(payload[key], ensure_ascii=False) if isinstance(payload[key], (dict, list)) else payload[key] for key in keys],
                lname="json_value",
                axis="",
            )

    width_mm, height_mm = plan.config["size_mm"]
    graph = base.new_graph(
        plan.config["graph"],
        plan.plot_id,
        common.mm_to_in(width_mm),
        common.mm_to_in(height_mm),
    )
    layer = graph[0]
    base.position_layer(layer, *plan.config["layer"])
    count = len(plan.rows)
    ylim = (0.5, count + 0.5)
    xlim = tuple(plan.config["xlim"])

    plot_indices: list[int] = []
    for row in plan.rows:
        y = count - row.category_order + 1
        if row.value > 0:
            plot = common.add_linked_line(
                layer,
                display_table,
                f"bar_{row.category_order}",
                [0.0, float(row.value)],
                [float(y), float(y)],
                row.color_hex,
                11.0,
            )
            try:
                plot.set_int("line.cap", 0)
            except Exception:
                pass
            plot_indices.append(len(layer.plot_list()))

    base.style_axis(
        layer,
        "",
        xlim=xlim,
        xstep=float(plan.config["xstep"]),
        ylim=ylim,
        ystep=1.0,
        show_y_labels=True,
        axis_title_size=AXIS_LABEL_FONT_PT,
        tick_size=BODY_FONT_PT,
    )
    disable_background_grids(layer)
    display_text = CATEGORY_DISPLAY_TEXT.get(plan.plot_id, {})
    labels = [display_text.get(row.display_label, row.display_label) for row in plan.rows]
    scientific_labels = [row.display_label for row in plan.rows]
    positions = [float(count - index) for index in range(count)]
    base.set_categorical_ticks(layer, display_table, labels, plan.plot_id, positions)
    # Place the standalone axis title below the numeric tick-label row.  At
    # 112% the title collided with the central tick after 600 dpi export.
    title = base.add_percent_label(
        layer,
        plan.config["xlabel"],
        50,
        119,
        size=AXIS_LABEL_FONT_PT,
        color=INK,
    )
    title.set_int("anchor", 5)

    span = xlim[1] - xlim[0]
    label_positions: list[dict[str, Any]] = []
    for row in plan.rows:
        y = count - row.category_order + 1
        offset = float(
            plan.config.get("zero_label_offset", plan.config["label_offset"])
            if row.value == 0
            else plan.config["label_offset"]
        )
        x = row.value + offset
        base.add_data_label(
            layer,
            row.value_label,
            x,
            y + 0.08,
            xlim=xlim,
            ylim=ylim,
            size=BODY_FONT_PT,
            bold=True,
            color=INK,
        )
        label_positions.append(
            {
                "category": row.display_label,
                "value": row.value,
                "label": row.value_label,
                "label_x": x,
                "offset_data_units": offset,
            }
        )
    disable_background_grids(layer)
    layer.lt_exec("layer.sauto=0; layer.maxpts=0; page.speedmode=0;")

    plot_audit = {
        "source_display_rows": len(plan.rows),
        "linked_nonzero_bar_series": len(plot_indices),
        "x_limits": list(xlim),
        "y_limits": list(ylim),
        "category_order_top_to_bottom": scientific_labels,
        "category_display_text_top_to_bottom": labels,
        "category_labels": "linked 12-pt categorical tick labels; wrapped only for layout",
        "background_gridlines": {
            "x_major": 0,
            "x_minor": 0,
            "y_major": 0,
            "y_minor": 0,
        },
        "group_separator_drawn": False,
        "embedded_panel_letter": False,
        "embedded_title": False,
        "legend": False,
        "bar_width_pt": 11.0,
        "font_sizes_pt": {
            "tick_category_value_body": BODY_FONT_PT,
            "axis_label": AXIS_LABEL_FONT_PT,
        },
        "layer_percent": list(plan.config["layer"]),
        "right_page_margin_percent": 100.0
        - float(plan.config["layer"][0])
        - float(plan.config["layer"][2]),
        "value_label_positions": label_positions,
        "value_label_clearance": (
            "rounded-cap radius plus approximately 0.75 mm white gap for nonzero bars; "
            "layer inset reserves page-edge clearance for enlarged labels"
        ),
    }
    write_validation_sheet(book, plan, plot_audit)
    pages = op.graph_list("p")
    if len(pages) != 1 or pages[0].name != plan.config["graph"]:
        raise RuntimeError(
            f"Expected one graph page named {plan.config['graph']}; found {[page.name for page in pages]}"
        )

    project = output_root / "origin_projects" / f"{plan.config['stem']}.opju"
    if not op.save(str(project.resolve())):
        raise RuntimeError(f"Origin failed to save project: {project}")
    exports = common.export_graph(graph, plan.plot_id, plan.config, output_root)
    report = {
        "status": "PASS",
        "analysis_id": ANALYSIS_ID,
        "plot_id": plan.plot_id,
        "origin_project": str(project.resolve()),
        "origin_project_sha256": sha256(project),
        "origin_graph_page": plan.config["graph"],
        "source_rows": int(
            plan.validations["authoritative_status_rows"]
            if "authoritative_status_rows" in plan.validations
            else plan.validations["authoritative_rows"]
        ),
        "source_copy_audit": copy_audit,
        "display_helper": str(helper_path.resolve()),
        "display_helper_sha256": sha256(helper_path),
        "specification_sha256": sha256(PACKAGE / plan.config["spec"]),
        "reference_assets": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in plan.reference_paths
        ],
        "style_sources": style_sources_metadata(),
        "scientific_validations": plan.validations,
        "page_dimensions_mm": list(plan.config["size_mm"]),
        "font": FONT,
        "font_sizes_pt": {
            "tick_category_value_body": BODY_FONT_PT,
            "axis_label": AXIS_LABEL_FONT_PT,
        },
        "background_gridlines": "disabled",
        "group_separator": "not drawn; matches PowerPoint-era source figure",
        "plot_audit": plot_audit,
        "exports": exports,
    }
    audit_path = output_root / "validation" / f"{plan.plot_id}_build_audit.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["build_audit"] = str(audit_path.resolve())
    return report


def dry_run_report(plan: PlotPlan, output_root: Path) -> dict[str, Any]:
    config = plan.config
    return {
        "status": "PASS",
        "mode": "dry-run-no-Origin",
        "analysis_id": ANALYSIS_ID,
        "plot_id": plan.plot_id,
        "scientific_status": plan.spec["status"],
        "manifest_action": plan.spec["required_action"],
        "display_rows": [row.__dict__ for row in plan.rows],
        "scientific_validations": plan.validations,
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in plan.source_paths
        ],
        "reference_assets": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in plan.reference_paths
        ],
        "style_sources": style_sources_metadata(),
        "planned_outputs": {
            "opju": str((output_root / "origin_projects" / f"{config['stem']}.opju").resolve()),
            "pdf": str((output_root / "pdf" / f"{config['stem']}.pdf").resolve()),
            "svg": str((output_root / "svg" / f"{config['stem']}.svg").resolve()),
            "png": str((output_root / "png" / f"{config['stem']}.png").resolve()),
            "audit": str((output_root / "validation" / f"{plan.plot_id}_build_audit.json").resolve()),
        },
        "style": {
            "page_dimensions_mm": list(config["size_mm"]),
            "font": FONT,
            "font_sizes_pt": {
                "tick_category_value_body": BODY_FONT_PT,
                "axis_label": AXIS_LABEL_FONT_PT,
            },
            "layer_percent": list(config["layer"]),
            "background_gridlines": "disabled",
            "group_separator": "not drawn; matches PowerPoint-era source figure",
        },
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
        # Import only after a real build was requested.  Closing here prevents
        # a hidden automation server from being stranded after a failure.
        try:
            import originpro as op

            if getattr(op, "oext", False) and (args.close or not succeeded):
                op.exit()
        except Exception:
            if not succeeded:
                traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
