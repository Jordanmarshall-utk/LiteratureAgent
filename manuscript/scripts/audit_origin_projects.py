"""Read-only reopen audit for the 18 available numeric Origin build targets.

The registry is assembled directly from the literal configuration dictionaries
in the three Origin builders. Reading those dictionaries with :mod:`ast`
avoids importing ``originpro`` during ``--static-check`` while still keeping
the auditor synchronized with each builder's graph name, project stem, source
CSV, and expected source-row count.

The real audit imports Origin only after registry validation.  Every OPJU is
opened with ``readonly=True``; no project, worksheet, graph, or data object is
created or saved.  A before/after file fingerprint additionally verifies that
the read-only audit did not change an Origin project on disk.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "2026-10-27_controlled_repaired_rerun"
ANALYSIS_VERSION = f"{ANALYSIS_ID}_v1"
PUBLIC_SOURCE_ROOT = ROOT / "source"
PACKAGE_ROOT = PUBLIC_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "build"

EXPECTED_PLOT_IDS = (
    "fig3a_historical_r2",
    "fig3b_historical_rmse",
    "fig3c_historical_mae",
    "fig3d_literatureagent_r2",
    "fig3e_literatureagent_rmse",
    "fig3f_literatureagent_mae",
    "fig4b_choice_shap",
    "fig5a_stability_classification",
    "fig5b_stability_regression",
    "figs1_global_shap",
    "figs2_solvent_shap",
    "figs3_local_shap",
    "figs4a_normalized_stability",
    "figs4b_normalization_sensitivity",
    "figs6a_benchmark",
    "figs6b_benchmark",
    "figs6c_benchmark",
    "figs6d_benchmark",
)


@dataclass(frozen=True)
class RegistrySource:
    builder: str
    path: Path
    symbol: str
    plot_ids: tuple[str, ...]


REGISTRY_SOURCES = (
    RegistrySource(
        builder="build_manuscript_plot",
        path=ROOT / "scripts" / "build_manuscript_plot.py",
        symbol="PLOT_CONFIG",
        plot_ids=EXPECTED_PLOT_IDS[0:12],
    ),
    RegistrySource(
        builder="build_figs4_origin_no_grid",
        path=ROOT / "scripts" / "build_figs4_origin_no_grid.py",
        symbol="PLOT_CONFIG",
        plot_ids=EXPECTED_PLOT_IDS[12:14],
    ),
    RegistrySource(
        builder="build_figs6_benchmark_origin",
        path=ROOT / "scripts" / "build_figs6_benchmark_origin.py",
        symbol="CONFIG",
        plot_ids=EXPECTED_PLOT_IDS[14:18],
    ),
)

SOURCE_KEY_ALIASES = ("source", "primary", "data")
EXPECTED_KEY_ALIASES = ("expected", "expected_rows")


@dataclass(frozen=True)
class ProjectSpec:
    plot_id: str
    stem: str
    graph: str
    builder: str
    builder_path: Path
    config_symbol: str
    source_key: str
    source_relative: Path
    output_source_relative: Path
    specification_relative: Path
    expected_rows: int
    source_headers: tuple[str, ...]
    source_anchor_index: int | None
    source_anchor_header: str | None
    sheet_roles: tuple[tuple[str, tuple[str, ...]], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def read_literal_mapping(path: Path, symbol: str) -> dict[str, dict[str, Any]]:
    """Read a top-level literal mapping without importing its builder module."""

    if not path.is_file():
        raise FileNotFoundError(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value_node: ast.expr | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == symbol for target in node.targets
        ):
            value_node = node.value
            break
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol
        ):
            value_node = node.value
            break
    if value_node is None:
        raise RuntimeError(f"Could not find literal {symbol} in {path}")
    try:
        payload = ast.literal_eval(value_node)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"{path}:{symbol} is no longer a literal mapping") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}:{symbol} must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise RuntimeError(f"{path}:{symbol} contains a non-string/non-mapping entry")
        result[key] = value
    return result


def read_csv_contract(path: Path) -> tuple[tuple[str, ...], int, int | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = tuple(reader.fieldnames or ())
        if not headers or len(headers) != len(set(headers)):
            raise RuntimeError(f"CSV headers must be present and unique: {path}")
        nonblank_counts = [0] * len(headers)
        rows = 0
        for row in reader:
            rows += 1
            for index, header in enumerate(headers):
                if is_nonblank(row.get(header)):
                    nonblank_counts[index] += 1
    if rows <= 0:
        raise RuntimeError(f"CSV contains no source rows: {path}")
    anchor_index = next(
        (index for index, count in enumerate(nonblank_counts) if count == rows), None
    )
    return headers, rows, anchor_index


def first_string(config: Mapping[str, Any], keys: Sequence[str], context: str) -> tuple[str, str]:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    raise RuntimeError(f"{context}: none of the supported keys {tuple(keys)} is present")


def unique_aliases(*values: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def source_role_aliases(role: str, relative: Path, *long_names: str) -> tuple[str, ...]:
    return unique_aliases(
        role,
        relative.stem,
        relative.name,
        f"Exact source: {relative.name}",
        *long_names,
    )


def sheet_roles_for(
    plot_id: str,
    registry_source: RegistrySource,
    config: Mapping[str, Any],
    source_relative: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Normalize worksheet roles plus aliases emitted after CSV import.

    Origin may replace a requested short sheet name with the imported CSV's
    basename.  Each role therefore accepts its builder name, long name, and
    deterministic input/helper basename.
    """

    validation = (
        "Validation",
        unique_aliases(
            "Validation",
            f"{plot_id} validation",
            f"{plot_id} build validation",
        ),
    )
    source = (
        "Source",
        source_role_aliases(
            "Source",
            source_relative,
            "Exact authoritative plotting table",
            "Exact authoritative source data",
            "Exact authoritative source table",
        ),
    )

    if registry_source.builder == "build_figs4_origin_no_grid":
        helper_name = str(config.get("helper_name", ""))
        if not helper_name:
            raise RuntimeError(f"{plot_id}: S4 builder helper_name is missing")
        helper = Path(helper_name)
        return (
            source,
            (
                "Geometry",
                unique_aliases(
                    "Geometry",
                    helper.stem,
                    helper.name,
                    "Deterministic linked plotting geometry",
                ),
            ),
            validation,
        )
    if registry_source.builder == "build_figs6_benchmark_origin":
        return (
            source,
            (
                "Display",
                unique_aliases(
                    "Display",
                    "origin_display_helper",
                    "origin_display_helper.csv",
                    "Explicit linked Origin plotting coordinates",
                ),
            ),
            validation,
        )
    if registry_source.builder == "build_manuscript_plot":
        family = str(config.get("family", ""))
        roles: list[tuple[str, tuple[str, ...]]] = [source]
        if family in {"shap_global", "shap_choice", "shap_solvent"}:
            roles.extend(
                [
                    (
                        "Points",
                        unique_aliases(
                            "Points",
                            "origin_point_helper",
                            "origin_point_helper.csv",
                            "Complete deterministic Origin point helper",
                        ),
                    ),
                    (
                        "Labels",
                        unique_aliases("Labels", "Categorical axis labels"),
                    ),
                ]
            )
            if family != "shap_global":
                roles.append(
                    (
                        "Summary",
                        unique_aliases(
                            "Summary",
                            "origin_summary_helper",
                            "origin_summary_helper.csv",
                            "Exact medians and IQR helpers",
                        ),
                    )
                )
        roles.append(validation)
        return tuple(roles)
    raise RuntimeError(f"Unsupported Origin builder: {registry_source.builder}")


def normalize_project(
    plot_id: str,
    config: Mapping[str, Any],
    registry_source: RegistrySource,
) -> ProjectSpec:
    stem = config.get("stem")
    graph = config.get("graph")
    if not isinstance(stem, str) or not stem:
        raise RuntimeError(f"{plot_id}: missing project stem")
    if not isinstance(graph, str) or not graph:
        raise RuntimeError(f"{plot_id}: missing Origin graph name")

    source_key, source_value = first_string(
        config, SOURCE_KEY_ALIASES, f"{plot_id} authoritative source"
    )
    source_relative = Path(source_value)
    if source_relative.suffix.lower() != ".csv":
        raise RuntimeError(f"{plot_id}: authoritative source must be CSV: {source_relative}")
    source_path = PACKAGE_ROOT / source_relative
    headers, source_rows, source_anchor_index = read_csv_contract(source_path)

    configured_expected: int | None = None
    for key in EXPECTED_KEY_ALIASES:
        if key in config:
            configured_expected = int(config[key])
            break
    if configured_expected is not None and configured_expected != source_rows:
        raise RuntimeError(
            f"{plot_id}: builder expects {configured_expected:,} source rows, "
            f"but controlled CSV contains {source_rows:,}"
        )

    spec_value = config.get("spec", f"specifications/{plot_id}.json")
    if not isinstance(spec_value, str) or not spec_value:
        raise RuntimeError(f"{plot_id}: invalid specification path")
    specification_relative = Path(spec_value)
    specification_path = PACKAGE_ROOT / specification_relative
    if not specification_path.is_file():
        raise FileNotFoundError(specification_path)
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    if specification.get("plot_id") != plot_id:
        raise RuntimeError(f"{plot_id}: specification identity mismatch")
    if specification.get("status") not in {"READY", "CURRENT"}:
        raise RuntimeError(f"{plot_id}: source specification must be READY or CURRENT")

    return ProjectSpec(
        plot_id=plot_id,
        stem=stem,
        graph=graph,
        builder=registry_source.builder,
        builder_path=registry_source.path,
        config_symbol=registry_source.symbol,
        source_key=source_key,
        source_relative=source_relative,
        output_source_relative=Path("data") / plot_id / source_relative.name,
        specification_relative=specification_relative,
        expected_rows=source_rows,
        source_headers=headers,
        source_anchor_index=source_anchor_index,
        source_anchor_header=(
            headers[source_anchor_index] if source_anchor_index is not None else None
        ),
        sheet_roles=sheet_roles_for(
            plot_id, registry_source, config, source_relative
        ),
    )


def load_registry() -> tuple[ProjectSpec, ...]:
    normalized: dict[str, ProjectSpec] = {}
    for registry_source in REGISTRY_SOURCES:
        mapping = read_literal_mapping(registry_source.path, registry_source.symbol)
        for plot_id in registry_source.plot_ids:
            if plot_id not in mapping:
                raise RuntimeError(
                    f"{plot_id} is missing from {registry_source.path.name}:"
                    f"{registry_source.symbol}"
                )
            if plot_id in normalized:
                raise RuntimeError(f"Duplicate audit registry entry: {plot_id}")
            normalized[plot_id] = normalize_project(
                plot_id, mapping[plot_id], registry_source
            )

    missing = [plot_id for plot_id in EXPECTED_PLOT_IDS if plot_id not in normalized]
    unexpected = [plot_id for plot_id in normalized if plot_id not in EXPECTED_PLOT_IDS]
    if missing or unexpected or len(normalized) != len(EXPECTED_PLOT_IDS):
        raise RuntimeError(
            f"Audit registry must contain exactly {len(EXPECTED_PLOT_IDS)} projects; missing={missing}, "
            f"unexpected={unexpected}, count={len(normalized)}"
        )
    return tuple(normalized[plot_id] for plot_id in EXPECTED_PLOT_IDS)


def project_path(output_root: Path, spec: ProjectSpec) -> Path:
    return output_root / "origin_projects" / f"{spec.stem}.opju"


def static_check_report(
    registry: Sequence[ProjectSpec], output_root: Path
) -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    missing_projects: list[str] = []
    for spec in registry:
        opju = project_path(output_root, spec)
        source_copy = output_root / spec.output_source_relative
        exists = opju.is_file()
        if not exists:
            missing_projects.append(spec.plot_id)
        projects.append(
            {
                "plot_id": spec.plot_id,
                "builder": spec.builder,
                "config_symbol": spec.config_symbol,
                "stem": spec.stem,
                "graph": spec.graph,
                "authoritative_source": str((PACKAGE_ROOT / spec.source_relative).resolve()),
                "output_source": str(source_copy.resolve()),
                "expected_rows": spec.expected_rows,
                "source_anchor_header": spec.source_anchor_header,
                "required_sheet_roles": {
                    role: list(aliases) for role, aliases in spec.sheet_roles
                },
                "project": str(opju.resolve()),
                "project_present": exists,
                "output_source_present": source_copy.is_file(),
            }
        )
    return {
        "status": "STATIC_CHECK_PASS",
        "origin_started": False,
        "analysis_id": ANALYSIS_ID,
        "analysis_version": ANALYSIS_VERSION,
        "registry_projects": len(registry),
        "expected_projects": len(EXPECTED_PLOT_IDS),
        "project_files_present": len(registry) - len(missing_projects),
        "project_files_missing": missing_projects,
        "ready_for_reopen_audit": not missing_projects,
        "projects": projects,
    }


def is_nonblank(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def worksheet_source_rows(
    source: Any, headers: Sequence[str], anchor_index: int | None
) -> tuple[int, bool]:
    """Count rows using only authoritative CSV columns, excluding helpers."""

    # Dense SHAP tables contain hundreds of thousands of rows.  Prefer a CSV
    # column proven nonblank for every controlled row so the reopen audit does
    # not materialize every worksheet column merely to establish row count.
    if anchor_index is not None:
        header = headers[anchor_index]
        try:
            values = list(source.to_list(anchor_index))
        except Exception as exc:
            raise RuntimeError(
                f"Could not read authoritative worksheet column "
                f"{anchor_index + 1} ({header})"
            ) from exc
        header_in_data = bool(values) and str(values[0]) == str(header)
        data = values[1:] if header_in_data else values
        return sum(1 for value in data if is_nonblank(value)), header_in_data

    # Fallback for a future controlled table with no universally populated
    # column: union occupancy across only the authoritative CSV columns.
    columns: list[list[Any]] = []
    for index, header in enumerate(headers):
        try:
            values = list(source.to_list(index))
        except Exception as exc:
            raise RuntimeError(
                f"Could not read authoritative worksheet column {index + 1} ({header})"
            ) from exc
        columns.append(values)
    header_in_data = bool(columns) and all(
        column and str(column[0]) == str(header)
        for column, header in zip(columns, headers)
    )
    offset = 1 if header_in_data else 0
    extent = max((len(column) for column in columns), default=0)
    rows = 0
    for row_index in range(offset, extent):
        if any(
            row_index < len(column) and is_nonblank(column[row_index])
            for column in columns
        ):
            rows += 1
    return rows, header_in_data


GENERIC_SHEET_ALIASES = {
    "display",
    "source",
    "source2",
    "source3",
    "geometry",
    "points",
    "labels",
    "summary",
    "validation",
}


def normalize_sheet_identity(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def sheet_identity_values(sheet: Any) -> tuple[str, ...]:
    values: list[str] = []
    for attribute in ("name", "lname"):
        try:
            value = str(getattr(sheet, attribute, "") or "").strip()
        except Exception:
            value = ""
        if value and value not in values:
            values.append(value)
    # Some Origin versions expose the imported file basename only through the
    # sheet range string, so retain it as a third identity when available.
    try:
        range_name = str(sheet.lt_range()).strip()
    except Exception:
        range_name = ""
    if range_name and range_name not in values:
        values.append(range_name)
    return tuple(values)


def sheet_alias_matches(identity: str, alias: str) -> bool:
    identity_key = normalize_sheet_identity(identity)
    alias_key = normalize_sheet_identity(alias)
    if not identity_key or not alias_key:
        return False
    if identity_key == alias_key:
        return True
    # Canonical role words must match exactly; otherwise "Source" would also
    # select every "Exact source: ..." support worksheet.
    if alias_key in GENERIC_SHEET_ALIASES:
        return False
    return alias_key in identity_key


def resolve_sheet_roles(
    sheets: Sequence[Any], spec: ProjectSpec
) -> dict[str, Any]:
    identities = [sheet_identity_values(sheet) for sheet in sheets]
    resolved: dict[str, Any] = {}
    used_indices: set[int] = set()
    for role, aliases in spec.sheet_roles:
        candidates = [
            index
            for index, values in enumerate(identities)
            if any(
                sheet_alias_matches(identity, alias)
                for identity in values
                for alias in aliases
            )
        ]
        if len(candidates) != 1:
            diagnostic = [
                {"index": index, "identities": list(values)}
                for index, values in enumerate(identities)
            ]
            raise RuntimeError(
                f"{spec.plot_id}: worksheet role {role!r} matched "
                f"{candidates}; aliases={list(aliases)}; sheets={diagnostic}"
            )
        index = candidates[0]
        if index in used_indices:
            raise RuntimeError(
                f"{spec.plot_id}: worksheet {list(identities[index])} matched "
                f"more than one required role"
            )
        used_indices.add(index)
        resolved[role] = sheets[index]
    return resolved


def inspect_project(op: Any, spec: ProjectSpec, output_root: Path) -> dict[str, Any]:
    project = project_path(output_root, spec)
    if not project.is_file():
        raise FileNotFoundError(project)
    before = file_fingerprint(project)
    if not op.open(str(project.resolve()), readonly=True, asksave=False):
        raise RuntimeError(f"Origin could not reopen {project}")

    graphs = op.graph_list("p")
    graph_names = [graph.name for graph in graphs]
    graph_lnames = [graph.lname for graph in graphs]
    if graph_names != [spec.graph]:
        raise RuntimeError(
            f"{spec.plot_id}: expected graph {[spec.graph]}, found {graph_names}"
        )
    if graph_lnames != [spec.plot_id]:
        raise RuntimeError(
            f"{spec.plot_id}: expected graph long name {[spec.plot_id]}, "
            f"found {graph_lnames}"
        )

    book = op.find_book("w", "PlotData")
    if book is None:
        raise RuntimeError(f"{spec.plot_id}: PlotData workbook is missing")
    sheets = list(book)
    sheet_names = [sheet.name for sheet in sheets]
    sheet_long_names = [sheet.lname for sheet in sheets]
    resolved_sheets = resolve_sheet_roles(sheets, spec)
    source = resolved_sheets["Source"]

    source_copy = output_root / spec.output_source_relative
    authoritative_source = PACKAGE_ROOT / spec.source_relative
    if not source_copy.is_file():
        raise FileNotFoundError(source_copy)
    source_copy_hash = sha256(source_copy)
    authoritative_hash = sha256(authoritative_source)
    if source_copy_hash != authoritative_hash:
        raise RuntimeError(
            f"{spec.plot_id}: output source copy differs from controlled source CSV"
        )

    source_rows, header_in_data = worksheet_source_rows(
        source, spec.source_headers, spec.source_anchor_index
    )
    worksheet_max_rows = int(source.rows)
    if source_rows != spec.expected_rows:
        raise RuntimeError(
            f"{spec.plot_id}: Source has {source_rows:,} rows; "
            f"expected {spec.expected_rows:,}"
        )

    after = file_fingerprint(project)
    if after != before:
        raise RuntimeError(
            f"{spec.plot_id}: OPJU changed during read-only audit; "
            f"before={before}, after={after}"
        )

    return {
        "plot_id": spec.plot_id,
        "status": "PASS",
        "builder": spec.builder,
        "project": str(project.resolve()),
        "project_fingerprint_before": before,
        "project_fingerprint_after": after,
        "read_only_fingerprint_unchanged": True,
        "graph_pages": graph_names,
        "graph_long_names": graph_lnames,
        "workbook": book.name,
        "worksheet_names": sheet_names,
        "worksheet_long_names": sheet_long_names,
        "worksheet_identities": [
            list(sheet_identity_values(sheet)) for sheet in sheets
        ],
        "resolved_worksheet_roles": {
            role: list(sheet_identity_values(sheet))
            for role, sheet in resolved_sheets.items()
        },
        "required_worksheet_roles_present": True,
        "source_sheet": source.name,
        "source_sheet_long_name": source.lname,
        "source_copy": str(source_copy.resolve()),
        "source_copy_sha256": source_copy_hash,
        "controlled_source_sha256": authoritative_hash,
        "source_copy_matches_controlled_source": True,
        "source_rows": source_rows,
        "source_anchor_header": spec.source_anchor_header,
        "csv_header_row_visible_in_source_data": header_in_data,
        "source_worksheet_max_rows_including_helpers": worksheet_max_rows,
        "expected_rows": spec.expected_rows,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_origin_audit(
    registry: Sequence[ProjectSpec], output_root: Path, report_path: Path
) -> dict[str, Any]:
    # Delayed import is mandatory: --static-check must never initialize or
    # launch Origin.
    import originpro as op  # type: ignore

    if getattr(op, "oext", False):
        op.set_show(False)
    results: list[dict[str, Any]] = []
    try:
        for spec in registry:
            results.append(inspect_project(op, spec, output_root))
        report = {
            "status": "PASS",
            "analysis_id": ANALYSIS_ID,
            "analysis_version": ANALYSIS_VERSION,
            "read_only": True,
            "checked_projects": len(results),
            "expected_projects": len(EXPECTED_PLOT_IDS),
            "checks": results,
        }
        write_json_atomic(report_path, report)
        print(json.dumps(report, indent=2))
        return report
    finally:
        if getattr(op, "oext", False):
            try:
                op.exit()
            except Exception:
                pass


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Delivery root containing origin_projects/ and data/.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Audit JSON path; defaults to OUTPUT_ROOT/validation/origin_project_reopen_audit.json.",
    )
    parser.add_argument(
        "--static-check",
        action="store_true",
        help="Validate the retained Origin build registry and controlled inputs without importing Origin.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    registry = load_registry()
    if args.static_check:
        print(json.dumps(static_check_report(registry, output_root), indent=2))
        return 0
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_root / "validation" / "origin_project_reopen_audit.json"
    )
    run_origin_audit(registry, output_root, report_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
