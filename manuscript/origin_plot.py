"""Send CSV data to a supported Origin installation and create a graph.

This module deliberately imports ``originpro`` only when a real plot is
requested.  That keeps ``--help`` and ``--dry-run`` useful on machines where
the Origin bridge has not been installed yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PLOT_TYPES = {
    "line": "l",
    "scatter": "s",
    "line-symbol": "y",
    "column": "c",
}


class PlotRequestError(ValueError):
    """Raised when input data or plot options are invalid."""


@dataclass(frozen=True)
class CsvData:
    headers: list[str]
    columns: dict[str, list[Any]]
    row_count: int


def _detect_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8-sig")[:8192]
    if not sample.strip():
        raise PlotRequestError(f"Input file is empty: {path}")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel


def _coerce_column(values: Sequence[str]) -> list[Any]:
    """Use numeric values when a whole non-empty column is numeric."""
    numeric: list[float | None] = []
    for value in values:
        stripped = value.strip()
        if not stripped:
            numeric.append(None)
            continue
        try:
            number = float(stripped)
        except ValueError:
            return [value.strip() for value in values]
        numeric.append(number)
    return numeric


def read_csv(path: Path) -> CsvData:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PlotRequestError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, dialect=_detect_dialect(path))
        try:
            headers = [value.strip() for value in next(reader)]
        except StopIteration as exc:
            raise PlotRequestError(f"Input file is empty: {path}") from exc

        if not headers or any(not header for header in headers):
            raise PlotRequestError("Every CSV column must have a non-empty header.")
        if len(headers) != len(set(headers)):
            raise PlotRequestError("CSV column headers must be unique.")

        raw_columns = {header: [] for header in headers}
        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(headers):
                raise PlotRequestError(
                    f"Row {line_number} has {len(row)} values; expected {len(headers)}."
                )
            for header, value in zip(headers, row):
                raw_columns[header].append(value)
            row_count += 1

    if row_count == 0:
        raise PlotRequestError("The CSV contains headers but no data rows.")

    columns = {name: _coerce_column(values) for name, values in raw_columns.items()}
    return CsvData(headers=headers, columns=columns, row_count=row_count)


def _is_numeric_column(values: Sequence[Any]) -> bool:
    observed = [value for value in values if value is not None]
    return bool(observed) and all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in observed
    )


def select_columns(
    data: CsvData, x_column: str | None, y_columns: Sequence[str] | None
) -> tuple[str | None, list[str]]:
    if x_column is not None and x_column not in data.columns:
        raise PlotRequestError(
            f"Unknown X column {x_column!r}. Available columns: {', '.join(data.headers)}"
        )
    if x_column is not None and not _is_numeric_column(data.columns[x_column]):
        raise PlotRequestError(f"X column {x_column!r} must be numeric.")

    if y_columns:
        selected_y = list(y_columns)
        missing = [name for name in selected_y if name not in data.columns]
        if missing:
            raise PlotRequestError(
                f"Unknown Y column(s): {', '.join(missing)}. "
                f"Available columns: {', '.join(data.headers)}"
            )
    else:
        selected_y = [
            name
            for name in data.headers
            if name != x_column and _is_numeric_column(data.columns[name])
        ]

    if not selected_y:
        raise PlotRequestError("No numeric Y columns were found or selected.")
    nonnumeric = [name for name in selected_y if not _is_numeric_column(data.columns[name])]
    if nonnumeric:
        raise PlotRequestError(f"Y column(s) must be numeric: {', '.join(nonnumeric)}")
    if len(selected_y) != len(set(selected_y)):
        raise PlotRequestError("Each Y column may be selected only once.")
    return x_column, selected_y


def _output_path(value: str | None, suffix: str | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if suffix and path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)
    return path


def build_plan(args: argparse.Namespace) -> tuple[CsvData, dict[str, Any]]:
    csv_path = Path(args.csv).expanduser().resolve()
    data = read_csv(csv_path)
    x_column, y_columns = select_columns(data, args.x, args.y)
    plan = {
        "csv": str(csv_path),
        "rows": data.row_count,
        "x": x_column or "row number",
        "y": y_columns,
        "plot": args.plot,
        "title": args.title or csv_path.stem,
        "template": args.template or "Origin default",
        "project": str(_output_path(args.project, ".opju")) if args.project else None,
        "image": str(_output_path(args.png, ".png")) if args.png else None,
        "visible": not args.hidden,
        "close_when_done": args.close,
    }
    return data, plan


def create_origin_plot(args: argparse.Namespace) -> dict[str, Any]:
    data, plan = build_plan(args)
    if args.dry_run:
        return {"status": "dry-run", **plan}

    try:
        import originpro as op
    except ImportError as exc:
        raise PlotRequestError(
            "The 'originpro' package is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    external_origin = bool(getattr(op, "oext", False))
    succeeded = False
    try:
        if external_origin:
            op.set_show(not args.hidden)

        sheet = op.new_sheet("w", lname=args.sheet_name)
        if sheet is None:
            raise RuntimeError("Origin did not create a worksheet.")

        x_column = args.x
        y_columns = plan["y"]
        for index, header in enumerate(data.headers):
            axis = "X" if header == x_column else ("Y" if header in y_columns else "")
            sheet.from_list(index, data.columns[header], lname=header, axis=axis)

        graph = op.new_graph(
            lname=plan["title"],
            template=args.template or "",
        )
        if graph is None:
            raise RuntimeError("Origin did not create a graph.")

        layer = graph[0]
        x_index: int | str = data.headers.index(x_column) if x_column else "#"
        for y_column in y_columns:
            layer.add_plot(
                sheet,
                coly=data.headers.index(y_column),
                colx=x_index,
                type=PLOT_TYPES[args.plot],
            )

        layer.axis("x").title = args.x_title or (x_column or "Row number")
        layer.axis("y").title = args.y_title or (
            y_columns[0] if len(y_columns) == 1 else "Value"
        )
        layer.rescale()
        op.wait()

        project_path = _output_path(args.project, ".opju")
        if project_path is not None:
            project_path.parent.mkdir(parents=True, exist_ok=True)
            if not op.save(str(project_path)):
                raise RuntimeError(f"Origin could not save project: {project_path}")

        image_path = _output_path(args.png, ".png")
        if image_path is not None:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            exported = graph.save_fig(str(image_path), type="png", width=args.width)
            if not exported:
                raise RuntimeError(f"Origin could not export graph: {image_path}")
            plan["image"] = exported

        succeeded = True
        return {"status": "created", **plan}
    finally:
        # On failure, do not strand a hidden Automation Server process. On
        # success, leave visible Origin open unless --close was requested.
        if external_origin and (not succeeded or args.close):
            op.exit()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an Origin graph from a CSV/TSV file via Origin's supported Python API."
    )
    parser.add_argument("csv", help="Input CSV or TSV file with a header row")
    parser.add_argument("--x", help="Header of the X column; omit to use row numbers")
    parser.add_argument(
        "--y",
        nargs="+",
        help="One or more Y-column headers; defaults to all numeric non-X columns",
    )
    parser.add_argument("--plot", choices=sorted(PLOT_TYPES), default="line")
    parser.add_argument("--title", help="Origin graph long name")
    parser.add_argument("--x-title", help="X-axis title")
    parser.add_argument("--y-title", help="Y-axis title")
    parser.add_argument("--sheet-name", default="LiteratureAgent Data", help="Origin workbook long name")
    parser.add_argument("--template", help="Origin graph template name or .otp/.otpu path")
    parser.add_argument("--project", help="Save the Origin project (.opju)")
    parser.add_argument("--png", help="Export the graph to PNG")
    parser.add_argument("--width", type=int, default=1200, help="PNG width in pixels")
    parser.add_argument("--hidden", action="store_true", help="Do not show Origin while plotting")
    parser.add_argument("--close", action="store_true", help="Close Origin after plotting")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print the plot plan without starting Origin",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.width <= 0:
        parser.error("--width must be greater than zero")
    try:
        result = create_origin_plot(args)
    except (PlotRequestError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
