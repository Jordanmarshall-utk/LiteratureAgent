"""Remove the Origin 2024 hovering-line text export artifact from an SVG.

The affected Origin exporter emits each unwanted bar as a one-segment ``path``
immediately after the corresponding live ``text`` element.  This script removes
only paths that also match the text position, font size, orientation, and color.
All plotted data geometry (polylines, symbols, error bars, and other paths) is
left unchanged.  An audit JSON records every removed element.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


TEXT_PATH_RE = re.compile(
    r"(?P<text><text\b[^>]*>(?:(?!<text\b).)*?</text>)"
    r"(?P<gap>\s*)(?P<path><path\b[^>]*/>)",
    re.IGNORECASE | re.DOTALL,
)
GROUPED_TEXT_PATH_RE = re.compile(
    r"(?P<group>(?P<gopen><g\b[^>]*>)"
    r"(?:(?!<g\b|</g>).)*?"
    r"(?P<text><text\b[^>]*>(?:(?!<text\b).)*?</text>)"
    r"(?:(?!<g\b|</g>).)*?</g>)"
    r"(?P<gap>\s*)(?P<path><path\b[^>]*/>)",
    re.IGNORECASE | re.DOTALL,
)
ATTR_RE = re.compile(r"([:\w-]+)\s*=\s*([\"'])(.*?)\2", re.DOTALL)
LINE_RE = re.compile(
    r"^\s*M\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*"
    r"L\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*$",
    re.IGNORECASE,
)
TSPAN_RE = re.compile(r"<[^>]+>|\s+", re.DOTALL)
MATRIX_RE = re.compile(r"matrix\(([^)]+)\)", re.IGNORECASE)
ROOT_SVG_RE = re.compile(r"<svg\b(?P<attrs>[^>]*)>", re.IGNORECASE)
VIEWBOX_RE = re.compile(
    r"viewBox\s*=\s*([\"'])\s*([-+\d.eE]+)\s*[,]?\s*([-+\d.eE]+)\s*[,]?\s*"
    r"([-+\d.eE]+)\s*[,]?\s*([-+\d.eE]+)\s*\1",
    re.IGNORECASE,
)


def attrs(tag: str) -> dict[str, str]:
    return {name: value for name, _, value in ATTR_RE.findall(tag)}


def style_map(style: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in style.split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            result[key.strip().lower()] = value.strip()
    return result


def number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def norm_color(value: str | None) -> str:
    compact = re.sub(r"\s+", "", (value or "").lower())
    aliases = {
        "black": "#000000",
        "#000": "#000000",
        "rgb(0,0,0)": "#000000",
        "white": "#ffffff",
        "#fff": "#ffffff",
        "rgb(255,255,255)": "#ffffff",
    }
    return aliases.get(compact, compact)


def visible_text(text_tag: str) -> str:
    inner = re.sub(r"^<text\b[^>]*>|</text>$", "", text_tag, flags=re.IGNORECASE | re.DOTALL)
    return TSPAN_RE.sub(" ", inner).strip()


def artifact_details(
    text_tag: str,
    path_tag: str,
    inherited: dict[str, str] | None = None,
) -> dict[str, object] | None:
    ta = attrs(text_tag)
    if inherited:
        for key, value in inherited.items():
            ta.setdefault(key, value)
    pa = attrs(path_tag)
    ps = style_map(pa.get("style", ""))
    match = LINE_RE.match(pa.get("d", ""))
    if match is None or pa.get("fill", "").lower() != "none":
        return None
    if ps.get("stroke-linecap", "").lower() != "butt":
        return None

    x1, y1, x2, y2 = (float(value) for value in match.groups())
    font_size = number(ta.get("font-size"))
    if font_size is None or font_size <= 0:
        return None
    text_color = norm_color(ta.get("fill"))
    path_color = norm_color(ps.get("stroke"))
    if text_color and path_color and text_color != path_color:
        return None

    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 0.2 * font_size:
        return None

    horizontal = abs(dy) <= 0.01 * max(1.0, length)
    vertical = abs(dx) <= 0.01 * max(1.0, length)
    text_x = number(ta.get("x"))
    text_y = number(ta.get("y"))
    transform = ta.get("transform", "")

    if horizontal and text_x is not None and text_y is not None:
        starts_at_text = abs(x1 - text_x) <= max(3.0, 0.08 * font_size)
        bar_offset = (text_y - y1) / font_size
        if not starts_at_text or not (0.45 <= bar_offset <= 1.15):
            return None
    elif vertical and transform:
        matrix_match = MATRIX_RE.search(transform)
        if matrix_match is None:
            return None
        values = [float(value) for value in re.split(r"[ ,]+", matrix_match.group(1).strip())]
        if len(values) != 6 or max(abs(values[1]), abs(values[2])) < 0.7:
            return None
        translate_x, translate_y = values[4], values[5]
        begins_at_baseline = abs(y1 - translate_y) <= max(3.0, 0.08 * font_size)
        side_offset = abs(translate_x - x1) / font_size
        if not begins_at_baseline or not (0.45 <= side_offset <= 1.15):
            return None
    else:
        return None

    return {
        "text": visible_text(text_tag),
        "path_d": pa.get("d", ""),
        "font_size": font_size,
        "orientation": "horizontal" if horizontal else "vertical",
        "stroke": ps.get("stroke", ""),
    }


def set_physical_width(svg: str, width_in: float | None) -> tuple[str, dict[str, float] | None]:
    if width_in is None:
        return svg, None
    root_match = ROOT_SVG_RE.search(svg)
    if root_match is None:
        raise ValueError("SVG root element not found")
    viewbox_match = VIEWBOX_RE.search(root_match.group(0))
    if viewbox_match is None:
        raise ValueError("SVG root viewBox not found")
    _, _, _, view_w, view_h = viewbox_match.groups()
    view_w_f, view_h_f = float(view_w), float(view_h)
    height_in = width_in * view_h_f / view_w_f
    root = root_match.group(0)
    root = re.sub(r"\swidth\s*=\s*([\"']).*?\1", f' width="{width_in:.6f}in"', root, count=1)
    root = re.sub(r"\sheight\s*=\s*([\"']).*?\1", f' height="{height_in:.6f}in"', root, count=1)
    svg = svg[: root_match.start()] + root + svg[root_match.end() :]
    return svg, {"width_in": width_in, "height_in": height_in, "viewbox_width": view_w_f, "viewbox_height": view_h_f}


def sanitize(source: Path, destination: Path, audit_path: Path, width_in: float | None) -> dict[str, object]:
    original = source.read_text(encoding="utf-8")
    removed: list[dict[str, object]] = []

    def replace(match: re.Match[str]) -> str:
        details = artifact_details(match.group("text"), match.group("path"))
        if details is None:
            return match.group(0)
        removed.append(details)
        return match.group("text") + match.group("gap")

    cleaned = TEXT_PATH_RE.sub(replace, original)

    def replace_grouped(match: re.Match[str]) -> str:
        details = artifact_details(
            match.group("text"),
            match.group("path"),
            attrs(match.group("gopen")),
        )
        if details is None:
            return match.group(0)
        details["wrapped_text_group"] = True
        removed.append(details)
        return match.group("group") + match.group("gap")

    # Origin wraps labels with White Out enabled in a <g> containing a white
    # rectangle and the live text.  The same unwanted text bar follows that
    # group, so apply the identical geometry/color checks while inheriting the
    # group's font size.  The whiteout rectangle itself is preserved.
    cleaned = GROUPED_TEXT_PATH_RE.sub(replace_grouped, cleaned)
    cleaned, dimensions = set_physical_width(cleaned, width_in)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8", newline="\n")
    audit = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_bytes": len(original.encode("utf-8")),
        "destination_bytes": len(cleaned.encode("utf-8")),
        "removed_artifact_count": len(removed),
        "removed_artifacts": removed,
        "physical_dimensions": dimensions,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--width-in", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.audit or args.destination.with_suffix(".svg_audit.json")
    audit = sanitize(args.source, args.destination, audit_path, args.width_in)
    print(json.dumps({"removed_artifact_count": audit["removed_artifact_count"], "output": str(args.destination)}))


if __name__ == "__main__":
    main()
