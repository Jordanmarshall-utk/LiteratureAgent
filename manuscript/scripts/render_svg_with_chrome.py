"""Render a local SVG to a publication-resolution PNG with headless Chrome.

The SVG is placed in a borderless white HTML viewport whose aspect ratio is
derived from the SVG viewBox.  Chrome therefore rasterizes live SVG text and
vector geometry at the requested pixel width without changing the artwork.
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path


VIEWBOX_RE = re.compile(
    r"viewBox\s*=\s*([\"'])\s*[-+\d.eE]+\s*[,]?\s*[-+\d.eE]+\s*[,]?\s*"
    r"([-+\d.eE]+)\s*[,]?\s*([-+\d.eE]+)\s*\1",
    re.IGNORECASE,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_PIXELS_PER_METER_600_DPI = 23622


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """Return one complete PNG chunk with its length and CRC."""
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def set_png_resolution(
    path: Path,
    pixels_per_meter: int = PNG_PIXELS_PER_METER_600_DPI,
) -> None:
    """Insert or replace PNG pHYs metadata without changing image data."""
    if not 0 < pixels_per_meter <= 0xFFFFFFFF:
        raise ValueError("pixels_per_meter must fit in an unsigned 32-bit integer")

    source = path.read_bytes()
    if not source.startswith(PNG_SIGNATURE):
        raise ValueError(f"Not a PNG file: {path}")

    physical_resolution = _png_chunk(
        b"pHYs",
        struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1),
    )
    output = bytearray(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    wrote_resolution = False
    found_idat = False
    found_iend = False

    while offset < len(source):
        if offset + 12 > len(source):
            raise ValueError(f"Truncated PNG chunk header: {path}")
        payload_length = struct.unpack(">I", source[offset : offset + 4])[0]
        chunk_end = offset + 12 + payload_length
        if chunk_end > len(source):
            raise ValueError(f"Truncated PNG chunk payload: {path}")

        chunk_type = source[offset + 4 : offset + 8]
        complete_chunk = source[offset:chunk_end]
        if chunk_type == b"IDAT":
            found_idat = True
            if not wrote_resolution:
                output.extend(physical_resolution)
                wrote_resolution = True
        if chunk_type == b"pHYs":
            if not wrote_resolution:
                output.extend(physical_resolution)
                wrote_resolution = True
        else:
            output.extend(complete_chunk)

        offset = chunk_end
        if chunk_type == b"IEND":
            found_iend = True
            output.extend(source[offset:])
            break

    if not found_idat or not found_iend:
        raise ValueError(f"PNG is missing IDAT or IEND: {path}")
    path.write_bytes(output)


def svg_dimensions(path: Path, width_px: int) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    match = VIEWBOX_RE.search(source)
    if match is None:
        raise ValueError(f"SVG viewBox not found: {path}")
    view_width, view_height = float(match.group(2)), float(match.group(3))
    if view_width <= 0 or view_height <= 0:
        raise ValueError(f"Invalid SVG viewBox: {path}")
    return width_px, max(1, round(width_px * view_height / view_width))


def chrome_path(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Neither Chrome nor Edge was found")


def render(source: Path, destination: Path, width_px: int, browser: Path | None) -> tuple[int, int]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    width, height = svg_dimensions(source, width_px)
    destination.parent.mkdir(parents=True, exist_ok=True)
    browser_exe = chrome_path(browser)

    # Chrome can keep OptimizationGuide profile files open for a fraction of a
    # second after the headless process returns on Windows.  The publication
    # PNG is already complete at that point, so do not fail an otherwise valid
    # render solely because that disposable profile cannot be removed yet.
    with tempfile.TemporaryDirectory(
        prefix="origin_svg_render_",
        dir=destination.parent,
        ignore_cleanup_errors=True,
    ) as temporary:
        temp = Path(temporary)
        profile = temp / "profile"
        wrapper = temp / "render.html"
        svg_uri = source.as_uri().replace("&", "&amp;").replace('"', "&quot;")
        wrapper.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#fff;}"
            "img{display:block;width:100vw;height:100vh;object-fit:fill;}</style></head>"
            f"<body><img src=\"{svg_uri}\"></body></html>",
            encoding="utf-8",
        )
        command = [
            str(browser_exe),
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-sync",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--default-background-color=ffffffff",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            f"--screenshot={destination}",
            wrapper.as_uri(),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not destination.is_file():
            raise RuntimeError(
                f"Browser render failed ({result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
    set_png_resolution(destination)
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--width-px", type=int, default=3900)
    parser.add_argument("--browser", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width, height = render(args.source, args.destination, args.width_px, args.browser)
    print(f"Rendered {args.destination} at {width} x {height} px")


if __name__ == "__main__":
    main()
