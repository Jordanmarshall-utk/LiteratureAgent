"""Clean Origin 2024's inherited bar-above text decoration in dense exports.

The complete, unmodified 600-dpi Origin PNG remains under
``validation/origin_raw_exports``.  This script creates a raster-backed SVG
from that exact render, masks only the reserved x-title band outside the data
layer, and redraws the same title as live Times New Roman SVG text.  Legends
are delivered separately for the current PowerPoint assembly workflow.  No
data-region pixel is covered or changed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from render_svg_with_chrome import set_png_resolution


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "2026-10-27_controlled_repaired_rerun"
PUBLIC_SOURCE_ROOT = ROOT / "source"
DEFAULT_OUTPUT = ROOT / "build"
TIMES_NEW_ROMAN = Path(r"C:\Windows\Fonts\times.ttf")
PNG_CLEANER = ROOT / "scripts" / "clean_dense_origin_png.ps1"
TITLE_FONT_PX = 108  # 12.96 pt at 600 dpi


@dataclass(frozen=True)
class DenseCleanup:
    plot_id: str
    stem: str
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    legend_mask: tuple[int, int, int, int] | None
    title_mask: tuple[int, int, int, int]
    legend_svg: str
    title_center_x: int
    title_baselines: tuple[int, int]


CONFIG = {
    "fig4b_choice_shap": DenseCleanup(
        plot_id="fig4b_choice_shap",
        stem="fig4b_choice_shap",
        width_px=4252,
        height_px=3661,
        width_mm=180,
        height_mm=155,
        legend_mask=None,
        # The graph builder reserves a lower title band, so this rectangle can
        # remove the complete inherited Origin title/decoration while staying
        # below the relocated numeric tick-label row.
        title_mask=(2200, 3220, 2052, 385),
        legend_svg="",
        title_center_x=3232,
        title_baselines=(3370, 3480),
    ),
    "figs1_global_shap": DenseCleanup(
        plot_id="figs1_global_shap",
        stem="figs1_global_shap",
        width_px=4252,
        height_px=3780,
        width_mm=180,
        height_mm=160,
        legend_mask=None,
        title_mask=(1700, 3364, 2500, 346),
        legend_svg="",
        title_center_x=3040,
        title_baselines=(3510, 3620),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def svg_for(config: DenseCleanup, raw_png: Path) -> str:
    payload = base64.b64encode(raw_png.read_bytes()).decode("ascii")
    tx, ty, tw, th = config.title_mask
    title_y1, title_y2 = config.title_baselines
    width_in = config.width_mm / 25.4
    height_in = config.height_mm / 25.4
    legend_cleanup = ""
    if config.legend_mask is not None:
        lx, ly, lw, lh = config.legend_mask
        legend_cleanup = f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#FFFFFF"/>{config.legend_svg}'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width_in:.6f}in" height="{height_in:.6f}in" viewBox="0 0 {config.width_px} {config.height_px}">
  <title>{config.plot_id} — complete Origin rendering with text-decoration cleanup</title>
  <desc>The embedded image is Origin's complete 600-dpi render with display reduction disabled. White masks are confined to reserved legend and x-title bands outside the data layer; the same text and legend keys are redrawn as live vector objects.</desc>
  <image x="0" y="0" width="{config.width_px}" height="{config.height_px}" href="data:image/png;base64,{payload}"/>
  <g id="origin-text-decoration-cleanup" font-family="Times New Roman" font-size="67" fill="#080808">
    {legend_cleanup}
    <rect x="{tx}" y="{ty}" width="{tw}" height="{th}" fill="#FFFFFF"/>
    <text x="{config.title_center_x}" y="{title_y1}" text-anchor="middle" dominant-baseline="middle" font-size="{TITLE_FONT_PX}">Grouped SHAP contribution</text>
    <text x="{config.title_center_x}" y="{title_y2}" text-anchor="middle" dominant-baseline="middle" font-size="{TITLE_FONT_PX}">to predicted PCE (percentage points)</text>
  </g>
</svg>
'''


def render_clean_png(raw_png: Path, png: Path, config: DenseCleanup) -> None:
    if not PNG_CLEANER.is_file():
        raise FileNotFoundError(PNG_CLEANER)
    tx, ty, tw, th = config.title_mask
    title_y1, title_y2 = config.title_baselines
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PNG_CLEANER),
        "-InputPng",
        str(raw_png),
        "-OutputPng",
        str(png),
        "-TitleMaskX",
        str(tx),
        "-TitleMaskY",
        str(ty),
        "-TitleMaskWidth",
        str(tw),
        "-TitleMaskHeight",
        str(th),
        "-TitleCenterX",
        str(config.title_center_x),
        "-TitleCenterY1",
        str(title_y1),
        "-TitleCenterY2",
        str(title_y2),
        "-TitleFontPixels",
        str(TITLE_FONT_PX),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not png.is_file() or png.stat().st_size == 0:
        raise RuntimeError(f"PowerShell PNG cleanup failed ({result.returncode}): {result.stderr}")
    set_png_resolution(png)


def png_to_pdf(png: Path, pdf: Path, width_in: float, height_in: float) -> None:
    """Embed a cleaned 24-bit PNG losslessly in a one-page PDF.

    The PowerShell helper deliberately saves RGB PNGs.  PDF supports the PNG
    row filters directly through FlateDecode with Predictor 15, so the image
    bytes can be reused without Pillow, Matplotlib, or a browser raster pass.
    """
    source = png.read_bytes()
    if not source.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"Not a PNG: {png}")
    offset = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while offset < len(source):
        length = struct.unpack(">I", source[offset : offset + 4])[0]
        chunk_type = source[offset + 4 : offset + 8]
        payload = source[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (bit_depth, color_type, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise RuntimeError(
                    f"Expected non-interlaced 8-bit RGB PNG for PDF embedding; got "
                    f"bit_depth={bit_depth}, color_type={color_type}, interlace={interlace}"
                )
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or not idat:
        raise RuntimeError(f"PNG is missing IHDR or IDAT: {png}")

    page_width = width_in * 72.0
    page_height = height_in * 72.0
    content = f"q\n{page_width:.6f} 0 0 {page_height:.6f} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.6f} {page_height:.6f}] "
            "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns {width} >> "
            f"/Length {len(idat)} >>\nstream\n"
        ).encode("ascii")
        + bytes(idat)
        + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"endstream",
    ]

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for object_offset in offsets[1:]:
        output.extend(f"{object_offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(output)


def clean_one(config: DenseCleanup, output_root: Path) -> dict[str, object]:
    raw_png = output_root / "validation" / "origin_raw_exports" / config.plot_id / f"{config.stem}.origin.png"
    svg = output_root / "svg" / f"{config.stem}.svg"
    png = output_root / "png" / f"{config.stem}.png"
    pdf = output_root / "pdf" / f"{config.stem}.pdf"
    audit_path = output_root / "validation" / f"{config.plot_id}_dense_text_cleanup.json"
    build_audit_path = output_root / "validation" / f"{config.plot_id}_build_audit.json"
    if not raw_png.is_file():
        raise FileNotFoundError(raw_png)
    svg.write_text(svg_for(config, raw_png), encoding="utf-8", newline="\n")
    render_clean_png(raw_png, png, config)
    png_to_pdf(png, pdf, config.width_mm / 25.4, config.height_mm / 25.4)
    audit: dict[str, object] = {
        "status": "PASS",
        "plot_id": config.plot_id,
        "raw_origin_png": str(raw_png.resolve()),
        "raw_origin_png_sha256": sha256(raw_png),
        "raw_origin_png_preserved": True,
        "data_region_modified": False,
        "legend_mask_px": list(config.legend_mask) if config.legend_mask is not None else None,
        "x_title_mask_px": list(config.title_mask),
        "cleanup_reason": "Origin 2024 inherited bar-above decoration on manually created legend/title text",
        "replacement": "same title redrawn as Times New Roman; legend remains separate",
        "replacement_title_font": {
            "pixels_at_600_dpi": TITLE_FONT_PX,
            "points": round(TITLE_FONT_PX * 72 / 600, 2),
        },
        "outputs": {
            "png": str(png.resolve()), "png_sha256": sha256(png), "png_bytes": png.stat().st_size,
            "svg": str(svg.resolve()), "svg_sha256": sha256(svg), "svg_bytes": svg.stat().st_size,
            "pdf": str(pdf.resolve()), "pdf_sha256": sha256(pdf), "pdf_bytes": pdf.stat().st_size,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if build_audit_path.is_file():
        build_audit = json.loads(build_audit_path.read_text(encoding="utf-8"))
        exports = build_audit["exports"]
        exports.update({
            "png_bytes": png.stat().st_size,
            "svg_bytes": svg.stat().st_size,
            "pdf_bytes": pdf.stat().st_size,
            "export_mode": "complete_origin_600dpi_raster_backed_svg_pdf_with_vector_text_cleanup",
            "dense_text_cleanup_audit": str(audit_path.resolve()),
        })
        build_audit_path.write_text(json.dumps(build_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plot_id", nargs="*")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    unknown = sorted(set(args.plot_id) - set(CONFIG))
    if unknown:
        parser.error(f"unknown plot_id: {', '.join(unknown)}")
    selected = args.plot_id or sorted(CONFIG)
    reports = [clean_one(CONFIG[plot_id], args.output_root.resolve()) for plot_id in selected]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
