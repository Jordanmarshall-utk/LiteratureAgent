#!/usr/bin/env python3
"""Shared print-size, export, and validation utilities for manuscript panels.

The physical panel size is part of the figure specification.  Panels are saved
without a tight bounding-box crop so that an 85 mm panel remains 85 mm when it
is inserted into a two-column, 180 mm manuscript figure.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image


MM_PER_INCH = 25.4
DEFAULT_DPI = 600
FONT_FAMILY = "Times New Roman"
FALLBACK_FONTS = [FONT_FAMILY, "Times", "Liberation Serif", "DejaVu Serif"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Locked, colorblind-safe manuscript palette. Light colors are reserved for
# grids and backgrounds; data marks use the saturated colors below.
BLUE = "#0072B2"
SKY = "#3C8DBC"
TEAL = "#009E73"
AMBER = "#E69F00"
ORANGE = "#D55E00"
PURPLE = "#7B61A8"
NEUTRAL = "#4B5563"
GRID = "#D9DEE5"
PALE = "#F4F6F8"
INK = "#202124"


def mm_to_inches(value_mm: float) -> float:
    return float(value_mm) / MM_PER_INCH


@dataclass(frozen=True)
class PanelSpec:
    """Physical and semantic contract for one independently exported panel."""

    panel_id: str
    stem: str
    title: str
    width_mm: float
    height_mm: float
    figure: str
    panel_label: str = ""
    role: str = "main"
    renderer: str = ""
    min_text_pt: float = 7.5
    min_tick_pt: float = 7.5
    min_axis_label_pt: float = 8.5
    dpi: int = DEFAULT_DPI

    @property
    def figsize(self) -> tuple[float, float]:
        return mm_to_inches(self.width_mm), mm_to_inches(self.height_mm)

    @property
    def pixel_size(self) -> tuple[int, int]:
        # Matplotlib's raster canvas truncates fractional device pixels.
        return tuple(int(value * self.dpi) for value in self.figsize)


PRINT_RCPARAMS = {
    "font.family": "serif",
    "font.serif": FALLBACK_FONTS,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "LiteratureAgent-manuscript-panels-v1",
}


@contextmanager
def publication_style(**overrides: object) -> Iterator[None]:
    """Apply the shared final-print style only within a renderer."""

    parameters = dict(PRINT_RCPARAMS)
    parameters.update(overrides)
    with mpl.rc_context(parameters):
        yield


def new_panel(spec: PanelSpec, **kwargs: object) -> tuple[plt.Figure, plt.Axes]:
    """Create one panel at its exact intended physical size."""

    return plt.subplots(figsize=spec.figsize, **kwargs)


def _visible_text(fig: plt.Figure) -> list[Text]:
    return [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and str(artist.get_text()).strip()
    ]


def validate_figure(fig: plt.Figure, spec: PanelSpec) -> dict[str, object]:
    """Validate physical dimensions and effective final-size typography."""

    width_in, height_in = fig.get_size_inches()
    width_mm = float(width_in * MM_PER_INCH)
    height_mm = float(height_in * MM_PER_INCH)
    if abs(width_mm - spec.width_mm) > 0.05 or abs(height_mm - spec.height_mm) > 0.05:
        raise ValueError(
            f"{spec.panel_id}: figure is {width_mm:.2f} x {height_mm:.2f} mm; "
            f"expected {spec.width_mm:.2f} x {spec.height_mm:.2f} mm"
        )

    # Matplotlib locators may emit a tick just outside the active limits.  The
    # corresponding label is not clipped and can leak beyond a fixed canvas.
    # Freeze each locator to ticks that are actually inside the visible range.
    for axis in fig.axes:
        x_low, x_high = sorted(axis.get_xlim())
        x_ticks = [
            tick
            for tick in axis.get_xticks()
            if x_low - 1e-12 <= float(tick) <= x_high + 1e-12
        ]
        if x_ticks:
            axis.set_xticks(x_ticks)
        y_low, y_high = sorted(axis.get_ylim())
        y_ticks = [
            tick
            for tick in axis.get_yticks()
            if y_low - 1e-12 <= float(tick) <= y_high + 1e-12
        ]
        if y_ticks:
            axis.set_yticks(y_ticks)

    fig.canvas.draw()
    texts = _visible_text(fig)
    sizes = [float(text.get_fontsize()) for text in texts]
    too_small = [
        (str(text.get_text()).replace("\n", " / ")[:90], float(text.get_fontsize()))
        for text in texts
        if float(text.get_fontsize()) + 1e-9 < spec.min_text_pt
    ]
    if too_small:
        sample = "; ".join(f"{text!r} ({size:g} pt)" for text, size in too_small[:6])
        raise ValueError(
            f"{spec.panel_id}: text below {spec.min_text_pt:g} pt at final size: {sample}"
        )

    tick_sizes: list[float] = []
    axis_label_sizes: list[float] = []
    for axis in fig.axes:
        tick_sizes.extend(
            float(label.get_fontsize())
            for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]
            if label.get_visible() and str(label.get_text()).strip()
        )
        axis_label_sizes.extend(
            float(label.get_fontsize())
            for label in (axis.xaxis.label, axis.yaxis.label)
            if label.get_visible() and str(label.get_text()).strip()
        )
    if tick_sizes and min(tick_sizes) + 1e-9 < spec.min_tick_pt:
        raise ValueError(
            f"{spec.panel_id}: smallest tick label is {min(tick_sizes):.2f} pt; "
            f"minimum is {spec.min_tick_pt:.2f} pt"
        )
    if axis_label_sizes and min(axis_label_sizes) + 1e-9 < spec.min_axis_label_pt:
        raise ValueError(
            f"{spec.panel_id}: smallest axis label is {min(axis_label_sizes):.2f} pt; "
            f"minimum is {spec.min_axis_label_pt:.2f} pt"
        )

    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    outside: list[str] = []
    tolerance_px = 3.0
    for text in texts:
        bounds = text.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < -tolerance_px
            or bounds.y0 < -tolerance_px
            or bounds.x1 > canvas_width + tolerance_px
            or bounds.y1 > canvas_height + tolerance_px
        ):
            outside.append(str(text.get_text()).replace("\n", " / ")[:90])
    if outside:
        raise ValueError(
            f"{spec.panel_id}: visible text extends beyond the fixed-size canvas: "
            + "; ".join(repr(value) for value in outside[:6])
        )

    return {
        "width_mm": round(width_mm, 3),
        "height_mm": round(height_mm, 3),
        "minimum_visible_text_pt": min(sizes) if sizes else None,
        "minimum_tick_text_pt": min(tick_sizes) if tick_sizes else None,
        "minimum_axis_label_pt": min(axis_label_sizes) if axis_label_sizes else None,
        "visible_text_count": len(texts),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_source_path(path: Path) -> str:
    """Return a project-relative source path when the source belongs to the repo."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def export_panel(
    fig: plt.Figure,
    out_dir: Path,
    spec: PanelSpec,
    *,
    source_files: list[Path] | None = None,
) -> dict[str, object]:
    """Validate and export a panel as exact-size PNG, SVG, and vector PDF."""

    out_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_figure(fig, spec)
    paths = {
        "png": out_dir / f"{spec.stem}.png",
        "svg": out_dir / f"{spec.stem}.svg",
        "pdf": out_dir / f"{spec.stem}.pdf",
    }
    common = {
        "facecolor": "white",
        "edgecolor": "none",
        "bbox_inches": None,
        "pad_inches": 0,
        # Vector backends use this resolution for rasterized artists while
        # preserving text, axes, and other vector-native elements.
        "dpi": spec.dpi,
    }
    fig.savefig(
        paths["png"],
        metadata={"Software": "LiteratureAgent manuscript panel renderer"},
        **common,
    )
    fig.savefig(
        paths["svg"],
        metadata={"Creator": "LiteratureAgent manuscript panel renderer", "Date": None},
        **common,
    )
    fig.savefig(
        paths["pdf"],
        metadata={
            "Creator": "LiteratureAgent manuscript panel renderer",
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    plt.close(fig)

    with Image.open(paths["png"]) as image:
        pixel_size = image.size
        dpi_info = tuple(round(float(value), 2) for value in image.info.get("dpi", (0.0, 0.0)))
    if pixel_size != spec.pixel_size:
        raise ValueError(
            f"{spec.panel_id}: PNG is {pixel_size}; expected {spec.pixel_size}. "
            "A tight crop or implicit resize changed the physical-size contract."
        )

    sources = []
    for source in source_files or []:
        resolved = source.resolve()
        sources.append(
            {
                "path": _portable_source_path(resolved),
                "sha256": _sha256(resolved) if resolved.is_file() else None,
            }
        )
    record = {
        **asdict(spec),
        **validation,
        "pixel_width": pixel_size[0],
        "pixel_height": pixel_size[1],
        "png_dpi_metadata": dpi_info,
        "exports": {
            extension: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for extension, path in paths.items()
        },
        "sources": sources,
        "physical_size_preserved": True,
        "tight_bbox_crop_used": False,
    }
    metadata_path = out_dir / f"{spec.stem}.panel.json"
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
