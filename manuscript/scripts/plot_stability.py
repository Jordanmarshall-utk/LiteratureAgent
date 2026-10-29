"""Render the two manuscript stability panels from the released estimate tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
PANELS = ("fig5a_stability_classification", "fig5b_stability_regression")
SERIES = {"baseline": ("Original", "s", "#33546E", -0.13), "integrated": ("Integrated", "o", "#F28A00", 0.13)}
LABELS = {
    "target_end_retention_ge80": "End retention\n≥80%",
    "target_end_retention_ge80_after_100h": "End ≥80%\n≥100 h test",
    "target_end_retention_ge80_after_500h": "End ≥80%\n≥500 h test",
    "target_T80_ge100h": "T80 ≥100 h",
    "target_T80_ge500h": "T80 ≥500 h",
    "target_after_1000h_ge80": "1000 h retention\n≥80%",
    "target_end_retention_pct": "End retention (%)",
    "target_after_1000h_retention_pct": "1000 h retention (%)",
    "target_log1p_T80_hours": "ln(1 + T80/h)",
    "target_phys_log_k_obs": "Calculated ln(k)\n(k in h⁻¹)",
    "target_phys_log_k_ref": "Normalized ln(k)\n(k in h⁻¹)",
    "target_phys_log1p_T80_ref_hours": "Norm. ln(1 + T80/h)",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 estimates in {path}, found {len(rows)}")
    keys = {(row["target"], row["dataset"]) for row in rows}
    if len(keys) != 12 or {row["dataset"] for row in rows} != set(SERIES):
        raise ValueError("Each endpoint requires one Original and one Integrated estimate.")
    for row in rows:
        value, low, high = (float(row[key]) for key in ("estimate", "ci_lower", "ci_upper"))
        if not low <= value <= high:
            raise ValueError(f"Invalid confidence interval for {row['target']}/{row['dataset']}")
        if "predicted" in row["feature_mode"]:
            raise ValueError("Predicted PCE is excluded from the main stability panels.")
    return rows


def render(panel: str, output: Path) -> dict[str, object]:
    source = ROOT / "source/data" / panel / "plot_data.csv"
    rows = read_rows(source)
    targets = sorted({row["target"] for row in rows}, key=lambda target: min(int(row["endpoint_order"]) for row in rows if row["target"] == target))
    if len(targets) != 6:
        raise ValueError("Each stability panel requires six endpoints.")
    fig = plt.figure(figsize=(85 / 25.4, 72 / 25.4), facecolor="white")
    ax = fig.add_axes((0.565, 0.185, 0.41, 0.77))
    for dataset, (label, marker, color, offset) in SERIES.items():
        selected = [next(row for row in rows if row["target"] == target and row["dataset"] == dataset) for target in targets]
        values = [float(row["estimate"]) for row in selected]
        lower = [value - float(row["ci_lower"]) for value, row in zip(values, selected)]
        upper = [float(row["ci_upper"]) - value for value, row in zip(values, selected)]
        ax.errorbar(values, [position + offset for position in range(6)], xerr=[lower, upper], fmt=marker,
                    color=color, ecolor=color, markerfacecolor="white", markeredgecolor=color,
                    markersize=4.5, markeredgewidth=0.8, elinewidth=1.5, capsize=2.3, capthick=1.1,
                    linestyle="none", label=label, zorder=3)
    ax.set_yticks(range(6), [LABELS[target] for target in targets], fontsize=12)
    ax.set_ylim(5.55, -0.55)
    ax.tick_params(axis="y", length=2.5, width=0.65, pad=5)
    ax.tick_params(axis="x", labelsize=12, length=3, width=0.65)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.65)
    ax.grid(False)
    if panel.startswith("fig5a"):
        ax.set_xlim(0.50, 0.715)
        ax.set_xticks((0.50, 0.60, 0.70))
        ax.set_xlabel("ROC-AUC", fontsize=13, labelpad=5)
    else:
        ax.set_xlim(-0.12, 0.18)
        ax.set_xticks((-0.10, 0.00, 0.10))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("R²", fontsize=13, labelpad=5)
        ax.axvline(0, color="#9A9A9A", linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    clipped = []
    for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label]:
        box = text.get_window_extent(renderer)
        if box.x0 < page.x0 or box.x1 > page.x1 or box.y0 < page.y0 or box.y1 > page.y1:
            clipped.append(text.get_text())
    if clipped:
        plt.close(fig)
        raise ValueError(f"Text extends beyond the figure boundary: {clipped}")
    paths = []
    for kind, extension in (("svg", "svg"), ("pdf", "pdf"), ("png_600dpi", "png"), ("editable", "svg")):
        destination = output / kind / f"{panel}.{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"Creator": "LiteratureAgent"} if extension in {"svg", "pdf"} else {"Software": "LiteratureAgent"}
        if extension == "svg":
            metadata["Date"] = None
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        fig.savefig(destination, dpi=600, facecolor="white", metadata=metadata)
        paths.append(destination.relative_to(output).as_posix())
    plt.close(fig)
    return {"panel": panel, "source": source.relative_to(ROOT).as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "estimates": 12, "endpoints": 6, "confidence_intervals": 12, "dimensions_mm": [85, 72],
            "font_family": "Times New Roman", "body_size_pt": 12, "axis_label_size_pt": 13,
            "editable_format": "SVG", "outputs": paths, "text_within_page": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", nargs="?", default="all", choices=("all", *PANELS))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "build/stability")
    args = parser.parse_args()
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42})
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = [render(panel, output) for panel in (PANELS if args.panel == "all" else (args.panel,))]
    report = {"status": "PASS", "analysis": "2026-10-28_controlled_repaired_completion", "panels": reports}
    (output / "stability_render_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
