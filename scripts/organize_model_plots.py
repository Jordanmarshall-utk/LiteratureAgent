#!/usr/bin/env python3
"""Create a categorized, non-destructive gallery of model-result plots."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def category_for(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    name = path.name.lower()
    if "before_after" in text or "comparison" in text or "signed_metric" in name:
        return "before_after_comparison"
    if "physical_layer" in text or "condition_normalized" in text or "physical_condition" in text:
        return "physical_stability"
    if "/pce/" in text or name.startswith("pce_"):
        return "pce"
    if "classification" in text or "score_distribution" in name or "roc" in name:
        return "stability_classification"
    if "stability" in text:
        return "stability_regression"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy all model plots into a clean categorized gallery.")
    parser.add_argument("--run-dir", required=True, help="Model or comparison result folder to scan")
    parser.add_argument("--out", default=None, help="Gallery output folder; defaults to RUN_DIR/plot_gallery")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir / "plot_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    image_paths = sorted(
        p for p in run_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        and out_dir not in p.parents
    )
    for source in image_paths:
        category = category_for(source)
        category_dir = out_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        relative = source.relative_to(run_dir)
        stem = "__".join(relative.with_suffix("").parts)
        destination = category_dir / f"{stem}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        rows.append({
            "category": category,
            "source_relative_path": str(relative),
            "gallery_relative_path": str(destination.relative_to(out_dir)),
        })

    with (out_dir / "plot_gallery_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "source_relative_path", "gallery_relative_path"])
        writer.writeheader()
        writer.writerows(rows)

    readme = [
        "Model Plot Gallery",
        "==================",
        "",
        f"Source folder: {run_dir}",
        f"Plots copied: {len(rows)}",
        "",
        "Folders:",
        "- before_after_comparison",
        "- pce",
        "- stability_regression",
        "- stability_classification",
        "- physical_stability",
        "- other",
        "",
        "This gallery contains copies. Original model artifacts were not moved or modified.",
        "See plot_gallery_index.csv for the original path of every plot.",
    ]
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Copied {len(rows)} plots into: {out_dir}")


if __name__ == "__main__":
    main()
