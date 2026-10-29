#!/usr/bin/env python3
"""Copy retained reference assets into the figure build directory.

The source list is read from PLOT_MANIFEST.csv.  Every copy is verified by SHA-256 and recorded in a JSON audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = ROOT / "source"
DEFAULT_PACKAGE_ROOT = PUBLIC_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "build"
ALLOWED_SUFFIXES = {".pdf", ".png", ".opju", ".svg"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_within(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path and reject paths that escape the package."""

    candidate = (root / Path(relative_path)).resolve()
    # Source manifests may refer directly to canonical assets in the sibling
    # figures directory; all paths must still remain in the manuscript tree.
    allowed_root = root.parent if root.name == "source" and root.parent.name == "manuscript" else root
    try:
        candidate.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes manuscript root: {relative_path}") from exc
    return candidate


def copy_retained_assets(package_root: Path, output_root: Path) -> dict:
    package_root = package_root.resolve()
    output_root = output_root.resolve()
    manifest_path = package_root / "PLOT_MANIFEST.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    retained_root = output_root / "retained_reference_assets"
    retained_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    retained_plot_ids: list[str] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {
            "plot_id",
            "required_action",
            "scientific_status",
            "editable_or_vector_reference_paths",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "PLOT_MANIFEST.csv is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            if row["required_action"].strip().upper() != "RETAIN":
                continue
            if row["scientific_status"].strip().upper() != "READY":
                continue

            plot_id = row["plot_id"].strip()
            if not plot_id:
                raise ValueError("READY+RETAIN manifest row has an empty plot_id")
            retained_plot_ids.append(plot_id)
            destination_dir = retained_root / plot_id
            destination_dir.mkdir(parents=True, exist_ok=True)

            source_entries = [
                item.strip()
                for item in row["editable_or_vector_reference_paths"].split(";")
                if item.strip()
            ]
            if not source_entries:
                raise ValueError(f"{plot_id} has no retained reference assets")

            destination_names: set[str] = set()
            for source_relative in source_entries:
                source = resolve_within(package_root, source_relative)
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Missing retained asset for {plot_id}: {source_relative}"
                    )
                if source.suffix.lower() not in ALLOWED_SUFFIXES:
                    raise ValueError(
                        f"Unsupported retained asset type for {plot_id}: {source.name}"
                    )
                destination_name = (Path(source.parent.name) / source.name).as_posix()
                if destination_name in destination_names:
                    raise ValueError(
                        f"Destination filename collision for {plot_id}: {source.name}"
                    )
                destination_names.add(destination_name)

                destination = destination_dir / destination_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_hash = sha256_file(source)
                status = "copied"
                if destination.is_file() and sha256_file(destination) == source_hash:
                    status = "already_identical"
                else:
                    shutil.copy2(source, destination)

                destination_hash = sha256_file(destination)
                if destination_hash != source_hash:
                    raise RuntimeError(
                        f"SHA-256 mismatch after copy: {source} -> {destination}"
                    )

                records.append(
                    {
                        "plot_id": plot_id,
                        "filename": source.name,
                        "source_relative_to_package": source_relative.replace("\\", "/"),
                        "destination_relative_to_output": destination.relative_to(
                            output_root
                        ).as_posix(),
                        "size_bytes": source.stat().st_size,
                        "sha256_source": source_hash,
                        "sha256_destination": destination_hash,
                        "copy_status": status,
                    }
                )

    if len(retained_plot_ids) != len(set(retained_plot_ids)):
        raise ValueError("PLOT_MANIFEST.csv contains duplicate READY+RETAIN plot IDs")

    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "copy_ready_retain_assets_without_regeneration",
        "source_manifest": str(manifest_path),
        "package_root": str(package_root),
        "output_root": str(output_root),
        "retained_asset_root": str(retained_root),
        "summary": {
            "ready_retain_plot_count": len(retained_plot_ids),
            "asset_count": len(records),
            "copied_count": sum(r["copy_status"] == "copied" for r in records),
            "already_identical_count": sum(
                r["copy_status"] == "already_identical" for r in records
            ),
            "sha256_mismatch_count": 0,
        },
        "retained_plot_ids": retained_plot_ids,
        "assets": records,
    }
    audit_path = retained_root / "retained_assets_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    audit["audit_path"] = str(audit_path)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help=f"Figure source directory (default: {DEFAULT_PACKAGE_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Delivery root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = copy_retained_assets(args.package_root, args.output_root)
    summary = audit["summary"]
    print(f"Retained plots: {summary['ready_retain_plot_count']}")
    print(f"Verified assets: {summary['asset_count']}")
    print(f"Copied: {summary['copied_count']}")
    print(f"Already identical: {summary['already_identical_count']}")
    print(f"Audit: {audit['audit_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
