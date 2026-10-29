#!/usr/bin/env python3
"""Copy and renumber the five existing shared-legend PNG/SVG pairs.

Copies are byte-exact.  The audit records the legacy source name, current
manuscript mapping, source/destination SHA-256, and copy status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = ROOT / "source"
DEFAULT_SOURCE = ROOT / "figures" / "legends"
DEFAULT_OUTPUT = ROOT / "build"

LEGENDS = (
    {
        "legacy_stem": "Legend_Figure3_system_benchmark",
        "delivery_stem": "legend_figs6_system_benchmark",
        "used_by": ["figs6a_benchmark", "figs6b_benchmark", "figs6c_benchmark", "figs6d_benchmark"],
    },
    {
        "legacy_stem": "Legend_Figure4_historical_models",
        "delivery_stem": "legend_fig3_historical_models",
        "used_by": ["fig3a_historical_r2", "fig3b_historical_rmse", "fig3c_historical_mae"],
    },
    {
        "legacy_stem": "Legend_Figure4_accepted_expansion",
        "delivery_stem": "legend_fig3_literatureagent_expansion",
        "used_by": ["fig3d_literatureagent_r2", "fig3e_literatureagent_rmse", "fig3f_literatureagent_mae"],
    },
    {
        "legacy_stem": "Legend_Figure5_SHAP_direction",
        "delivery_stem": "legend_shap_direction",
        "used_by": ["fig4b_choice_shap", "figs1_global_shap", "figs2_solvent_shap"],
    },
    {
        "legacy_stem": "Legend_Figure6_stability_models",
        "delivery_stem": "legend_fig5_stability_models",
        "used_by": ["fig5a_stability_classification", "fig5b_stability_regression"],
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    status = "copied"
    if destination.is_file() and sha256(destination) == source_hash:
        status = "already_identical"
    else:
        shutil.copy2(source, destination)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"SHA-256 mismatch after copy: {source} -> {destination}")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "bytes": source.stat().st_size,
        "sha256_source": source_hash,
        "sha256_destination": destination_hash,
        "copy_status": status,
    }


def source_asset(
    source_root: Path,
    mapping: dict[str, object],
    extension: str,
) -> Path:
    """Resolve either a packaged canonical legend or the legacy flat asset."""

    packaged = (
        source_root
        / ("png_600dpi" if extension == ".png" else "svg")
        / f"{mapping['delivery_stem']}{extension}"
    )
    if packaged.is_file():
        return packaged
    return source_root / f"{mapping['legacy_stem']}{extension}"


def run(source_root: Path, output_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    legend_root = output_root / "legends"
    (legend_root / "origin_projects").mkdir(parents=True, exist_ok=True)
    extension_dirs = {".png": legend_root / "PNG_600dpi", ".svg": legend_root / "SVG"}
    assets: list[dict[str, object]] = []
    for mapping in LEGENDS:
        for extension in (".png", ".svg"):
            source = source_asset(source_root, mapping, extension)
            destination = extension_dirs[extension] / f"{mapping['delivery_stem']}{extension}"
            record = copy_verified(source, destination)
            record.update(
                {
                    "legacy_stem": mapping["legacy_stem"],
                    "delivery_stem": mapping["delivery_stem"],
                    "used_by": mapping["used_by"],
                    "format": extension.removeprefix(".").upper(),
                }
            )
            assets.append(record)

    pair_checks: list[dict[str, object]] = []
    for mapping in LEGENDS:
        matching = [asset for asset in assets if asset["delivery_stem"] == mapping["delivery_stem"]]
        formats = sorted(str(asset["format"]) for asset in matching)
        pair_checks.append(
            {
                "delivery_stem": mapping["delivery_stem"],
                "formats": formats,
                "pair_complete": formats == ["PNG", "SVG"],
            }
        )
    if not all(bool(item["pair_complete"]) for item in pair_checks):
        raise RuntimeError("One or more shared-legend pairs is incomplete")

    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "byte_exact_copy_and_current_figure_renaming",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "legend_root": str(legend_root),
        "origin_projects_root": str((legend_root / "origin_projects").resolve()),
        "summary": {
            "legend_pair_count": len(LEGENDS),
            "asset_count": len(assets),
            "copied_count": sum(asset["copy_status"] == "copied" for asset in assets),
            "already_identical_count": sum(asset["copy_status"] == "already_identical" for asset in assets),
            "sha256_mismatch_count": 0,
        },
        "pair_checks": pair_checks,
        "assets": assets,
    }
    audit_dir = legend_root / "validation"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "existing_shared_legend_copy_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit["audit_path"] = str(audit_path.resolve())
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        checks = []
        for mapping in LEGENDS:
            for extension in (".png", ".svg"):
                source = source_asset(args.source_root, mapping, extension)
                if not source.is_file() or source.stat().st_size == 0:
                    raise FileNotFoundError(source)
                checks.append(
                    {
                        "source": str(source.resolve()),
                        "sha256": sha256(source),
                        "planned_destination": str(
                            (
                                args.output_root
                                / "legends"
                                / ("PNG_600dpi" if extension == ".png" else "SVG")
                                / f"{mapping['delivery_stem']}{extension}"
                            ).resolve()
                        ),
                    }
                )
        print(json.dumps({"status": "PASS", "mode": "dry-run", "assets": checks}, indent=2))
        return 0
    audit = run(args.source_root, args.output_root)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
