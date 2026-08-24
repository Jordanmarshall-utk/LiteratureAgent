#!/usr/bin/env python3
"""Archive old LiteratureAgent run outputs before a fresh campaign.

Default behavior is dry-run. With --apply, this moves stale run outputs into an
archive folder instead of permanently deleting them. That keeps benchmarking,
knowledge graph, manuscript figure bundles, code, config, data, and secrets.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_KEEP_NAMES = {
    ".agents",
    ".git",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "config",
    "data",
    "docs",
    "literature_agent",
    "models",
    "polaris_integration",
    "scripts",
    "secrets",
    "literature_agent_full_end_to_end_v21_3_english_sanitizer.py",
    "paper_materials_packet_full_20260716",
    "paper_materials_packet_full_20260716.zip",
}

ARTIFACT_KEEP_PREFIXES = (
    "manuscript_figure_bundle",
    "kg_all_reasoning_policy_comparison",
    "weekly_report_knowledge_graph",
    "presentation_model_plots",
)

ARTIFACT_KEEP_CONTAINS = (
    "benchmark",
)

OUTPUT_KEEP_NAMES = {
    "README_OUTPUTS.md",
    "pce_then_stability_same_approach.py",
}

OUTPUT_KEEP_PREFIXES = (
    "benchmark",
    "kg_",
)


def should_keep_project_child(path: Path) -> tuple[bool, str]:
    if path.name in PROJECT_KEEP_NAMES:
        return True, "core_project_or_manuscript_packet"
    if path.name == "artifacts":
        return True, "artifacts_parent_filtered_inside"
    if path.name == "__pycache__":
        return False, "python_cache"
    return False, "stale_project_output_or_temp"


def should_keep_artifact_child(path: Path) -> tuple[bool, str]:
    name = path.name
    if name.startswith(ARTIFACT_KEEP_PREFIXES):
        return True, "kept_manuscript_or_knowledge_graph_artifact"
    if any(token in name.lower() for token in ARTIFACT_KEEP_CONTAINS):
        return True, "kept_benchmark_artifact"
    return False, "stale_diagnostic_or_old_report_artifact"


def should_keep_output_child(path: Path) -> tuple[bool, str]:
    name = path.name
    if name in OUTPUT_KEEP_NAMES:
        return True, "kept_output_reference_file"
    if name.startswith(OUTPUT_KEEP_PREFIXES):
        return True, "kept_benchmark_or_kg_output"
    return False, "old_literatureagent_run_output"


def unique_dest(base: Path, name: str) -> Path:
    dest = base / name
    if not dest.exists():
        return dest
    i = 2
    while True:
        candidate = base / f"{name}__{i}"
        if not candidate.exists():
            return candidate
        i += 1


def collect_actions(project_root: Path, output_root: Path, archive_root: Path) -> list[dict]:
    actions = []

    for child in sorted(project_root.iterdir(), key=lambda p: p.name.lower()):
        keep, reason = should_keep_project_child(child)
        if child.name == "artifacts":
            continue
        actions.append({
            "root": "project",
            "path": str(child),
            "name": child.name,
            "kind": "dir" if child.is_dir() else "file",
            "action": "keep" if keep else "archive",
            "reason": reason,
            "destination": "" if keep else str(unique_dest(archive_root / "project_root", child.name)),
        })

    artifacts = project_root / "artifacts"
    if artifacts.exists():
        for child in sorted(artifacts.iterdir(), key=lambda p: p.name.lower()):
            keep, reason = should_keep_artifact_child(child)
            actions.append({
                "root": "project_artifacts",
                "path": str(child),
                "name": child.name,
                "kind": "dir" if child.is_dir() else "file",
                "action": "keep" if keep else "archive",
                "reason": reason,
                "destination": "" if keep else str(unique_dest(archive_root / "project_artifacts", child.name)),
            })

    if output_root.exists():
        for child in sorted(output_root.iterdir(), key=lambda p: p.name.lower()):
            keep, reason = should_keep_output_child(child)
            actions.append({
                "root": "output_root",
                "path": str(child),
                "name": child.name,
                "kind": "dir" if child.is_dir() else "file",
                "action": "keep" if keep else "archive",
                "reason": reason,
                "destination": "" if keep else str(unique_dest(archive_root / "literature_output_root", child.name)),
            })

    return actions


def apply_actions(actions: list[dict]) -> None:
    for row in actions:
        if row["action"] != "archive":
            continue
        src = Path(row["path"])
        dest = Path(row["destination"])
        if not src.exists():
            row["applied"] = "source_missing"
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        row["applied"] = "moved"


def write_manifest(actions: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["root", "path", "name", "kind", "action", "reason", "destination", "applied"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in actions:
            writer.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "LITERATURE_AGENT_RUNTIME_ROOT",
            str(Path(__file__).resolve().parents[2] / "LiteratureAgent"),
        ),
    )
    parser.add_argument("--archive-root", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = Path(args.archive_root).resolve() if args.archive_root else project_root / f"_archive_before_fresh_campaign_{stamp}"

    actions = collect_actions(project_root, output_root, archive_root)
    if args.apply:
        apply_actions(actions)

    manifest = project_root / "artifacts" / f"fresh_campaign_cleanup_manifest_{stamp}.csv"
    write_manifest(actions, manifest)

    archive_count = sum(1 for row in actions if row["action"] == "archive")
    keep_count = sum(1 for row in actions if row["action"] == "keep")
    print(f"Cleanup {'APPLIED' if args.apply else 'DRY RUN'}")
    print(f"Project root: {project_root}")
    print(f"Output root: {output_root}")
    print(f"Archive root: {archive_root}")
    print(f"Keep entries: {keep_count}")
    print(f"Archive entries: {archive_count}")
    print(f"Manifest: {manifest}")
    print("Kept artifact/output entries include benchmarking, manuscript figure bundles, and knowledge graph outputs.")


if __name__ == "__main__":
    main()
