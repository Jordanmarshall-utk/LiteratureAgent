from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


BAD_REPAIR_METHODS = {
    "failed",
    "family_chunk_skipped",
    "family_skipped",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_failure_slugs(work_dir: Path, min_bad_events: int) -> tuple[set[str], dict[str, dict[str, Any]]]:
    details: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "bad_events": 0,
        "stages": Counter(),
        "repair_methods": Counter(),
        "sources": set(),
    })

    report = work_dir / "failed_json_outputs" / "json_failure_report.csv"
    if report.exists():
        with report.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                slug = (row.get("paper_slug") or "").strip()
                method = (row.get("repair_method") or "").strip()
                if not slug:
                    continue
                if method in BAD_REPAIR_METHODS:
                    details[slug]["bad_events"] += 1
                    details[slug]["stages"][row.get("stage") or "unknown"] += 1
                    details[slug]["repair_methods"][method] += 1
                    details[slug]["sources"].add(str(report))

    timing = work_dir / "timing_logs" / "paper_timing.jsonl"
    if timing.exists():
        with timing.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("event") != "paper_complete":
                    continue
                slug = str(event.get("paper_slug") or "").strip()
                status = str(event.get("status") or "").lower()
                error = str(event.get("error") or "")
                if slug and (status == "error" or "Model returned empty content" in error):
                    details[slug]["bad_events"] += max(1, min_bad_events)
                    details[slug]["stages"]["paper_timing_error"] += 1
                    details[slug]["repair_methods"]["timing_error"] += 1
                    details[slug]["sources"].add(str(timing))

    selected = {
        slug for slug, info in details.items()
        if int(info["bad_events"]) >= min_bad_events
    }
    normalized = {}
    for slug, info in details.items():
        normalized[slug] = {
            "bad_events": int(info["bad_events"]),
            "stages": dict(info["stages"]),
            "repair_methods": dict(info["repair_methods"]),
            "sources": sorted(info["sources"]),
            "selected_for_retry": slug in selected,
        }
    return selected, normalized


def registry_entries_for_slugs(registry_path: Path, slugs: set[str]) -> list[tuple[str, dict[str, Any]]]:
    data = load_json(registry_path)
    if not isinstance(data, dict):
        return []
    papers = data.get("papers")
    if not isinstance(papers, dict):
        return []
    out = []
    for key, entry in papers.items():
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug") or "")
        if slug in slugs or any(str(v).find(slug) >= 0 for slug in slugs for v in [key]):
            out.append((key, entry))
    return out


def candidate_artifacts(work_dir: Path, slugs: set[str]) -> list[Path]:
    files: list[Path] = []
    dirs: list[Path] = []
    for slug in sorted(slugs):
        for child in work_dir.rglob(f"*{slug}*"):
            if "_retry_quarantine_" in str(child):
                continue
            if child.is_dir():
                dirs.append(child)
            elif child.is_file():
                files.append(child)
    # Move deepest dirs first so nested paper_store directories do not collide.
    dirs = sorted(set(dirs), key=lambda p: len(p.parts), reverse=True)
    files = sorted(set(files), key=lambda p: len(p.parts), reverse=True)
    return files + dirs


def quarantine_paths(paths: list[Path], work_dir: Path, quarantine_dir: Path) -> list[dict[str, str]]:
    moved = []
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(work_dir)
        dest = quarantine_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            suffix = 1
            while dest.with_name(dest.name + f".{suffix}").exists():
                suffix += 1
            dest = dest.with_name(dest.name + f".{suffix}")
        shutil.move(str(path), str(dest))
        moved.append({"from": str(path), "to": str(dest)})
    return moved


def remove_registry_entries(registry_path: Path, slugs: set[str], apply: bool) -> list[str]:
    data = load_json(registry_path)
    if not isinstance(data, dict) or not isinstance(data.get("papers"), dict):
        return []
    papers = data["papers"]
    remove_keys = []
    for key, entry in list(papers.items()):
        if isinstance(entry, dict) and str(entry.get("slug") or "") in slugs:
            remove_keys.append(key)
        elif any(slug in str(key) for slug in slugs):
            remove_keys.append(key)
    if apply and remove_keys:
        for key in remove_keys:
            papers.pop(key, None)
        save_json(registry_path, data)
    return remove_keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quarantine partial/failed LiteratureAgent paper outputs so resume will retry them."
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--min-bad-events", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-slug", action="append", default=[],
                        help="Force a slug onto the retry list. Can be repeated.")
    args = parser.parse_args()

    work_dir = args.work_dir
    if not work_dir.exists():
        raise SystemExit(f"Work dir does not exist: {work_dir}")

    slugs, details = collect_failure_slugs(work_dir, args.min_bad_events)
    slugs.update(s.strip() for s in args.include_slug if s.strip())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir = work_dir / f"_retry_quarantine_{stamp}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    registry_paths = [work_dir / "paper_registry.json", work_dir / "combined_full_coverage" / "paper_registry.json"]
    registry_removals = {}
    for reg in registry_paths:
        if reg.exists():
            registry_removals[str(reg)] = remove_registry_entries(reg, slugs, apply=args.apply)

    artifacts = candidate_artifacts(work_dir, slugs)
    moved = quarantine_paths(artifacts, work_dir, audit_dir) if args.apply else []

    manifest = {
        "work_dir": str(work_dir),
        "apply": bool(args.apply),
        "min_bad_events": args.min_bad_events,
        "retry_slugs": sorted(slugs),
        "failure_details": details,
        "registry_removals": registry_removals,
        "artifact_count": len(artifacts),
        "moved_count": len(moved),
        "moved": moved,
    }
    manifest_path = audit_dir / "retry_prepare_manifest.json"
    save_json(manifest_path, manifest)

    csv_path = audit_dir / "retry_slugs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_slug", "bad_events", "selected_for_retry", "repair_methods"])
        writer.writeheader()
        for slug in sorted(details):
            info = details[slug]
            writer.writerow({
                "paper_slug": slug,
                "bad_events": info.get("bad_events", 0),
                "selected_for_retry": slug in slugs,
                "repair_methods": json.dumps(info.get("repair_methods", {}), ensure_ascii=False),
            })

    print("Retry preparation complete")
    print(f"Work dir: {work_dir}")
    print(f"Apply: {args.apply}")
    print(f"Retry slugs: {len(slugs)}")
    for slug in sorted(slugs)[:30]:
        print(f"  {slug}")
    if len(slugs) > 30:
        print(f"  ... {len(slugs) - 30} more")
    print(f"Artifact matches: {len(artifacts)}")
    print(f"Moved artifacts: {len(moved)}")
    print(f"Manifest: {manifest_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
