#!/usr/bin/env python
"""Run controlled, resumable LiteratureAgent expansion batches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from math import ceil
from pathlib import Path


DEFAULT_BATCHES = {"pilot": 100, "scale": 750, "full": 1000}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(
    os.environ.get("LITERATURE_AGENT_RUNTIME_ROOT", PROJECT_ROOT.parent / "LiteratureAgent")
).expanduser()
DEFAULT_MODEL_SCRIPT = PROJECT_ROOT / "models" / "pce_then_stability_same_approach.py"
VERIFIED_PROCESSED_REGISTRY = (
    PROJECT_ROOT
    / "data"
    / "expansion_processed_papers"
    / "processed_papers_registry_v21_12_clean.csv"
)
LEGACY_PROCESSED_REGISTRY = PROJECT_ROOT / "data" / "expansion_processed_papers" / "processed_papers_registry.csv"
DEFAULT_PROCESSED_REGISTRY = (
    VERIFIED_PROCESSED_REGISTRY if VERIFIED_PROCESSED_REGISTRY.exists() else LEGACY_PROCESSED_REGISTRY
)
FULL_ORIGINAL_BASE_CSV = PROJECT_ROOT / "data" / "Perovskite_database_content_all_data.csv"
FILTERED_BASE_CSV = PROJECT_ROOT / "data" / "Perovskite_database_content_2018plus.csv"
TRUSTED_GOOGLE_DRIVE_UPDATED_CSV = (
    RUNTIME_ROOT
    / "artifacts_literature_dataset_update"
    / "updated_perovskite_database_shared_schema_control.csv"
)
TRUSTED_GOOGLE_DRIVE_UPDATED_FALLBACK_CSV = (
    RUNTIME_ROOT
    / "artifacts_literature_dataset_update"
    / "updated_perovskite_database_with_literature_agent.csv"
)


def first_existing_path(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


DEFAULT_BASE_CSV = Path(
    os.environ.get(
        "LITERATURE_AGENT_BASE_CSV",
        first_existing_path(
            FULL_ORIGINAL_BASE_CSV,
            TRUSTED_GOOGLE_DRIVE_UPDATED_CSV,
            TRUSTED_GOOGLE_DRIVE_UPDATED_FALLBACK_CSV,
            FILTERED_BASE_CSV,
        ),
    )
).expanduser()
DEFAULT_CAMPAIGN_ROOT = Path(
    os.environ.get("LITERATURE_AGENT_CAMPAIGN_ROOT", RUNTIME_ROOT / "expansion_campaign")
).expanduser()
DEFAULT_WORK_DIR = Path(
    os.environ.get("LITERATURE_AGENT_WORK_DIR", RUNTIME_ROOT / "lit_outputs")
).expanduser()
TARGET_QUERIES = {
    "model_ready": (
        '"perovskite solar cell" "Table 1" "power conversion efficiency" '
        '"Voc" "Jsc" "fill factor" "stability" "retention"'
    ),
    "model_ready_pce_table": (
        '"perovskite solar cell" "Table" "PCE" "Voc" "Jsc" '
        '"fill factor" "device structure"'
    ),
    "model_ready_stability_table": (
        '"perovskite solar cell" "stability test" "retention" "hours" '
        '"T80" "initial PCE" "encapsulation"'
    ),
    "model_ready_mpp": (
        '"perovskite solar cell" "maximum power point" "operational stability" '
        '"T80" "retention" "initial efficiency"'
    ),
    "model_ready_aging_conditions": (
        '"perovskite solar cell" "aging" "humidity" "temperature" '
        '"illumination" "retention" "PCE"'
    ),
    "pce_stability": (
        '"perovskite solar cell" "power conversion efficiency" device '
        '"J-V" "Voc" "Jsc" "fill factor" stability T80 retention'
    ),
    "pce": (
        '"perovskite solar cell" "power conversion efficiency" device '
        '"J-V" "Voc" "Jsc" "fill factor" "device architecture"'
    ),
    "stability": (
        '"perovskite solar cell" stability T80 retention lifetime aging '
        'humidity illumination encapsulation "initial PCE"'
    ),
    "processing": (
        '"perovskite solar cell" device PCE additive passivation annealing '
        'solvent "spin coating" "device performance"'
    ),
    "high_efficiency": (
        '"perovskite solar cell" "power conversion efficiency" "certified" '
        '"device" "Voc" "Jsc" "fill factor"'
    ),
    "operational_stability": (
        '"perovskite solar cell" "operational stability" "maximum power point" '
        'T80 T95 retention "initial PCE"'
    ),
    "thermal_humidity_stability": (
        '"perovskite solar cell" stability humidity thermal temperature '
        'encapsulation retention aging "device"'
    ),
    "interface_passivation": (
        '"perovskite solar cell" interface passivation ETL HTL device '
        'PCE stability retention'
    ),
}
SWEEP_TARGET_MODES = [
    "model_ready",
    "model_ready_pce_table",
    "model_ready_stability_table",
    "model_ready_mpp",
    "model_ready_aging_conditions",
    "pce_stability",
    "pce",
    "stability",
    "operational_stability",
    "thermal_humidity_stability",
    "high_efficiency",
    "interface_passivation",
    "processing",
]
STABILITY_SWEEP_TARGET_MODES = [
    "model_ready_stability_table",
    "model_ready_mpp",
    "model_ready_aging_conditions",
    "operational_stability",
    "thermal_humidity_stability",
    "stability",
    "pce_stability",
]
STABILITY_TARGET_MODES = set(STABILITY_SWEEP_TARGET_MODES[:-1])

# The original sweep above is intentionally compact, but it is too shallow for
# scale campaigns: the same highly ranked Crossref results recur across most
# queries. These broader topic queries expose different parts of the device
# literature while the downstream model_ready_strict gate still decides which
# candidates are worth retrieving and extracting.
EXPANDED_QUERY_TOPICS = [
    ("inverted", "inverted p-i-n"),
    ("regular", "regular n-i-p"),
    ("planar", "planar heterojunction"),
    ("mesoporous", "mesoporous"),
    ("carbon_electrode", "carbon electrode"),
    ("flexible", "flexible device"),
    ("tandem", "tandem wide bandgap"),
    ("fapbi3", "FAPbI3"),
    ("cspbi3", "CsPbI3"),
    ("mixed_cation", "mixed cation mixed halide"),
    ("tin_lead", "tin lead narrow bandgap"),
    ("additive", "additive engineering"),
    ("solvent", "solvent engineering antisolvent"),
    ("crystallization", "crystallization nucleation"),
    ("grain_boundary", "grain boundary passivation"),
    ("buried_interface", "buried interface passivation"),
    ("surface_passivation", "surface passivation"),
    ("self_assembled_monolayer", "self assembled monolayer"),
    ("electron_transport", "electron transport layer ETL"),
    ("hole_transport", "hole transport layer HTL"),
    ("nickel_oxide", "nickel oxide NiOx"),
    ("sno2", "SnO2 electron transport"),
    ("spiro", "Spiro-OMeTAD"),
    ("annealing", "annealing temperature processing"),
    ("blade_coating", "blade coating scalable"),
    ("slot_die", "slot die coating module"),
    ("operational", "operational stability maximum power point"),
    ("light_soaking", "light soaking stability"),
    ("humidity", "humidity aging retention"),
    ("thermal", "thermal aging retention"),
    ("shelf_life", "shelf life stability"),
    ("encapsulation", "encapsulation stability"),
    ("damp_heat", "damp heat stability"),
    ("t80", "T80 lifetime"),
    ("retention_1000h", "1000 hours retention"),
]


def _expanded_query_plan() -> list[tuple[str, str]]:
    plan: list[tuple[str, str]] = []
    for name, topic in EXPANDED_QUERY_TOPICS:
        plan.append((
            f"expanded_pce_{name}",
            f"perovskite solar cell {topic} device PCE Voc Jsc fill factor",
        ))
        plan.append((
            f"expanded_stability_{name}",
            f"perovskite solar cell {topic} stability retention hours PCE",
        ))
    return plan


def _expanded_stability_query_plan() -> list[tuple[str, str]]:
    return [
        (mode, query)
        for mode, query in _expanded_query_plan()
        if mode.startswith("expanded_stability_")
    ]


def run(command: list[str], dry_run: bool) -> None:
    print("\n>", subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    print(f"[EXPANSION] Started command at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    proc = subprocess.Popen(command, env=env)
    return_code = proc.wait()
    elapsed = time.time() - started
    print(f"[EXPANSION] Command finished with exit={return_code} elapsed={elapsed/60:.1f} min", flush=True)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def find_csv(folder: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_previous_updated_csv(campaign_root: Path, current_batch: Path) -> Path | None:
    candidates = [
        path
        for path in campaign_root.glob("*/integration/updated_perovskite_database_with_literature_agent.csv")
        if current_batch not in path.parents
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def find_previous_updated_csv_for_base(campaign_root: Path, current_batch: Path, base_csv: Path) -> Path | None:
    """Return the newest previous integration output only if it used the same baseline CSV."""
    base_norm = str(base_csv.resolve()).lower() if base_csv.exists() else str(base_csv).lower()
    candidates = [
        path
        for path in campaign_root.glob("*/integration/updated_perovskite_database_with_literature_agent.csv")
        if current_batch not in path.parents
    ]
    compatible: list[Path] = []
    for path in candidates:
        summary = path.parent / "literature_update_summary.json"
        if not summary.exists():
            continue
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            previous_base = Path(str(payload.get("base_csv") or ""))
            previous_norm = str(previous_base.resolve()).lower() if previous_base.exists() else str(previous_base).lower()
        except Exception:
            continue
        if previous_norm == base_norm:
            compatible.append(path)
    return max(compatible, key=lambda path: path.stat().st_mtime) if compatible else None


def find_incomplete_cumulative_batch(
    campaign_root: Path,
    stage: str,
    batch_size: int,
) -> Path | None:
    """Find the newest unfinished batch with recoverable round snapshots."""
    batches_dir = campaign_root / "batches"
    if not batches_dir.exists():
        return None
    prefix = f"{stage}_{batch_size}_"
    candidates = []
    for path in batches_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        if (path / "batch_manifest.json").exists():
            continue
        rounds_dir = path / "raw_literature_records" / "rounds"
        progress = path / "raw_literature_records" / "all_records_for_this_batch.progress.csv"
        if progress.exists() or (rounds_dir.exists() and any(rounds_dir.glob("round_*.csv"))):
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def round_index_from_snapshot(path: Path) -> int:
    match = re.match(r"round_(\d+)_", path.name)
    return int(match.group(1)) if match else 0


def newest_records_csv(work_dir: Path) -> Path:
    candidates = [
        work_dir / "csv" / "all_records_from_manager.csv",
        work_dir / "combined_full_coverage" / "all_records_from_manager.csv",
        work_dir / "csv" / "all_records.csv",
        work_dir / "combined_full_coverage" / "csv" / "all_records.csv",
        work_dir / "all_records_from_manager.csv",
        work_dir / "all_records.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No LiteratureAgent records CSV found under {work_dir}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def per_paper_record_csvs(work_dir: Path) -> list[Path]:
    """Return individual paper record CSVs, excluding aggregate/summary/audit files."""
    csv_dir = work_dir / "csv"
    if not csv_dir.exists():
        return []
    blocked_prefixes = (
        "all_records",
        "paper_summaries",
        "accepted",
        "rejected",
        "duplicate",
        "integration",
        "audit",
    )
    out = []
    for path in sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime):
        name = path.name.lower()
        if any(name.startswith(prefix) for prefix in blocked_prefixes):
            continue
        out.append(path)
    return out


def snapshot_records(work_dir: Path, batch_dir: Path) -> Path:
    raw_dir = batch_dir / "raw_literature_records"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "all_records_for_this_batch.csv"
    per_paper = per_paper_record_csvs(work_dir)
    try:
        source = newest_records_csv(work_dir)
        aggregate_rows = count_csv_rows(source) or 0
    except FileNotFoundError:
        source = None
        aggregate_rows = 0
    per_paper_rows = sum(count_csv_rows(path) or 0 for path in per_paper)
    if per_paper and per_paper_rows > aggregate_rows:
        merge_record_csvs(per_paper, target)
    elif source:
        target.write_bytes(source.read_bytes())
    else:
        merge_record_csvs(per_paper, target)
    return target


def snapshot_round_records(work_dir: Path, batch_dir: Path, round_index: int, query_mode: str) -> Path:
    raw_dir = batch_dir / "raw_literature_records" / "rounds"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_mode = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in query_mode)
    target = raw_dir / f"round_{round_index:02d}_{safe_mode}.csv"
    per_paper = per_paper_record_csvs(work_dir)
    try:
        source = newest_records_csv(work_dir)
        aggregate_rows = count_csv_rows(source) or 0
    except FileNotFoundError:
        source = None
        aggregate_rows = 0
    per_paper_rows = sum(count_csv_rows(path) or 0 for path in per_paper)
    if per_paper and per_paper_rows > aggregate_rows:
        merge_record_csvs(per_paper, target)
    elif source:
        target.write_bytes(source.read_bytes())
    else:
        merge_record_csvs(per_paper, target)
    return target


def merge_record_csvs(csv_paths: list[Path], target: Path) -> Path:
    """Merge per-round LiteratureAgent CSVs into one batch CSV, deduping exact row keys."""
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    seen_rows: set[tuple[str, str, str, str]] = set()

    for path in csv_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in seen_fields:
                    seen_fields.add(field)
                    fieldnames.append(field)
            for row in reader:
                key = (
                    str(row.get("Ref_DOI_number") or "").strip().lower(),
                    str(row.get("Ref_internal_sample_id") or "").strip().lower(),
                    str(row.get("Ref_original_filename_data_upload") or "").strip().lower(),
                    str(row.get("Cell_stack_sequence") or "").strip().lower(),
                )
                if key in seen_rows:
                    continue
                seen_rows.add(key)
                rows.append(row)

    if not fieldnames:
        fieldnames = ["Ref_DOI_number", "Ref_original_filename_data_upload"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


def install_batch_records_for_integration(batch_csv: Path, work_dir: Path) -> None:
    """Place the merged batch CSV where the controller integration stage reads records."""
    csv_dir = work_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name in ["all_records.csv", "all_records_from_manager.csv"]:
        target = csv_dir / name
        if target.exists():
            backup = target.with_name(f"{target.stem}_pre_batch_merge_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}")
            target.replace(backup)
        target.write_bytes(batch_csv.read_bytes())


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            rows = sum(1 for _ in reader)
        return max(0, rows - 1)
    except Exception:
        return None


def count_processed_papers(csv_path: Path) -> int:
    """Count unique processed papers from a LiteratureAgent records CSV."""
    if not csv_path.exists():
        return 0
    seen: set[tuple[str, str]] = set()
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                doi = str(row.get("Ref_DOI_number") or "").strip().lower()
                title = str(row.get("Ref_original_filename_data_upload") or "").strip().lower()
                slugish = str(row.get("Ref_internal_sample_id") or "").strip().lower()
                key = (doi, title or slugish)
                if key != ("", ""):
                    seen.add(key)
    except Exception:
        return 0
    return len(seen)


def normalize_title(value: str) -> str:
    import re

    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(value: str) -> str:
    import re

    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip(" .;")


def update_processed_registry_from_records(records_csv: Path, registry_csv: Path, source_label: str) -> int:
    """Append processed-paper identifiers to a central registry used across output folders."""
    if not records_csv.exists():
        return 0
    registry_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "doi",
        "title_key",
        "title",
        "paper_slug",
        "source_label",
        "first_seen",
        "last_seen",
    ]
    existing: dict[tuple[str, str, str], dict[str, str]] = {}
    if registry_csv.exists():
        try:
            with registry_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    key = (
                        normalize_doi(row.get("doi") or ""),
                        str(row.get("title_key") or "").strip(),
                        str(row.get("paper_slug") or "").strip().lower(),
                    )
                    if key != ("", "", ""):
                        existing[key] = {field: str(row.get(field) or "") for field in fields}
        except Exception:
            existing = {}

    added = 0
    now = datetime.now().isoformat(timespec="seconds")
    try:
        with records_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                doi = normalize_doi(row.get("Ref_DOI_number") or row.get("doi") or row.get("DOI") or "")
                title = str(row.get("Ref_original_filename_data_upload") or row.get("title") or row.get("Title") or "").strip()
                title_key = normalize_title(title)
                paper_slug = str(row.get("Ref_internal_sample_id") or row.get("paper_slug") or "").strip().lower()
                key = (doi, title_key if len(title_key) >= 24 else "", paper_slug)
                if key == ("", "", "") or key in existing:
                    continue
                existing[key] = {
                    "doi": doi,
                    "title_key": key[1],
                    "title": title[:500],
                    "paper_slug": paper_slug,
                    "source_label": source_label,
                    "first_seen": now,
                    "last_seen": now,
                }
                added += 1
    except Exception as exc:
        print(f"[WARN] Could not update processed registry from {records_csv}: {exc}", flush=True)
        return 0

    with registry_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(existing.values(), key=lambda r: (r.get("doi") or "", r.get("title_key") or "", r.get("paper_slug") or "")):
            writer.writerow(row)
    return added


def sync_batch_attempt_registry(
    retrieval_report: Path,
    registry_csv: Path,
    seed_registry: Path | None,
    source_label: str,
) -> int:
    """
    Record every candidate actually contacted during this campaign.

    This registry is batch-local: it prevents overlapping query rounds from
    repeatedly downloading the same inaccessible DOI without permanently
    blacklisting that DOI from future campaigns.
    """
    registry_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "doi",
        "title_key",
        "title",
        "paper_slug",
        "source_label",
        "first_seen",
        "last_seen",
    ]
    existing: dict[tuple[str, str], dict[str, str]] = {}

    for path in [seed_registry, registry_csv]:
        if path is None or not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    doi = normalize_doi(row.get("doi") or "")
                    title = str(row.get("title") or "").strip()
                    title_key = str(row.get("title_key") or "").strip() or normalize_title(title)
                    key = ("doi", doi) if doi else ("title", title_key)
                    if key[1]:
                        existing[key] = {
                            field: str(row.get(field) or "")
                            for field in fields
                        }
        except Exception as exc:
            print(f"[WARN] Could not seed batch attempt registry from {path}: {exc}", flush=True)

    added = 0
    now = datetime.now().isoformat(timespec="seconds")
    if retrieval_report.exists():
        try:
            with retrieval_report.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    doi = normalize_doi(row.get("doi") or "")
                    title = str(row.get("title") or "").strip()
                    title_key = normalize_title(title)
                    key = ("doi", doi) if doi else ("title", title_key)
                    if not key[1] or key in existing:
                        continue
                    existing[key] = {
                        "doi": doi,
                        "title_key": title_key if len(title_key) >= 24 else "",
                        "title": title[:500],
                        "paper_slug": str(row.get("slug") or "").strip().lower(),
                        "source_label": source_label,
                        "first_seen": now,
                        "last_seen": now,
                    }
                    added += 1
        except Exception as exc:
            print(
                f"[WARN] Could not update batch attempt registry from {retrieval_report}: {exc}",
                flush=True,
            )
            return 0

    with registry_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            existing.values(),
            key=lambda item: (
                item.get("doi") or "",
                item.get("title_key") or "",
                item.get("paper_slug") or "",
            ),
        ):
            writer.writerow(row)
    return added


def current_literature_row_count(work_dir: Path) -> int | None:
    try:
        return count_csv_rows(newest_records_csv(work_dir))
    except Exception:
        return None


def build_query_plan(target_mode: str, search_query: str | None, query_sweep: bool, max_rounds: int) -> list[tuple[str, str]]:
    if search_query:
        return [("custom", search_query)]
    if not query_sweep:
        return [(target_mode, TARGET_QUERIES[target_mode])]

    if target_mode in STABILITY_TARGET_MODES:
        modes = [target_mode] + [
            mode for mode in STABILITY_SWEEP_TARGET_MODES if mode != target_mode
        ]
    else:
        modes = [target_mode] + [mode for mode in SWEEP_TARGET_MODES if mode != target_mode]
    plan = [(mode, TARGET_QUERIES[mode]) for mode in modes]
    if target_mode in STABILITY_TARGET_MODES:
        plan.extend(_expanded_stability_query_plan())
    else:
        plan.extend(_expanded_query_plan())

    deduplicated: list[tuple[str, str]] = []
    seen_queries: set[str] = set()
    for mode, query in plan:
        query_key = " ".join(query.lower().split())
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        deduplicated.append((mode, query))
    plan = deduplicated
    if max_rounds > 0:
        plan = plan[:max_rounds]
    return plan


def registry_paper_count(work_dir: Path) -> int | None:
    registry = work_dir / "paper_registry.json"
    if not registry.exists():
        return None
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ["papers", "records", "items"]:
                if isinstance(payload.get(key), (list, dict)):
                    return len(payload[key])
            return len(payload)
        if isinstance(payload, list):
            return len(payload)
    except Exception:
        return None
    return None


def read_added_rows_estimate(summary_path: Path) -> int | None:
    if not summary_path.exists():
        return None


def read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def publish_cumulative_campaign_state(
    *,
    campaign_root: Path,
    batch_name: str,
    original_base_csv: Path,
    previous_base_csv: Path,
    updated_csv: Path,
    integration_summary: Path,
    qa_summary: Path,
    processed_registry: Path,
) -> Path:
    """Publish one stable database and cumulative ledger after a completed batch."""
    if not updated_csv.exists():
        raise FileNotFoundError(f"Cannot publish cumulative state; updated CSV is missing: {updated_csv}")

    current_dir = campaign_root / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    canonical_csv = current_dir / "updated_perovskite_database_with_literature_agent.csv"
    temporary_csv = canonical_csv.with_suffix(".csv.tmp")
    shutil.copy2(updated_csv, temporary_csv)
    temporary_csv.replace(canonical_csv)

    integration = read_json_object(integration_summary)
    qa = read_json_object(qa_summary)
    state_path = campaign_root / "cumulative_campaign_state.json"
    previous_state = read_json_object(state_path)
    original_rows = previous_state.get("original_database_rows")
    if original_rows is None:
        original_rows = count_csv_rows(original_base_csv)
    current_rows = count_csv_rows(canonical_csv)
    original_support = previous_state.get("original_model_support")
    if not isinstance(original_support, dict):
        original_support = summarize_model_support_csv(original_base_csv)
    current_support = summarize_model_support_csv(canonical_csv)
    cumulative_pce_minimal = (
        int(current_support.get("pce_minimal_rows", 0))
        - int(original_support.get("pce_minimal_rows", 0))
    )
    cumulative_pce_strict = (
        int(current_support.get("pce_strict_rows", 0))
        - int(original_support.get("pce_strict_rows", 0))
    )
    cumulative_stability_minimal = (
        int(current_support.get("stability_minimal_rows", 0))
        - int(original_support.get("stability_minimal_rows", 0))
    )
    batches_completed = int(previous_state.get("batches_completed") or 0) + 1

    ledger_path = campaign_root / "cumulative_batch_ledger.csv"
    ledger_fields = [
        "completed_at",
        "batch_name",
        "input_database",
        "input_rows",
        "raw_literature_rows",
        "accepted_rows",
        "rejected_rows",
        "output_rows",
        "cumulative_rows_added",
        "cumulative_pce_model_ready_minimal_net_change",
        "cumulative_pce_model_ready_strict_net_change",
        "cumulative_stability_model_ready_minimal_net_change",
        "accepted_pce_usable_target_rows",
        "accepted_pce_model_ready_minimal",
        "accepted_pce_model_ready_strict",
        "accepted_stability_model_ready_minimal",
        "qa_pass",
    ]
    ledger_row = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "batch_name": batch_name,
        "resumed_incomplete_batch": bool(resumed_batch),
        "input_database": str(previous_base_csv),
        "input_rows": integration.get("base_rows", count_csv_rows(previous_base_csv)),
        "raw_literature_rows": integration.get("literature_raw_rows", ""),
        "accepted_rows": integration.get("literature_accepted_rows", ""),
        "rejected_rows": integration.get("literature_rejected_rows", ""),
        "output_rows": current_rows,
        "cumulative_rows_added": (
            current_rows - original_rows
            if isinstance(current_rows, int) and isinstance(original_rows, int)
            else ""
        ),
        "cumulative_pce_model_ready_minimal_net_change": cumulative_pce_minimal,
        "cumulative_pce_model_ready_strict_net_change": cumulative_pce_strict,
        "cumulative_stability_model_ready_minimal_net_change": cumulative_stability_minimal,
        "accepted_pce_usable_target_rows": qa.get("accepted_pce_usable_target_rows", ""),
        "accepted_pce_model_ready_minimal": qa.get("accepted_pce_model_ready_minimal", ""),
        "accepted_pce_model_ready_strict": qa.get("accepted_pce_model_ready_strict", ""),
        "accepted_stability_model_ready_minimal": qa.get("accepted_stability_model_ready_minimal", ""),
        "qa_pass": qa.get("qa_pass", ""),
    }
    ledger_exists = ledger_path.exists() and ledger_path.stat().st_size > 0
    with ledger_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields)
        if not ledger_exists:
            writer.writeheader()
        writer.writerow(ledger_row)

    state = {
        "campaign_mode": "cumulative",
        "original_base_csv": str(original_base_csv),
        "original_database_rows": original_rows,
        "current_database_csv": str(canonical_csv),
        "current_database_rows": current_rows,
        "cumulative_rows_added": ledger_row["cumulative_rows_added"],
        "original_model_support": original_support,
        "current_model_support": current_support,
        "cumulative_pce_model_ready_minimal_net_change": cumulative_pce_minimal,
        "cumulative_pce_model_ready_strict_net_change": cumulative_pce_strict,
        "cumulative_stability_model_ready_minimal_net_change": cumulative_stability_minimal,
        "batches_completed": batches_completed,
        "latest_batch_name": batch_name,
        "latest_batch_input_csv": str(previous_base_csv),
        "latest_batch_accepted_rows": integration.get("literature_accepted_rows"),
        "latest_batch_rejected_rows": integration.get("literature_rejected_rows"),
        "latest_batch_pce_model_ready_minimal": qa.get("accepted_pce_model_ready_minimal"),
        "latest_batch_pce_model_ready_strict": qa.get("accepted_pce_model_ready_strict"),
        "latest_batch_qa_pass": qa.get("qa_pass"),
        "processed_registry": str(processed_registry),
        "batch_ledger": str(ledger_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return canonical_csv
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(payload.get("added_rows_estimate", 0))
    except Exception:
        return None


def _normalized_candidate_key(row: dict) -> str:
    doi = str(
        row.get("doi")
        or row.get("DOI")
        or row.get("Ref_DOI_number")
        or ""
    ).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):].strip()
    if doi:
        return f"doi:{doi}"
    text = str(
        row.get("title")
        or row.get("Title")
        or row.get("slug")
        or row.get("paper_slug")
        or row.get("Ref_original_filename_data_upload")
        or ""
    ).lower()
    normalized = " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())
    return f"title:{normalized}" if normalized else ""


def _report_candidate_keys(
    path: Path,
    *,
    require_allowed: bool = False,
    require_ok: bool = False,
) -> set[str]:
    if not path.exists() or path.stat().st_size <= 0:
        return set()
    keys: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if require_allowed and str(row.get("allowed") or "").strip().lower() not in {
                    "1", "true", "yes",
                }:
                    continue
                if require_ok and str(row.get("ok") or "").strip().lower() not in {
                    "1", "true", "yes",
                }:
                    continue
                key = _normalized_candidate_key(row)
                if key:
                    keys.add(key)
    except Exception:
        return set()
    return keys


def _numeric_in_range(row: dict, fields: tuple[str, ...], low: float, high: float | None = None) -> bool:
    for field in fields:
        try:
            value = float(str(row.get(field) or "").strip())
        except Exception:
            continue
        if value < low:
            continue
        if high is not None and value > high:
            continue
        return True
    return False


def _field_present(row: dict, fields: tuple[str, ...]) -> bool:
    missing = {"", "nan", "none", "null", "unknown", "n/a"}
    return any(str(row.get(field) or "").strip().lower() not in missing for field in fields)


def summarize_model_support_csv(path: Path | None) -> dict[str, int]:
    summary = {
        "rows": 0,
        "unique_papers": 0,
        "pce_target_rows": 0,
        "pce_minimal_rows": 0,
        "pce_strict_rows": 0,
        "stability_target_rows": 0,
        "stability_minimal_rows": 0,
        "t80_rows": 0,
    }
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return summary
    paper_keys: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            summary["rows"] += 1
            key = _normalized_candidate_key(row)
            if key:
                paper_keys.add(key)
            pce = _numeric_in_range(
                row,
                ("JV_default_PCE", "JV_reverse_scan_PCE", "JV_forward_scan_PCE", "Stabilised_performance_PCE"),
                0.01,
                40.0,
            )
            composition = _field_present(
                row,
                ("Perovskite_composition_short_form", "Perovskite_composition_long_form"),
            )
            device = _field_present(
                row,
                ("Cell_architecture", "Cell_stack_sequence", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence"),
            )
            processing = _field_present(
                row,
                (
                    "Perovskite_deposition_procedure",
                    "Perovskite_deposition_solvents",
                    "Perovskite_deposition_quenching_media",
                    "Perovskite_deposition_thermal_annealing_temperature",
                    "Perovskite_deposition_thermal_annealing_time",
                ),
            )
            stability_target = _numeric_in_range(
                row,
                ("Stability_PCE_T80", "Stability_PCE_T95", "Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h"),
                0.0,
            )
            stability_time = _numeric_in_range(
                row,
                ("Stability_time_total_exposure", "Stability_PCE_T80", "Stability_PCE_T95"),
                0.0,
            )
            stability_conditions = _field_present(
                row,
                (
                    "Stability_temperature_range",
                    "Stability_relative_humidity_average_value",
                    "Stability_light_intensity",
                    "Stability_atmosphere",
                    "Encapsulation",
                ),
            )
            summary["pce_target_rows"] += int(pce)
            summary["pce_minimal_rows"] += int(pce and (composition or device))
            summary["pce_strict_rows"] += int(pce and composition and device and processing)
            summary["stability_target_rows"] += int(stability_target)
            summary["stability_minimal_rows"] += int(
                stability_target
                and stability_time
                and (composition or device or stability_conditions)
            )
            summary["t80_rows"] += int(
                _numeric_in_range(row, ("Stability_PCE_T80",), 0.0)
            )
    summary["unique_papers"] = len(paper_keys)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "scale", "full", "audit-only"], required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--candidate-attempt-multiplier",
        type=int,
        default=12,
        help=(
            "Search expansion over-sampling factor. --batch-size is the desired processed-paper target; "
            "the controller may try batch_size * multiplier candidates because many web papers lack usable full text."
        ),
    )
    parser.add_argument(
        "--max-papers-per-query-round",
        type=int,
        default=100,
        help=(
            "Maximum candidates sent through retrieval/extraction in one query "
            "round before progress is audited. Default 100; small remaining "
            "targets automatically use smaller rounds."
        ),
    )
    parser.add_argument("--target-total", type=int, default=5000)
    parser.add_argument(
        "--target-processed-papers",
        type=int,
        default=None,
        help=(
            "Stop extraction only after this many unique papers produce LiteratureAgent records. "
            "Defaults to --batch-size."
        ),
    )
    parser.add_argument(
        "--target-model-ready-rows",
        type=int,
        default=0,
        help=(
            "Optional additional stop condition: require at least this many "
            "minimally model-ready PCE or stability rows in the batch. "
            "The campaign must satisfy both this and --target-processed-papers."
        ),
    )
    parser.add_argument(
        "--max-candidate-attempts",
        type=int,
        default=5000,
        help="Hard safety cap on approximate search/download/extraction candidate attempts for this batch.",
    )
    parser.add_argument(
        "--target-mode",
        choices=sorted(TARGET_QUERIES),
        default="model_ready",
        help="Use a target-rich query preset unless --search-query is supplied.",
    )
    parser.add_argument(
        "--search-query",
        nargs="+",
        default=None,
        help="Target-rich Crossref/RSS query words for this batch. Multiple arguments are joined safely.",
    )
    parser.add_argument(
        "--query-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run several target-rich search presets in one batch. This is on by default because many web "
            "candidates lack retrievable full text; use --no-query-sweep for one query only."
        ),
    )
    parser.add_argument(
        "--max-query-rounds",
        type=int,
        default=0,
        help="Maximum query presets to try when --query-sweep is enabled. 0 means all presets.",
    )
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument(
        "--cumulative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Maintain one canonical current database and automatically use it as the next batch baseline. "
            "Enabled by default; use --no-cumulative only for an isolated experiment."
        ),
    )
    parser.add_argument(
        "--resume-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume the newest unfinished cumulative batch with saved round snapshots. "
            "Enabled by default; use --no-resume-incomplete to deliberately start a new batch."
        ),
    )
    parser.add_argument("--controller", type=Path, default=Path(__file__).parents[1] / "literature_agent_full_end_to_end_v21_3_english_sanitizer.py")
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument(
        "--cumulative-original-base-csv",
        type=Path,
        default=None,
        help=(
            "Original pre-LiteratureAgent database used only for cumulative growth accounting. "
            "Set this when seeding a cumulative campaign from an already integrated database."
        ),
    )
    parser.add_argument(
        "--ontology-path",
        type=Path,
        default=PROJECT_ROOT / "config" / "perovskite_ontology_library_v19.json",
    )
    parser.add_argument("--oauth-secrets", type=Path, help="Optional Google Drive OAuth client-secrets JSON.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--model-script", type=Path, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--model-out-dir", type=Path, default=None)
    parser.add_argument(
        "--run-model-on-empty-integration",
        action="store_true",
        help="Run the downstream model even when integration added zero rows. Default: skip to avoid wasted runs.",
    )
    parser.add_argument(
        "--enable-missing-field-recovery",
        action="store_true",
        help="Enable one targeted recovery pass for missing device/PCE/stability fields.",
    )
    parser.add_argument(
        "--target-finalization-enable",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Opt in to legacy broad source-text regex recovery after extraction. "
            "Default 0 because normal extraction now preserves field evidence and integration withholds unsupported targets."
        ),
    )
    parser.add_argument(
        "--family-gating",
        choices=["off", "moderate", "strict"],
        default="strict",
        help="Family gating mode passed to LiteratureAgent. Use moderate when source targeting is already highly model-ready.",
    )
    parser.add_argument(
        "--expansion-candidate-filter",
        choices=["off", "target_ready", "stability_target_ready", "model_ready_strict"],
        default="model_ready_strict",
        help=(
            "Candidate filter passed to LiteratureAgent. stability_target_ready retrieves PSC device papers "
            "with explicit stability/lifetime metadata and defers numeric target validation to full text. "
            "model_ready_strict requires explicit PCE/JV/stability/device cues before retrieval."
        ),
    )
    parser.add_argument(
        "--expansion-require-oa",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Require Unpaywall open-access status before attempting extraction. "
            "Default is 0 for model-ready campaigns because many useful device papers expose retrievable landing/HTML text without OA PDFs."
        ),
    )
    parser.add_argument(
        "--expansion-skip-known-candidates",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Skip Crossref/RSS candidates whose DOI/title already appears in the current base CSV "
            "or current work-dir records before full-text retrieval and LLM extraction. Default: 1."
        ),
    )
    parser.add_argument(
        "--expansion-skip-base-known-candidates",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Include the baseline CSV in pre-extraction duplicate skipping. Set to 0 for "
            "targeted enrichment of legacy database papers while continuing to skip papers "
            "already completed by LiteratureAgent. Default: 1."
        ),
    )
    parser.add_argument(
        "--processed-registry",
        type=Path,
        default=DEFAULT_PROCESSED_REGISTRY,
        help=(
            "Central CSV registry of expansion papers successfully processed across output folders. "
            "Default: data/expansion_processed_papers/processed_papers_registry.csv."
        ),
    )
    parser.add_argument(
        "--expansion-skip-terminal-retrieval-failures",
        type=int,
        choices=[0, 1],
        default=1,
        help="Skip candidates that previously exhausted several retrieval routes without usable text. Default: 1.",
    )
    parser.add_argument(
        "--expansion-terminal-failure-min-routes",
        type=int,
        default=4,
        help="Distinct failed retrieval routes required before terminal caching. Default: 4.",
    )
    parser.add_argument(
        "--automated-retrieval-enable",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Enable automated DOI/OpenAlex/EuropePMC/arXiv/landing/PDF resolver fallback for expansion candidates "
            "whose normal full-text retrieval is too short. Default: 1."
        ),
    )
    parser.add_argument(
        "--retrieval-max-urls-per-candidate",
        type=int,
        default=8,
        help="Maximum automated retrieval URLs to try per candidate before moving on.",
    )
    parser.add_argument(
        "--retrieval-supplementary-enable",
        type=int,
        choices=[0, 1],
        default=1,
        help="Retrieve linked supporting-information tables/text for aligned HTML articles. Default: 1.",
    )
    parser.add_argument(
        "--retrieval-max-supplementary-files",
        type=int,
        default=2,
        help="Maximum supporting-information files to retrieve per article. Default: 2.",
    )
    parser.add_argument(
        "--retrieval-max-pdf-bytes",
        type=int,
        default=25_000_000,
        help="Reject retrieved PDFs larger than this before text extraction.",
    )
    parser.add_argument(
        "--retrieval-max-pdf-pages",
        type=int,
        default=80,
        help="Reject retrieved PDFs with more pages than this before text extraction.",
    )
    parser.add_argument(
        "--expansion-min-fulltext-chars",
        type=int,
        default=6000,
        help="Character threshold for treating a retrieved source as full text.",
    )
    parser.add_argument(
        "--expansion-min-extraction-text-chars",
        type=int,
        default=1500,
        help=(
            "Minimum characters for a source-aligned partial HTML/abstract "
            "record to enter audited extraction. Default: 1500."
        ),
    )
    parser.add_argument(
        "--web-request-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for Crossref/RSS/Unpaywall/PDF/HTML retrieval.",
    )
    parser.add_argument(
        "--model-min-publication-year",
        type=int,
        default=2018,
        help="Publication-year cutoff used by the optional downstream model check. Default: 2018.",
    )
    parser.add_argument(
        "--model-n-estimators",
        type=int,
        default=300,
        help="Number of trees/estimators used by the optional downstream model check. Default: 300.",
    )
    parser.add_argument(
        "--model-min-completeness",
        type=float,
        default=0.15,
        help="Row-completeness threshold used by the optional downstream model check. Default: 0.15.",
    )
    parser.add_argument(
        "--model-min-completeness-column-coverage",
        type=float,
        default=0.005,
        help=(
            "Minimum column coverage used when computing source-aware row completeness in the optional "
            "downstream model check. Default: 0.005."
        ),
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess selected candidates even if they are already in the LiteratureAgent registry.",
    )
    parser.add_argument(
        "--use-reasoning-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Scientific Reasoning Policy Layer for summary, extraction, schema, and uncertainty LLM calls. Default: enabled.",
    )
    parser.add_argument(
        "--reasoning-policy-mode",
        choices=["off", "single", "multi", "auto"],
        default="auto",
        help="Reasoning-policy routing mode passed to LiteratureAgent. Default: auto.",
    )
    parser.add_argument(
        "--figure-report-enable",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Enable text/OCR figure reports during extract_batch. Use 0 for high-throughput expansion "
            "and run a separate vision/figure pass later. Default: 1."
        ),
    )
    parser.add_argument("--run-model-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.search_query = (
        " ".join(args.search_query).strip()
        if args.search_query
        else None
    )
    requested_base_csv = args.base_csv
    cumulative_state_path = args.campaign_root / "cumulative_campaign_state.json"
    cumulative_state = read_json_object(cumulative_state_path) if args.cumulative else {}
    cumulative_current_csv = (
        args.campaign_root
        / "current"
        / "updated_perovskite_database_with_literature_agent.csv"
    )
    if args.cumulative and cumulative_current_csv.exists():
        args.base_csv = cumulative_current_csv
        print(
            f"[CUMULATIVE] Continuing from canonical database: {args.base_csv}",
            flush=True,
        )
    original_base_csv = Path(
        str(
            cumulative_state.get("original_base_csv")
            or args.cumulative_original_base_csv
            or requested_base_csv
        )
    )
    if not args.base_csv.exists():
        if DEFAULT_BASE_CSV.exists():
            print(
                f"[WARN] Base CSV not found: {args.base_csv}. "
                f"Using trusted default baseline instead: {DEFAULT_BASE_CSV}",
                flush=True,
            )
            args.base_csv = DEFAULT_BASE_CSV
        else:
            raise FileNotFoundError(f"Base CSV not found: {args.base_csv}")
    query_plan = build_query_plan(
        target_mode=args.target_mode,
        search_query=args.search_query,
        query_sweep=bool(args.query_sweep),
        max_rounds=max(0, int(args.max_query_rounds)),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = args.batch_size or DEFAULT_BATCHES.get(args.stage, 0)
    target_processed_papers = int(args.target_processed_papers or batch_size)
    target_model_ready_rows = max(0, int(args.target_model_ready_rows or 0))
    max_candidate_attempts = max(target_processed_papers, int(args.max_candidate_attempts or 0))
    per_round_attempt_budget = max(batch_size, batch_size * max(1, args.candidate_attempt_multiplier))
    resumed_batch = None
    if args.cumulative and args.resume_incomplete:
        resumed_batch = find_incomplete_cumulative_batch(
            args.campaign_root,
            args.stage,
            batch_size,
        )
    batch_name = resumed_batch.name if resumed_batch else f"{args.stage}_{batch_size}_{timestamp}"
    batch_dir = resumed_batch or (
        args.campaign_root / "batches" / batch_name
        if args.cumulative
        else args.campaign_root / batch_name
    )
    if resumed_batch:
        print(f"[CUMULATIVE] Resuming incomplete batch: {batch_dir}", flush=True)
    if args.cumulative:
        args.work_dir = args.work_dir / "batches" / batch_name
    integration_dir = batch_dir / "integration"
    qa_dir = batch_dir / "qa"
    updated = integration_dir / "updated_perovskite_database_with_literature_agent.csv"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_attempt_registry = batch_dir / "candidate_attempt_registry.csv"
    sync_batch_attempt_registry(
        retrieval_report=args.work_dir / "retrieval_reports" / "retrieval_candidate_report.csv",
        registry_csv=batch_attempt_registry,
        seed_registry=args.processed_registry,
        source_label=f"{batch_name}:campaign_start",
    )
    papers_before = registry_paper_count(args.work_dir)
    audit_baseline = (
        args.base_csv
        if args.cumulative
        else find_previous_updated_csv_for_base(args.campaign_root, batch_dir, args.base_csv) or args.base_csv
    )
    effective_skip_known_candidates = 0 if args.force_reprocess else int(args.expansion_skip_known_candidates)

    common = [
        sys.executable,
        str(args.controller),
        "--base_csv", str(args.base_csv),
        "--ontology_path", str(args.ontology_path),
        "--work_dir", str(args.work_dir),
        "--integration_out_dir", str(integration_dir),
        "--run_mode", "expand",
        "--family_gating", args.family_gating,
        "--llm_cache_enable", "1",
        "--use_reasoning_layer", "1" if args.use_reasoning_layer else "0",
        "--reasoning_policy_mode", args.reasoning_policy_mode,
        "--web_request_timeout", str(args.web_request_timeout),
        "--vision_enable", "0",
        "--inline_vision", "0",
        "--figure_report_enable", str(args.figure_report_enable),
        "--missing_field_recovery", "1" if args.enable_missing_field_recovery else "0",
        "--target_finalization_enable", str(args.target_finalization_enable),
        "--allow_embedded_reset", "0",
        "--disable_google_drive", "1",
        "--expansion_candidate_filter", args.expansion_candidate_filter,
        "--expansion_require_oa", str(args.expansion_require_oa),
        "--expansion_skip_known_candidates", str(effective_skip_known_candidates),
        "--expansion_skip_base_known_candidates", str(args.expansion_skip_base_known_candidates),
        "--expansion_processed_registry", str(batch_attempt_registry),
        "--expansion_skip_terminal_retrieval_failures", str(args.expansion_skip_terminal_retrieval_failures),
        "--expansion_terminal_failure_min_routes", str(args.expansion_terminal_failure_min_routes),
        "--automated_retrieval_enable", str(args.automated_retrieval_enable),
        "--retrieval_max_urls_per_candidate", str(args.retrieval_max_urls_per_candidate),
        "--retrieval_supplementary_enable", str(args.retrieval_supplementary_enable),
        "--retrieval_max_supplementary_files", str(args.retrieval_max_supplementary_files),
        "--retrieval_max_pdf_bytes", str(args.retrieval_max_pdf_bytes),
        "--retrieval_max_pdf_pages", str(args.retrieval_max_pdf_pages),
        "--expansion_min_fulltext_chars", str(args.expansion_min_fulltext_chars),
        "--expansion_min_extraction_text_chars", str(args.expansion_min_extraction_text_chars),
        "--force_reprocess", "1" if args.force_reprocess else "0",
        "--no_require_doi",
    ]
    if args.oauth_secrets:
        common += ["--google_drive_oauth_client_secrets", str(args.oauth_secrets)]

    if args.stage != "audit-only":
        extraction_rounds = []
        rounds_dir = batch_dir / "raw_literature_records" / "rounds"
        round_record_csvs: list[Path] = sorted(
            rounds_dir.glob("round_*.csv"),
            key=round_index_from_snapshot,
        ) if resumed_batch else []
        rows_before = current_literature_row_count(args.work_dir)
        candidate_filter_report = args.work_dir / "expansion_candidate_filter_report.csv"
        retrieval_report = args.work_dir / "retrieval_reports" / "retrieval_candidate_report.csv"
        seen_candidates = _report_candidate_keys(candidate_filter_report)
        allowed_candidates = _report_candidate_keys(
            candidate_filter_report,
            require_allowed=True,
        )
        retrieval_candidates = _report_candidate_keys(retrieval_report)
        retrieved_candidates = _report_candidate_keys(
            retrieval_report,
            require_ok=True,
        )
        batch_retrieval_candidates: set[str] = set(retrieval_candidates)
        attempts_used = len(batch_retrieval_candidates)
        round_index = max(
            (round_index_from_snapshot(path) for path in round_record_csvs),
            default=0,
        )
        search_capacity_requested = round_index * per_round_attempt_budget
        existing_progress = (
            batch_dir
            / "raw_literature_records"
            / "all_records_for_this_batch.progress.csv"
        )
        if round_record_csvs and not existing_progress.exists():
            existing_progress = merge_record_csvs(round_record_csvs, existing_progress)
        processed_so_far = count_processed_papers(existing_progress)
        existing_support = summarize_model_support_csv(existing_progress)
        model_ready_so_far = (
            existing_support["pce_minimal_rows"]
            + existing_support["stability_minimal_rows"]
        )
        if resumed_batch:
            print(
                "[CUMULATIVE RESUME] "
                f"completed_rounds={round_index} "
                f"papers={processed_so_far}/{target_processed_papers} "
                f"rows={existing_support['rows']} "
                f"model_ready={model_ready_so_far}/{target_model_ready_rows} "
                f"retrieval_attempts={attempts_used}/{max_candidate_attempts}",
                flush=True,
            )
        while (
            round_index < len(query_plan)
            and attempts_used < max_candidate_attempts
            and (
            processed_so_far < target_processed_papers
            or model_ready_so_far < target_model_ready_rows
            )
        ):
            query_mode, query_text = query_plan[round_index]
            round_index += 1
            attempts_this_round = min(per_round_attempt_budget, max_candidate_attempts - attempts_used)
            remaining_target = max(
                target_processed_papers - processed_so_far,
                target_model_ready_rows - model_ready_so_far,
                1,
            )
            papers_this_round = min(
                attempts_this_round,
                max(
                    5,
                    min(
                        max(1, int(args.max_papers_per_query_round)),
                        remaining_target * 2,
                    ),
                ),
            )
            # Crossref's public REST API rejects cursorless requests once the
            # offset reaches 10000. With 100 rows/page, page 100 is the last
            # safe page because it uses offset 9900.
            crossref_max_pages = min(100, max(20, ceil(attempts_this_round / 100)))
            extraction = common + [
                "--pipeline_stage", "extract_batch",
                "--full_literature_run",
                "--max_papers", str(papers_this_round),
                "--drive_process_all_files", "0",
                "--google_drive_max_files_per_run", str(batch_size),
                "--crossref_rows_per_page", "100",
                "--crossref_max_pages", str(crossref_max_pages),
                "--search_query", query_text,
            ]
            print(
                f"\n[EXPANSION] Query round {round_index}: {query_mode} | "
                f"processed={processed_so_far}/{target_processed_papers} | "
                f"model_ready={model_ready_so_far}/{target_model_ready_rows} | "
                f"round_paper_limit={papers_this_round} | "
                f"retrieval_attempts={attempts_used}/{max_candidate_attempts} | "
                f"search_capacity_requested={search_capacity_requested}"
            )
            run(extraction, args.dry_run)
            search_capacity_requested += attempts_this_round
            round_csv = None
            if not args.dry_run:
                round_csv = snapshot_round_records(args.work_dir, batch_dir, round_index, query_mode)
                round_record_csvs.append(round_csv)
                added_registry = update_processed_registry_from_records(
                    round_csv,
                    args.processed_registry,
                    source_label=f"{batch_name}:round_{round_index}:{query_mode}",
                )
                if added_registry:
                    print(
                        f"[EXPANSION] Added {added_registry} paper key(s) to processed registry: "
                        f"{args.processed_registry}",
                        flush=True,
                    )
                merged_progress = merge_record_csvs(
                    round_record_csvs,
                    batch_dir / "raw_literature_records" / "all_records_for_this_batch.progress.csv",
                )
                processed_so_far = count_processed_papers(merged_progress)
                support = summarize_model_support_csv(merged_progress)
                model_ready_so_far = (
                    support["pce_minimal_rows"]
                    + support["stability_minimal_rows"]
                )
                print(
                    "[EXPANSION SUPPORT] "
                    f"papers={support['unique_papers']} rows={support['rows']} "
                    f"pce_target={support['pce_target_rows']} "
                    f"pce_minimal={support['pce_minimal_rows']} "
                    f"pce_strict={support['pce_strict_rows']} "
                    f"stability_target={support['stability_target_rows']} "
                    f"stability_minimal={support['stability_minimal_rows']} "
                    f"t80={support['t80_rows']} "
                    f"model_ready_total={model_ready_so_far}/{target_model_ready_rows}",
                    flush=True,
                )
            current_seen = _report_candidate_keys(candidate_filter_report)
            current_allowed = _report_candidate_keys(
                candidate_filter_report,
                require_allowed=True,
            )
            current_retrieval = _report_candidate_keys(retrieval_report)
            current_retrieved = _report_candidate_keys(
                retrieval_report,
                require_ok=True,
            )
            retrieval_attempts_this_round = (
                current_retrieval - retrieval_candidates
            )
            batch_retrieval_candidates.update(retrieval_attempts_this_round)
            attempts_used = len(batch_retrieval_candidates)
            newly_registered_attempts = sync_batch_attempt_registry(
                retrieval_report=retrieval_report,
                registry_csv=batch_attempt_registry,
                seed_registry=args.processed_registry,
                source_label=f"{batch_name}:round_{round_index}:{query_mode}",
            )
            if newly_registered_attempts:
                print(
                    f"[EXPANSION] Added {newly_registered_attempts} newly attempted "
                    f"candidate(s) to batch-local skip registry: {batch_attempt_registry}",
                    flush=True,
                )
            rows_after = current_literature_row_count(args.work_dir)
            extraction_rounds.append({
                "round": round_index,
                "target_mode": query_mode,
                "search_query": query_text,
                "literature_rows_before_batch": rows_before,
                "literature_rows_after_round": rows_after,
                "round_records_csv": str(round_csv) if round_csv else None,
                "candidate_attempts_this_round": len(retrieval_attempts_this_round),
                "paper_processing_limit_this_round": papers_this_round,
                "candidate_attempts_used": attempts_used,
                "search_capacity_requested_this_round": attempts_this_round,
                "search_capacity_requested_cumulative": search_capacity_requested,
                "unique_candidates_seen_this_round": len(current_seen - seen_candidates),
                "unique_candidates_allowed_this_round": len(current_allowed - allowed_candidates),
                "unique_candidates_retrieval_attempted_this_round": len(current_retrieval - retrieval_candidates),
                "unique_candidates_retrieved_this_round": len(current_retrieved - retrieved_candidates),
                "unique_candidates_seen_cumulative": len(current_seen),
                "unique_candidates_allowed_cumulative": len(current_allowed),
                "unique_candidates_retrieval_attempted_cumulative": len(current_retrieval),
                "unique_candidates_retrieved_cumulative": len(current_retrieved),
                "processed_papers_after_round": processed_so_far,
                "model_ready_rows_after_round": model_ready_so_far,
            })
            seen_candidates = current_seen
            allowed_candidates = current_allowed
            retrieval_candidates = current_retrieval
            retrieved_candidates = current_retrieved
            if (
                processed_so_far >= target_processed_papers
                and model_ready_so_far >= target_model_ready_rows
            ):
                print(
                    "[EXPANSION] Reached batch targets: "
                    f"processed={processed_so_far}/{target_processed_papers}, "
                    f"model_ready={model_ready_so_far}/{target_model_ready_rows}."
                )
                break

        if (
            processed_so_far < target_processed_papers
            or model_ready_so_far < target_model_ready_rows
        ):
            if attempts_used >= max_candidate_attempts:
                stop_reason = (
                    "retrieval-attempt safety cap reached: "
                    f"{attempts_used}/{max_candidate_attempts}"
                )
            else:
                stop_reason = (
                    "query plan exhausted: "
                    f"{round_index}/{len(query_plan)} unique query rounds"
                )
            print(
                f"[EXPANSION] Stopped before targets: {stop_reason}; "
                f"processed={processed_so_far}/{target_processed_papers}, "
                f"model_ready={model_ready_so_far}/{target_model_ready_rows}, "
                f"search_capacity_requested={search_capacity_requested}."
            )

        if args.dry_run:
            raw = args.work_dir / "csv" / "all_records.csv"
        else:
            raw = merge_record_csvs(
                round_record_csvs,
                batch_dir / "raw_literature_records" / "all_records_for_this_batch.csv",
            )
            added_registry = update_processed_registry_from_records(
                raw,
                args.processed_registry,
                source_label=f"{batch_name}:final_batch_records",
            )
            if added_registry:
                print(
                    f"[EXPANSION] Added {added_registry} final paper key(s) to processed registry: "
                    f"{args.processed_registry}",
                    flush=True,
                )
            install_batch_records_for_integration(raw, args.work_dir)

        candidate_accounting = {
            "search_capacity_requested": search_capacity_requested,
            "unique_candidates_seen": len(seen_candidates),
            "unique_candidates_allowed": len(allowed_candidates),
            "unique_candidates_retrieval_attempted": attempts_used,
            "unique_candidates_retrieved": len(retrieved_candidates),
            "unique_papers_processed": count_processed_papers(raw) if not args.dry_run else None,
            "model_support": summarize_model_support_csv(raw) if not args.dry_run else {},
            "target_processed_papers": target_processed_papers,
            "target_model_ready_rows": target_model_ready_rows,
            "note": (
                "search_capacity_requested is the Crossref/RSS request budget. "
                "unique_candidates_retrieval_attempted is the batch-local count "
                "used by the retrieval-attempt safety cap."
            ),
        }
        (batch_dir / "candidate_accounting.json").write_text(
            json.dumps(candidate_accounting, indent=2),
            encoding="utf-8",
        )
        if extraction_rounds:
            fieldnames = list(extraction_rounds[0])
            with (batch_dir / "candidate_accounting_by_round.csv").open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(extraction_rounds)
        print(
            "[EXPANSION ACCOUNTING] "
            f"search_capacity={candidate_accounting['search_capacity_requested']} "
            f"unique_seen={candidate_accounting['unique_candidates_seen']} "
            f"unique_allowed={candidate_accounting['unique_candidates_allowed']} "
            f"retrieval_attempted={candidate_accounting['unique_candidates_retrieval_attempted']} "
            f"retrieved={candidate_accounting['unique_candidates_retrieved']} "
            f"processed={candidate_accounting['unique_papers_processed']}",
            flush=True,
        )

        integration = common + ["--pipeline_stage", "integrate_and_model", "--skip_literature_agent"]
        run(integration, args.dry_run)

        integration_summary = integration_dir / "literature_update_summary.json"
        added_rows = read_added_rows_estimate(integration_summary)
        if args.run_model_check:
            if args.dry_run:
                print("[EXPANSION] Dry-run: downstream model check would run after integration.")
            elif (added_rows or 0) <= 0 and not args.run_model_on_empty_integration:
                print(
                    "[EXPANSION] Skipping downstream model check because integration added "
                    f"{added_rows or 0} rows. Use --run-model-on-empty-integration to force it.",
                    flush=True,
                )
            else:
                model_out = args.model_out_dir or (batch_dir / "model")
                model_cmd = [
                    sys.executable,
                    str(args.model_script),
                    "--csv",
                    str(updated),
                    "--out",
                    str(model_out),
                    "--min-publication-year",
                    str(args.model_min_publication_year),
                    "--n-estimators",
                    str(args.model_n_estimators),
                    "--min-completeness",
                    str(args.model_min_completeness),
                    "--min-completeness-column-coverage",
                    str(args.model_min_completeness_column_coverage),
                ]
                run(model_cmd, args.dry_run)
    else:
        extraction_rounds = []
        raw = snapshot_records(args.work_dir, batch_dir) if not args.dry_run else args.work_dir / "csv" / "all_records.csv"

    accepted = find_csv(integration_dir, ["*accepted*.csv"])
    rejected = find_csv(integration_dir, ["*rejected*.csv"])
    audit = [
        sys.executable,
        str(Path(__file__).with_name("audit_expansion_batch.py")),
        "--batch-name", batch_name,
        "--base-csv", str(audit_baseline),
        "--updated-csv", str(updated),
        "--raw-literature-csv", str(raw),
        "--paper-type-report", str(args.work_dir / "paper_type_gate_report.csv"),
        "--out-dir", str(qa_dir),
    ]
    if accepted:
        audit += ["--accepted-csv", str(accepted)]
    if rejected:
        audit += ["--rejected-csv", str(rejected)]
    run(audit, args.dry_run)

    canonical_updated_csv = None
    if args.cumulative and not args.dry_run:
        canonical_updated_csv = publish_cumulative_campaign_state(
            campaign_root=args.campaign_root,
            batch_name=batch_name,
            original_base_csv=original_base_csv,
            previous_base_csv=args.base_csv,
            updated_csv=updated,
            integration_summary=integration_dir / "literature_update_summary.json",
            qa_summary=qa_dir / "batch_summary.json",
            processed_registry=args.processed_registry,
        )
        print(
            f"[CUMULATIVE] Canonical database updated: {canonical_updated_csv}",
            flush=True,
        )
        print(
            f"[CUMULATIVE] State: {args.campaign_root / 'cumulative_campaign_state.json'}",
            flush=True,
        )

    manifest = {
        "batch_name": batch_name,
        "cumulative": bool(args.cumulative),
        "original_base_csv": str(original_base_csv),
        "effective_base_csv": str(args.base_csv),
        "canonical_updated_csv": str(canonical_updated_csv) if canonical_updated_csv else None,
        "stage": args.stage,
        "batch_size": batch_size,
        "target_processed_papers": target_processed_papers,
        "target_model_ready_rows": target_model_ready_rows,
        "processed_papers_in_batch": count_processed_papers(raw) if not args.dry_run else None,
        "model_ready_rows_in_batch": (
            (
                summarize_model_support_csv(raw)["pce_minimal_rows"]
                + summarize_model_support_csv(raw)["stability_minimal_rows"]
            )
            if not args.dry_run
            else None
        ),
        "max_candidate_attempts": max_candidate_attempts,
        "candidate_attempt_multiplier": args.candidate_attempt_multiplier,
        "max_papers_per_query_round": int(args.max_papers_per_query_round),
        "candidate_attempt_budget": per_round_attempt_budget,
        "search_capacity_requested": (
            search_capacity_requested if args.stage != "audit-only" else 0
        ),
        "unique_candidates_seen": len(seen_candidates) if args.stage != "audit-only" else None,
        "unique_candidates_allowed": len(allowed_candidates) if args.stage != "audit-only" else None,
        "unique_candidates_retrieval_attempted": len(retrieval_candidates) if args.stage != "audit-only" else None,
        "unique_candidates_retrieved": len(retrieved_candidates) if args.stage != "audit-only" else None,
        "created": datetime.now().isoformat(),
        "work_dir": str(args.work_dir),
        "integration_dir": str(integration_dir),
        "qa_dir": str(qa_dir),
        "vision_inline": False,
        "family_gating": args.family_gating,
        "expansion_candidate_filter": args.expansion_candidate_filter,
        "expansion_require_oa": int(args.expansion_require_oa),
        "expansion_skip_known_candidates": int(effective_skip_known_candidates),
        "expansion_skip_base_known_candidates": int(args.expansion_skip_base_known_candidates),
        "processed_registry": str(args.processed_registry),
        "expansion_skip_terminal_retrieval_failures": int(args.expansion_skip_terminal_retrieval_failures),
        "expansion_terminal_failure_min_routes": int(args.expansion_terminal_failure_min_routes),
        "automated_retrieval_enable": int(args.automated_retrieval_enable),
        "retrieval_max_urls_per_candidate": int(args.retrieval_max_urls_per_candidate),
        "retrieval_supplementary_enable": int(args.retrieval_supplementary_enable),
        "retrieval_max_supplementary_files": int(args.retrieval_max_supplementary_files),
        "retrieval_max_pdf_bytes": int(args.retrieval_max_pdf_bytes),
        "retrieval_max_pdf_pages": int(args.retrieval_max_pdf_pages),
        "expansion_min_fulltext_chars": int(args.expansion_min_fulltext_chars),
        "expansion_min_extraction_text_chars": int(args.expansion_min_extraction_text_chars),
        "force_reprocess": bool(args.force_reprocess),
        "target_mode": args.target_mode,
        "missing_field_recovery": bool(args.enable_missing_field_recovery),
        "target_finalization_enable": int(args.target_finalization_enable),
        "use_reasoning_layer": bool(args.use_reasoning_layer),
        "reasoning_policy_mode": args.reasoning_policy_mode,
        "figure_report_enable": int(args.figure_report_enable),
        "raw_literature_csv_snapshot": str(raw),
        "model_script": str(args.model_script),
        "model_out_dir": str(args.model_out_dir or (batch_dir / "model")),
        "model_min_publication_year": int(args.model_min_publication_year),
        "model_n_estimators": int(args.model_n_estimators),
        "model_min_completeness": float(args.model_min_completeness),
        "model_min_completeness_column_coverage": float(args.model_min_completeness_column_coverage),
        "run_model_on_empty_integration": bool(args.run_model_on_empty_integration),
        "dry_run": args.dry_run,
        "target_total_papers": args.target_total,
        "registered_papers_before": papers_before,
        "registered_papers_after": registry_paper_count(args.work_dir),
        "audit_baseline_csv": str(audit_baseline),
        "search_query": args.search_query,
        "query_sweep": bool(args.query_sweep),
        "query_plan": [{"target_mode": mode, "search_query": query} for mode, query in query_plan],
        "extraction_rounds": extraction_rounds,
    }
    (batch_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nBatch folder: {batch_dir}")


if __name__ == "__main__":
    main()
