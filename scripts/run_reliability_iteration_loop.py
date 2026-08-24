#!/usr/bin/env python
"""Run the LiteratureAgent target-finalization, integration, modeling, and audit loop.

This script is intentionally conservative. It only applies target/schema
finalization rows that pass the validation report, then reruns integration,
three-way model comparison, and the comprehensive audit so the next decision is
based on model-used target support rather than raw extracted row counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER = PROJECT_ROOT / "literature_agent_full_end_to_end_v21_3_english_sanitizer.py"
DEFAULT_MODEL_SCRIPT = PROJECT_ROOT / "models" / "pce_then_stability_same_approach.py"
DEFAULT_BASE_CSV = PROJECT_ROOT / "data" / "Perovskite_database_content_all_data.csv"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "config" / "perovskite_ontology_library_v19.json"


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n>", subprocess.list2cmdline(cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with path.open("r", encoding=enc, newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            pass
    return []


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count_model_rows(metrics_csv: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in read_csv_rows(metrics_csv):
        dataset = row.get("dataset_version", "")
        target = row.get("target", "")
        status = row.get("status", "")
        n = row.get("n", "")
        key = f"{dataset}::{target}"
        if status.startswith("trained") or status.startswith("skipped"):
            try:
                out[key] = int(float(n))
            except Exception:
                out[key] = 0
    return out


def write_decision(
    out_dir: Path,
    integration_summary: dict[str, Any],
    audit_summary: dict[str, Any],
    metrics_csv: Path,
    min_pce_added: int,
    min_stability_added: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = count_model_rows(metrics_csv)

    original_pce = metrics.get("01_original_database::JV_default_PCE", 0)
    integrated_pce = metrics.get("03_integrated_database::JV_default_PCE", 0)
    original_ret = metrics.get("01_original_database::target_end_retention_pct", 0)
    integrated_ret = metrics.get("03_integrated_database::target_end_retention_pct", 0)
    original_t80 = metrics.get("01_original_database::target_log1p_T80_hours", 0)
    integrated_t80 = metrics.get("03_integrated_database::target_log1p_T80_hours", 0)

    pce_added = integrated_pce - original_pce
    stability_added = max(integrated_ret - original_ret, integrated_t80 - original_t80)
    rejected_rows = int(integration_summary.get("literature_rejected_rows", 0) or 0)
    accepted_rows = int(integration_summary.get("literature_accepted_rows", 0) or 0)
    raw_rows = int(integration_summary.get("literature_raw_rows", 0) or 0)

    checks = {
        "accepted_rows_positive": accepted_rows > 0,
        "pce_model_support_increased": pce_added >= min_pce_added,
        "stability_model_support_increased": stability_added >= min_stability_added,
        "rejected_rows_not_all_records": rejected_rows < raw_rows if raw_rows else False,
    }
    decision = "pass" if all(checks.values()) else "review"

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "checks": checks,
        "thresholds": {
            "min_pce_model_rows_added": min_pce_added,
            "min_stability_model_rows_added": min_stability_added,
        },
        "integration": {
            "raw_rows": raw_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "updated_rows": integration_summary.get("updated_rows"),
            "backfilled_rows": integration_summary.get("backfilled_rows"),
        },
        "model_support_delta": {
            "pce_rows_added": pce_added,
            "retention_rows_added": integrated_ret - original_ret,
            "t80_rows_added": integrated_t80 - original_t80,
        },
        "audit": audit_summary,
    }
    (out_dir / "reliability_iteration_decision.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Reliability Iteration Decision",
        "",
        f"- Decision: {decision}",
        f"- Accepted rows: {accepted_rows}",
        f"- Rejected rows: {rejected_rows}",
        f"- PCE model-used rows added: {pce_added}",
        f"- Retention model-used rows added: {integrated_ret - original_ret}",
        f"- T80 model-used rows added: {integrated_t80 - original_t80}",
        "",
        "## Checks",
    ]
    for key, value in checks.items():
        md.append(f"- {key}: {value}")
    (out_dir / "RELIABILITY_ITERATION_DECISION.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--ontology-path", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--model-script", type=Path, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--integration-out-dir", type=Path, required=True)
    parser.add_argument("--comparison-out-dir", type=Path, required=True)
    parser.add_argument("--audit-out-dir", type=Path, required=True)
    parser.add_argument("--min-pce-score", type=int, default=11)
    parser.add_argument("--min-pce-model-rows-added", type=int, default=1)
    parser.add_argument("--min-stability-model-rows-added", type=int, default=1)
    parser.add_argument("--model-min-publication-year", type=int, default=2018)
    parser.add_argument("--model-n-estimators", type=int, default=300)
    parser.add_argument("--model-min-completeness", type=float, default=0.20)
    parser.add_argument("--model-min-completeness-column-coverage", type=float, default=0.005)
    parser.add_argument(
        "--start-at",
        choices=["full", "final_audit"],
        default="full",
        help="Use final_audit to resume after integration and three-way model comparison have already completed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_report_dir = args.work_dir / "target_recovery_reports"
    target_report = target_report_dir / "target_recovery_report.csv"
    validation_report = args.audit_out_dir / "target_recovery_validation.csv"
    validation_summary = args.audit_out_dir / "target_recovery_validation_summary.csv"

    if args.start_at == "full":
        run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "recover_literatureagent_targets_from_source_text.py"),
            "--work-dir",
            str(args.work_dir),
            "--min-pce-score",
            str(args.min_pce_score),
        ], args.dry_run)

        run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_target_recovery_report.py"),
            "--report-csv",
            str(target_report),
            "--out-csv",
            str(validation_report),
            "--summary-csv",
            str(validation_summary),
        ], args.dry_run)

        run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "recover_literatureagent_targets_from_source_text.py"),
            "--work-dir",
            str(args.work_dir),
            "--min-pce-score",
            str(args.min_pce_score),
            "--validated-report",
            str(validation_report),
            "--apply",
        ], args.dry_run)

        run([
            sys.executable,
            str(args.controller),
            "--base_csv",
            str(args.base_csv),
            "--ontology_path",
            str(args.ontology_path),
            "--work_dir",
            str(args.work_dir),
            "--integration_out_dir",
            str(args.integration_out_dir),
            "--model_out_dir",
            str(args.comparison_out_dir / "single_dataset_model"),
            "--model_script",
            str(args.model_script),
            "--run_mode",
            "initial",
            "--pipeline_stage",
            "integrate_and_model",
            "--figure_report_enable",
            "1",
            "--use_reasoning_layer",
            "1",
            "--reasoning_policy_mode",
            "multi",
            "--llm_cache_enable",
            "1",
            "--target_finalization_enable",
            "0",
            "--no_require_doi",
        ], args.dry_run)

    accepted_csv = args.integration_out_dir / "literature_update_accepted_rows.csv"
    updated_csv = args.integration_out_dir / "updated_perovskite_database_with_literature_agent.csv"

    if args.start_at == "full":
        run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_three_way_pce_stability_comparison.py"),
            "--original-csv",
            str(args.base_csv),
            "--extracted-csv",
            str(accepted_csv),
            "--integrated-csv",
            str(updated_csv),
            "--out-root",
            str(args.comparison_out_dir),
            "--model-script",
            str(args.model_script),
            "--min-publication-year",
            str(args.model_min_publication_year),
            "--n-estimators",
            str(args.model_n_estimators),
            "--min-completeness",
            str(args.model_min_completeness),
            "--extracted-min-completeness",
            str(args.model_min_completeness),
            "--min-completeness-column-coverage",
            str(args.model_min_completeness_column_coverage),
        ], args.dry_run)

    run([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "comprehensive_reliability_audit.py"),
        "--work-dir",
        str(args.work_dir),
        "--integration-dir",
        str(args.integration_out_dir),
        "--model-comparison-dir",
        str(args.comparison_out_dir),
        "--out-dir",
        str(args.audit_out_dir),
    ], args.dry_run)

    if not args.dry_run:
        write_decision(
            out_dir=args.audit_out_dir,
            integration_summary=read_json(args.integration_out_dir / "literature_update_summary.json"),
            audit_summary=read_json(args.audit_out_dir / "comprehensive_reliability_audit.json"),
            metrics_csv=args.audit_out_dir / "model_metric_summary_snapshot.csv",
            min_pce_added=args.min_pce_model_rows_added,
            min_stability_added=args.min_stability_model_rows_added,
        )
        print(f"\nReliability loop complete: {args.audit_out_dir}", flush=True)


if __name__ == "__main__":
    main()
