#!/usr/bin/env python3
"""Build current plus reasoning-policy knowledge graph variants for comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


POLICIES = [
    "current",
    "socratic_uncertainty",
    "cartesian_decomposition",
    "kantian_constraints",
    "humean_evidence",
    "aristotelian_classification",
    "hegelian_conflict_resolution",
    "platonic_abstraction",
]


def run_builder(
    builder_script: Path,
    records: Path,
    ontology: Path,
    work_dir: Path,
    out_dir: Path,
    policy: str,
    max_records: int,
) -> None:
    cmd = [
        sys.executable,
        str(builder_script),
        "--records",
        str(records),
        "--ontology",
        str(ontology),
        "--work-dir",
        str(work_dir),
        "--out",
        str(out_dir),
        "--reasoning-policy",
        policy,
    ]
    if max_records > 0:
        cmd.extend(["--max-records", str(max_records)])
    print("\n[KG POLICY]", policy)
    print("> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_summary(policy: str, variant_dir: Path) -> dict:
    summary_path = variant_dir / "reports" / "graph_summary.json"
    policy_summary_path = variant_dir / "reports" / "reasoning_policy_summary.csv"
    row = {"policy": policy, "variant_dir": str(variant_dir)}
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        row.update({
            "nodes": data.get("nodes", 0),
            "relationships": data.get("relationships", 0),
            "scientific_claims": data.get("scientific_claims", 0),
            "learned_directional_relationships": data.get("learned_directional_relationships", 0),
            "dangling_relationships": data.get("dangling_relationships", 0),
        })
    if policy_summary_path.exists():
        policy_df = pd.read_csv(policy_summary_path)
        if not policy_df.empty:
            first = policy_df.iloc[0].to_dict()
            for key in ["evidence_linked_claims", "summary_only_claims", "reasoning_findings"]:
                row[key] = first.get(key, 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KG variants under each scientific reasoning policy.")
    parser.add_argument("--records", required=True, type=Path, help="LiteratureAgent all_records.csv")
    parser.add_argument("--ontology", required=True, type=Path, help="Perovskite ontology JSON")
    parser.add_argument("--work-dir", required=True, type=Path, help="LiteratureAgent output directory")
    parser.add_argument("--out-root", required=True, type=Path, help="Output root for policy comparison folders")
    parser.add_argument("--max-records", type=int, default=0, help="Optional test limit; 0 processes all records")
    parser.add_argument(
        "--policies",
        nargs="*",
        default=POLICIES,
        choices=POLICIES,
        help="Policies to build. Defaults to current plus all philosophy reasoning schools.",
    )
    args = parser.parse_args()

    builder_script = Path(__file__).resolve().with_name("build_literature_knowledge_graph.py")
    args.out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for policy in args.policies:
        variant_dir = args.out_root / policy
        run_builder(
            builder_script=builder_script,
            records=args.records,
            ontology=args.ontology,
            work_dir=args.work_dir,
            out_dir=variant_dir,
            policy=policy,
            max_records=args.max_records,
        )
        summaries.append(read_summary(policy, variant_dir))

    comparison = pd.DataFrame(summaries)
    comparison_path = args.out_root / "reasoning_policy_kg_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    readme = """Reasoning Policy Knowledge Graph Comparison
==========================================

This folder contains separate graph builds from the same LiteratureAgent records:

- current: baseline evidence-grounded knowledge graph.
- socratic_uncertainty: constructs gap, uncertainty, and next-evidence nodes.
- cartesian_decomposition: constructs stepwise device/material/process/metric chains.
- kantian_constraints: constructs schema and target-completeness constraint checks.
- humean_evidence: flags direct evidence support versus summary-only claims.
- aristotelian_classification: constructs category and definition groupings.
- hegelian_conflict_resolution: constructs tension/conflict/synthesis nodes.
- platonic_abstraction: constructs abstract principle/pattern nodes.

Use reasoning_policy_kg_comparison.csv for the quick comparison table.
Each variant keeps the same standard KG outputs: Neo4j CSV, GraphML, JSONL,
reports, derived views, and plots.
"""
    (args.out_root / "README.txt").write_text(readme, encoding="utf-8")
    print(f"\nWrote reasoning-policy KG comparison to: {args.out_root.resolve()}")
    print(f"Comparison CSV: {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
