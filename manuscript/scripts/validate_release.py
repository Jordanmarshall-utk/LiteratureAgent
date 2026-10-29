#!/usr/bin/env python3
"""Validate the manuscript data and figure release without fitting models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
MAX_RELATIVE_PATH_LENGTH = 180
GENERATED_REPORTS = {
    "manuscript/validation/FILE_MANIFEST.csv",
    "manuscript/validation/VALIDATION.json",
}
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "build", "dist",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".svg", ".xml",
    ".ps1", ".cmd", ".bat", ".yml", ".yaml", ".toml", ".ini", ".bib", ".ris",
}
TEXT_NAMES = {"VERSION", "SHA256SUMS", ".gitignore", ".gitattributes"}
AUTHORING_PATTERN = re.compile(
    r"(?<![a-z])(?:" + "|".join((
        "co" + "dex", "pro" + "fessor", "hand" + "off", "co" + "worker",
        "repackag" + r"(?:e|ed|es|ing)",
    )) + r")(?![a-z])", re.IGNORECASE,
)
PERSONAL_PATH_PATTERN = re.compile(
    r"(?:[a-z]:[\\/]+(?:Users[\\/]+[^<\\/\s\"']+|LiteratureAgentProject)"
    r"|/(?:Users|home)/[^<\s/\"']+)", re.IGNORECASE,
)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"hf_[A-Za-z0-9]{25,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{25,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not set(path.relative_to(root).parts).intersection(EXCLUDED_DIRECTORIES)
        and path.relative_to(root).as_posix() not in GENERATED_REPORTS
        and not path.name.startswith("~$")
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for _ in reader)


def within_root(root: Path, relative: str) -> Path:
    if re.match(r"^[A-Za-z]:", relative) or relative.startswith(("/", "\\")):
        raise ValueError("manifest and asset paths must be relative to the release directory")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path leaves the release directory")
    return path


def resolve_asset(root: Path, relative: str) -> Path:
    for prefix in ("", "manuscript/", "manuscript/figures/"):
        path = within_root(root, prefix + relative)
        if path.is_file():
            return path
    return within_root(root, relative)


def hygiene_issues(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root).as_posix()
    issues = []
    if len(relative) > MAX_RELATIVE_PATH_LENGTH:
        issues.append(f"relative path length {len(relative)} exceeds limit {MAX_RELATIVE_PATH_LENGTH}")
    if AUTHORING_PATTERN.search(relative):
        issues.append("authoring language in filename")
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
        return issues
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    # Decode escaped separators used by JSON and embedded Python strings.
    normalized = content
    for _ in range(3):
        normalized = normalized.replace("\\\\", "\\")
    if AUTHORING_PATTERN.search(content):
        issues.append("authoring language in source text")
    if PERSONAL_PATH_PATTERN.search(normalized):
        issues.append("personal absolute path")
    if CONTROL_PATTERN.search(content):
        issues.append("unexpected control character")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        issues.append("credential pattern")
    return issues


def numeric_equal(left: str, right: str) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (ValueError, TypeError):
        return False


def stability_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row.get("dataset", ""), row.get("feature_mode", row.get("mode", "")), row.get("target", "")


def compare_stability_rows(
    plotted: list[dict[str, str]], reference: list[dict[str, str]],
) -> list[str]:
    errors = []
    lookup = {stability_key(row): row for row in reference}
    seen = set()
    for row in plotted:
        key = stability_key(row)
        if key in seen:
            errors.append(f"duplicate stability key {key}")
        seen.add(key)
        if "predicted_pce" in key[1]:
            errors.append(f"predicted PCE included in main stability mode {key}")
        expected = lookup.get(key)
        if expected is None:
            errors.append(f"stability key absent from current reference {key}")
            continue
        for plot_field, reference_field in (
            ("estimate", "value"), ("ci_lower", "ci_low"), ("ci_upper", "ci_high"),
            ("eligible_records", "n"), ("doi_groups", "doi_groups"),
        ):
            if not numeric_equal(row.get(plot_field, ""), expected.get(reference_field, "")):
                errors.append(f"stability {plot_field} differs for {key}")
        if row.get("metric") != expected.get("metric"):
            errors.append(f"stability metric differs for {key}")
    return errors


def verify_file_manifest(root: Path, manifest: Path) -> list[str]:
    errors = []
    listed = set()
    for row in csv_rows(manifest):
        relative = row.get("path", row.get("relative_path", ""))
        try:
            path = within_root(root, relative)
        except ValueError:
            errors.append("manifest path leaves release directory")
            continue
        if relative in listed:
            errors.append(f"duplicate manifest path: {relative}")
        listed.add(relative)
        if not path.is_file():
            errors.append(f"manifest file missing: {relative}")
        elif str(path.stat().st_size) != row.get("size_bytes") or sha256(path) != row.get("sha256", "").lower():
            errors.append(f"manifest content differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in release_files(root)}
    for relative in sorted(actual.difference(listed)):
        errors.append(f"file missing from manifest: {relative}")
    for relative in sorted(listed.difference(actual)):
        errors.append(f"manifest contains excluded or unavailable path: {relative}")
    return errors


def validate_release(root: Path, *, verify_manifest: bool = True) -> dict[str, Any]:
    root = root.resolve()
    report: dict[str, Any] = {"status": "PASS", "checks": [], "errors": [], "visual_review_performed": False}

    def check(name: str, action: Any) -> None:
        try:
            details = action()
            report["checks"].append({"check": name, "status": "PASS", "details": details})
        except (AssertionError, ValueError, KeyError, OSError, StopIteration, ElementTree.ParseError, csv.Error) as exc:
            report["checks"].append({"check": name, "status": "FAIL"})
            report["errors"].append({"check": name, "message": str(exc)})

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def figure_assets() -> dict[str, int]:
        directory = root / "manuscript/figures"
        combined_index = directory / "ALL_FIGURES.csv"
        rows = csv_rows(combined_index if combined_index.is_file() else directory / "FIGURE_INDEX.csv")
        require(len(rows) == 24, f"figure index has {len(rows)} rows; expected 24")
        require(len({row["plot_id"] for row in rows}) == 24, "figure plot IDs are not unique")
        count = 0
        for row in rows:
            for kind in ("png", "svg", "pdf", "editable"):
                require(bool(row.get(kind)), f"figure {row['plot_id']} lacks {kind} path")
                asset = resolve_asset(root, row[kind])
                require(asset.is_file() and asset.stat().st_size > 0, f"figure asset missing: {row[kind]}")
                if kind == "png":
                    with asset.open("rb") as handle:
                        require(handle.read(8) == b"\x89PNG\r\n\x1a\n", f"invalid PNG header: {row[kind]}")
                elif kind == "pdf":
                    with asset.open("rb") as handle:
                        require(handle.read(5) == b"%PDF-", f"invalid PDF header: {row[kind]}")
                elif kind == "svg" or asset.suffix.lower() == ".svg":
                    require(ElementTree.parse(asset).getroot().tag.endswith("svg"), f"invalid SVG root: {row[kind]}")
                count += 1
        return {"panels": len(rows), "asset_references": count}

    def stability() -> dict[str, int]:
        analysis = root / "manuscript/analysis/stability"
        all_rows = csv_rows(analysis / "all_stability_metrics_and_sensitivity.csv")
        require(len(all_rows) == 66, f"all stability table has {len(all_rows)} rows; expected 66")
        require(len({stability_key(row) for row in all_rows}) == 66, "all stability keys are not unique")
        for row in all_rows:
            source = row.get("oof_source", "")
            require(bool(source), f"stability prediction source absent: {stability_key(row)}")
            require(resolve_asset(root, source).is_file(), f"stability prediction source missing: {source}")
        shared = csv_rows(root / "manuscript/source/data/shared_stability/primary_repaired_stability_metrics.csv")
        plotted = []
        for panel in ("fig5a_stability_classification", "fig5b_stability_regression"):
            rows = csv_rows(root / f"manuscript/source/data/{panel}/plot_data.csv")
            require(len(rows) == 12, f"{panel} has {len(rows)} rows; expected 12")
            for reference in (all_rows, shared):
                errors = compare_stability_rows(rows, reference)
                require(not errors, "; ".join(errors))
            plotted.extend(rows)
        selected = {stability_key(row) for row in all_rows if row.get("manuscript_selected", "").lower() == "true"}
        require(selected == {stability_key(row) for row in plotted}, "main stability selections disagree with plot inputs")
        requested = csv_rows(analysis / "requested_24_updated_stability_estimates.csv")
        require(len(requested) == 24, "requested alternatives table must contain 24 estimates")
        require(len({stability_key(row) for row in requested}) == 24, "requested alternatives have duplicate keys")
        expected_modes = {
            (mode, target, dataset)
            for mode in ("design_only", "design_plus_predicted_pce")
            for target in ("target_T80_ge100h", "target_T80_ge500h", "target_end_retention_ge80")
            for dataset in ("baseline", "integrated")
        } | {
            (mode, target, dataset)
            for mode in ("design_only", "design_plus_predicted_pce", "design_plus_measured_initial_performance")
            for target in ("target_end_retention_pct", "target_log1p_T80_hours")
            for dataset in ("baseline", "integrated")
        }
        require({(row["mode"], row["target"], row["dataset"]) for row in requested} == expected_modes,
                "requested alternative combinations do not match the established analysis")
        for row in requested:
            corresponding = next(item for item in all_rows if stability_key(item) == stability_key(row))
            require(all(numeric_equal(row[field], corresponding[field]) for field in ("value", "ci_low", "ci_high", "n", "doi_groups")),
                    f"alternative estimate differs from full table: {stability_key(row)}")
        feature_rows = csv_rows(analysis / "stability_feature_lists.csv")
        main_modes = {stability_key(row)[:2] for row in plotted}
        main_features = [row for row in feature_rows if (row["dataset"], row["mode"]) in main_modes]
        require(bool(main_features), "main stability feature lists are absent")
        require(not any(row.get("predicted_pce_feature", "").lower() == "true" or "predicted_pce" in row.get("feature", "").lower() for row in main_features),
                "predicted PCE feature appears in a main stability feature list")
        return {"full_estimates": len(all_rows), "main_plot_rows": len(plotted), "requested_alternatives": len(requested)}

    def adjudication() -> dict[str, int]:
        expected = {
            "original_and_adjudicated_evidence_classifications.csv": 173,
            "typed_repair_ledger.csv": 232,
            "doi_verification_and_duplicate_decisions.csv": 10,
        }
        for filename, count in expected.items():
            actual = csv_count(root / "manuscript/data_adjudication" / filename)
            require(actual == count, f"{filename} has {actual} rows; expected {count}")
        return expected

    def attachments() -> dict[str, int]:
        directory = root / "manuscript/supplementary_data"
        rows = csv_rows(directory / "submission_attachment_manifest.csv")
        labels = {"Data S1", "Data S1a", "Data S1b", "Data S1c", "Data S1d", "Data S2"}
        numbered = [row for row in rows if row["manuscript_label"] in labels]
        require(len(numbered) == 6 and {row["manuscript_label"] for row in numbered} == labels,
                "submission attachment manifest does not contain all six numbered attachments exactly once")
        for row in rows:
            path = within_root(root, "manuscript/supplementary_data/" + row["packaged_filename"])
            require(path.is_file(), f"attachment missing: {row['packaged_filename']}")
            require(sha256(path) == row["sha256"], f"attachment checksum mismatch: {row['packaged_filename']}")
            require(path.stat().st_size == int(row["size_bytes"]), f"attachment size mismatch: {row['packaged_filename']}")
        return {"numbered_attachments": len(numbered), "all_manifest_entries": len(rows)}

    def shap() -> dict[str, int]:
        directory = root / "manuscript/source/data/shared_shap"
        validation = json.loads((directory / "pce_grouped_shap_validation.json").read_text(encoding="utf-8-sig"))
        expected = {"training_rows": 21426, "publication_group_count": 4080, "encoded_feature_count": 6262, "scientific_group_count": 218}
        require(validation["status"] == "PASS", "SHAP source validation is not PASS")
        for field, count in expected.items():
            require(validation.get(field) == count and validation["model_contract"].get(field) == count,
                    f"current SHAP contract differs for {field}")
        for filename, count in (
            ("pce_grouped_shap_values.csv", 21426),
            ("pce_grouped_shap_row_metadata.csv", 21426),
            ("pce_grouped_shap_importance.csv", 218),
        ):
            require(csv_count(directory / filename) == count, f"SHAP row count differs: {filename}")
        require(validation.get("out_of_fold_explanation") is False, "SHAP attribution scope is not the final fitted cohort")
        require(validation.get("explained_rows") == 21426, "SHAP coverage is incomplete")
        snapshot = root / "models/frozen/pce_then_stability_same_approach_sha256_19f58acd.py"
        require(sha256(snapshot) == validation["model_contract"]["model_script_sha256"], "frozen model snapshot checksum differs")
        return expected

    def graph() -> dict[str, int]:
        directory = root / "manuscript/source/graph_assets/fig6_integrated_graph"
        summary = json.loads((directory / "repaired_graph_summary.json").read_text(encoding="utf-8-sig"))
        expected = {"nodes": 91134, "relationships": 173911}
        for field, count in expected.items():
            require(summary[field] == count, f"graph {field} total differs")
        require(sum(summary["node_counts"].values()) == 91134, "graph node labels do not reconcile")
        require(sum(summary["relationship_counts"].values()) == 173911, "graph relationship types do not reconcile")
        for filename, key_field, counts_field in (
            ("effective_node_counts.csv", "node_type", "node_counts"),
            ("relationship_counts.csv", "relationship_type", "relationship_counts"),
        ):
            observed = {row[key_field]: int(row["count"]) for row in csv_rows(directory / filename)}
            require(observed == summary[counts_field], f"graph count table differs: {filename}")
        return expected

    def files_and_text() -> dict[str, Any]:
        files = release_files(root)
        problems = []
        longest = max((path.relative_to(root).as_posix() for path in files), key=len, default="")
        report["longest_relative_path"] = {
            "path": longest,
            "length": len(longest),
            "limit": MAX_RELATIVE_PATH_LENGTH,
        }
        for path in files:
            relative = path.relative_to(root).as_posix()
            if path.stat().st_size >= 100 * 1024 * 1024:
                problems.append({"path": relative, "issue": "file reaches the 100 MiB repository limit"})
            for issue in hygiene_issues(path, root):
                problems.append({"path": relative, "issue": issue})
        report["file_issues"] = problems
        require(not problems, f"{len(problems)} source or file issues; see file_issues (paths and issue types only)")
        return {"files_checked": len(files), "longest_relative_path_length": len(longest)}

    check("figure_assets", figure_assets)
    check("stability_data_dependencies", stability)
    check("repair_evidence_counts", adjudication)
    check("supplementary_attachments", attachments)
    check("current_shap_contract", shap)
    check("graph_count_reconciliation", graph)
    check("file_size_and_source_text", files_and_text)
    manifest = root / "manuscript/validation/FILE_MANIFEST.csv"
    if verify_manifest and manifest.is_file():
        def manifest_integrity() -> dict[str, int]:
            errors = verify_file_manifest(root, manifest)
            require(not errors, "; ".join(errors))
            return {"verified_entries": csv_count(manifest)}
        check("file_manifest_checksums", manifest_integrity)
    if report["errors"]:
        report["status"] = "FAIL"
    return report


def write_reports(root: Path, report: dict[str, Any]) -> None:
    directory = root / "manuscript/validation"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "FILE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        for path in release_files(root):
            writer.writerow({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    report["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
    (directory / "VALIDATION.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository release root.")
    parser.add_argument("--write-manifest", action="store_true", help="Write the release checksum manifest and validation report.")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = validate_release(root, verify_manifest=not args.write_manifest)
    if args.write_manifest:
        write_reports(root, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
