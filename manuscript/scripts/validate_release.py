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
from collections import Counter, defaultdict
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
CURRENT_PLOT_IDS = frozenset({
    "fig3a_historical_r2", "fig3b_historical_rmse", "fig3c_historical_mae",
    "fig3d_literatureagent_r2", "fig3e_literatureagent_rmse", "fig3f_literatureagent_mae",
    "fig4b_choice_shap", "fig5a_stability_classification", "fig5b_stability_regression",
    "figs1_global_shap", "figs2_solvent_shap", "figs3_local_shap",
    "figs4a_normalized_stability", "figs4b_normalization_sensitivity", "figs5_reasoning_graph",
    "figs6a_benchmark", "figs6b_benchmark", "figs6c_benchmark", "figs6d_benchmark",
})
AUTHOR_GRAPHIC_IDS = frozenset({"fig1_workflow", "fig2a_campaign_accounting", "fig2b_campaign_readiness",
                                 "fig4a_five_paper_graph", "fig6_integrated_graph"})
GRAPHICAL_SUFFIXES = {".png", ".svg", ".pdf", ".opju", ".pptx", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".emf", ".wmf", ".eps"}
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
STABILITY_VALUES_SHA256 = "97e3f238847336a3f7167f0886fc37868f2e763031e981ff0b5068488a0faffa"


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


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def canonical_publication(value: str, *, pooled_missing: bool = False) -> str:
    raw = value.strip()
    if raw.casefold() in {"", "nan", "none", "null", "na", "n/a", "no_group", "missing_group"}:
        if pooled_missing:
            return "NO_GROUP"
        raise ValueError("missing publication identity cannot be merged into a stability group")
    candidate = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw.lstrip("\ufeff").casefold())
    candidate = re.sub(r"^doi:\s*", "", candidate).rstrip(" .;,")
    return candidate if re.fullmatch(r"10\.\d{4,9}/\S+", candidate) else raw


def canonical_sample_key(value: str) -> str:
    publication, separator, sample = value.partition("|")
    if not separator:
        raise ValueError("sample key lacks its publication separator")
    return canonical_publication(publication, pooled_missing=True) + separator + sample


def pce_identity_errors(predictions: list[dict[str, str]], membership: list[dict[str, str]]) -> list[str]:
    errors = []
    test_keys = [row["canonical_row_key"] for row in predictions]
    if len(set(test_keys)) != len(test_keys):
        errors.append("canonical test sample identities collide")
    if any(canonical_sample_key(row["raw_frozen_row_key"]) != row["canonical_row_key"] for row in predictions):
        errors.append("canonicalization changed a sample suffix or publication contract")
    if any(canonical_sample_key(row["raw_row_key"]) != row["canonical_row_key"] for row in membership):
        errors.append("training identity crosswalk does not follow the sample-key contract")
    if len({row["canonical_row_key"] for row in membership}) != len(membership):
        errors.append("eligible canonical sample identities collide")
    training_keys = {row["canonical_row_key"] for row in membership if truth(row["in_training"])}
    if set(test_keys) & training_keys:
        errors.append("the same canonical sample appears in training and test")
    matched = {row["canonical_row_key"] for row in membership if truth(row["is_frozen_test_sample"])}
    if matched != set(test_keys):
        errors.append("training audit does not identify every frozen test sample")
    return errors


def publication_fold_errors(rows: list[dict[str, str]]) -> list[str]:
    folds: dict[str, set[str]] = defaultdict(set)
    errors = []
    for row in rows:
        identity = row["canonical_publication_identity"]
        if not identity:
            errors.append("stability publication identity is missing")
            continue
        if canonical_publication(row["raw_publication_label"]) != identity:
            errors.append("stability canonical publication does not match its retained source label")
        folds[identity].add(row["final_test_fold"])
    if any(len(values) != 1 for values in folds.values()):
        errors.append("a canonical publication crosses historical test folds")
    return errors


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("invalid regression prediction alignment")
    count = len(actual)
    mean = math.fsum(actual) / count
    residual = math.fsum((left - right) ** 2 for left, right in zip(actual, predicted))
    total = math.fsum((value - mean) ** 2 for value in actual)
    return {"r2": 1 - residual / total if total else float(residual == 0),
            "rmse": math.sqrt(residual / count),
            "mae": math.fsum(abs(left - right) for left, right in zip(actual, predicted)) / count}


def classification_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    ordered = sorted((float(row["prediction"]), int(float(row["actual"]))) for row in rows)
    positives = sum(actual for _, actual in ordered)
    negatives = len(ordered) - positives
    if not positives or not negatives or any(actual not in {0, 1} for _, actual in ordered):
        raise ValueError("ROC-AUC requires both binary outcome classes")
    concordant, previous_negatives, start = 0.0, 0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        tied_positive = sum(actual for _, actual in ordered[start:end])
        tied_negative = end - start - tied_positive
        concordant += tied_positive * (previous_negatives + 0.5 * tied_negative)
        previous_negatives += tied_negative
        start = end
    pairs = [(int(float(row["actual"])), int(float(row["predicted_class"]))) for row in rows]
    true_positive = sum(actual == predicted == 1 for actual, predicted in pairs)
    true_negative = sum(actual == predicted == 0 for actual, predicted in pairs)
    false_positive = sum(actual == 0 and predicted == 1 for actual, predicted in pairs)
    false_negative = sum(actual == 1 and predicted == 0 for actual, predicted in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return {"roc_auc": concordant / (positives * negatives), "accuracy": (true_positive + true_negative) / len(rows),
            "balanced_accuracy": 0.5 * (true_positive / positives + true_negative / negatives),
            "f1": 2 * true_positive / denominator if denominator else 0.0}


def stability_values_fingerprint(rows: list[dict[str, str]]) -> str:
    values = [
        [row["dataset"], row["mode"], row["target"], row["metric"],
         *(format(float(row[field]), ".12g") for field in ("value", "ci_low", "ci_high")),
         str(int(float(row["n"]))), str(int(float(row["canonical_group_count"]))), str(int(float(row["bootstrap_seed"])))]
        for row in sorted(rows, key=lambda item: (item["dataset"], item["mode"], item["target"]))
    ]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def bootstrap_regression(
    actual: list[float], predictions: dict[str, list[float]], groups: list[str], contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    import numpy as np

    ordered = list(dict.fromkeys(groups))
    count = len(ordered)
    if contract["groups"] != count or contract["iterations"] != 2000 or contract["seed"] != 42:
        raise ValueError("PCE bootstrap count, iterations or seed changed")
    if contract["sampler"] != "np.random.default_rng(42).multinomial(G, np.full(G, 1/G), size=2000)" or contract["group_order"] != "first appearance in frozen test CSV order":
        raise ValueError("PCE bootstrap sampler or group-order declaration differs")
    lookup = {group: index for index, group in enumerate(ordered)}
    inverse = np.array([lookup[group] for group in groups])
    draws = np.random.default_rng(42).multinomial(count, np.full(count, 1 / count), size=2000)
    if hashlib.sha256(draws.astype(np.int64).tobytes(order="C")).hexdigest() != contract["multiplicity_sha256_int64_c"]:
        raise ValueError("PCE first-seen multinomial group multiplicities differ")
    actual_array = np.asarray(actual, dtype=float)
    size = draws @ np.bincount(inverse)
    total = draws @ np.bincount(inverse, weights=actual_array)
    total_square = draws @ np.bincount(inverse, weights=actual_array ** 2)
    total_variance = total_square - total ** 2 / size
    results = {}
    for condition, predicted in predictions.items():
        error = actual_array - np.asarray(predicted, dtype=float)
        square_error = draws @ np.bincount(inverse, weights=error ** 2)
        absolute_error = draws @ np.bincount(inverse, weights=np.abs(error))
        results[condition] = {"r2": 1 - square_error / total_variance,
                              "rmse": np.sqrt(square_error / size), "mae": absolute_error / size}
    return results, ordered


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


def figure_identity_errors(rows: list[dict[str, str]]) -> list[str]:
    identifiers = [row["plot_id"] for row in rows]
    errors = []
    if len(identifiers) != len(set(identifiers)):
        errors.append("figure plot IDs are not unique")
    if len(rows) != len(CURRENT_PLOT_IDS) or set(identifiers) != CURRENT_PLOT_IDS:
        errors.append("figure index must contain exactly the 19 manuscript-used source panels")
    if set(identifiers) & AUTHOR_GRAPHIC_IDS:
        errors.append("generated alternatives to author-created graphics must not be indexed")
    return errors


def unexpected_graphics(root: Path, allowed_assets: set[Path]) -> list[str]:
    allowed = {path.resolve() for path in allowed_assets}
    return [path.relative_to(root).as_posix() for path in release_files(root)
            if path.suffix.lower() in GRAPHICAL_SUFFIXES and path.resolve() not in allowed]


def validate_release(root: Path, *, verify_manifest: bool = True) -> dict[str, Any]:
    root = root.resolve()
    report: dict[str, Any] = {"status": "PASS", "checks": [], "errors": [], "visual_review_performed": False}

    def check(name: str, action: Any) -> None:
        try:
            details = action()
            report["checks"].append({"check": name, "status": "PASS", "details": details})
        except (AssertionError, ValueError, KeyError, OSError, StopIteration, ElementTree.ParseError, csv.Error, ImportError) as exc:
            report["checks"].append({"check": name, "status": "FAIL"})
            report["errors"].append({"check": name, "message": str(exc)})

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def figure_assets() -> dict[str, int]:
        directory = root / "manuscript/figures"
        combined_index = directory / "ALL_FIGURES.csv"
        rows = csv_rows(combined_index if combined_index.is_file() else directory / "FIGURE_INDEX.csv")
        require(not figure_identity_errors(rows), "; ".join(figure_identity_errors(rows)))
        count = 0
        allowed_assets: set[Path] = set()
        for row in rows:
            for kind in ("png", "svg", "pdf", "editable"):
                require(bool(row.get(kind)), f"figure {row['plot_id']} lacks {kind} path")
                asset = resolve_asset(root, row[kind])
                require(asset.is_file() and asset.stat().st_size > 0, f"figure asset missing: {row[kind]}")
                allowed_assets.add(asset.resolve())
                if kind == "png":
                    with asset.open("rb") as handle:
                        require(handle.read(8) == b"\x89PNG\r\n\x1a\n", f"invalid PNG header: {row[kind]}")
                elif kind == "pdf":
                    with asset.open("rb") as handle:
                        require(handle.read(5) == b"%PDF-", f"invalid PDF header: {row[kind]}")
                elif kind == "svg" or asset.suffix.lower() == ".svg":
                    require(ElementTree.parse(asset).getroot().tag.endswith("svg"), f"invalid SVG root: {row[kind]}")
                count += 1
        manual = csv_rows(directory / "AUTHOR_GRAPHICS.csv")
        require(len(manual) == 4, "author-graphics scope must identify Figures 1, 2, 4a and 6")
        require(all(row["status"] == "author_composed_not_bundled" for row in manual),
                "author-created graphics must not imply that an unverified editable master is bundled")
        manual_ids = {value.strip() for row in manual for value in row["excluded_plot_ids"].split(";") if value.strip()}
        require(manual_ids == AUTHOR_GRAPHIC_IDS, "author-graphics scope does not match the excluded generated alternatives")
        for row in csv_rows(directory / "legends/LEGEND_INDEX.csv"):
            require(row["legend_id"] != "legend_fig4a_knowledge_graph", "unused generated author-graphic legend remains indexed")
            for kind in ("png", "svg", "editable"):
                if not row.get(kind):
                    continue
                relative = row[kind].replace("\\", "/")
                candidate = (directory / "legends" / relative).resolve()
                require(candidate.is_relative_to(root), "legend asset path leaves the release directory")
                if not candidate.is_file():
                    candidate = resolve_asset(root, relative)
                require(candidate.is_file(), f"legend asset missing: {relative}")
                allowed_assets.add(candidate.resolve())
        extra = unexpected_graphics(root, allowed_assets)
        require(not extra, "unlisted or non-manuscript graphical assets: " + "; ".join(extra))
        return {"panels": len(rows), "asset_references": count, "author_created_graphics": len(manual),
                "graphical_files": len(allowed_assets), "unlisted_graphical_files": len(extra)}

    def stability() -> dict[str, int]:
        analysis = root / "manuscript/analysis/stability"
        all_rows = csv_rows(analysis / "all_stability_metrics_and_sensitivity.csv")
        require(len(all_rows) == 66, f"all stability table has {len(all_rows)} rows; expected 66")
        require(len({stability_key(row) for row in all_rows}) == 66, "all stability keys are not unique")
        require(stability_values_fingerprint(all_rows) == STABILITY_VALUES_SHA256,
                "stability numerical values differ from the verified canonical-publication export")
        require(sum(truth(row["interval_changed"]) for row in all_rows) == 64, "canonical stability interval change count differs")
        require(sum(truth(row["group_count_changed"]) for row in all_rows) == 7, "canonical stability group-count change count differs")
        require(not any(truth(row["point_changed"]) or truth(row["new_model_fit"]) for row in all_rows),
                "saved stability point estimates or model-fit scope changed")
        seed_prefixes = Counter()
        fold_metric_count = 0
        for row in all_rows:
            source = row.get("oof_source", "")
            require(bool(source), f"stability prediction source absent: {stability_key(row)}")
            for source_field, hash_field in (("oof_source", "oof_sha256"), ("fold_source", "fold_sha256"), ("fold_metric_source", "fold_metric_sha256")):
                path = resolve_asset(root, row[source_field])
                require(path.is_file() and sha256(path) == row[hash_field], f"stability source or hash differs: {row[source_field]}")
            predictions = csv_rows(resolve_asset(root, source))
            folds = csv_rows(resolve_asset(root, row["fold_source"]))
            require(len(predictions) == len(folds) == int(float(row["n"])), f"stability source population differs: {stability_key(row)}")
            require(not publication_fold_errors(predictions), f"invalid canonical stability partition: {stability_key(row)}")
            require(len({item["canonical_publication_identity"] for item in predictions}) == int(float(row["canonical_group_count"])),
                    f"canonical stability publication count differs: {stability_key(row)}")
            for predicted, fold in zip(predictions, folds):
                require(all(predicted[field] == fold[field] for field in ("row_position", "source_row_index", "stable_sample_key", "canonical_publication_identity", "final_test_fold")),
                        f"stability row-level fold alignment differs: {stability_key(row)}")
            primary = classification_metrics(predictions) if row["metric"] == "roc_auc" else regression_metrics(
                [float(item["actual"]) for item in predictions], [float(item["prediction"]) for item in predictions])
            require(numeric_equal(primary[row["metric"]], row["value"]), f"stability point does not recompute: {stability_key(row)}")
            by_fold: dict[str, list[dict[str, str]]] = defaultdict(list)
            for item in predictions:
                by_fold[item["final_test_fold"]].append(item)
            saved_fold_metrics = csv_rows(resolve_asset(root, row["fold_metric_source"]))
            require(len(saved_fold_metrics) == len(by_fold) == 5, "stability evaluation must retain five verified folds")
            for fold in saved_fold_metrics:
                held = by_fold[fold["fold"]]
                values = classification_metrics(held) if row["metric"] == "roc_auc" else regression_metrics(
                    [float(item["actual"]) for item in held], [float(item["prediction"]) for item in held])
                for metric, value in values.items():
                    require(numeric_equal(value, fold[metric]), f"saved stability fold metric does not recompute: {stability_key(row)}/{metric}")
                    fold_metric_count += 1
            seed_parts = row["bootstrap_seed_parts"]
            seed = 42 + int(hashlib.sha256(seed_parts.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000
            require(seed == int(float(row["bootstrap_seed"])) and int(float(row["bootstrap_iterations"])) == 2000,
                    "stability deterministic bootstrap seed or replicate count differs")
            seed_prefixes["completion" if seed_parts.startswith("ref28_exclusion|") else "carried"] += 1
            if truth(row["group_count_changed"]):
                require(row["dataset"] == "baseline" and int(float(row["n"])) == 4986 and int(float(row["raw_group_count"])) == 1822 and int(float(row["canonical_group_count"])) == 1821,
                        "case-alias stability population differs")
                aliases = [item for item in predictions if item["canonical_publication_identity"] == "10.1039/c8se00314a"]
                require(len(aliases) == 2 and {item["Ref_ID"] for item in aliases} == {"17636", "33875"}
                        and {item["row_position"] for item in aliases} == {"2157", "3882"}
                        and {item["final_test_fold"] for item in aliases} == {"2"}, "verified case-alias source identities or historical folds differ")
        require(seed_prefixes == {"completion": 44, "carried": 22}, "stability completion/carried bootstrap seed scopes differ")
        dependency = csv_rows(analysis / "stability_dependency_change_table.csv")
        require(len(dependency) == 66 and {stability_key(row) for row in dependency} == {stability_key(row) for row in all_rows},
                "stability retained-dependency decisions do not cover every final estimate")
        require(all(truth(row["source_order_and_targets_match_saved"]) and row["historical_folds_reconstruction"] == "PASS"
                    and int(float(row["canonical_groups_crossing_verified_folds"])) == 0 and not truth(row["requires_new_model_refit"]) for row in dependency),
                "stability retained OOF lacks an unchanged-source/verified-partition basis")
        require((analysis / "canonical/historical_fold_reconstruction_metrics.csv").is_file(), "historical fold reconstruction metric proof is absent")
        shared = csv_rows(root / "manuscript/source/data/shared_stability/primary_repaired_stability_metrics.csv")
        plotted = []
        for panel in ("fig5a_stability_classification", "fig5b_stability_regression"):
            rows = csv_rows(root / f"manuscript/source/data/{panel}/plot_data.csv")
            require(len(rows) == 12, f"{panel} has {len(rows)} rows; expected 12")
            for reference in (all_rows, shared):
                errors = compare_stability_rows(rows, reference)
                require(not errors, "; ".join(errors))
            plotted.extend(rows)
        require(all(truth(row["interval_changed"]) and not truth(row["point_changed"]) for row in plotted),
                "all 24 main stability intervals must update without changing point estimates")
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
        table = csv_rows(analysis / "TableS10_updated.csv")
        require(len(table) == 33, "Table S10 must contain 33 Original/Integrated outcome/input-set comparisons")
        lookup = {stability_key(row): row for row in all_rows}
        for comparison in table:
            values = {}
            for dataset, prefix in (("baseline", "original"), ("integrated", "integrated")):
                estimate = lookup[(dataset, comparison["model_mode"], comparison["endpoint_name"])]
                for table_field, estimate_field in (("database_performance", "value"), ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("eligible_rows", "n"), ("doi_groups", "canonical_group_count")):
                    require(numeric_equal(comparison[f"{prefix}_{table_field}"], estimate[estimate_field]), "Table S10 value/population differs from its canonical estimate")
                for field in ("oof_source", "oof_sha256", "fold_source", "fold_sha256", "fold_metric_source", "fold_metric_sha256"):
                    require(comparison[f"{prefix}_{field}"] == estimate[field], "Table S10 source reference differs from its canonical estimate")
                values[dataset] = float(estimate["value"])
            require(numeric_equal(comparison["difference_integrated_minus_original"], values["integrated"] - values["baseline"]),
                    "Table S10 descriptive point difference does not recompute")
            require("not paired" in comparison["comparison_basis"], "Table S10 must identify the separately eligible population comparison")
        return {"full_estimates": len(all_rows), "main_plot_rows": len(plotted), "requested_alternatives": len(requested),
                "table_s10_rows": len(table), "recomputed_fold_metric_values": fold_metric_count, "canonical_publications_crossing_folds": 0}

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
        reconciliation = root / "manuscript/source/data/shared_graph"
        rule = json.loads((reconciliation / "counting_rule.json").read_text(encoding="utf-8-sig"))
        patch = json.loads((reconciliation / "snapshot_metadata_patch.json").read_text(encoding="utf-8-sig"))
        metadata = patch["set_properties"]
        counts = {"field_evidence_count": 37663, "source_input_field_evidence_mapping_count": 37663,
                  "current_field_evidence_link_count": 37805, "current_observation_field_evidence_link_count": 31116,
                  "current_fallback_field_evidence_link_count": 6689, "current_repair_field_evidence_link_count": 142,
                  "current_claim_evidence_link_count": 2694}
        require(all(metadata.get(field) == count for field, count in counts.items()), "graph evidence subtotals differ from the traced builder definition")
        require(rule["unique_frozen_input_mappings"] == rule["unique_frozen_field_evidence_ids"] == 37663
                and rule["frozen_exact_sample_mapping_rows"] == 37282 and rule["frozen_doi_field_fallback_rows"] == 381,
                "graph historical mapping inputs do not reconcile")
        require(31116 + 6689 == 37663 + 142 == 37805 and 31116 + 2694 == 33810, "graph field/claim evidence arithmetic differs")
        require(patch["topology_changes"] == [] and patch["field_evidence_count_numeric_value_changed"] is False,
                "graph companion patch must preserve historical count and topology")
        require(len(csv_rows(reconciliation / "metadata_enrichment_node_updates.csv")) == 21,
                "preceding graph metadata-only enrichment count differs")
        return {**expected, "historical_field_mappings": 37663, "current_field_links": 37805, "repair_field_links": 142}

    def pce() -> dict[str, int]:
        import numpy as np

        directory = root / "manuscript/analysis/pce/row_random"
        audits = {row["condition"]: row for row in csv_rows(directory / "training_audit.csv")}
        require(set(audits) == {"original", "curated", "expanded"}, "row-random training conditions differ")
        membership = csv_rows(directory / "training_membership.csv")
        frozen = csv_rows(directory / "references/original_row_random_saved.csv")
        require(len(frozen) == 4159, "saved frozen row-random reference must contain 4,159 records")
        frames, point_values = {}, {}
        expected_train = {"original": 16634, "curated": 16741, "expanded": 17267}
        for condition, training_rows in expected_train.items():
            predictions = csv_rows(directory / f"predictions_{condition}.csv")
            frames[condition] = predictions
            require(len(predictions) == 4159, "row-random test record count changed")
            own_membership = [row for row in membership if row["condition"] == condition]
            errors = pce_identity_errors(predictions, own_membership)
            require(not errors, f"{condition}: " + "; ".join(errors))
            require(sum(truth(row["in_training"]) for row in own_membership) == training_rows,
                    f"row-random training membership differs: {condition}")
            for index, (prediction, saved) in enumerate(zip(predictions, frozen)):
                require(int(prediction["frozen_test_order"]) == index and prediction["raw_frozen_row_key"] == saved["row_key"]
                        and numeric_equal(prediction["actual"], saved["actual"]), "frozen row-random identities, order or targets changed")
                require(canonical_publication(prediction["raw_publication_label"], pooled_missing=True) == prediction["normalized_publication_identity"],
                        "row-random publication crosswalk differs from canonical identity")
            groups = [row["normalized_publication_identity"] for row in predictions]
            require(len(set(groups)) == 2243 and groups.count("NO_GROUP") == 48, "row-random missing-identity cluster convention changed")
            require(len({row["raw_publication_label"] for row in predictions}) == 2248, "raw row-random publication-label count changed")
            training_groups = {row["normalized_publication_identity"] for row in own_membership if truth(row["in_training"])}
            require(len((set(groups) & training_groups) - {"NO_GROUP"}) == 2161 and "NO_GROUP" in training_groups,
                    "row-random publication overlap accounting differs")
            audit = audits[condition]
            expected = {"training_rows": training_rows, "matched_frozen_test_samples": 4159, "canonical_test_rows_in_training": 0,
                        "canonical_key_collisions": 0, "identified_test_publications_in_training": 2161,
                        "canonical_bootstrap_groups_in_test": 2243, "identified_publications_in_test": 2242, "unresolved_identity_records_in_test": 48}
            require(all(int(float(audit[field])) == value for field, value in expected.items()), "row-random audit counts differ from verified identities")
            point_values[condition] = regression_metrics([float(row["actual"]) for row in predictions], [float(row["predicted"]) for row in predictions])
        require(int(audits["expanded"]["eligible_rows"]) == 21426 and int(audits["expanded"]["raw_key_exclusion_training_rows"]) == 17269,
                "expanded repaired PCE cohort or two-row correction accounting differs")
        for metric, expected in {"r2": 0.6257189513816943, "rmse": 2.5321764043399457, "mae": 1.9206519643601392}.items():
            require(numeric_equal(point_values["expanded"][metric], expected), "expanded row-random point is not the corrected saved model")
        contract = json.loads((directory / "row_random_bootstrap_contract.json").read_text(encoding="utf-8-sig"))
        draws, ordered = bootstrap_regression([float(row["actual"]) for row in frames["original"]],
                                              {condition: [float(row["predicted"]) for row in rows] for condition, rows in frames.items()},
                                              [row["normalized_publication_identity"] for row in frames["original"]], contract)
        saved_order = csv_rows(directory / "row_random_bootstrap_group_order.csv")
        require([row["normalized_publication_identity"] for row in saved_order] == ordered, "row-random bootstrap groups are not in first-seen order")
        marginal = csv_rows(directory / "row_random_metrics.csv")
        require(len(marginal) == 9, "row-random marginal table must contain nine metrics")
        for row in marginal:
            condition, metric = row["condition"], row["metric"]
            require(numeric_equal(row["value"], point_values[condition][metric]), "row-random point does not recompute from exported predictions")
            low, high = np.percentile(draws[condition][metric], [2.5, 97.5])
            require(numeric_equal(low, row["ci_low"]) and numeric_equal(high, row["ci_high"]), "row-random marginal interval does not reproduce")

        def paired_checks(rows: list[dict[str, str]], bootstrap: dict[str, dict[str, Any]], points: dict[str, dict[str, float]], group_count: int, record_count: int) -> None:
            require(len(rows) == 6 and len({(row["comparison"], row["metric"]) for row in rows}) == 6, "paired PCE table must contain six unique comparison metrics")
            for row in rows:
                condition = row["comparison"].split(" minus ")[0]
                metric = row["metric"]
                difference = bootstrap[condition][metric] - bootstrap["original"][metric]
                low, high = np.percentile(difference, [2.5, 97.5])
                expected = {"direct_metric_difference": points[condition][metric] - points["original"][metric],
                            "median_bootstrap_difference": float(np.median(difference)), "ci_low": low, "ci_high": high}
                require(all(numeric_equal(row[field], value) for field, value in expected.items()), "paired PCE direct/median difference or interval does not reproduce")
                require(int(row["bootstrap_groups"]) == group_count and int(row["test_records"]) == record_count
                        and int(row["bootstrap_iterations"]) == 2000 and int(row["seed"]) == 42, "paired PCE bootstrap population or seed differs")

        paired_checks(csv_rows(directory / "row_random_paired_differences.csv"), draws, point_values, 2243, 4159)
        historical = {condition: csv_rows(directory / f"historical_predictions_{condition}.csv") for condition in expected_train}
        historical_original = historical["original"]
        require(len(historical_original) == 4092, "historical main test membership count changed")
        for rows in historical.values():
            require(len(rows) == 4092 and all(row["canonical_row_key"] == baseline["canonical_row_key"] and numeric_equal(row["actual"], baseline["actual"])
                    and row["normalized_publication_identity"] == baseline["normalized_publication_identity"] for row, baseline in zip(rows, historical_original)),
                    "historical main test samples, targets or group alignment changed")
        historical_contract = json.loads((directory / "historical_741_bootstrap_contract.json").read_text(encoding="utf-8-sig"))
        historical_draws, historical_order = bootstrap_regression([float(row["actual"]) for row in historical_original],
                                                                 {condition: [float(row["predicted"]) for row in rows] for condition, rows in historical.items()},
                                                                 [row["normalized_publication_identity"] for row in historical_original], historical_contract)
        require(len(historical_order) == 741, "historical main bootstrap must retain 741 canonical publication groups")
        historical_points = {condition: regression_metrics([float(row["actual"]) for row in rows], [float(row["predicted"]) for row in rows]) for condition, rows in historical.items()}
        paired_checks(csv_rows(directory / "historical_741_paired_differences.csv"), historical_draws, historical_points, 741, 4092)
        main_intervals = csv_rows(directory / "main_Figure3_S6A_interval_validation.csv")
        require(len(main_intervals) == 9, "main Figure 3/S6A validation must contain nine retained intervals")
        condition_ids = {"original": "01_original_database", "curated": "02_after_google_drive", "expanded": "03_after_google_drive_plus_expansion"}
        panel_names = {"r2": "fig3a_historical_r2", "rmse": "fig3b_historical_rmse", "mae": "fig3c_historical_mae"}
        main_plots = {metric: {row["dataset_id"]: row for row in csv_rows(root / f"manuscript/source/data/{panel}/plot_data.csv")} for metric, panel in panel_names.items()}
        for row in main_intervals:
            condition, metric = row["condition"], row["metric"]
            low, high = np.percentile(historical_draws[condition][metric], [2.5, 97.5])
            require(row["status"] == "PASS" and numeric_equal(historical_points[condition][metric], row["retained_value"])
                    and numeric_equal(low, row["retained_ci_low"]) and numeric_equal(high, row["retained_ci_high"]), "retained main Figure 3/S6A interval does not reproduce")
            plotted = main_plots[metric][condition_ids[condition]]
            require(all(numeric_equal(plotted[field], row[reference]) for field, reference in (("estimate", "retained_value"), ("ci_lower", "retained_ci_low"), ("ci_upper", "retained_ci_high"))),
                    "validated main Figure 3 source values were changed")
        return {"frozen_row_random_records": 4159, "corrected_expanded_training_records": 17267, "identified_publications": 2242,
                "pooled_unresolved_records": 48, "row_random_clusters": 2243, "main_historical_records": 4092,
                "main_historical_groups": 741, "retained_main_intervals": 9, "bootstrap_iterations": 2000}

    def predictors() -> dict[str, int]:
        directory = root / "manuscript/supplementary_data"
        inventory = csv_rows(directory / "DataS2_PCE_predictor_inventory.csv")
        require(len(inventory) == len({row["original_field_name"] for row in inventory}) == 461, "included/excluded predictor universe must contain 461 exact field names")
        included = [row for row in inventory if row["inclusion_status"] == "included"]
        excluded = [row for row in inventory if row["inclusion_status"] == "excluded"]
        require(len(included) == 278 and len(excluded) == 183, "included/excluded predictor membership counts differ")
        require(Counter(row["frozen_type"] for row in included) == {"numeric": 113, "categorical": 165}, "frozen 113/165 predictor typing changed")
        boolean = [row for row in included if row["retained_inventory_type"].lower().startswith("boolean")]
        require(len(boolean) == 16 and all(row["frozen_type"] == "categorical" for row in boolean), "Boolean predictor categorical encoding changed")
        require(sum(truth(row["frozen_overrides_dynamic"]) for row in included) == 14, "frozen/dynamic typing overrides changed")
        require(sum(row["original_field_name"].startswith("Stability_") for row in excluded) == 44
                and sum(row["original_field_name"].startswith("Outdoor_") for row in excluded) == 40, "direct Stability_/Outdoor_ exclusion counts differ")
        lineage = csv_rows(root / "manuscript/analysis/pce/predictor_lineage.csv")
        indirect = {row["original_field_name"] for row in lineage if not truth(row["included_as_direct_frozen_predictor"])}
        require(len(lineage) == 21 and indirect == {"Ref_publication_date", "JV_certification_institute", "Outdoor_location_country", "Outdoor_location_city", "Outdoor_location_coordinates", "Outdoor_location_climate_zone"},
                "descriptor source lineage no longer identifies the six directly excluded fields")
        require(indirect.issubset({row["original_field_name"] for row in excluded}), "descriptor source lineage contradicts direct predictor membership")
        return {"inventory_fields": 461, "included": 278, "excluded": 183, "numeric": 113, "categorical": 165, "descriptor_source_fields": 21, "indirect_excluded_fields": 6}

    def figure5_embedded_values() -> dict[str, int]:
        from PIL import Image

        index_path = root / "manuscript/figures/ALL_FIGURES.csv"
        figures = {row["plot_id"]: row for row in csv_rows(index_path)}
        for panel in ("fig5a_stability_classification", "fig5b_stability_regression"):
            source = root / f"manuscript/source/data/{panel}/plot_data.csv"
            expected = csv_rows(source)
            svg = resolve_asset(root, figures[panel]["svg"])
            document = ElementTree.parse(svg).getroot()
            require(not any(node.tag.rsplit("}", 1)[-1] == "image" for node in document.iter()), "Figure 5 SVG must retain vector-editable data marks")
            descriptions = [node.text for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "description" and node.text]
            require(len(descriptions) == 1, "Figure 5 SVG lacks a unique scientific source description")
            metadata = json.loads(descriptions[0])
            require(metadata["source_sha256"] == sha256(source), "Figure 5 source changed after rendering")
            embedded = metadata["plot_records"]
            require(len(embedded) == len(expected) == 12, "Figure 5 rendered record count differs")
            lookup = {stability_key(row): row for row in expected}
            require(len({stability_key(row) for row in embedded}) == 12, "Figure 5 embedded scientific keys collide")
            for row in embedded:
                plotted = lookup[stability_key(row)]
                for embedded_field, source_field in (("estimate", "estimate"), ("ci_lower", "ci_lower"), ("ci_upper", "ci_upper"), ("record_count", "eligible_records"), ("canonical_group_count", "canonical_group_count")):
                    require(numeric_equal(row[embedded_field], plotted[source_field]), "rendered Figure 5 estimate, interval or population differs from verified CSV")
            with Image.open(resolve_asset(root, figures[panel]["png"])) as image:
                require(json.loads(image.info["Description"]) == metadata, "Figure 5 PNG scientific metadata differs from its SVG")
                require(min(image.size) >= 1500 and min(image.info.get("dpi", (0, 0))) >= 599, "Figure 5 PNG resolution is below the publication export contract")
            require(metadata["source_sha256"].encode("ascii") in resolve_asset(root, figures[panel]["pdf"]).read_bytes(), "Figure 5 PDF omits the verified scientific source identity")
        return {"individual_panels": 2, "rendered_estimates_reconciled": 24, "vector_svg_raster_images": 0}

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
    check("canonical_pce_holdout_and_bootstrap", pce)
    check("frozen_predictor_inventory_and_lineage", predictors)
    check("figure5_rendered_scientific_values", figure5_embedded_values)
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
