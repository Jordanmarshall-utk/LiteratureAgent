"""Regression checks for the release validator's data and integrity gates."""
from __future__ import annotations

import csv
import ast
import importlib.util
import os
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "manuscript/scripts/validate_release.py"
SPEC = importlib.util.spec_from_file_location("manuscript_release_validator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def example_plot() -> dict[str, str]:
    return {
        "dataset": "integrated", "feature_mode": "physical_condition_features",
        "target": "target_end_retention_pct", "metric": "r2", "estimate": "0.05",
        "ci_lower": "0.02", "ci_upper": "0.08", "eligible_records": "5100",
        "doi_groups": "1909",
    }


def example_reference() -> dict[str, str]:
    return {
        "dataset": "integrated", "mode": "physical_condition_features",
        "target": "target_end_retention_pct", "metric": "r2", "value": "0.05",
        "ci_low": "0.02", "ci_high": "0.08", "n": "5100", "doi_groups": "1909",
    }


def test_stability_comparison_checks_estimates_intervals_and_population() -> None:
    plotted = example_plot()
    reference = example_reference()
    assert validator.compare_stability_rows([plotted], [reference]) == []
    for field in ("estimate", "ci_lower", "ci_upper", "eligible_records", "doi_groups"):
        changed = {**plotted, field: "-1"}
        assert validator.compare_stability_rows([changed], [reference])


def test_stability_comparison_rejects_missing_duplicate_and_predicted_modes() -> None:
    plotted = example_plot()
    reference = example_reference()
    assert validator.compare_stability_rows([plotted], [])
    assert validator.compare_stability_rows([plotted, plotted], [reference])
    predicted = {**plotted, "feature_mode": "design_plus_predicted_pce"}
    expected = {**reference, "mode": "design_plus_predicted_pce"}
    assert validator.compare_stability_rows([predicted], [expected])


def test_release_paths_remain_inside_the_repository(tmp_path: Path) -> None:
    assert validator.within_root(tmp_path, "manuscript/data.csv") == tmp_path / "manuscript/data.csv"
    with pytest.raises(ValueError):
        validator.within_root(tmp_path, "../outside.csv")
    with pytest.raises(ValueError):
        validator.within_root(tmp_path, "C:" + "\\outside.csv")


def test_hygiene_keeps_scientific_model_descriptions(tmp_path: Path) -> None:
    source = tmp_path / "methods.md"
    source.write_text("Qwen and ChatGPT benchmark results use language-model extraction.\n", encoding="utf-8")
    assert validator.hygiene_issues(source, tmp_path) == []


def test_relative_file_path_limit_protects_windows_checkouts(tmp_path: Path) -> None:
    long_path = tmp_path / ("a" * 90) / ("b" * 90) / "values.bin"
    length = len(long_path.relative_to(tmp_path).as_posix())
    assert validator.hygiene_issues(long_path, tmp_path) == [f"relative path length {length} exceeds limit 180"]
    assert validator.hygiene_issues(tmp_path / "values.bin", tmp_path) == []


def test_hygiene_reports_paths_without_exposing_matched_values(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("Co" + "dex instructions for the pro" + "fessor.\n", encoding="utf-8")
    assert validator.hygiene_issues(source, tmp_path) == ["authoring language in source text"]
    source.write_text("C:" + "\\Users\\" + "researcher\\workspace\n", encoding="utf-8")
    assert validator.hygiene_issues(source, tmp_path) == ["personal absolute path"]
    source.write_text("gh" + "p_" + "a" * 40, encoding="utf-8")
    assert validator.hygiene_issues(source, tmp_path) == ["credential pattern"]
    source.write_text("measurement" + chr(7), encoding="utf-8")
    assert validator.hygiene_issues(source, tmp_path) == ["unexpected control character"]


def test_generated_reports_and_transient_files_are_not_manifest_inputs(tmp_path: Path) -> None:
    source = tmp_path / "methods.md"
    source.write_text("Methods\n", encoding="utf-8")
    for relative in ("manuscript/validation/FILE_MANIFEST.csv", "manuscript/validation/VALIDATION.json", "build/cache.txt", ".git/config", "__pycache__/module.pyc"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated", encoding="utf-8")
    assert validator.release_files(tmp_path) == [source]


def test_manifest_verification_detects_changed_content_and_extra_files(tmp_path: Path) -> None:
    source = tmp_path / "methods.md"
    source.write_text("Methods\n", encoding="utf-8")
    manifest = tmp_path / "manuscript/validation/FILE_MANIFEST.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerow({"path": "methods.md", "size_bytes": source.stat().st_size, "sha256": validator.sha256(source)})
    assert validator.verify_file_manifest(tmp_path, manifest) == []
    source.write_text("Changed\n", encoding="utf-8")
    assert "manifest content differs: methods.md" in validator.verify_file_manifest(tmp_path, manifest)
    (tmp_path / "extra.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert "file missing from manifest: extra.csv" in validator.verify_file_manifest(tmp_path, manifest)


def test_manifest_verification_rejects_escaping_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "FILE_MANIFEST.csv"
    manifest.write_text("path,size_bytes,sha256\n../outside.txt,0,invalid\n", encoding="utf-8")
    errors = validator.verify_file_manifest(tmp_path, manifest)
    assert "manifest path leaves release directory" in errors


def test_benchmark_root_is_portable_and_preserves_existing_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = SCRIPT.parents[2] / "literature_agent_pipeline.py"
    tree = ast.parse(pipeline.read_text(encoding="utf-8"))
    embedded = next(
        node.value.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name) and target.id == "_LITERATURE_AGENT_SOURCE" for target in node.targets)
    )
    benchmark = next(node for node in ast.parse(embedded).body if isinstance(node, ast.ClassDef) and node.name == "BenchmarkingAgent")
    constructor = next(node for node in benchmark.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    root_setup = compile(ast.Module(body=constructor.body[:2], type_ignores=[]), "benchmark_root_setup", "exec")
    instance = SimpleNamespace()
    namespace = {"self": instance, "Path": Path, "os": os}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BENCHMARK_ROOT", raising=False)
    monkeypatch.delenv("LITERATURE_AGENT_BENCHMARK_ROOT", raising=False)
    exec(root_setup, namespace)
    assert instance.root == tmp_path / "data/benchmark"
    monkeypatch.setenv("LITERATURE_AGENT_BENCHMARK_ROOT", str(tmp_path / "selected"))
    exec(root_setup, namespace)
    assert instance.root == tmp_path / "selected"
    monkeypatch.setenv("BENCHMARK_ROOT", str(tmp_path / "existing"))
    exec(root_setup, namespace)
    assert instance.root == tmp_path / "existing"


def test_publication_normalization_preserves_opaque_identities_and_missing_policy() -> None:
    assert validator.canonical_publication("\ufeffDOI: 10.1039/C8SE00314A") == "10.1039/c8se00314a"
    assert validator.canonical_publication("https://doi.org/10.1039/C8SE00314A") == "10.1039/c8se00314a"
    assert validator.canonical_publication("RetainedSource:ABC-1") == "RetainedSource:ABC-1"
    with pytest.raises(ValueError, match="missing publication identity"):
        validator.canonical_publication("")
    assert validator.canonical_publication("", pooled_missing=True) == "NO_GROUP"


def test_canonical_sample_keys_preserve_numeric_looking_suffix_strings() -> None:
    actual = validator.canonical_sample_key("doi: 10.1126/science.adm9474|0.0|01|")
    assert actual == "10.1126/science.adm9474|0.0|01|"
    assert actual != validator.canonical_sample_key("10.1126/science.adm9474|0|01|")
    with pytest.raises(ValueError, match="separator"):
        validator.canonical_sample_key("10.1126/science.adm9474")


def sample_identity_fixture() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    predictions = [{"raw_frozen_row_key": "doi: 10.1126/science.adm9474|0.0|01|", "canonical_row_key": "10.1126/science.adm9474|0.0|01|"}]
    membership = [
        {"raw_row_key": "10.1126/science.adm9474|0.0|01|", "canonical_row_key": predictions[0]["canonical_row_key"], "in_training": "False", "is_frozen_test_sample": "True"},
        {"raw_row_key": "10.1126/science.adm9474|2|02|", "canonical_row_key": "10.1126/science.adm9474|2|02|", "in_training": "True", "is_frozen_test_sample": "False"},
    ]
    return predictions, membership


def test_same_sample_overlap_is_not_hidden_by_a_doi_prefix() -> None:
    predictions, membership = sample_identity_fixture()
    assert validator.pce_identity_errors(predictions, membership) == []
    membership[0]["in_training"] = "True"
    assert "the same canonical sample appears in training and test" in validator.pce_identity_errors(predictions, membership)


def test_pce_identity_gate_rejects_missing_test_matches_and_collisions() -> None:
    predictions, membership = sample_identity_fixture()
    membership[0]["is_frozen_test_sample"] = "False"
    assert "training audit does not identify every frozen test sample" in validator.pce_identity_errors(predictions, membership)
    assert "canonical test sample identities collide" in validator.pce_identity_errors(predictions * 2, membership)
    membership.append(membership[1].copy())
    assert "eligible canonical sample identities collide" in validator.pce_identity_errors(predictions, membership)


def test_canonical_publication_fold_check_rejects_case_alias_crossing() -> None:
    rows = [
        {"raw_publication_label": "10.1039/C8SE00314A", "canonical_publication_identity": "10.1039/c8se00314a", "final_test_fold": "2"},
        {"raw_publication_label": "10.1039/c8se00314a", "canonical_publication_identity": "10.1039/c8se00314a", "final_test_fold": "2"},
    ]
    assert validator.publication_fold_errors(rows) == []
    rows[1]["final_test_fold"] = "3"
    assert "a canonical publication crosses historical test folds" in validator.publication_fold_errors(rows)


def test_canonical_publication_fold_check_rejects_inconsistent_labels() -> None:
    rows = [{"raw_publication_label": "10.1039/C8SE00314A", "canonical_publication_identity": "10.1039/C8SE00314A", "final_test_fold": "2"}]
    assert "stability canonical publication does not match its retained source label" in validator.publication_fold_errors(rows)


def test_exported_regression_metrics_are_recomputed_without_estimators() -> None:
    values = validator.regression_metrics([2.0, 4.0], [2.0, 3.0])
    assert values == pytest.approx({"r2": 0.5, "rmse": 0.5 ** 0.5, "mae": 0.5})
    assert validator.regression_metrics([1.0, 1.0], [1.0, 1.0])["r2"] == 1.0
    assert validator.regression_metrics([1.0, 1.0], [1.0, 2.0])["r2"] == 0.0
    with pytest.raises(ValueError):
        validator.regression_metrics([], [])


def test_exported_classification_metrics_use_tie_aware_auc() -> None:
    rows = [
        {"actual": "0", "prediction": "0.1", "predicted_class": "0"},
        {"actual": "0", "prediction": "0.4", "predicted_class": "0"},
        {"actual": "1", "prediction": "0.4", "predicted_class": "0"},
        {"actual": "1", "prediction": "0.8", "predicted_class": "1"},
    ]
    assert validator.classification_metrics(rows) == pytest.approx({"roc_auc": 0.875, "accuracy": 0.75, "balanced_accuracy": 0.75, "f1": 2 / 3})
    with pytest.raises(ValueError, match="both binary"):
        validator.classification_metrics(rows[:2])


def fingerprint_fixture() -> list[dict[str, str]]:
    return [{"dataset": "baseline", "mode": "design_only", "target": "target_end_retention_pct", "metric": "r2",
             "value": "0.05", "ci_low": "0.02", "ci_high": "0.08", "n": "4986", "canonical_group_count": "1821", "bootstrap_seed": "42",
             "oof_source": "analysis/predictions.csv", "outcome_label": "Final retention"}]


def test_verified_stability_value_fingerprint_excludes_portable_paths_and_labels() -> None:
    rows = fingerprint_fixture()
    digest = validator.stability_values_fingerprint(rows)
    rows[0].update({"oof_source": "analysis/canonical/predictions.csv", "outcome_label": "Final retention (%)", "n": "4986.0"})
    assert validator.stability_values_fingerprint(rows) == digest
    rows[0]["ci_low"] = "0.021"
    assert validator.stability_values_fingerprint(rows) != digest


def test_first_seen_multinomial_bootstrap_matches_literal_row_resampling() -> None:
    import numpy as np

    actual = [1.0, 2.0, 4.0, 3.0]
    predictions = {"original": [1.3, 2.1, 3.5, 2.9]}
    groups = ["publication_b", "NO_GROUP", "publication_b", "NO_GROUP"]
    multiplicity = np.random.default_rng(42).multinomial(2, np.full(2, 0.5), size=2000)
    contract = {"groups": 2, "iterations": 2000, "seed": 42,
                "sampler": "np.random.default_rng(42).multinomial(G, np.full(G, 1/G), size=2000)",
                "group_order": "first appearance in frozen test CSV order",
                "multiplicity_sha256_int64_c": hashlib.sha256(multiplicity.astype(np.int64).tobytes(order="C")).hexdigest()}
    values, order = validator.bootstrap_regression(actual, predictions, groups, contract)
    assert order == ["publication_b", "NO_GROUP"]
    for draw_index in (0, 7, 42, 1999):
        drawn = np.repeat(np.arange(2), multiplicity[draw_index])
        rows = [index for group in drawn for index, name in enumerate(groups) if name == order[group]]
        literal = validator.regression_metrics([actual[index] for index in rows], [predictions["original"][index] for index in rows])
        assert {metric: bootstrap[draw_index] for metric, bootstrap in values["original"].items()} == pytest.approx(literal, abs=1e-12)
    with pytest.raises(ValueError, match="seed changed"):
        validator.bootstrap_regression(actual, predictions, groups, {**contract, "seed": 43})
    with pytest.raises(ValueError, match="multiplicities differ"):
        validator.bootstrap_regression(actual, predictions, groups, {**contract, "multiplicity_sha256_int64_c": "0" * 64})
