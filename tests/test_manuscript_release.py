"""Regression checks for the release validator's data and integrity gates."""
from __future__ import annotations

import csv
import ast
import importlib.util
import os
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
