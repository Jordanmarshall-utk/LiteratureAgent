#!/usr/bin/env python3
"""Reproduce and audit the production PCE encoding and grouped TreeSHAP.

This is a sidecar analysis: the production training script and its leakage,
population, grouping, and completeness rules are imported rather than copied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_SCRIPT = ROOT / "models" / "pce_then_stability_same_approach.py"
PROJECT_FINAL_MODEL_SCRIPT = (
    ROOT
    / "artifacts/literature_agent_pce_stability/source_data/model_results/code_snapshots"
    / "pce_then_stability_same_approach_sha256_19f58acd.py"
)
BUNDLED_FINAL_MODEL_SCRIPT = (
    ROOT / "models/frozen/pce_then_stability_same_approach_sha256_19f58acd.py"
)
FINAL_MODEL_SCRIPT = (
    PROJECT_FINAL_MODEL_SCRIPT
    if PROJECT_FINAL_MODEL_SCRIPT.is_file()
    else BUNDLED_FINAL_MODEL_SCRIPT
)
FINAL_HARMONIZATION_SCRIPT = ROOT / "scripts/evaluate_pce_harmonized_expansion.py"
PROJECT_FINAL_TYPING = (
    ROOT
    / "artifacts/literature_agent_pce_stability/source_data/model_results"
    / "frozen_holdout_harmonized_v3_final_typing/final_feature_typing_inventory.csv"
)
BUNDLED_FINAL_TYPING = ROOT / "data/figure5/final_feature_typing_inventory.csv"
DEFAULT_FINAL_TYPING = (
    PROJECT_FINAL_TYPING if PROJECT_FINAL_TYPING.is_file() else BUNDLED_FINAL_TYPING
)
DEFAULT_DATA = ROOT / "artifacts/literature_agent_pce_stability/source_data/data_provenance/datasets/final_integrated_database.csv"
DEFAULT_MANIFEST = ROOT / "artifacts/literature_agent_pce_stability/source_data/submission_audit_v1/pce_feature_columns_278.csv"
DEFAULT_OUT = ROOT / "artifacts/literature_agent_pce_stability/source_data/pce_feature_interpretability_audit"
BOOLEAN_VALUES = frozenset({"true", "false", "yes", "no", "0", "1"})
EXACT_STRICT_ABSOLUTE_THRESHOLD = 1e-5
EXACT_COHORT_MAE_LIMIT = 1e-6
EXACT_MINIMUM_STRICT_PASS_FRACTION = 0.9999
EXACT_MAXIMUM_ABSOLUTE_ERROR = 0.005
EXACT_MAXIMUM_RELATIVE_ERROR = 2e-4
FINAL_EXPECTED_TRAINING_ROWS = 21434
FINAL_EXPECTED_PUBLICATION_GROUPS = 4085
FINAL_EXPECTED_ENCODED_FEATURES = 6263
FINAL_EXPECTED_SCIENTIFIC_GROUPS = 218
FINAL_EXPECTED_POPULATION_FINGERPRINT = (
    "a932e882ecb4f37c015e4e06de66856ebeda18eaee4915d3fcd14eed30538f9e"
)
FINAL_EXPECTED_SOURCE_SHA256 = {
    "model_script": "19f58acdfb6143a75df5d034be08f1de070d43713b716cba6c77d4b422898115",
    "feature_typing_inventory": "f2a193a2818279bb889145d91ebe438cf7ae173e39e0dba71c6d28145219ebdc",
    "external_harmonization_script": "1c7373ed8a85448afc9ba7edb30c1f314347bc1168121dc9d539adb33386df60",
    "data": "88d26d9c46c5142f546dd640db6e091713a7dc38b19399bb7887cdd751b599db",
    "predictor_manifest": "56a7c34a370df4aee5cd83d2bea2e56dc3d3d5b4c1d46e7518dacd108afc9524",
}
STALE_PLOT_DERIVED_NAMES = (
    "pce_grouped_shap_top15.csv",
    "pce_grouped_shap_top20.csv",
    "pce_choice_stratified_shap_summary.csv",
    "pce_solvent_formulation_shap_values.csv",
    "pce_solvent_formulation_shap_summary.csv",
    "pce_grouped_shap_local_random_selection.json",
    "pce_feature_prevalence_audit.csv",
)
SHAP_CHECKPOINT_SCHEMA_VERSION = 1
SHAP_CHECKPOINT_MANIFEST_NAME = "manifest.json"
SHAP_CHECKPOINT_PROGRESS_NAME = "progress.json"

CHOICE_BUNDLE_SPECS = (
    {
        "feature_family": "Precursor solvent formulation",
        "context_column": "context_solvent_formulation",
        "raw_source_fields": (
            "Perovskite_deposition_solvents",
            "Perovskite_deposition_solvents_mixing_ratios",
        ),
    },
    {
        "feature_family": "ETL stack and deposition",
        "context_column": "context_etl",
        "raw_source_fields": ("ETL_stack_sequence", "ETL_deposition_procedure"),
    },
    {
        "feature_family": "HTL stack and deposition",
        "context_column": "context_htl",
        "raw_source_fields": ("HTL_stack_sequence", "HTL_deposition_procedure"),
    },
    {
        "feature_family": "Absorber composition identity",
        "context_column": "context_absorber_composition",
        "raw_source_fields": (
            "Perovskite_composition_a_ions",
            "Perovskite_composition_a_ions_coefficients",
            "Perovskite_composition_b_ions",
            "Perovskite_composition_b_ions_coefficients",
            "Perovskite_composition_c_ions",
            "Perovskite_composition_c_ions_coefficients",
            "Perovskite_composition_short_form",
            "Perovskite_composition_long_form",
            "Perovskite_composition_inorganic",
            "Perovskite_composition_leadfree",
        ),
    },
    {
        "feature_family": "Quenching-induced crystallization",
        "context_column": "context_quenching",
        "raw_source_fields": ("Perovskite_deposition_quenching_induced_crystallisation",),
    },
    {
        "feature_family": "Back-contact material",
        "context_column": "context_back_contact",
        "raw_source_fields": ("Backcontact_stack_sequence",),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_final_contract_source_hashes(
    *,
    model_script: Path,
    feature_typing_inventory: Path,
    data: Path,
    predictor_manifest: Path,
    _expected_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Reject a byte-different source while using the frozen Figure 4 label.

    ``_expected_hashes`` is dependency injection for focused unit tests only.
    Production callers omit it and therefore use the immutable canonical
    source hashes above.  A byte-identical copy at another path is allowed.
    """

    expected = FINAL_EXPECTED_SOURCE_SHA256 if _expected_hashes is None else _expected_hashes
    paths = {
        "model_script": model_script,
        "feature_typing_inventory": feature_typing_inventory,
        "external_harmonization_script": FINAL_HARMONIZATION_SCRIPT,
        "data": data,
        "predictor_manifest": predictor_manifest,
    }
    missing_expected = sorted(set(paths).difference(expected))
    if missing_expected:
        raise RuntimeError(
            f"Frozen final-contract source hash inventory is incomplete: {missing_expected}"
        )
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": str(expected[name]), "observed": digest, "path": str(paths[name])}
        for name, digest in observed.items()
        if digest != str(expected[name]).strip().casefold()
    }
    if mismatches:
        raise RuntimeError(
            "Refusing to label byte-different sources as the frozen final Figure 4 contract: "
            f"{mismatches}"
        )
    return observed


def load_production_module(model_script: Path = MODEL_SCRIPT):
    spec = importlib.util.spec_from_file_location("production_pce", model_script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_harmonization_module():
    spec = importlib.util.spec_from_file_location(
        "pce_final_contract_harmonization", FINAL_HARMONIZATION_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean_token(x: object) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().strip("[]'\"")
    return None if not s or s.casefold() in {"nan", "none", "unknown", "null"} else s


def is_boolean_predictor(series: pd.Series) -> bool:
    """Return whether every observed source value has Boolean semantics."""
    observed = set(series.dropna().astype(str).str.strip().str.casefold().unique())
    return bool(observed) and observed <= BOOLEAN_VALUES


def simple_aligned_list(x: object) -> list[str] | None:
    """Parse one schema list only; reject layer/step delimiters as ambiguous."""
    s = clean_token(x)
    if s is None or "|" in s or ">>" in s:
        return None
    vals = [v.strip().strip("'\"") for v in s.split(";")]
    return vals if vals and all(vals) else None


def aligned_fractions(ions: object, coefficients: object) -> dict[str, float] | None:
    names, raw = simple_aligned_list(ions), simple_aligned_list(coefficients)
    if names is None or raw is None or len(names) != len(raw):
        return None
    try:
        nums = np.asarray([float(v.replace(",", ".")) for v in raw], dtype=float)
    except ValueError:
        return None
    if not np.all(np.isfinite(nums)) or np.any(nums < 0) or nums.sum() <= 0:
        return None
    return {re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_"): float(v / nums.sum()) for n, v in zip(names, nums)}


def solvent_descriptor(solvents: object, ratios: object) -> tuple[str | None, dict[str, float] | None]:
    names = simple_aligned_list(solvents)
    if names is None:
        return None, None
    raw = simple_aligned_list(ratios)
    if len(names) == 1:
        # Database convention: ratio=1 for one solvent means pure solvent.
        return f"{names[0]} only", {names[0]: 1.0}
    if raw is None or len(raw) != len(names):
        return None, None
    try:
        vals = np.asarray([float(v.replace(",", ".")) for v in raw], dtype=float)
    except ValueError:
        return None, None
    if np.any(vals < 0) or not np.all(np.isfinite(vals)) or vals.sum() <= 0:
        return None, None
    label = f"{':'.join(names)} = {':'.join(raw)}"
    return label, {n: float(v / vals.sum()) for n, v in zip(names, vals)}


def family(name: str) -> str:
    n = name.casefold()
    rules = [("composition", "Perovskite composition"), ("solvent", "Solvent/process chemistry"),
             ("quenching", "Crystallization/quenching"), ("backcontact", "Back contact"),
             ("htl", "Hole-transport layer"), ("etl", "Electron-transport layer"),
             ("cell_", "Device architecture"), ("substrate", "Substrate")]
    return next((label for key, label in rules if key in n), "Other design/process")


def scientific_group(raw: str) -> str:
    n = raw.casefold()
    if "perovskite_composition_a_ions" in n: return "A-site composition"
    if "perovskite_composition_b_ions" in n: return "B-site composition"
    if "perovskite_composition_c_ions" in n: return "X-site/halide composition"
    if n in {"perovskite_deposition_solvents", "perovskite_deposition_solvents_mixing_ratios"}: return "Perovskite precursor solvent formulation"
    if n == "perovskite_deposition_solvents_supplier": return "Perovskite solvent supplier"
    if n == "perovskite_deposition_solvents_purity": return "Perovskite solvent purity"
    if "quenching_induced_crystallisation" in n: return "Quenching-induced crystallization"
    if "composition_inorganic" in n: return "Fully inorganic absorber"
    if "backcontact_deposition_procedure" in n: return "Back-contact deposition method"
    if "backcontact_stack" in n or "backcontact_compounds" in n: return "Back-contact material"
    if n.startswith("htl_"): return "HTL material/stack and processing"
    if n.startswith("etl_"): return "ETL material/stack and processing"
    return raw.replace("_", " ")


def raw_from_encoded(name: str, num_cols: list[str], cat_cols: list[str]) -> tuple[str, str, str]:
    s = str(name)
    if s.startswith("num__"):
        return s[5:], "numeric: standard scaling after median imputation", "scaled numeric"
    tail = s[5:] if s.startswith("cat__") else s
    # Longest prefix is necessary when category strings contain underscores.
    matches = [c for c in cat_cols if tail == c or tail.startswith(c + "_")]
    raw = max(matches, key=len) if matches else tail
    category = tail[len(raw) + 1:] if tail.startswith(raw + "_") else ""
    return raw, "categorical: harmonize, most-frequent impute, one-hot (min_frequency=5)", category


def prepare_population(
    mod,
    data: Path,
    manifest: Path,
    out: Path,
    *,
    harmonize_categorical_schema: bool = False,
):
    cfg = mod.Config(INPUT_CSV=str(data), OUTPUT_DIR=str(out), N_ESTIMATORS=300,
                     ENABLE_SHAP=False, SHAP_MAX_ROWS=None, CV_STRATEGY="grouped",
                     PCE_ONLY=True, MODEL_BACKEND="extra_trees", RANDOM_STATE=42,
                     HARMONIZE_CATEGORICAL_SCHEMA=harmonize_categorical_schema)
    df = mod.clean_df(mod.robust_read_csv(data))
    df = mod.apply_publication_year_filter(df, cfg, out)
    df = mod.add_additive_descriptor_features(df, out)
    df, target = mod.prepare_pce_target(df, cfg)
    features = pd.read_csv(manifest)["feature_column"].tolist()
    missing = [c for c in features if c not in df]
    if missing: raise RuntimeError(f"Manifest columns absent from fitted data: {missing}")
    (out / "population").mkdir(parents=True, exist_ok=True)
    df = mod.apply_row_completeness(df, features, cfg, out / "population")
    return cfg, df, target, features, mod.find_group_col(df)


def freeze_predictor_typing(mod, inventory_path: Path, features: list[str]):
    inventory = pd.read_csv(inventory_path)
    required = {"predictor_name", "final_frozen_model_block"}
    missing = required.difference(inventory.columns)
    if missing:
        raise RuntimeError(f"Frozen typing inventory is missing columns: {sorted(missing)}")
    if inventory["predictor_name"].duplicated().any():
        raise RuntimeError("Frozen typing inventory contains duplicate predictors")
    block_by_feature = inventory.set_index("predictor_name")["final_frozen_model_block"]
    missing_features = [feature for feature in features if feature not in block_by_feature]
    extra_features = [feature for feature in block_by_feature.index if feature not in set(features)]
    if missing_features or extra_features:
        raise RuntimeError(
            "Frozen typing inventory and the 278-predictor manifest differ: "
            f"missing={missing_features}, extra={extra_features}"
        )
    ordered_blocks = block_by_feature.reindex(features).astype(str).str.strip().str.casefold()
    unexpected = sorted(set(ordered_blocks).difference({"numeric", "categorical"}))
    if unexpected:
        raise RuntimeError(f"Frozen typing inventory contains unsupported blocks: {unexpected}")
    numeric_columns = [
        feature for feature, block in zip(features, ordered_blocks) if block == "numeric"
    ]
    categorical_columns = [
        feature for feature, block in zip(features, ordered_blocks) if block == "categorical"
    ]
    if (len(numeric_columns), len(categorical_columns)) != (113, 165):
        raise RuntimeError(
            "Final Figure 4 typing contract must contain 113 numeric and 165 categorical predictors"
        )

    def frozen_inventory_split(_frame, _cfg):
        return numeric_columns.copy(), categorical_columns.copy()

    mod.split_num_cat = frozen_inventory_split
    return numeric_columns, categorical_columns


def fit_frozen_pipeline(mod, df, target, features, cfg):
    """Fit the frozen production pipeline once on the complete eligible cohort."""
    X = df[features].copy()
    pre, num, cat = mod.make_preprocessor(X, cfg)
    pipe = Pipeline([("pre", pre), ("model", mod.make_model("regression", cfg))])
    pipe.fit(X, df[target])
    names = mod.get_feature_names(pipe.named_steps["pre"])
    return pipe, names, num, cat


def _normalize_tree_shap_values(values, expected_rows: int, expected_features: int) -> np.ndarray:
    """Normalize SHAP's version-dependent single-output return conventions."""
    if isinstance(values, list):
        if len(values) != 1:
            raise RuntimeError(f"Expected one regression SHAP output, received {len(values)}")
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != (expected_rows, expected_features):
        raise RuntimeError(
            f"Unexpected SHAP shape {values.shape}; expected {(expected_rows, expected_features)}"
        )
    return values


def _aggregation_matrix(mapping: pd.DataFrame, group_names: list[str], encoded_count: int):
    group_lookup = {name: position for position, name in enumerate(group_names)}
    rows = mapping["scientific_grouping"].map(group_lookup).to_numpy(dtype=int)
    cols = mapping["encoded_feature_index"].to_numpy(dtype=int)
    matrix = sparse.csr_matrix(
        (np.ones(len(mapping), dtype=float), (rows, cols)),
        shape=(len(group_names), encoded_count),
    )
    coverage = np.asarray(matrix.sum(axis=0)).ravel()
    if len(mapping) != encoded_count or not np.array_equal(coverage, np.ones(encoded_count)):
        raise RuntimeError("Every encoded model input must map to exactly one scientific group")
    return matrix


def _choice_aggregation_matrix(mapping: pd.DataFrame, encoded_count: int):
    mapped_raw_fields = set(mapping["raw_source_predictor"].astype(str))
    claimed_raw_fields: set[str] = set()
    definitions: list[dict[str, object]] = []
    rows: list[int] = []
    cols: list[int] = []
    for bundle_position, spec in enumerate(CHOICE_BUNDLE_SPECS):
        raw_source_fields = tuple(str(field) for field in spec["raw_source_fields"])
        overlap = claimed_raw_fields.intersection(raw_source_fields)
        if overlap:
            raise RuntimeError(f"Choice bundles overlap on raw fields: {sorted(overlap)}")
        claimed_raw_fields.update(raw_source_fields)
        missing_fields = set(raw_source_fields).difference(mapped_raw_fields)
        if missing_fields:
            raise RuntimeError(f"Choice bundle fields are absent from encoded mapping: {sorted(missing_fields)}")
        part = mapping.loc[mapping["raw_source_predictor"].isin(raw_source_fields)]
        rows.extend([bundle_position] * len(part))
        cols.extend(part["encoded_feature_index"].astype(int).tolist())
        definitions.append(
            {
                "feature_family": str(spec["feature_family"]),
                "context_column": str(spec["context_column"]),
                "raw_source_fields": list(raw_source_fields),
                "raw_source_field_count": len(raw_source_fields),
                "encoded_input_count": len(part),
            }
        )
    if len(cols) != len(set(cols)):
        raise RuntimeError("An encoded model input is claimed by more than one choice bundle")
    matrix = sparse.csr_matrix(
        (np.ones(len(cols), dtype=float), (np.asarray(rows), np.asarray(cols))),
        shape=(len(CHOICE_BUNDLE_SPECS), encoded_count),
    )
    return matrix, definitions


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    """Durably write JSON and expose it only after the file is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    """Durably write one compressed batch shard without exposing a partial ZIP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = _canonical_json_bytes({"dtype": value.dtype.str, "shape": list(value.shape)})
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def sparse_matrix_sha256(matrix) -> str:
    """Hash a sparse aggregation matrix in a platform-stable CSR representation."""
    value = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    value.sort_indices()
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({"shape": list(value.shape)}))
    for array in (
        np.asarray(value.indptr, dtype="<i8"),
        np.asarray(value.indices, dtype="<i8"),
        np.asarray(value.data, dtype="<f8"),
    ):
        digest.update(_canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)}))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fitted_tree_model_sha256(model) -> str:
    """Fingerprint fitted tree structure and leaf values, independent of pickle metadata."""
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        raise RuntimeError("Cannot checkpoint SHAP without fitted tree estimators")
    digest = hashlib.sha256()
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    digest.update(
        _canonical_json_bytes(
            {
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "parameters": params,
                "n_features_in": int(getattr(model, "n_features_in_", -1)),
                "estimator_count": len(estimators),
            }
        )
    )
    for position, estimator in enumerate(estimators):
        tree = estimator.tree_
        digest.update(_canonical_json_bytes({"estimator_position": position, "node_count": tree.node_count}))
        for name, dtype in (
            ("children_left", "<i8"),
            ("children_right", "<i8"),
            ("feature", "<i8"),
            ("threshold", "<f8"),
            ("value", "<f8"),
            ("impurity", "<f8"),
            ("n_node_samples", "<i8"),
            ("weighted_n_node_samples", "<f8"),
        ):
            array = np.ascontiguousarray(getattr(tree, name), dtype=dtype)
            digest.update(_canonical_json_bytes({"name": name, "shape": list(array.shape)}))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_shard_path(checkpoint_dir: Path, batch_number: int) -> Path:
    return checkpoint_dir / f"batch_{batch_number:05d}.npz"


def _load_checkpoint_shard(
    path: Path,
    *,
    manifest_sha256: str,
    batch_number: int,
    start: int,
    stop: int,
    encoded_count: int,
    group_count: int,
    choice_count: int,
) -> dict[str, np.ndarray]:
    expected_names = {
        "manifest_sha256",
        "batch_number",
        "start",
        "stop",
        "grouped_values",
        "choice_values",
        "predictions",
        "encoded_nonzero_count",
        "encoded_abs_shap_sum",
        "encoded_residuals",
        "grouped_residuals",
    }
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != expected_names:
                raise RuntimeError(
                    f"checkpoint shard fields differ: expected={sorted(expected_names)}, "
                    f"observed={sorted(loaded.files)}"
                )
            scalar_manifest = str(np.asarray(loaded["manifest_sha256"]).item())
            scalar_batch = int(np.asarray(loaded["batch_number"]).item())
            scalar_start = int(np.asarray(loaded["start"]).item())
            scalar_stop = int(np.asarray(loaded["stop"]).item())
            arrays = {
                name: np.asarray(loaded[name]).copy()
                for name in expected_names.difference(
                    {"manifest_sha256", "batch_number", "start", "stop"}
                )
            }
    except Exception as exc:
        raise RuntimeError(f"Cannot safely load SHAP checkpoint shard {path}: {exc}") from exc
    if (
        scalar_manifest != manifest_sha256
        or scalar_batch != batch_number
        or scalar_start != start
        or scalar_stop != stop
    ):
        raise RuntimeError(f"SHAP checkpoint shard identity mismatch: {path}")
    row_count = stop - start
    expected_shapes = {
        "grouped_values": (row_count, group_count),
        "choice_values": (row_count, choice_count),
        "predictions": (row_count,),
        "encoded_nonzero_count": (encoded_count,),
        "encoded_abs_shap_sum": (encoded_count,),
        "encoded_residuals": (row_count,),
        "grouped_residuals": (row_count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(
                f"SHAP checkpoint shard {path} has {name} shape {arrays[name].shape}; expected {shape}"
            )
    if arrays["encoded_nonzero_count"].dtype != np.dtype("int64"):
        raise RuntimeError(f"SHAP checkpoint shard {path} has a non-int64 prevalence count")
    for name, value in arrays.items():
        if name == "encoded_nonzero_count":
            if np.any(value < 0) or np.any(value > row_count):
                raise RuntimeError(f"SHAP checkpoint shard {path} has invalid prevalence counts")
        else:
            if value.dtype != np.dtype("float64") or not np.isfinite(value).all():
                raise RuntimeError(f"SHAP checkpoint shard {path} has invalid {name} values")
    return arrays


def _checkpoint_progress(
    checkpoint_dir: Path,
    *,
    manifest_sha256: str,
    batch_count: int,
    shard_sha256: dict[int, str],
) -> None:
    completed = sorted(shard_sha256)
    _atomic_write_json(
        checkpoint_dir / SHAP_CHECKPOINT_PROGRESS_NAME,
        {
            "schema_version": SHAP_CHECKPOINT_SCHEMA_VERSION,
            "manifest_sha256": manifest_sha256,
            "batch_count": batch_count,
            "completed_batches": completed,
            "completed_batch_count": len(completed),
            "complete": len(completed) == batch_count,
            "shard_sha256": {
                str(batch): shard_sha256[batch] for batch in completed
            },
        },
    )


def compute_batched_grouped_shap(
    pipe,
    X: pd.DataFrame,
    group_matrix,
    choice_matrix,
    *,
    batch_size: int,
    shap_method: str,
    background_positions: np.ndarray | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_identity: dict[str, object] | None = None,
    _after_checkpoint: Callable[[int, Path], None] | None = None,
):
    """Explain every fitted row while retaining only governed aggregates.

    When ``checkpoint_dir`` is supplied, each completed batch is written as an
    atomic compressed shard.  A resumed run accepts a shard only after the
    frozen run manifest, shard identity, shapes, dtypes, and recorded checksum
    all validate.  Final arrays are assembled in batch order, so resuming does
    not change floating-point accumulation order or downstream table schemas.
    ``_after_checkpoint`` is an internal test hook called only after a shard and
    its progress record are durable.
    """
    if batch_size < 1:
        raise ValueError("SHAP batch size must be positive")
    if not X.index.is_unique:
        raise RuntimeError("Production row indices must be unique for deterministic SHAP provenance")
    if resume and checkpoint_dir is None:
        raise ValueError("SHAP resume requires a checkpoint directory")
    import shap

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    encoded_count = group_matrix.shape[1]
    row_count = len(X)
    grouped_values = np.empty((row_count, group_matrix.shape[0]), dtype=float)
    choice_values = np.empty((row_count, choice_matrix.shape[0]), dtype=float)
    predictions = np.empty(row_count, dtype=float)
    encoded_nonzero_count = np.zeros(encoded_count, dtype=np.int64)
    encoded_abs_shap_sum = np.zeros(encoded_count, dtype=float)
    encoded_residuals = np.empty(row_count, dtype=float)
    grouped_residuals = np.empty(row_count, dtype=float)

    approximate = shap_method == "approximate-path-dependent"
    background_details: dict[str, object] | None = None
    background_sha256: str | None = None
    if shap_method == "exact-interventional":
        if background_positions is None or len(background_positions) == 0:
            raise ValueError("Exact interventional TreeSHAP requires a non-empty background cohort")
        background = pre.transform(X.iloc[background_positions])
        background = background.toarray() if sparse.issparse(background) else np.asarray(background)
        background = np.asarray(background, dtype=float)
        background_sha256 = _array_sha256(background)
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",
        )
        background_details = {
            "background_rows": int(len(background_positions)),
            "background_positions": background_positions.astype(int).tolist(),
            "feature_perturbation": "interventional",
        }
        del background
    elif shap_method in {"approximate-path-dependent", "exact-path-dependent"}:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
    else:
        raise ValueError(f"Unsupported SHAP method: {shap_method}")
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    batch_count = math.ceil(row_count / batch_size)

    checkpoint_manifest_sha256: str | None = None
    shard_sha256: dict[int, str] = {}
    existing_batches: set[int] = set()
    resumed_batch_count = 0
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_identity is None:
            raise ValueError("Checkpointing requires a complete checkpoint_identity")
        required_identity = {
            "population_fingerprint_sha256",
            "fitted_model_sha256",
            "source_sha256",
            "encoded_feature_names_sha256",
            "encoded_mapping_sha256",
            "scientific_group_names_sha256",
            "group_aggregation_matrix_sha256",
            "choice_aggregation_matrix_sha256",
        }
        missing_identity = sorted(required_identity.difference(checkpoint_identity))
        if missing_identity:
            raise ValueError(
                f"Checkpoint identity is missing governed fields: {missing_identity}"
            )
        index_hash_values = pd.util.hash_pandas_object(
            X.index.to_series(index=X.index), index=False
        ).to_numpy(dtype="<u8", copy=False)
        run_manifest = {
            "schema_version": SHAP_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_identity": checkpoint_identity,
            "row_count": row_count,
            "row_index_dtype": str(X.index.dtype),
            "row_index_sha256": hashlib.sha256(index_hash_values.tobytes()).hexdigest(),
            "raw_input_column_names_sha256": _json_sha256(list(map(str, X.columns))),
            "encoded_feature_count": encoded_count,
            "scientific_group_count": int(group_matrix.shape[0]),
            "choice_bundle_count": int(choice_matrix.shape[0]),
            "shap_method": shap_method,
            "approximate": bool(approximate),
            "batch_size": batch_size,
            "batch_count": batch_count,
            "background_positions": (
                None
                if background_positions is None
                else np.asarray(background_positions, dtype=int).tolist()
            ),
            "background_encoded_sha256": background_sha256,
            "expected_value_float_hex": expected.hex(),
            "shap_version": str(getattr(shap, "__version__", "unknown")),
        }
        checkpoint_manifest_sha256 = _json_sha256(run_manifest)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_dir / SHAP_CHECKPOINT_MANIFEST_NAME
        progress_path = checkpoint_dir / SHAP_CHECKPOINT_PROGRESS_NAME
        discovered_paths: dict[int, Path] = {}
        for path in checkpoint_dir.glob("batch_*.npz"):
            match = re.fullmatch(r"batch_(\d{5})\.npz", path.name)
            if match is None:
                raise RuntimeError(f"Unrecognized SHAP checkpoint shard name: {path}")
            batch_number = int(match.group(1))
            if not 1 <= batch_number <= batch_count:
                raise RuntimeError(f"Out-of-range SHAP checkpoint shard: {path}")
            discovered_paths[batch_number] = path
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"SHAP checkpoint manifest already exists at {manifest_path}; "
                    "use resume=True or a new checkpoint directory"
                )
            try:
                observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Cannot read SHAP checkpoint manifest {manifest_path}: {exc}") from exc
            if observed_manifest != run_manifest:
                raise RuntimeError(
                    "SHAP checkpoint manifest mismatch; refusing stale or mixed shards "
                    f"(expected {_json_sha256(run_manifest)}, observed {_json_sha256(observed_manifest)}). "
                    "Use a new checkpoint directory for the changed run contract."
                )
        else:
            if discovered_paths or progress_path.exists():
                raise RuntimeError(
                    "SHAP checkpoint shards/progress exist without their governing manifest; "
                    "refusing unverifiable resume"
                )
            _atomic_write_json(manifest_path, run_manifest)
        if sha256_file(manifest_path) != hashlib.sha256(
            (json.dumps(run_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest():
            raise RuntimeError("SHAP checkpoint manifest bytes changed after atomic write")
        existing_batches = set(discovered_paths)
        shard_sha256 = {
            batch_number: sha256_file(path)
            for batch_number, path in discovered_paths.items()
        }
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Cannot read SHAP checkpoint progress {progress_path}: {exc}") from exc
            if (
                progress.get("schema_version") != SHAP_CHECKPOINT_SCHEMA_VERSION
                or progress.get("manifest_sha256") != checkpoint_manifest_sha256
                or progress.get("batch_count") != batch_count
            ):
                raise RuntimeError("SHAP checkpoint progress does not match the governed manifest")
            recorded = {int(value) for value in progress.get("completed_batches", [])}
            if not recorded.issubset(existing_batches):
                raise RuntimeError("SHAP checkpoint progress names a missing batch shard")
            recorded_hashes = progress.get("shard_sha256", {})
            for batch_number in recorded:
                if recorded_hashes.get(str(batch_number)) != shard_sha256[batch_number]:
                    raise RuntimeError(
                        f"SHAP checkpoint checksum mismatch for batch {batch_number}"
                    )
        if existing_batches:
            print(
                f"SHAP checkpoint resume: found {len(existing_batches)}/{batch_count} batch shards; "
                "validating each shard before reuse",
                flush=True,
            )

    for batch_number, start in enumerate(range(0, row_count, batch_size), start=1):
        stop = min(start + batch_size, row_count)
        shard_path = (
            _checkpoint_shard_path(checkpoint_dir, batch_number)
            if checkpoint_dir is not None
            else None
        )
        if shard_path is not None and batch_number in existing_batches:
            assert checkpoint_manifest_sha256 is not None
            batch_arrays = _load_checkpoint_shard(
                shard_path,
                manifest_sha256=checkpoint_manifest_sha256,
                batch_number=batch_number,
                start=start,
                stop=stop,
                encoded_count=encoded_count,
                group_count=group_matrix.shape[0],
                choice_count=choice_matrix.shape[0],
            )
            resumed_batch_count += 1
        else:
            encoded = pre.transform(X.iloc[start:stop])
            if sparse.issparse(encoded):
                batch_nonzero_count = np.asarray(encoded.getnnz(axis=0), dtype=np.int64)
                encoded = encoded.toarray()
            else:
                encoded = np.asarray(encoded)
                batch_nonzero_count = np.asarray(
                    np.count_nonzero(encoded, axis=0), dtype=np.int64
                )
            encoded = np.asarray(encoded, dtype=np.float64)
            values = explainer.shap_values(
                encoded,
                check_additivity=False,
                approximate=approximate,
            )
            values = _normalize_tree_shap_values(values, stop - start, encoded_count)
            prediction = np.asarray(model.predict(encoded), dtype=np.float64)
            group_batch = np.asarray((group_matrix @ values.T).T, dtype=np.float64)
            choice_batch = np.asarray((choice_matrix @ values.T).T, dtype=np.float64)
            if not (
                np.isfinite(values).all()
                and np.isfinite(prediction).all()
                and np.isfinite(group_batch).all()
                and np.isfinite(choice_batch).all()
            ):
                raise RuntimeError(f"Non-finite SHAP result in batch {batch_number}")
            batch_arrays = {
                "grouped_values": group_batch,
                "choice_values": choice_batch,
                "predictions": prediction,
                "encoded_nonzero_count": batch_nonzero_count,
                "encoded_abs_shap_sum": np.asarray(
                    np.abs(values).sum(axis=0), dtype=np.float64
                ),
                "encoded_residuals": np.asarray(
                    expected + values.sum(axis=1) - prediction, dtype=np.float64
                ),
                "grouped_residuals": np.asarray(
                    expected + group_batch.sum(axis=1) - prediction, dtype=np.float64
                ),
            }
            if shard_path is not None:
                assert checkpoint_manifest_sha256 is not None
                _atomic_savez_compressed(
                    shard_path,
                    manifest_sha256=np.asarray(checkpoint_manifest_sha256),
                    batch_number=np.asarray(batch_number, dtype=np.int64),
                    start=np.asarray(start, dtype=np.int64),
                    stop=np.asarray(stop, dtype=np.int64),
                    **batch_arrays,
                )
                shard_sha256[batch_number] = sha256_file(shard_path)
                _checkpoint_progress(
                    checkpoint_dir,
                    manifest_sha256=checkpoint_manifest_sha256,
                    batch_count=batch_count,
                    shard_sha256=shard_sha256,
                )
                if _after_checkpoint is not None:
                    _after_checkpoint(batch_number, shard_path)
            del encoded, values, prediction, group_batch, choice_batch

        grouped_values[start:stop] = batch_arrays["grouped_values"]
        choice_values[start:stop] = batch_arrays["choice_values"]
        predictions[start:stop] = batch_arrays["predictions"]
        encoded_nonzero_count += batch_arrays["encoded_nonzero_count"]
        encoded_abs_shap_sum += batch_arrays["encoded_abs_shap_sum"]
        encoded_residuals[start:stop] = batch_arrays["encoded_residuals"]
        grouped_residuals[start:stop] = batch_arrays["grouped_residuals"]
        if (
            shard_path is None
            and (batch_number == 1 or batch_number % 10 == 0 or batch_number == batch_count)
        ):
            print(
                f"SHAP batch {batch_number}/{batch_count}: explained {stop:,}/{row_count:,} rows",
                flush=True,
            )
        elif shard_path is not None and batch_number not in existing_batches:
            print(
                f"SHAP batch {batch_number}/{batch_count}: checkpoint saved after "
                f"{stop:,}/{row_count:,} rows",
                flush=True,
            )
        del batch_arrays

    checkpoint_details = None
    if checkpoint_dir is not None:
        assert checkpoint_manifest_sha256 is not None
        _checkpoint_progress(
            checkpoint_dir,
            manifest_sha256=checkpoint_manifest_sha256,
            batch_count=batch_count,
            shard_sha256=shard_sha256,
        )
        checkpoint_details = {
            "enabled": True,
            "directory": str(checkpoint_dir.resolve()),
            "schema_version": SHAP_CHECKPOINT_SCHEMA_VERSION,
            "manifest_sha256": checkpoint_manifest_sha256,
            "resumed_batch_count": resumed_batch_count,
            "completed_batch_count": batch_count,
            "all_shards_validated": True,
        }
        if resumed_batch_count:
            print(
                f"SHAP checkpoint resume complete: reused {resumed_batch_count}/{batch_count} batches",
                flush=True,
            )

    return {
        "grouped_values": grouped_values,
        "choice_values": choice_values,
        "predictions": predictions,
        "expected_value": expected,
        "encoded_prevalence": encoded_nonzero_count.astype(float) / row_count,
        "encoded_mean_absolute_shap": encoded_abs_shap_sum / row_count,
        "encoded_reconstruction_mae": float(np.mean(np.abs(encoded_residuals))),
        "encoded_reconstruction_max_abs_error": float(np.max(np.abs(encoded_residuals))),
        "grouped_reconstruction_mae": float(np.mean(np.abs(grouped_residuals))),
        "grouped_reconstruction_max_abs_error": float(np.max(np.abs(grouped_residuals))),
        "batch_size": batch_size,
        "batch_count": batch_count,
        "approximate": bool(approximate),
        "exact_tree_shap": not approximate,
        "shap_method_key": shap_method,
        "background": background_details,
        "shap_version": str(getattr(shap, "__version__", "unknown")),
        "checkpoint": checkpoint_details,
    }


def _json_scalar(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_exact_interventional_reconstruction_validation(
    residuals: np.ndarray,
    predictions: np.ndarray,
    production_row_indices,
    publication_group_ids,
    *,
    shap_version: str,
) -> dict[str, object]:
    """Apply the governed numerical-additivity criteria without altering SHAP values."""

    residuals = np.asarray(residuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    production_row_indices = np.asarray(production_row_indices)
    publication_group_ids = np.asarray(publication_group_ids, dtype=object)
    row_count = len(residuals)
    if not (
        row_count
        and len(predictions) == row_count
        and len(production_row_indices) == row_count
        and len(publication_group_ids) == row_count
    ):
        raise RuntimeError("Reconstruction governance inputs are empty or misaligned")
    if not np.isfinite(residuals).all() or not np.isfinite(predictions).all():
        raise RuntimeError("Reconstruction governance received non-finite values")

    absolute_errors = np.abs(residuals)
    relative_errors = absolute_errors / np.maximum(
        np.abs(predictions),
        np.finfo(float).eps,
    )
    strict_pass = absolute_errors <= EXACT_STRICT_ABSOLUTE_THRESHOLD
    exception_positions = np.flatnonzero(~strict_pass)
    strict_pass_count = int(strict_pass.sum())
    strict_pass_fraction = float(strict_pass_count / row_count)
    cohort_mae = float(absolute_errors.mean())
    maximum_absolute_error = float(absolute_errors.max())
    maximum_relative_error = float(relative_errors.max())
    criteria = {
        "cohort_mae": cohort_mae <= EXACT_COHORT_MAE_LIMIT,
        "strict_pass_fraction": strict_pass_fraction >= EXACT_MINIMUM_STRICT_PASS_FRACTION,
        "maximum_absolute_error": maximum_absolute_error <= EXACT_MAXIMUM_ABSOLUTE_ERROR,
        "maximum_relative_error": maximum_relative_error <= EXACT_MAXIMUM_RELATIVE_ERROR,
    }
    if not all(criteria.values()):
        failed = [name for name, passed in criteria.items() if not passed]
        raise RuntimeError(
            "Exact-interventional SHAP failed governed numerical-additivity criteria: "
            f"{failed}; MAE={cohort_mae:.12g}, strict_pass_fraction={strict_pass_fraction:.12g}, "
            f"max_abs={maximum_absolute_error:.12g}, max_relative={maximum_relative_error:.12g}"
        )

    exception_production_rows = [
        _json_scalar(production_row_indices[position]) for position in exception_positions
    ]
    exception_publications = list(
        dict.fromkeys(str(publication_group_ids[position]) for position in exception_positions)
    )
    exception_records = [
        {
            "position_zero_based": int(position),
            "production_row_index": _json_scalar(production_row_indices[position]),
            "publication_group_id": str(publication_group_ids[position]),
            "prediction_pce_points": float(predictions[position]),
            "signed_reconstruction_residual_pce_points": float(residuals[position]),
            "absolute_reconstruction_error_pce_points": float(absolute_errors[position]),
            "relative_reconstruction_error": float(relative_errors[position]),
        }
        for position in exception_positions
    ]
    exception_count = int(len(exception_positions))
    status = "PASS" if exception_count == 0 else "PASS_WITH_DOCUMENTED_NUMERICAL_TOLERANCE"
    disclosure = (
        "Exact-interventional TreeSHAP contributions were retained without modification and no rows were "
        f"excluded. In this SHAP {shap_version} runtime, the interventional C++ path uses single-precision "
        "internal tree thresholds and values; records at a float32 split boundary can therefore show bounded "
        "numerical additivity residuals even though SHAP and the fitted scikit-learn model use the same "
        "prediction. The governed cohort-level, strict-pass-fraction, maximum-absolute, and maximum-relative "
        "criteria all passed. Contributions were not redistributed to force exact reconstruction."
    )
    return {
        "status": status,
        "strict_absolute_threshold_pce_points": EXACT_STRICT_ABSOLUTE_THRESHOLD,
        "cohort_mae_limit_pce_points": EXACT_COHORT_MAE_LIMIT,
        "minimum_strict_pass_fraction": EXACT_MINIMUM_STRICT_PASS_FRACTION,
        "maximum_absolute_error_limit_pce_points": EXACT_MAXIMUM_ABSOLUTE_ERROR,
        "maximum_relative_error_limit": EXACT_MAXIMUM_RELATIVE_ERROR,
        "relative_error_denominator": "max(abs(predicted_pce), machine_epsilon)",
        "cohort_mae_pce_points": cohort_mae,
        "maximum_absolute_error_pce_points": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "strict_pass_count": strict_pass_count,
        "strict_pass_fraction": strict_pass_fraction,
        "exception_count": exception_count,
        "exception_fraction": float(exception_count / row_count),
        "exception_production_row_indices": exception_production_rows,
        "exception_publication_group_ids": exception_publications,
        "exception_records": exception_records,
        "rows_excluded": 0,
        "contributions_modified": False,
        "numerical_issue_disclosure": disclosure,
    }


def compute_population_fingerprint(
    production_row_indices,
    publication_group_ids,
    observed_pce,
) -> str:
    frame = pd.DataFrame(
        {
            "production_row_index": np.asarray(production_row_indices).astype(str),
            "publication_group_id": list(publication_group_ids),
            "observed_pce": np.asarray(observed_pce, dtype=float),
        }
    )
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    ).hexdigest()


def cleanup_stale_plot_outputs(out_dir: Path) -> None:
    for name in STALE_PLOT_DERIVED_NAMES:
        (out_dir / name).unlink(missing_ok=True)


def refresh_interpretability_audit_after_finalization(
    out_dir: Path,
    validation: dict[str, object],
    *,
    data_path: Path,
    manifest_path: Path,
) -> None:
    """Refresh the narrative audit while preserving representation-ablation context."""

    report_path = out_dir / "pce_feature_interpretability_audit.md"
    if not report_path.is_file():
        # A versioned final-harmonized Figure 5-only run intentionally has no
        # legacy pre-harmonizer representation-ablation report to preserve.
        # Keep its self-contained contract audit and add an idempotent record
        # of the existing-table validation instead of importing stale prose.
        harmonized_report_path = out_dir / "pce_final_harmonized_shap_audit.md"
        if not harmonized_report_path.is_file():
            raise FileNotFoundError(
                "Existing-SHAP finalization requires either the prior interpretability "
                "audit or the self-contained final-harmonized SHAP audit"
            )
        finalization = validation.get("finalization")
        reconstruction = validation.get("reconstruction_validation")
        if not isinstance(finalization, dict) or not isinstance(reconstruction, dict):
            raise RuntimeError(
                "Finalized validation lacks its finalization or reconstruction record"
            )
        previous = harmonized_report_path.read_text(encoding="utf-8")
        marker = "## Exact-interventional reconstruction governance"
        marker_position = previous.find(marker)
        if marker_position >= 0:
            previous = previous[:marker_position].rstrip()
        report = f"""{previous.rstrip()}

## Exact-interventional reconstruction governance

- Reconstruction status: **{reconstruction['status']}**.
- Cohort MAE: {float(reconstruction['cohort_mae_pce_points']):.12g} PCE points (limit {float(reconstruction['cohort_mae_limit_pce_points']):g}).
- Strict per-row threshold: {float(reconstruction['strict_absolute_threshold_pce_points']):g} PCE points; {int(reconstruction['strict_pass_count']):,}/{int(validation['explained_rows']):,} rows pass.
- Maximum absolute residual: {float(reconstruction['maximum_absolute_error_pce_points']):.12g} PCE points (limit {float(reconstruction['maximum_absolute_error_limit_pce_points']):g}).
- Rows excluded: **{int(reconstruction['rows_excluded'])}**; contributions modified or redistributed: **no**.

{reconstruction['numerical_issue_disclosure']}

## Existing-table finalization

- Persisted full-cohort contributions rewritten or redistributed: **no**.
- Current population, frozen mapping, model predictions, and governed background reproduced: **yes**.
- DOI-diverse exact-interventional spot check: {int(finalization['spot_check_rows']):,} rows.
- Grouped maximum absolute spot-check difference: {float(finalization['spot_check_grouped_max_abs_difference']):.12g} PCE points.
- Choice-bundle maximum absolute spot-check difference: {float(finalization['spot_check_choice_max_abs_difference']):.12g} PCE points.
- Reconstruction status: **{reconstruction['status']}**; cohort MAE {float(reconstruction['cohort_mae_pce_points']):.12g} PCE points; maximum absolute residual {float(reconstruction['maximum_absolute_error_pce_points']):.12g} PCE points.
"""
        harmonized_report_path.write_text(report, encoding="utf-8")
        return
    previous = report_path.read_text(encoding="utf-8")
    findings_start = previous.find("## Findings and corrections")
    outputs_start = previous.find("## Outputs", findings_start)
    if outputs_start < 0:
        outputs_start = previous.find("## Current audited outputs", findings_start)
    if findings_start < 0 or outputs_start < 0:
        raise RuntimeError("Prior interpretability audit does not contain the governed report sections")
    preservation_note_start = previous.find(
        "The feature-representation findings and model-comparison/ablation section above are preserved",
        findings_start,
    )
    preserved_end = (
        preservation_note_start
        if findings_start <= preservation_note_start < outputs_start
        else outputs_start
    )
    preserved_findings_and_ablation = previous[findings_start:preserved_end].rstrip()

    reconstruction = validation["reconstruction_validation"]
    if not isinstance(reconstruction, dict):
        raise RuntimeError("Finalized validation lacks reconstruction governance")
    raw_inventory = pd.read_csv(out_dir / "pce_raw_predictor_inventory.csv")
    raw_count = len(raw_inventory)
    numeric_count = int(raw_inventory["type"].astype(str).eq("numeric").sum())
    categorical_count = raw_count - numeric_count
    output_names = [
        "pce_raw_predictor_inventory.csv",
        "pce_encoded_feature_dictionary.csv",
        "pce_grouped_shap_values.csv",
        "pce_grouped_shap_importance.csv",
        "pce_grouped_shap_importance_doi_balanced.csv",
        "pce_grouped_shap_row_metadata.csv",
        "pce_choice_stratified_shap_values.csv",
        "pce_grouped_shap_validation.json",
        "pce_existing_shap_finalization_report.md",
        "pce_structured_parse_audit.csv",
        "pce_encoding_ablation_comparison.csv",
    ]
    existing_outputs = [name for name in output_names if (out_dir / name).is_file()]
    output_list = "\n".join(f"- `{name}`" for name in existing_outputs)
    exception_rows = ", ".join(
        str(value) for value in reconstruction["exception_production_row_indices"]
    ) or "none"
    exception_publications = ", ".join(
        str(value) for value in reconstruction["exception_publication_group_ids"]
    ) or "none"

    report = f"""# PCE feature interpretability audit

## Reproduction

- Production script: `{MODEL_SCRIPT}`
- Integrated fitted dataset: `{data_path}`
- Frozen leakage-audited manifest: `{manifest_path}`
- Population: {int(validation['training_rows']):,} rows; {int(validation['publication_group_count']):,} DOI/publication groups.
- Exact production inventory: {raw_count} raw predictors ({numeric_count} numerical; {categorical_count} categorically encoded), {int(validation['encoded_feature_count']):,} encoded inputs, and {int(validation['scientific_group_count']):,} source-level scientific groups.
- Estimator: Extra Trees, 300 estimators, `min_samples_leaf=2`, `max_features=sqrt`, seed 42; DOI-aware grouped cross-validation remains the separate performance-evaluation design.
- Exact interventional TreeSHAP covers all {int(validation['explained_rows']):,} final-refit training rows with no explanation sampling. The deterministic background contains {int(validation['interventional_background']['background_rows']):,} records from {int(validation['interventional_background']['distinct_publications']):,} DOI/publication groups.
- These are in-sample explanations of the final all-data refit, not out-of-fold performance estimates or causal effects.

## Exact-interventional reconstruction governance

- Top-level validation: **{validation['status']}**; nested reconstruction status: **{reconstruction['status']}**.
- Strict threshold: {float(reconstruction['strict_absolute_threshold_pce_points']):g} PCE points; {int(reconstruction['strict_pass_count']):,}/{int(validation['explained_rows']):,} rows pass ({float(reconstruction['strict_pass_fraction']):.8%}; required minimum {float(reconstruction['minimum_strict_pass_fraction']):.5%}).
- Cohort MAE: {float(reconstruction['cohort_mae_pce_points']):.12g} PCE points (limit {float(reconstruction['cohort_mae_limit_pce_points']):g}).
- Maximum absolute residual: {float(reconstruction['maximum_absolute_error_pce_points']):.12g} PCE points (limit {float(reconstruction['maximum_absolute_error_limit_pce_points']):g}).
- Maximum relative residual: {float(reconstruction['maximum_relative_error']):.12g} (limit {float(reconstruction['maximum_relative_error_limit']):g}; denominator `{reconstruction['relative_error_denominator']}`).
- Strict-threshold exceptions: {int(reconstruction['exception_count']):,} rows ({float(reconstruction['exception_fraction']):.8%}); production rows: {exception_rows}; DOI/publication groups: {exception_publications}.
- Rows excluded: {int(reconstruction['rows_excluded'])}; contributions modified or redistributed: **no**.

{reconstruction['numerical_issue_disclosure']}

{preserved_findings_and_ablation}

The feature-representation findings and model-comparison/ablation section above are preserved from the prior audit. Existing-SHAP finalization revalidated their cohort and feature mapping but did not rerun the non-Figure-5 ablation.

## Current audited outputs

{output_list}

Plot-derived summaries, local-selection records, and rendered figures were intentionally removed as stale. `build_curated_publication_figures.py` regenerates them from the governed full-cohort source tables after finalization.

## Manuscript guardrails

- Methods must distinguish the complete {int(validation['scientific_group_count'])}-group audit from the six narrow, predefined choice bundles used for the main figure.
- Caption/methods must state that all {int(validation['explained_rows']):,} final-refit training records were explained, only the top {int(validation['importance_display_group_count'])} global groups are displayed, and attribution is in-sample.
- Report the nested numerical-tolerance status and its documented strict-threshold exceptions without dropping rows or reallocating contributions.
- Interpret SHAP as fitted-model attribution, not held-out performance, substitution effects, or causal evidence.
"""
    report_path.write_text(report, encoding="utf-8")


def select_interventional_background_positions(
    publication_group_ids,
    observed_target,
    requested_rows: int,
) -> np.ndarray:
    """Select a deterministic DOI/publication-group-diverse background spanning observed PCE."""
    if requested_rows < 1:
        raise ValueError("Interventional background row count must be positive")
    candidates = pd.DataFrame(
        {
            "position": np.arange(len(observed_target), dtype=int),
            "publication_group_id": publication_group_ids,
            "observed_pce": np.asarray(observed_target, dtype=float),
        }
    )
    candidates = (
        candidates.sort_values(["publication_group_id", "position"], kind="stable")
        .drop_duplicates("publication_group_id", keep="first")
        .sort_values(["observed_pce", "position"], kind="stable")
        .reset_index(drop=True)
    )
    count = min(int(requested_rows), len(candidates))
    selected = np.unique(np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int))
    return candidates.iloc[selected]["position"].to_numpy(dtype=int)


def compare_approximate_with_exact(
    pipe,
    X: pd.DataFrame,
    group_matrix,
    group_names: list[str],
    production_row_indices,
    publication_group_ids,
    predictions,
    requested_rows: int,
):
    """Compare the full-cohort approximation with exact TreeSHAP on a DOI-diverse audit set."""
    if requested_rows <= 0:
        return None, pd.DataFrame()
    import shap

    candidates = pd.DataFrame(
        {
            "position": np.arange(len(X), dtype=int),
            "production_row_index": production_row_indices,
            "publication_group_id": publication_group_ids,
            "predicted_pce": predictions,
        }
    )
    candidates = (
        candidates.sort_values(["publication_group_id", "position"], kind="stable")
        .drop_duplicates("publication_group_id", keep="first")
        .sort_values(["predicted_pce", "position"], kind="stable")
        .reset_index(drop=True)
    )
    audit_rows = min(int(requested_rows), len(candidates))
    selected_candidate_positions = np.unique(
        np.rint(np.linspace(0, len(candidates) - 1, audit_rows)).astype(int)
    )
    selected = candidates.iloc[selected_candidate_positions].copy()
    positions = selected["position"].to_numpy(dtype=int)
    encoded = pipe.named_steps["pre"].transform(X.iloc[positions])
    encoded = encoded.toarray() if sparse.issparse(encoded) else np.asarray(encoded)
    encoded = np.asarray(encoded, dtype=float)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    approximate_values = _normalize_tree_shap_values(
        explainer.shap_values(encoded, check_additivity=False, approximate=True),
        len(selected),
        group_matrix.shape[1],
    )
    print(f"Exact-agreement audit: computing exact TreeSHAP for {len(selected):,} DOI-diverse rows", flush=True)
    exact_values = _normalize_tree_shap_values(
        explainer.shap_values(encoded, check_additivity=False, approximate=False),
        len(selected),
        group_matrix.shape[1],
    )
    approximate_grouped = np.asarray((group_matrix @ approximate_values.T).T, dtype=float)
    exact_grouped = np.asarray((group_matrix @ exact_values.T).T, dtype=float)
    approximate_importance = np.mean(np.abs(approximate_grouped), axis=0)
    exact_importance = np.mean(np.abs(exact_grouped), axis=0)
    importance_spearman = float(spearmanr(approximate_importance, exact_importance).statistic)
    importance_kendall = float(kendalltau(approximate_importance, exact_importance).statistic)
    exact_top20 = set(np.argsort(exact_importance)[-20:])
    approximate_top20 = set(np.argsort(approximate_importance)[-20:])
    top20_overlap = len(exact_top20.intersection(approximate_top20))
    absolute_difference = np.abs(approximate_grouped - exact_grouped)
    exact_scale = float(np.mean(np.abs(exact_grouped)))
    relative_mae = float(np.mean(absolute_difference) / max(exact_scale, np.finfo(float).eps))
    flattened_pearson = float(np.corrcoef(approximate_grouped.ravel(), exact_grouped.ravel())[0, 1])
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    selected_predictions = np.asarray(pipe.named_steps["model"].predict(encoded), dtype=float)
    exact_reconstruction_max = float(
        np.max(np.abs(expected + exact_values.sum(axis=1) - selected_predictions))
    )
    thresholds = {
        "minimum_importance_rank_spearman": 0.90,
        "minimum_top20_overlap_count": 15,
        "maximum_relative_grouped_mae": 0.25,
    }
    agreement_pass = bool(
        importance_spearman >= thresholds["minimum_importance_rank_spearman"]
        and top20_overlap >= thresholds["minimum_top20_overlap_count"]
        and relative_mae <= thresholds["maximum_relative_grouped_mae"]
        and exact_reconstruction_max <= 1e-8
    )
    comparison = pd.DataFrame(
        {
            "scientific_group": group_names,
            "approximate_mean_absolute_grouped_shap_pce_points": approximate_importance,
            "exact_mean_absolute_grouped_shap_pce_points": exact_importance,
            "absolute_importance_difference_pce_points": np.abs(approximate_importance - exact_importance),
        }
    ).sort_values("exact_mean_absolute_grouped_shap_pce_points", ascending=False)
    metrics = {
        "status": "PASS" if agreement_pass else "REVIEW_REQUIRED",
        "selection_method": (
            "one deterministic row per publication, then evenly spaced across the fitted prediction range"
        ),
        "requested_rows": int(requested_rows),
        "compared_rows": len(selected),
        "distinct_publications": int(selected["publication_group_id"].nunique()),
        "production_row_indices": selected["production_row_index"].tolist(),
        "importance_rank_spearman": importance_spearman,
        "importance_rank_kendall": importance_kendall,
        "top20_overlap_count": top20_overlap,
        "top20_overlap_fraction": top20_overlap / 20,
        "flattened_grouped_value_pearson": flattened_pearson,
        "relative_grouped_mean_absolute_difference": relative_mae,
        "exact_reconstruction_max_abs_error": exact_reconstruction_max,
        "governance_thresholds": thresholds,
    }
    return metrics, comparison


def display_token(value: object) -> str:
    token = clean_token(value)
    return token if token is not None else "Not reported"


def boolean_context_label(value: object) -> str:
    token = clean_token(value)
    if token is None:
        return "Not reported"
    normalized = token.casefold()
    if normalized in {"true", "yes", "1", "1.0"}:
        return "Yes"
    if normalized in {"false", "no", "0", "0.0"}:
        return "No"
    return token


def additional_row_context_metadata(sample: pd.DataFrame) -> dict[str, object]:
    add_desc_length = pd.to_numeric(sample["ADD_DESC_text_length"], errors="coerce")
    return {
        "context_fully_inorganic": sample["Perovskite_composition_inorganic"]
        .map(boolean_context_label)
        .to_numpy(),
        "context_perovskite_long_form": sample["Perovskite_composition_long_form"]
        .map(display_token)
        .to_numpy(),
        "context_add_desc_text_length": [
            f"metadata proxy; {value:g} characters" if pd.notna(value) else "Not reported"
            for value in add_desc_length
        ],
        "context_b_site_composition": [
            composition_label(ions, coefficients)
            for ions, coefficients in zip(
                sample["Perovskite_composition_b_ions"],
                sample["Perovskite_composition_b_ions_coefficients"],
            )
        ],
        "context_substrate_stack_sequence": sample["Substrate_stack_sequence"]
        .map(display_token)
        .to_numpy(),
        "context_lead_free": sample["Perovskite_composition_leadfree"]
        .map(boolean_context_label)
        .to_numpy(),
        "context_perovskite_deposition_procedure": sample["Perovskite_deposition_procedure"]
        .map(display_token)
        .to_numpy(),
        "context_perovskite_additives": sample["Perovskite_additives_compounds"]
        .map(display_token)
        .to_numpy(),
        "context_perovskite_synthesis_atmosphere": sample[
            "Perovskite_deposition_synthesis_atmosphere"
        ]
        .map(display_token)
        .to_numpy(),
    }


def composition_label(ions: object, coefficients: object) -> str:
    names = simple_aligned_list(ions)
    values = simple_aligned_list(coefficients)
    if not names:
        return "Not reported"
    if values and len(names) == len(values):
        pairs = [name if value in {"1", "1.0"} else f"{name} {value}" for name, value in zip(names, values)]
        return " / ".join(pairs)
    return " / ".join(names)


def solvent_formulation_label(solvents: object, ratios: object) -> str:
    solvent_text = clean_token(solvents)
    ratio_text = clean_token(ratios)
    if solvent_text is None:
        return "Not reported"
    solvent_stages = [stage.strip() for stage in solvent_text.split(">>")]
    ratio_stages = [stage.strip() for stage in ratio_text.split(">>")] if ratio_text else []
    rendered: list[str] = []
    for index, stage in enumerate(solvent_stages):
        names = [name.strip() for name in stage.split(";") if name.strip()]
        ratio_stage = ratio_stages[index] if index < len(ratio_stages) else ""
        values = [value.strip() for value in ratio_stage.split(";") if value.strip()]
        identity = "/".join(names)
        if len(names) == 1 and values in (["1"], ["1.0"]):
            rendered.append(identity)
        elif values and len(values) == len(names):
            rendered.append(f"{identity} ({':'.join(values)})")
        elif ratio_stage:
            rendered.append(f"{identity} (reported ratio {ratio_stage.replace(';', ':')})")
        else:
            rendered.append(f"{identity} (ratio not reported)")
    return " → ".join(rendered)


def joined_context(*values: object) -> str:
    tokens = [display_token(value) for value in values]
    present = [token for token in tokens if token != "Not reported"]
    return "; ".join(present) if present else "Not reported"


def absorber_identity_label(long_form: object, short_form: object) -> str:
    return display_token(long_form) if clean_token(long_form) is not None else display_token(short_form)


def normalized_choice_state(feature_family: str, value: object) -> tuple[str, bool, bool]:
    state = display_token(value)
    reported = state != "Not reported"
    if feature_family != "Quenching-induced crystallization":
        return state, reported, reported
    normalized = state.casefold()
    if normalized in {"true", "yes", "1"}:
        return "Yes", True, True
    if normalized in {"false", "no", "0"}:
        return "No", True, True
    return ("Not reported" if not reported else "Invalid/unknown"), reported, False


def finalize_existing_exact_interventional_shap(
    *,
    out_dir: Path,
    contract_provenance: dict[str, object],
    figure4_contract_aligned: bool,
    pipe,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    names: list[str],
    numeric_raw_feature_count: int,
    categorical_raw_feature_count: int,
    boolean_categorical_raw_feature_count: int,
    interpretation_sample: pd.DataFrame,
    dictionary: pd.DataFrame,
    mapping: pd.DataFrame,
    group_names: list[str],
    group_matrix,
    choice_matrix,
    choice_definition_templates: list[dict[str, object]],
    sample_idx: np.ndarray,
    sample: pd.DataFrame,
    publication_group_ids: list[str],
    background_positions: np.ndarray,
    batch_size: int,
    background_requested_rows: int,
    spot_check_rows: int,
    data_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Finalize fully written exact SHAP tables after strict provenance checks.

    This path never changes or redistributes a contribution. It refits the frozen
    deterministic model, validates persisted tables against the current cohort,
    and recomputes a small exact-interventional spot check before issuing new
    validation provenance.
    """

    import shap

    required_files = {
        "grouped_values": out_dir / "pce_grouped_shap_values.csv",
        "row_metadata": out_dir / "pce_grouped_shap_row_metadata.csv",
        "importance": out_dir / "pce_grouped_shap_importance.csv",
        "doi_importance": out_dir / "pce_grouped_shap_importance_doi_balanced.csv",
        "choice_values": out_dir / "pce_choice_stratified_shap_values.csv",
        "encoded_dictionary": out_dir / "pce_encoded_feature_dictionary.csv",
        "raw_inventory": out_dir / "pce_raw_predictor_inventory.csv",
    }
    missing_files = [str(path) for path in required_files.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Cannot finalize incomplete SHAP tables: {missing_files}")

    grouped_values = pd.read_csv(required_files["grouped_values"])
    metadata = pd.read_csv(required_files["row_metadata"])
    importance = pd.read_csv(required_files["importance"])
    doi_importance = pd.read_csv(required_files["doi_importance"])
    choice_values = pd.read_csv(required_files["choice_values"])
    saved_dictionary = pd.read_csv(required_files["encoded_dictionary"])
    raw_inventory = pd.read_csv(required_files["raw_inventory"])

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"Existing exact SHAP finalization rejected: {message}")

    expected_rows = len(df)
    require(expected_rows > 0, "current fitted population is empty")
    require(len(grouped_values) == expected_rows, "grouped values do not cover the current fitted population")
    require(len(metadata) == expected_rows, "row metadata do not cover the current fitted population")
    require("production_row_index" in grouped_values, "grouped values lack production_row_index")
    require("production_row_index" in metadata, "row metadata lack production_row_index")
    expected_row_ids = [str(_json_scalar(value)) for value in sample_idx]
    require(
        grouped_values["production_row_index"].map(lambda value: str(_json_scalar(value))).tolist()
        == expected_row_ids,
        "grouped values are stale, reordered, or from another population",
    )
    require(
        metadata["production_row_index"].map(lambda value: str(_json_scalar(value))).tolist()
        == expected_row_ids,
        "row metadata are stale, reordered, or from another population",
    )
    require(not grouped_values["production_row_index"].duplicated().any(), "grouped row identifiers are duplicated")
    require(not metadata["production_row_index"].duplicated().any(), "metadata row identifiers are duplicated")

    actual_group_columns = [column for column in grouped_values if column != "production_row_index"]
    require(actual_group_columns == group_names, "grouped-value columns do not match the current ordered mapping")
    require(np.isfinite(grouped_values[group_names].to_numpy(dtype=float)).all(), "grouped values are non-finite")
    required_metadata = {
        "publication_group_id",
        "observed_pce",
        "predicted_pce",
        "base_value",
        "solvent_formulation",
    }
    require(required_metadata <= set(metadata), "row metadata contract is incomplete")
    require(
        metadata["publication_group_id"].astype(str).tolist()
        == [str(value) for value in publication_group_ids],
        "publication-group lineage does not match the current population",
    )
    require(
        np.allclose(
            metadata["observed_pce"].to_numpy(dtype=float),
            sample[target].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ),
        "observed PCE lineage does not match the current population",
    )

    expected_dictionary_columns = [
        "encoded_feature_index",
        "exact_encoded_model_feature_name",
        "raw_source_predictor",
        "scientific_grouping",
    ]
    require(set(expected_dictionary_columns) <= set(saved_dictionary.columns), "encoded dictionary contract is incomplete")
    require(len(saved_dictionary) == len(names), "encoded dictionary row count is stale")
    for column in expected_dictionary_columns:
        expected_values = dictionary[column]
        actual_values = saved_dictionary[column]
        if column == "encoded_feature_index":
            matches = np.array_equal(actual_values.to_numpy(dtype=int), expected_values.to_numpy(dtype=int))
        else:
            matches = actual_values.astype(str).tolist() == expected_values.astype(str).tolist()
        require(matches, f"encoded dictionary column {column} differs from the current mapping")
    require(
        raw_inventory["raw_predictor_name"].astype(str).tolist() == [str(value) for value in features],
        "raw predictor inventory differs from the frozen manifest",
    )

    current_predictions = np.asarray(pipe.predict(df[features].copy()), dtype=float)
    # Use Python's exact decimal-to-binary parser for the persisted prediction
    # lineage.  Pandas' fast CSV parser can select an adjacent float for a few
    # shortest-round-trip strings, which would change a cryptographic hash even
    # though the displayed value and governed numerical comparison are equal.
    with required_files["row_metadata"].open(
        "r", encoding="utf-8", newline=""
    ) as metadata_handle:
        metadata_reader = csv.DictReader(metadata_handle)
        if "predicted_pce" not in (metadata_reader.fieldnames or []):
            raise RuntimeError(
                "Existing exact SHAP finalization rejected: row metadata lack predicted_pce"
            )
        saved_predictions = np.asarray(
            [float(row["predicted_pce"]) for row in metadata_reader], dtype=float
        )
    require(
        len(saved_predictions) == len(metadata),
        "exactly parsed saved predictions do not cover the metadata rows",
    )
    require(
        np.allclose(saved_predictions, current_predictions, rtol=0, atol=1e-12),
        "saved predictions do not reproduce from the deterministic frozen-model refit",
    )
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    background = pre.transform(df.iloc[background_positions][features])
    background = background.toarray() if sparse.issparse(background) else np.asarray(background)
    background = np.asarray(background, dtype=float)
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    require(
        np.allclose(metadata["base_value"].to_numpy(dtype=float), expected_value, rtol=0, atol=1e-12),
        "saved base value does not match the governed interventional background expectation",
    )
    del background

    grouped_array = grouped_values[group_names].to_numpy(dtype=float)
    residuals = expected_value + grouped_array.sum(axis=1) - saved_predictions
    reconstruction = build_exact_interventional_reconstruction_validation(
        residuals,
        saved_predictions,
        sample_idx,
        publication_group_ids,
        shap_version=str(getattr(shap, "__version__", "unknown")),
    )

    expected_absolute = pd.Series(np.mean(np.abs(grouped_array), axis=0), index=group_names)
    expected_signed = pd.Series(np.mean(grouped_array, axis=0), index=group_names)
    importance_indexed = importance.set_index("scientific_group")
    require(set(importance_indexed.index) == set(group_names), "importance groups differ from the current mapping")
    require(
        np.allclose(
            importance_indexed.loc[group_names, "mean_absolute_grouped_shap_pce_points"],
            expected_absolute,
            rtol=0,
            atol=1e-12,
        ),
        "device-weighted mean absolute importance does not reproduce",
    )
    require(
        np.allclose(
            importance_indexed.loc[group_names, "signed_mean_grouped_shap_pce_points"],
            expected_signed,
            rtol=0,
            atol=1e-12,
        ),
        "device-weighted signed importance does not reproduce",
    )
    expected_encoded_counts = mapping.groupby("scientific_grouping", sort=False).size().reindex(group_names)
    require(
        np.array_equal(
            importance_indexed.loc[group_names, "encoded_input_count"].to_numpy(dtype=int),
            expected_encoded_counts.to_numpy(dtype=int),
        ),
        "importance encoded-input counts differ from the current mapping",
    )

    grouped_for_balance = pd.DataFrame(grouped_array, columns=group_names)
    grouped_for_balance["publication_group_id"] = [str(value) for value in publication_group_ids]
    expected_doi_absolute = (
        grouped_for_balance.assign(
            **{name: grouped_for_balance[name].abs() for name in group_names}
        )
        .groupby("publication_group_id", sort=False)[group_names]
        .mean()
        .mean(axis=0)
    )
    expected_doi_signed = (
        grouped_for_balance.groupby("publication_group_id", sort=False)[group_names]
        .mean()
        .mean(axis=0)
    )
    doi_indexed = doi_importance.set_index("scientific_group")
    require(set(doi_indexed.index) == set(group_names), "DOI-balanced importance groups differ from mapping")
    require(
        np.allclose(
            doi_indexed.loc[group_names, "doi_balanced_mean_absolute_grouped_shap_pce_points"],
            expected_doi_absolute,
            rtol=0,
            atol=1e-12,
        ),
        "DOI-balanced mean absolute importance does not reproduce",
    )
    require(
        np.allclose(
            doi_indexed.loc[group_names, "doi_balanced_signed_mean_grouped_shap_pce_points"],
            expected_doi_signed,
            rtol=0,
            atol=1e-12,
        ),
        "DOI-balanced signed importance does not reproduce",
    )
    for column, expected_values in (
        ("doi_balanced_mean_absolute_grouped_shap_pce_points", expected_doi_absolute),
        ("doi_balanced_signed_mean_grouped_shap_pce_points", expected_doi_signed),
    ):
        require(column in importance_indexed, f"merged importance lacks {column}")
        require(
            np.allclose(importance_indexed.loc[group_names, column], expected_values, rtol=0, atol=1e-12),
            f"merged importance column {column} does not reproduce",
        )

    choice_required = {
        "production_row_index",
        "publication_group_id",
        "feature_family",
        "state",
        "bundle_shap_pce_points",
        "state_is_reported",
        "state_is_valid",
        "raw_source_fields",
        "raw_source_field_count",
        "encoded_input_count",
    }
    require(choice_required <= set(choice_values), "choice-stratified table contract is incomplete")
    expected_choice_rows = expected_rows * len(choice_definition_templates)
    require(len(choice_values) == expected_choice_rows, "choice-stratified row count does not reproduce N x bundles")
    require(
        not choice_values.duplicated(["production_row_index", "feature_family"]).any(),
        "choice-stratified record/bundle keys are duplicated",
    )
    context_values = {
        "Precursor solvent formulation": [
            solvent_formulation_label(solvent, ratio)
            for solvent, ratio in zip(
                interpretation_sample["Perovskite_deposition_solvents"],
                interpretation_sample[
                    "Perovskite_deposition_solvents_mixing_ratios"
                ],
            )
        ],
        "ETL stack and deposition": [
            joined_context(stack, procedure)
            for stack, procedure in zip(
                interpretation_sample["ETL_stack_sequence"],
                interpretation_sample["ETL_deposition_procedure"],
            )
        ],
        "HTL stack and deposition": [
            joined_context(stack, procedure)
            for stack, procedure in zip(
                interpretation_sample["HTL_stack_sequence"],
                interpretation_sample["HTL_deposition_procedure"],
            )
        ],
        "Absorber composition identity": [
            absorber_identity_label(long_form, short_form)
            for long_form, short_form in zip(
                interpretation_sample["Perovskite_composition_long_form"],
                interpretation_sample["Perovskite_composition_short_form"],
            )
        ],
        "Quenching-induced crystallization": interpretation_sample[
            "Perovskite_deposition_quenching_induced_crystallisation"
        ].map(display_token).tolist(),
        "Back-contact material": interpretation_sample[
            "Backcontact_stack_sequence"
        ].map(display_token).tolist(),
    }
    bundle_definitions: list[dict[str, object]] = []
    for template in choice_definition_templates:
        family_name = str(template["feature_family"])
        family_rows = choice_values.loc[choice_values["feature_family"].astype(str).eq(family_name)].copy()
        require(len(family_rows) == expected_rows, f"choice family {family_name} does not cover every row")
        require(
            family_rows["production_row_index"].map(lambda value: str(_json_scalar(value))).tolist()
            == expected_row_ids,
            f"choice family {family_name} is stale or reordered",
        )
        require(
            family_rows["publication_group_id"].astype(str).tolist()
            == [str(value) for value in publication_group_ids],
            f"choice family {family_name} publication lineage differs",
        )
        expected_fields = " | ".join(str(value) for value in template["raw_source_fields"])
        require(set(family_rows["raw_source_fields"].astype(str)) == {expected_fields}, f"choice family {family_name} raw fields differ")
        require(
            set(family_rows["raw_source_field_count"].astype(int)) == {int(template["raw_source_field_count"])},
            f"choice family {family_name} raw-field count differs",
        )
        require(
            set(family_rows["encoded_input_count"].astype(int)) == {int(template["encoded_input_count"])},
            f"choice family {family_name} encoded-input count differs",
        )
        expected_states = [
            normalized_choice_state(family_name, value) for value in context_values[family_name]
        ]
        require(
            family_rows["state"].astype(str).tolist() == [state[0] for state in expected_states],
            f"choice family {family_name} states do not reproduce from current source fields",
        )
        require(
            family_rows["state_is_reported"].astype(str).str.casefold().tolist()
            == [str(state[1]).casefold() for state in expected_states],
            f"choice family {family_name} reported-state flags do not reproduce",
        )
        require(
            family_rows["state_is_valid"].astype(str).str.casefold().tolist()
            == [str(state[2]).casefold() for state in expected_states],
            f"choice family {family_name} validity flags do not reproduce",
        )
        require(
            np.isfinite(family_rows["bundle_shap_pce_points"].to_numpy(dtype=float)).all(),
            f"choice family {family_name} contains non-finite contributions",
        )
        bundle_definitions.append(
            {key: value for key, value in template.items() if key != "context_column"}
            | {
                "mean_absolute_bundle_shap_pce_points": float(
                    family_rows["bundle_shap_pce_points"].abs().mean()
                )
            }
        )

    exception_positions = [
        int(record["position_zero_based"]) for record in reconstruction["exception_records"]
    ]
    requested_spot_rows = max(1, int(spot_check_rows))
    diverse_spot_positions = select_interventional_background_positions(
        publication_group_ids,
        sample[target].to_numpy(dtype=float),
        requested_spot_rows,
    ).tolist()
    spot_positions = list(dict.fromkeys(exception_positions + diverse_spot_positions))
    encoded_spot = pre.transform(df.iloc[spot_positions][features])
    encoded_spot = encoded_spot.toarray() if sparse.issparse(encoded_spot) else np.asarray(encoded_spot)
    encoded_spot = np.asarray(encoded_spot, dtype=float)
    spot_values = _normalize_tree_shap_values(
        explainer.shap_values(encoded_spot, check_additivity=False, approximate=False),
        len(spot_positions),
        len(names),
    )
    spot_grouped = np.asarray((group_matrix @ spot_values.T).T, dtype=float)
    grouped_spot_difference = float(
        np.max(np.abs(spot_grouped - grouped_array[np.asarray(spot_positions, dtype=int)]))
    )
    require(grouped_spot_difference <= 1e-10, "exact-interventional grouped spot check does not reproduce")
    spot_choice = np.asarray((choice_matrix @ spot_values.T).T, dtype=float)
    saved_spot_choice = np.empty_like(spot_choice)
    for bundle_position, template in enumerate(choice_definition_templates):
        family_rows = choice_values.loc[
            choice_values["feature_family"].astype(str).eq(str(template["feature_family"])),
            "bundle_shap_pce_points",
        ].to_numpy(dtype=float)
        saved_spot_choice[:, bundle_position] = family_rows[np.asarray(spot_positions, dtype=int)]
    choice_spot_difference = float(np.max(np.abs(spot_choice - saved_spot_choice)))
    require(choice_spot_difference <= 1e-10, "exact-interventional choice-bundle spot check does not reproduce")
    del encoded_spot, spot_values, spot_grouped, spot_choice, explainer

    publication_group_series = pd.Series(publication_group_ids, name="publication_group_id")
    row_importance = expected_absolute.reindex(group_names)
    doi_weighted_importance = expected_doi_absolute.reindex(group_names)
    rank_spearman = float(spearmanr(row_importance, doi_weighted_importance).statistic)
    rank_kendall = float(kendalltau(row_importance, doi_weighted_importance).statistic)
    row_top20 = set(row_importance.nlargest(20).index)
    doi_top20 = set(doi_weighted_importance.nlargest(20).index)
    top20_overlap = len(row_top20.intersection(doi_top20))
    background_validation = {
        "background_rows": int(len(background_positions)),
        "background_positions": background_positions.astype(int).tolist(),
        "feature_perturbation": "interventional",
        "requested_rows": int(background_requested_rows),
        "selection_method": (
            "one deterministic row per publication, then evenly spaced across observed fitted-cohort PCE"
        ),
        "distinct_publications": int(publication_group_series.iloc[background_positions].nunique()),
        "production_row_indices": [
            _json_scalar(value) for value in sample_idx[background_positions]
        ],
        "observed_pce_min": float(sample[target].iloc[background_positions].min()),
        "observed_pce_max": float(sample[target].iloc[background_positions].max()),
    }
    population_hash = compute_population_fingerprint(
        sample_idx,
        publication_group_ids,
        sample[target].to_numpy(dtype=float),
    )
    encoded_mapping_sha256 = hashlib.sha256(
        mapping.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    model_prediction_sha256 = hashlib.sha256(
        np.asarray(saved_predictions, dtype="<f8").tobytes()
    ).hexdigest()
    finalized_contract = dict(contract_provenance)
    finalized_contract.update(
        {
            "status": "PASS",
            "population_fingerprint_sha256": population_hash,
            "scientific_group_count": len(group_names),
            "encoded_inputs_mapped_exactly_once": True,
            "encoded_mapping_sha256": encoded_mapping_sha256,
            "prediction_sha256": model_prediction_sha256,
        }
    )
    context_metadata = additional_row_context_metadata(interpretation_sample)
    added_context_columns: list[str] = []
    for column, expected_values in context_metadata.items():
        expected_series = pd.Series(expected_values, index=metadata.index).astype(str)
        if column in metadata:
            require(
                metadata[column].astype(str).tolist() == expected_series.tolist(),
                f"row metadata context column {column} differs from the current source population",
            )
        else:
            metadata[column] = expected_values
            added_context_columns.append(column)
    if added_context_columns:
        metadata.to_csv(required_files["row_metadata"], index=False)
    table_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in required_files.items()
    }
    method_label = (
        "exact TreeExplainer TreeSHAP (feature_perturbation='interventional', model_output='raw')"
    )
    validation = {
        "status": "PASS",
        "model_contract": finalized_contract,
        "model_contract_label": finalized_contract["contract_name"],
        "figure4_contract_aligned": bool(figure4_contract_aligned),
        "categorical_schema_harmonization": bool(figure4_contract_aligned),
        "raw_feature_count": len(features),
        "numeric_raw_feature_count": int(numeric_raw_feature_count),
        "categorical_raw_feature_count": int(categorical_raw_feature_count),
        "boolean_categorical_raw_feature_count": int(
            boolean_categorical_raw_feature_count
        ),
        "analysis_population": "complete final fitted-training cohort",
        "explanation_scope": "in-sample explanation of the final model refit on every eligible row",
        "out_of_fold_explanation": False,
        "training_rows": expected_rows,
        "fitted_training_rows": expected_rows,
        "population_rows": expected_rows,
        "aggregation_method": (
            "sum signed encoded-input SHAP contributions within each source-level scientific group "
            "for every explained row, then rank groups by mean absolute contribution"
        ),
        "explained_rows": expected_rows,
        "explanation_coverage_fraction": 1.0,
        "sampling_method": "none",
        "publication_group_count": int(publication_group_series.nunique()),
        "population_fingerprint_sha256": population_hash,
        "scientific_group_count": len(group_names),
        "encoded_feature_count": len(names),
        "encoded_features_mapped_exactly_once": True,
        "encoded_feature_names_sha256": finalized_contract[
            "encoded_feature_names_sha256"
        ],
        "encoded_mapping_sha256": encoded_mapping_sha256,
        "model_prediction_sha256": model_prediction_sha256,
        "shap_method": method_label,
        "shap_version": str(getattr(shap, "__version__", "unknown")),
        "shap_batch_size": int(batch_size),
        "shap_batch_count": int(math.ceil(expected_rows / batch_size)),
        "exact_tree_shap": True,
        "feature_perturbation": "interventional",
        "interventional_background": background_validation,
        "reconstruction_validation_tolerance": EXACT_MAXIMUM_ABSOLUTE_ERROR,
        "base_value_pce_points": expected_value,
        "prediction_reconstruction_mae": reconstruction["cohort_mae_pce_points"],
        "prediction_reconstruction_max_abs_error": reconstruction[
            "maximum_absolute_error_pce_points"
        ],
        "prediction_reconstruction_max_relative_error": reconstruction["maximum_relative_error"],
        "grouped_prediction_reconstruction_mae": reconstruction["cohort_mae_pce_points"],
        "grouped_prediction_reconstruction_max_abs_error": reconstruction[
            "maximum_absolute_error_pce_points"
        ],
        "grouped_prediction_reconstruction_max_relative_error": reconstruction[
            "maximum_relative_error"
        ],
        "reconstruction_mae": reconstruction["cohort_mae_pce_points"],
        "reconstruction_validation": reconstruction,
        "importance_display_group_count": 20,
        "importance_full_table_group_count": len(group_names),
        "doi_balanced_importance": {
            "publication_weighting": (
                "equal weight per publication after averaging absolute row contributions within publication"
            ),
            "rank_spearman_vs_device_weighted": rank_spearman,
            "rank_kendall_vs_device_weighted": rank_kendall,
            "top20_overlap_count": top20_overlap,
            "top20_overlap_fraction": top20_overlap / 20,
        },
        "choice_bundle_count": len(bundle_definitions),
        "choice_value_rows": len(choice_values),
        "choice_state_selection": (
            "within each predefined feature family, select the three most prevalent reported valid states "
            "with at least five explained rows; selection does not use SHAP magnitude"
        ),
        "choice_bundle_definitions": bundle_definitions,
        "interpretation_note": (
            "These are in-sample attributions for the final model fitted on the complete eligible cohort. "
            "They explain model predictions, are not out-of-fold performance evidence, and do not prove causality."
        ),
        "approximation_disclosure": (
            "Exact interventional TreeSHAP was used with a deterministic DOI/publication-group-diverse "
            "background spanning the observed fitted-cohort PCE range. "
            + str(reconstruction["numerical_issue_disclosure"])
        ),
        "finalization": {
            "mode": "validated existing exact-interventional tables without recomputing full-cohort SHAP",
            "contribution_tables_rewritten": False,
            "row_metadata_context_columns_added": added_context_columns,
            "row_metadata_context_columns_validated": list(context_metadata),
            "current_model_predictions_reproduced": True,
            "current_mapping_reproduced": True,
            "current_population_fingerprint_reproduced": True,
            "current_background_expectation_reproduced": True,
            "spot_check_rows": len(spot_positions),
            "spot_check_positions": spot_positions,
            "spot_check_production_row_indices": [
                _json_scalar(sample_idx[position]) for position in spot_positions
            ],
            "spot_check_grouped_max_abs_difference": grouped_spot_difference,
            "spot_check_choice_max_abs_difference": choice_spot_difference,
            "reconstruction_metrics_source": (
                "recomputed from persisted full-cohort grouped contributions, base values, and "
                "current-refit predictions"
            ),
            "source_table_sha256": table_hashes,
        },
    }

    cleanup_stale_plot_outputs(out_dir)
    (out_dir / "pce_shap_exact_agreement_by_group.csv").unlink(missing_ok=True)
    validation_path = out_dir / "pce_grouped_shap_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    exception_lines = "\n".join(
        "| {production_row_index} | {publication_group_id} | {absolute_reconstruction_error_pce_points:.12g} | "
        "{relative_reconstruction_error:.12g} |".format(**record)
        for record in reconstruction["exception_records"]
    ) or "| None | None | 0 | 0 |"
    report = f"""# Existing exact-interventional SHAP finalization report

## Outcome

- Top-level status: **PASS**.
- Reconstruction status: **{reconstruction['status']}**.
- Existing contributions modified: **no**.
- Rows excluded: **0**.
- Complete fitted cohort: {expected_rows:,} records from {int(publication_group_series.nunique()):,} publication groups.
- Exact-interventional spot check: {len(spot_positions):,} records; grouped maximum difference {grouped_spot_difference:.3g}; choice-bundle maximum difference {choice_spot_difference:.3g}.

## Governed reconstruction checks

- Cohort MAE: {float(reconstruction['cohort_mae_pce_points']):.12g} PCE points (limit {EXACT_COHORT_MAE_LIMIT:g}).
- Strict per-row threshold: {EXACT_STRICT_ABSOLUTE_THRESHOLD:g} PCE points; {int(reconstruction['strict_pass_count']):,}/{expected_rows:,} rows pass ({float(reconstruction['strict_pass_fraction']):.8%}; minimum {EXACT_MINIMUM_STRICT_PASS_FRACTION:.5%}).
- Maximum absolute residual: {float(reconstruction['maximum_absolute_error_pce_points']):.12g} PCE points (limit {EXACT_MAXIMUM_ABSOLUTE_ERROR:g}).
- Maximum relative residual: {float(reconstruction['maximum_relative_error']):.12g} (limit {EXACT_MAXIMUM_RELATIVE_ERROR:g}).
- Documented strict-threshold exceptions: {int(reconstruction['exception_count']):,} rows across {len(reconstruction['exception_publication_group_ids']):,} publication groups.

| Production row | Publication group | Absolute residual (PCE points) | Relative residual |
| ---: | --- | ---: | ---: |
{exception_lines}

## Numerical disclosure

{reconstruction['numerical_issue_disclosure']}

The finalizer rejected any mismatch in cohort order, population fingerprint, frozen mapping, model predictions, interventional background expectation, device-weighted importance, DOI-balanced importance, or the complete choice-table record/bundle grid before writing this report.
"""
    (out_dir / "pce_existing_shap_finalization_report.md").write_text(report, encoding="utf-8")
    refresh_interpretability_audit_after_finalization(
        out_dir,
        validation,
        data_path=data_path,
        manifest_path=manifest_path,
    )
    return validation


def cv_metrics(mod, df, target, features, group_col, cfg):
    groups = df[group_col].fillna("NO_GROUP").astype(str) if group_col else pd.Series(df.index.astype(str), index=df.index)
    pred = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=cfg.N_SPLITS).split(df, df[target], groups):
        Xtr, Xte = df.iloc[tr][features].copy(), df.iloc[te][features].copy()
        pre, _, _ = mod.make_preprocessor(Xtr, cfg)
        pipe = Pipeline([("pre", pre), ("model", mod.make_model("regression", cfg))])
        pipe.fit(Xtr, df.iloc[tr][target]); pred[te] = pipe.predict(Xte)
    y = df[target].to_numpy()
    return {"grouped_cv_r2": r2_score(y, pred), "grouped_cv_rmse": mean_squared_error(y, pred) ** .5,
            "grouped_cv_mae": mean_absolute_error(y, pred)}


def revise_representation(df: pd.DataFrame, features: list[str]):
    out, revised = df.copy(), list(features)
    provenance = []
    for site in ("a", "b", "c"):
        ion, coef = f"Perovskite_composition_{site}_ions", f"Perovskite_composition_{site}_ions_coefficients"
        parsed = [aligned_fractions(a, b) for a, b in zip(out[ion], out[coef])]
        tokens = sorted({k for d in parsed if d for k in d})
        for token in tokens:
            col = f"SCI_{site.upper()}_{token}_fraction"; out[col] = [d.get(token, 0.0) if d else np.nan for d in parsed]; revised.append(col)
        valid = f"SCI_{site.upper()}_composition_parse_valid"; out[valid] = [1.0 if d else 0.0 for d in parsed]; revised.append(valid)
        revised = [c for c in revised if c not in {ion, coef}]
        provenance.append({"representation": f"{site.upper()}-site", "raw_fields": f"{ion}; {coef}", "generated_features": len(tokens)+1,
                           "unambiguous_rows": sum(d is not None for d in parsed)})
    solvent_pairs = []
    for solv in features:
        if not solv.endswith("_deposition_solvents"):
            continue
        ratio = f"{solv}_mixing_ratios"
        if ratio in features:
            solvent_pairs.append((solv, ratio))
    for solv, ratio in solvent_pairs:
        parsed_s = [solvent_descriptor(a, b) for a, b in zip(out[solv], out[ratio])]
        prefix = solv.removesuffix("_deposition_solvents")
        col = f"SCI_{prefix}_solvent_system"
        out[col] = [x[0] for x in parsed_s]
        revised = [c for c in revised if c not in {solv, ratio}] + [col]
        provenance.append({"representation": f"{prefix.replace('_', ' ')} solvent system", "raw_fields": f"{solv}; {ratio}", "generated_features": 1,
                           "unambiguous_rows": sum(x[0] is not None for x in parsed_s)})
    for raw, stem in [("Perovskite_composition_inorganic", "Fully_inorganic_absorber"),
                      ("Perovskite_deposition_quenching_induced_crystallisation", "Quenching_induced_crystallization")]:
        txt = out[raw].astype(str).str.strip().str.casefold()
        out[f"SCI_{stem}"] = np.where(txt.isin(["true", "yes", "1"]), 1.0, np.where(txt.isin(["false", "no", "0"]), 0.0, np.nan))
        out[f"SCI_{stem}_missing"] = out[f"SCI_{stem}"].isna().astype(float)
        revised = [c for c in revised if c != raw] + [f"SCI_{stem}", f"SCI_{stem}_missing"]
    return out, revised, pd.DataFrame(provenance)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, default=DEFAULT_DATA); ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT); ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument(
        "--final-harmonized-contract",
        action="store_true",
        help=(
            "Use the final Figure 4 model snapshot, frozen 113/165 predictor typing, and "
            "categorical-schema harmonizer for a versioned full-cohort interpretability refit."
        ),
    )
    ap.add_argument(
        "--model-script",
        type=Path,
        default=None,
        help="Optional model-script override; the final snapshot is selected automatically with --final-harmonized-contract.",
    )
    ap.add_argument(
        "--feature-typing-inventory",
        type=Path,
        default=None,
        help="Optional frozen predictor-typing inventory; required by and defaulted for the final harmonized contract.",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Fit and validate the selected full-cohort model contract without computing SHAP values.",
    )
    ap.add_argument(
        "--shap-batch-size",
        type=int,
        default=256,
        help="Rows transformed and explained at once; all eligible rows are always explained.",
    )
    ap.add_argument(
        "--shap-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory for atomic per-batch SHAP shards. By default, final-harmonized runs use "
            "<out>/_shap_checkpoints; other runs checkpoint only when this option is supplied."
        ),
    )
    resume_group = ap.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-shap",
        dest="resume_shap",
        action="store_true",
        help="Validate the checkpoint manifest and reuse completed SHAP batch shards.",
    )
    resume_group.add_argument(
        "--no-resume-shap",
        dest="resume_shap",
        action="store_false",
        help="Start only in a checkpoint directory that contains no prior manifest or shards.",
    )
    ap.set_defaults(resume_shap=None)
    ap.add_argument(
        "--shap-method",
        choices=(
            "exact-interventional",
            "approximate-path-dependent",
            "exact-path-dependent",
        ),
        default="exact-interventional",
        help=(
            "TreeExplainer attribution method. Exact interventional TreeSHAP is the manuscript default; "
            "the path-dependent modes are retained for governed sensitivity/diagnostic runs."
        ),
    )
    ap.add_argument(
        "--interventional-background-rows",
        type=int,
        default=100,
        help="DOI/publication-group-diverse fitted records spanning observed PCE used as the exact interventional background.",
    )
    ap.add_argument(
        "--exact-agreement-rows",
        type=int,
        default=16,
        help=(
            "For approximate full-cohort SHAP, compare against exact TreeSHAP on this many "
            "deterministically selected DOI-diverse rows; use 0 only to skip the governed audit."
        ),
    )
    ap.add_argument(
        "--figure5-only",
        action="store_true",
        help="Regenerate the frozen-model Figure 5 SHAP tables and figures, then skip the encoding ablation.",
    )
    ap.add_argument(
        "--finalize-existing-shap",
        action="store_true",
        help=(
            "Validate already-written exact-interventional full-cohort SHAP tables against the current "
            "population, frozen mapping, model refit, and governed background; write fresh validation "
            "provenance without recomputing full-cohort SHAP."
        ),
    )
    ap.add_argument(
        "--finalize-spot-check-rows",
        type=int,
        default=16,
        help=(
            "DOI-diverse rows recomputed with exact-interventional TreeSHAP when finalizing existing tables; "
            "strict-threshold exception rows are always included in addition."
        ),
    )
    args = ap.parse_args()
    resume_shap = (
        bool(args.final_harmonized_contract or args.shap_checkpoint_dir is not None)
        if args.resume_shap is None
        else bool(args.resume_shap)
    )
    shap_checkpoint_dir = args.shap_checkpoint_dir
    if shap_checkpoint_dir is None and args.final_harmonized_contract and not args.preflight_only:
        shap_checkpoint_dir = args.out / "_shap_checkpoints"
    if resume_shap and shap_checkpoint_dir is None and not args.preflight_only:
        raise ValueError("--resume-shap requires --shap-checkpoint-dir outside the final harmonized contract")
    if args.final_harmonized_contract and args.out.resolve() == DEFAULT_OUT.resolve():
        raise ValueError(
            "The final harmonized run must use a new versioned output directory; "
            "the archived pre-harmonizer audit is immutable"
        )
    if args.final_harmonized_contract and not args.preflight_only:
        mixed_outputs = [
            name
            for name in (
                "pce_grouped_shap_values.csv",
                "pce_grouped_shap_importance.csv",
                "pce_choice_stratified_shap_values.csv",
                "pce_grouped_shap_validation.json",
            )
            if (args.out / name).exists()
        ]
        governed_resume_available = bool(
            resume_shap
            and shap_checkpoint_dir is not None
            and (shap_checkpoint_dir / SHAP_CHECKPOINT_MANIFEST_NAME).is_file()
        )
        if mixed_outputs and not (args.finalize_existing_shap or governed_resume_available):
            raise FileExistsError(
                "Refusing to mix a final harmonized SHAP run with existing attribution outputs: "
                f"{mixed_outputs}"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    shap_method = args.shap_method
    if args.finalize_existing_shap and shap_method != "exact-interventional":
        raise ValueError("--finalize-existing-shap only accepts --shap-method exact-interventional")
    if args.finalize_spot_check_rows < 1:
        raise ValueError("--finalize-spot-check-rows must be positive")
    model_script = args.model_script or (
        FINAL_MODEL_SCRIPT if args.final_harmonized_contract else MODEL_SCRIPT
    )
    typing_inventory_path = args.feature_typing_inventory or (
        DEFAULT_FINAL_TYPING if args.final_harmonized_contract else None
    )
    if not model_script.is_file():
        raise FileNotFoundError(model_script)
    if args.final_harmonized_contract and typing_inventory_path is None:
        raise ValueError("The final harmonized contract requires a frozen typing inventory")
    if typing_inventory_path is not None and not typing_inventory_path.is_file():
        raise FileNotFoundError(typing_inventory_path)
    if args.final_harmonized_contract:
        assert typing_inventory_path is not None
        validate_final_contract_source_hashes(
            model_script=model_script,
            feature_typing_inventory=typing_inventory_path,
            data=args.data,
            predictor_manifest=args.manifest,
        )
    mod = load_production_module(model_script)
    cfg, df, target, features, group_col = prepare_population(
        mod,
        args.data,
        args.manifest,
        args.out,
        harmonize_categorical_schema=args.final_harmonized_contract,
    )
    frozen_numeric = frozen_categorical = None
    if typing_inventory_path is not None:
        frozen_numeric, frozen_categorical = freeze_predictor_typing(
            mod, typing_inventory_path, features
        )
    # Preserve the source spellings and capitalization exclusively for figure
    # labels and row context.  The fitted model receives the separately
    # harmonized frame below; human-facing labels must not be derived from its
    # case-folded categorical representation.
    source_context_df = df
    if args.final_harmonized_contract:
        harmonization = load_harmonization_module()
        df = harmonization.harmonize(df, frozen_numeric, frozen_categorical)
    pipe, names, num, cat = fit_frozen_pipeline(mod, df, target, features, cfg)
    if len(features) != 278:
        raise RuntimeError(f"Production reproduction mismatch: raw={len(features)}")
    if args.final_harmonized_contract:
        if (len(num), len(cat)) != (113, 165):
            raise RuntimeError(
                f"Final harmonized typing mismatch: numeric={len(num)}, categorical={len(cat)}"
            )
        if num != frozen_numeric or cat != frozen_categorical:
            raise RuntimeError("Fitted preprocessing order differs from the frozen typing inventory")
        if len(names) != FINAL_EXPECTED_ENCODED_FEATURES:
            raise RuntimeError(
                "Final harmonized encoding mismatch: expected "
                f"{FINAL_EXPECTED_ENCODED_FEATURES:,} inputs, found {len(names):,}"
            )
    elif len(names) != 6292:
        raise RuntimeError(
            f"Archived pre-harmonizer reproduction mismatch: raw={len(features)}, encoded={len(names)}"
        )
    if not df.index.is_unique:
        raise RuntimeError("The complete fitted cohort does not have unique production row indices")

    contract_provenance = {
        "contract_name": (
            "final Figure 4 harmonized integrated full-cohort interpretability refit"
            if args.final_harmonized_contract
            else "archived pre-harmonizer full-cohort interpretability refit"
        ),
        "figure4_alignment": (
            "same frozen predictor order, 113/165 typing, categorical harmonizer, and Extra Trees "
            "hyperparameters used for final Figure 4; refitted on all eligible rows for interpretation"
            if args.final_harmonized_contract
            else "shares the raw-predictor manifest but predates the final Figure 4 categorical harmonizer"
        ),
        "is_one_of_figure4_evaluation_estimators": False,
        "full_cohort_post_evaluation_refit": True,
        "categorical_schema_harmonized": bool(args.final_harmonized_contract),
        "model_script": str(model_script.resolve()),
        "model_script_sha256": sha256_file(model_script),
        "feature_typing_inventory": (
            str(typing_inventory_path.resolve()) if typing_inventory_path is not None else None
        ),
        "feature_typing_inventory_sha256": (
            sha256_file(typing_inventory_path) if typing_inventory_path is not None else None
        ),
        "external_harmonization_script": (
            str(FINAL_HARMONIZATION_SCRIPT.resolve())
            if args.final_harmonized_contract
            else None
        ),
        "external_harmonization_script_sha256": (
            sha256_file(FINAL_HARMONIZATION_SCRIPT)
            if args.final_harmonized_contract
            else None
        ),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "predictor_manifest": str(args.manifest.resolve()),
        "predictor_manifest_sha256": sha256_file(args.manifest),
        "raw_feature_count": len(features),
        "numeric_raw_feature_count": len(num),
        "categorical_raw_feature_count": len(cat),
        "encoded_feature_count": len(names),
        "training_rows": len(df),
        "publication_group_count": int(
            df[group_col].fillna("NO_GROUP").astype(str).nunique() if group_col else len(df)
        ),
        "estimator": {
            "class": type(pipe.named_steps["model"]).__name__,
            "n_estimators": int(pipe.named_steps["model"].n_estimators),
            "min_samples_leaf": int(pipe.named_steps["model"].min_samples_leaf),
            "max_features": pipe.named_steps["model"].max_features,
            "max_depth": pipe.named_steps["model"].max_depth,
            "random_state": int(pipe.named_steps["model"].random_state),
        },
        "encoded_feature_names_sha256": hashlib.sha256(
            json.dumps(list(map(str, names)), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    preflight_predictions = None
    if args.preflight_only:
        preflight_predictions = np.asarray(pipe.predict(df[features].copy()), dtype=float)
        if not np.isfinite(preflight_predictions).all():
            raise RuntimeError("Preflight predictions contain non-finite values")

    rows = []
    boolean_raws: set[str] = set()
    for c in features:
        s, vc = df[c], df[c].dropna().astype(str).value_counts()
        boolean_like = is_boolean_predictor(s)
        if boolean_like and c not in num:
            boolean_raws.add(c)
        paired = ""
        if c.endswith("_coefficients"): paired = c.removesuffix("_coefficients")
        elif c.endswith("_mixing_ratios"): paired = c.removesuffix("_mixing_ratios")
        concern = "Requires paired identity field" if paired else ("Complementary Boolean encoding" if boolean_like else "")
        rows.append({"raw_predictor_name": c, "scientific_field_family": family(c), "type": "numeric" if c in num else "categorical/structured",
                     "included_in_production_model": "yes", "leakage_audit_classification": "allowed design/process input", "coverage": s.notna().mean(),
                     "missing_fraction": s.isna().mean(), "unique_value_count": s.nunique(dropna=True), "most_common_values": " | ".join(f"{k} ({v})" for k,v in vc.head(5).items()),
                     "most_common_level_prevalence_nonmissing": (vc.iloc[0]/vc.sum() if len(vc) else np.nan), "requires_another_field": "yes" if paired else "no",
                     "paired_dependent_field_name": paired, "interpretability_concern": concern, "recommended_treatment": "couple before scientific interpretation" if paired else "retain and group SHAP",
                     "reason_for_retention_or_exclusion": "retained by frozen leakage-audited production manifest"})
        if boolean_like and c not in num:
            rows[-1]["type"] = "Boolean (categorically encoded in production)"
    raw_inventory = pd.DataFrame(rows)
    if not args.finalize_existing_shap:
        raw_inventory.to_csv(args.out / "pce_raw_predictor_inventory.csv", index=False)

    dict_rows = []
    raw_for_name = []
    for i, name in enumerate(names):
        raw, transform, category = raw_from_encoded(name, num, cat); raw_for_name.append(raw)
        is_bool = raw in boolean_raws
        dict_rows.append({"encoded_feature_index": i, "exact_encoded_model_feature_name": name, "raw_source_predictor": raw,
                          "transformation_encoder": transform, "encoded_category_or_numerical_transformation": category,
                          "human_readable_scientific_label": f"{raw.replace('_',' ')}: {category}" if category else raw.replace('_',' '),
                          "scientific_grouping": scientific_group(raw), "prevalence_in_fitted_population": np.nan,
                          "is_boolean_complement": "yes" if is_bool else "no", "is_common_baseline_category": "no",
                          "scientifically_interpretable_by_itself": "no" if is_bool or raw.endswith(("_coefficients", "_mixing_ratios")) else "yes",
                          "required_context_field": raw.removesuffix("_coefficients") if raw.endswith("_coefficients") else (raw.removesuffix("_mixing_ratios") if raw.endswith("_mixing_ratios") else "")})
    dictionary = pd.DataFrame(dict_rows)
    if len(dictionary) != len(names): raise RuntimeError("Encoded dictionary is incomplete")

    mapping = dictionary[["encoded_feature_index","exact_encoded_model_feature_name","raw_source_predictor","scientific_grouping"]]
    group_names = list(dict.fromkeys(mapping["scientific_grouping"].astype(str)))
    group_matrix = _aggregation_matrix(mapping, group_names, len(names))
    choice_matrix, choice_definition_templates = _choice_aggregation_matrix(mapping, len(names))
    sample_idx = df.index.to_numpy(copy=True)
    sample = df.loc[sample_idx]
    interpretation_sample = source_context_df.loc[sample_idx]
    if group_col:
        publication_group_ids = [
            display_token(value) if clean_token(value) is not None else "NO_GROUP"
            for value in sample[group_col]
        ]
    else:
        publication_group_ids = [f"row:{row_index}" for row_index in sample_idx]
    population_fingerprint = compute_population_fingerprint(
        sample_idx,
        publication_group_ids,
        sample[target].to_numpy(dtype=float),
    )
    if args.final_harmonized_contract:
        observed_contract = {
            "training_rows": len(df),
            "publication_group_count": len(set(publication_group_ids)),
            "encoded_feature_count": len(names),
            "scientific_group_count": len(group_names),
            "population_fingerprint_sha256": population_fingerprint,
        }
        expected_contract = {
            "training_rows": FINAL_EXPECTED_TRAINING_ROWS,
            "publication_group_count": FINAL_EXPECTED_PUBLICATION_GROUPS,
            "encoded_feature_count": FINAL_EXPECTED_ENCODED_FEATURES,
            "scientific_group_count": FINAL_EXPECTED_SCIENTIFIC_GROUPS,
            "population_fingerprint_sha256": FINAL_EXPECTED_POPULATION_FINGERPRINT,
        }
        mismatches = {
            key: {"expected": expected_contract[key], "observed": observed_contract[key]}
            for key in expected_contract
            if expected_contract[key] != observed_contract[key]
        }
        if mismatches:
            raise RuntimeError(
                f"Final Figure 4 contract changed before SHAP computation: {mismatches}"
            )
    background_positions = None
    if shap_method == "exact-interventional":
        background_positions = select_interventional_background_positions(
            publication_group_ids,
            sample[target].to_numpy(dtype=float),
            args.interventional_background_rows,
        )
    if args.preflight_only:
        if args.final_harmonized_contract:
            expected_contract = {
                "training_rows": FINAL_EXPECTED_TRAINING_ROWS,
                "publication_group_count": FINAL_EXPECTED_PUBLICATION_GROUPS,
                "raw_feature_count": 278,
                "numeric_raw_feature_count": 113,
                "categorical_raw_feature_count": 165,
                "encoded_feature_count": FINAL_EXPECTED_ENCODED_FEATURES,
                "scientific_group_count": FINAL_EXPECTED_SCIENTIFIC_GROUPS,
                "population_fingerprint_sha256": FINAL_EXPECTED_POPULATION_FINGERPRINT,
            }
            observed_contract = {
                "training_rows": len(df),
                "publication_group_count": len(set(publication_group_ids)),
                "raw_feature_count": len(features),
                "numeric_raw_feature_count": len(num),
                "categorical_raw_feature_count": len(cat),
                "encoded_feature_count": len(names),
                "scientific_group_count": len(group_names),
                "population_fingerprint_sha256": population_fingerprint,
            }
            mismatches = {
                key: {"expected": expected_contract[key], "observed": observed_contract[key]}
                for key in expected_contract
                if expected_contract[key] != observed_contract[key]
            }
            if mismatches:
                raise RuntimeError(f"Final Figure 4 contract preflight mismatch: {mismatches}")
        if len(mapping) != len(names) or mapping["encoded_feature_index"].nunique() != len(names):
            raise RuntimeError("Preflight encoded-to-scientific-group mapping is incomplete")
        if background_positions is None:
            raise RuntimeError("Preflight requires the governed exact-interventional background")
        if len(background_positions) != args.interventional_background_rows:
            raise RuntimeError("Preflight background row count differs from the governed request")
        if pd.Series(publication_group_ids).iloc[background_positions].nunique() != len(
            background_positions
        ):
            raise RuntimeError("Preflight background is not publication-group diverse")
        repeat_pipe, repeat_names, repeat_num, repeat_cat = fit_frozen_pipeline(
            mod, df, target, features, cfg
        )
        repeat_predictions = np.asarray(
            repeat_pipe.predict(df[features].copy()), dtype=float
        )
        maximum_repeat_difference = float(
            np.max(np.abs(preflight_predictions - repeat_predictions))
        )
        if names != repeat_names or num != repeat_num or cat != repeat_cat:
            raise RuntimeError("Repeated preflight fit changed preprocessing structure")
        repeat_fit_tolerance = 1e-12
        if maximum_repeat_difference > repeat_fit_tolerance:
            raise RuntimeError(
                "Repeated seeded preflight fit changed predictions: "
                f"max absolute difference={maximum_repeat_difference}"
            )
        contract_provenance.update(
            {
                "status": "PASS",
                "population_fingerprint_sha256": population_fingerprint,
                "scientific_group_count": len(group_names),
                "encoded_inputs_mapped_exactly_once": True,
                "choice_bundle_count": int(choice_matrix.shape[0]),
                "prediction_sha256": hashlib.sha256(
                    preflight_predictions.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "prediction_min": float(preflight_predictions.min()),
                "prediction_max": float(preflight_predictions.max()),
                "prediction_mean": float(preflight_predictions.mean()),
                "repeat_fit_maximum_prediction_difference": maximum_repeat_difference,
                "repeat_fit_prediction_tolerance": repeat_fit_tolerance,
                "encoded_mapping_sha256": hashlib.sha256(
                    mapping.to_csv(index=False).encode("utf-8")
                ).hexdigest(),
                "interventional_background": {
                    "requested_rows": int(args.interventional_background_rows),
                    "background_rows": int(len(background_positions)),
                    "distinct_publications": int(
                        pd.Series(publication_group_ids).iloc[background_positions].nunique()
                    ),
                    "background_positions": background_positions.astype(int).tolist(),
                    "production_row_indices": sample_idx[background_positions].tolist(),
                    "observed_pce_min": float(sample[target].iloc[background_positions].min()),
                    "observed_pce_max": float(sample[target].iloc[background_positions].max()),
                },
            }
        )
        preflight_path = args.out / "pce_final_model_contract_preflight.json"
        preflight_path.write_text(
            json.dumps(contract_provenance, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(preflight_path), **contract_provenance}, indent=2))
        return
    if args.finalize_existing_shap:
        assert background_positions is not None
        validation = finalize_existing_exact_interventional_shap(
            out_dir=args.out,
            contract_provenance=contract_provenance,
            figure4_contract_aligned=args.final_harmonized_contract,
            pipe=pipe,
            df=df,
            target=target,
            features=features,
            names=names,
            numeric_raw_feature_count=len(num),
            categorical_raw_feature_count=len(cat),
            boolean_categorical_raw_feature_count=len(boolean_raws),
            interpretation_sample=interpretation_sample,
            dictionary=dictionary,
            mapping=mapping,
            group_names=group_names,
            group_matrix=group_matrix,
            choice_matrix=choice_matrix,
            choice_definition_templates=choice_definition_templates,
            sample_idx=sample_idx,
            sample=sample,
            publication_group_ids=publication_group_ids,
            background_positions=background_positions,
            batch_size=args.shap_batch_size,
            background_requested_rows=args.interventional_background_rows,
            spot_check_rows=args.finalize_spot_check_rows,
            data_path=args.data,
            manifest_path=args.manifest,
        )
        print(
            json.dumps(
                {
                    "output": str(args.out),
                    "status": validation["status"],
                    "reconstruction_status": validation["reconstruction_validation"]["status"],
                    "explained_rows": validation["explained_rows"],
                    "exception_count": validation["reconstruction_validation"]["exception_count"],
                    "maximum_absolute_error_pce_points": validation["reconstruction_validation"][
                        "maximum_absolute_error_pce_points"
                    ],
                    "scope": "finalize-existing-exact-interventional-shap",
                },
                indent=2,
            )
        )
        return
    checkpoint_identity = None
    if shap_checkpoint_dir is not None:
        checkpoint_identity = {
            "population_fingerprint_sha256": population_fingerprint,
            "target_values_sha256": _array_sha256(
                sample[target].to_numpy(dtype=np.float64)
            ),
            "publication_group_ids_sha256": _json_sha256(
                list(map(str, publication_group_ids))
            ),
            "fitted_model_sha256": fitted_tree_model_sha256(
                pipe.named_steps["model"]
            ),
            "source_sha256": {
                "model_script": contract_provenance["model_script_sha256"],
                "feature_typing_inventory": contract_provenance[
                    "feature_typing_inventory_sha256"
                ],
                "external_harmonization_script": contract_provenance[
                    "external_harmonization_script_sha256"
                ],
                "data": contract_provenance["data_sha256"],
                "predictor_manifest": contract_provenance[
                    "predictor_manifest_sha256"
                ],
            },
            "encoded_feature_names_sha256": contract_provenance[
                "encoded_feature_names_sha256"
            ],
            "encoded_mapping_sha256": hashlib.sha256(
                mapping.to_csv(index=False).encode("utf-8")
            ).hexdigest(),
            "scientific_group_names_sha256": _json_sha256(group_names),
            "group_aggregation_matrix_sha256": sparse_matrix_sha256(group_matrix),
            "choice_aggregation_matrix_sha256": sparse_matrix_sha256(choice_matrix),
            "choice_bundle_definitions_sha256": _json_sha256(
                choice_definition_templates
            ),
            "final_harmonized_contract": bool(args.final_harmonized_contract),
        }
    shap_result = compute_batched_grouped_shap(
        pipe,
        df[features].copy(),
        group_matrix,
        choice_matrix,
        batch_size=args.shap_batch_size,
        shap_method=shap_method,
        background_positions=background_positions,
        checkpoint_dir=shap_checkpoint_dir,
        resume=resume_shap,
        checkpoint_identity=checkpoint_identity,
    )
    grouped = shap_result["grouped_values"]
    bundle_values_all = shap_result["choice_values"]
    pred = shap_result["predictions"]
    expected = shap_result["expected_value"]
    recon = shap_result["grouped_reconstruction_mae"]
    fitted_prevalence = shap_result["encoded_prevalence"]
    categorical_mask = dictionary["encoded_category_or_numerical_transformation"].astype(str).ne("")
    dictionary.loc[categorical_mask, "prevalence_in_fitted_population"] = fitted_prevalence[categorical_mask.to_numpy()]
    dictionary["is_common_baseline_category"] = np.where(
        dictionary["prevalence_in_fitted_population"].ge(0.80), "yes", "no"
    )
    dictionary.to_csv(args.out / "pce_encoded_feature_dictionary.csv", index=False)

    importance = []
    for group_position, group in enumerate(group_names):
        part = mapping.loc[mapping["scientific_grouping"].eq(group)]
        vals = grouped[:, group_position]
        importance.append({"scientific_group": group, "mean_absolute_grouped_shap_pce_points": np.mean(np.abs(vals)),
                           "signed_mean_grouped_shap_pce_points": np.mean(vals), "encoded_input_count": len(part),
                           "encoded_inputs": " | ".join(part.exact_encoded_model_feature_name)})
    grouped_values = (
        pd.DataFrame(grouped, index=sample_idx, columns=group_names)
        .rename_axis("production_row_index")
        .reset_index()
    )
    grouped_values.to_csv(args.out / "pce_grouped_shap_values.csv", index=False)
    imp = pd.DataFrame(importance).sort_values(
        "mean_absolute_grouped_shap_pce_points", ascending=False
    )

    exact_agreement = None
    if shap_result["approximate"] and args.exact_agreement_rows > 0:
        exact_agreement, exact_comparison = compare_approximate_with_exact(
            pipe,
            df[features].copy(),
            group_matrix,
            group_names,
            sample_idx,
            publication_group_ids,
            pred,
            args.exact_agreement_rows,
        )
        exact_comparison.to_csv(args.out / "pce_shap_exact_agreement_by_group.csv", index=False)
    publication_group_series = pd.Series(publication_group_ids, name="publication_group_id")
    grouped_for_balance = pd.DataFrame(grouped, columns=group_names)
    grouped_for_balance["publication_group_id"] = publication_group_series.to_numpy()
    doi_abs = (
        grouped_for_balance.assign(**{name: grouped_for_balance[name].abs() for name in group_names})
        .groupby("publication_group_id", sort=False)[group_names]
        .mean()
        .mean(axis=0)
    )
    doi_signed = (
        grouped_for_balance.groupby("publication_group_id", sort=False)[group_names]
        .mean()
        .mean(axis=0)
    )
    doi_imp = pd.DataFrame(
        {
            "scientific_group": group_names,
            "doi_balanced_mean_absolute_grouped_shap_pce_points": [float(doi_abs[name]) for name in group_names],
            "doi_balanced_signed_mean_grouped_shap_pce_points": [float(doi_signed[name]) for name in group_names],
        }
    ).sort_values("doi_balanced_mean_absolute_grouped_shap_pce_points", ascending=False)
    imp = imp.merge(doi_imp, on="scientific_group", how="left", validate="one_to_one")
    imp.to_csv(args.out / "pce_grouped_shap_importance.csv", index=False)
    doi_imp.to_csv(args.out / "pce_grouped_shap_importance_doi_balanced.csv", index=False)
    row_importance = imp.set_index("scientific_group")["mean_absolute_grouped_shap_pce_points"].reindex(group_names)
    doi_importance = doi_imp.set_index("scientific_group")["doi_balanced_mean_absolute_grouped_shap_pce_points"].reindex(group_names)
    rank_spearman = float(spearmanr(row_importance, doi_importance).statistic)
    rank_kendall = float(kendalltau(row_importance, doi_importance).statistic)
    row_top20 = set(row_importance.nlargest(20).index)
    doi_top20 = set(doi_importance.nlargest(20).index)
    top20_overlap = len(row_top20.intersection(doi_top20))
    metadata = pd.DataFrame(
        {
            "production_row_index": sample_idx,
            "publication_group_id": publication_group_ids,
            "observed_pce": sample[target].to_numpy(dtype=float),
            "predicted_pce": pred,
            "base_value": expected,
            "solvent_identity": interpretation_sample["Perovskite_deposition_solvents"].map(display_token).to_numpy(),
            "solvent_mixing_ratio": interpretation_sample["Perovskite_deposition_solvents_mixing_ratios"].map(display_token).to_numpy(),
            "solvent_formulation": [
                solvent_formulation_label(solvent, ratio)
                for solvent, ratio in zip(
                    interpretation_sample["Perovskite_deposition_solvents"],
                    interpretation_sample["Perovskite_deposition_solvents_mixing_ratios"],
                )
            ],
            "context_solvent_formulation": [
                solvent_formulation_label(solvent, ratio)
                for solvent, ratio in zip(
                    interpretation_sample["Perovskite_deposition_solvents"],
                    interpretation_sample["Perovskite_deposition_solvents_mixing_ratios"],
                )
            ],
            "context_a_site_composition": [
                composition_label(ions, coefficients)
                for ions, coefficients in zip(
                    interpretation_sample["Perovskite_composition_a_ions"],
                    interpretation_sample["Perovskite_composition_a_ions_coefficients"],
                )
            ],
            "context_x_site_composition": [
                composition_label(ions, coefficients)
                for ions, coefficients in zip(
                    interpretation_sample["Perovskite_composition_c_ions"],
                    interpretation_sample["Perovskite_composition_c_ions_coefficients"],
                )
            ],
            "context_absorber_composition": [
                absorber_identity_label(long_form, short_form)
                for long_form, short_form in zip(
                    interpretation_sample["Perovskite_composition_long_form"],
                    interpretation_sample["Perovskite_composition_short_form"],
                )
            ],
            "context_etl": [
                joined_context(stack, procedure)
                for stack, procedure in zip(interpretation_sample["ETL_stack_sequence"], interpretation_sample["ETL_deposition_procedure"])
            ],
            "context_htl": [
                joined_context(stack, procedure)
                for stack, procedure in zip(interpretation_sample["HTL_stack_sequence"], interpretation_sample["HTL_deposition_procedure"])
            ],
            "context_quenching": interpretation_sample[
                "Perovskite_deposition_quenching_induced_crystallisation"
            ].map(display_token).to_numpy(),
            "context_cell_stack": interpretation_sample["Cell_stack_sequence"].map(display_token).to_numpy(),
            "context_back_contact": interpretation_sample["Backcontact_stack_sequence"].map(display_token).to_numpy(),
            "context_back_contact_deposition": interpretation_sample[
                "Backcontact_deposition_procedure"
            ].map(display_token).to_numpy(),
            "context_perovskite_short_form": interpretation_sample[
                "Perovskite_composition_short_form"
            ].map(display_token).to_numpy(),
            "context_annealing_time": [
                f"{value:g} min" if pd.notna(value) else "Not reported"
                for value in pd.to_numeric(
                    interpretation_sample["Perovskite_deposition_thermal_annealing_time"], errors="coerce"
                )
            ],
            "context_annealing_temperature": [
                f"{value:g} °C" if pd.notna(value) else "Not reported"
                for value in pd.to_numeric(
                    interpretation_sample["Perovskite_deposition_thermal_annealing_temperature"], errors="coerce"
                )
            ],
            **additional_row_context_metadata(interpretation_sample),
        }
    )
    metadata.to_csv(args.out / "pce_grouped_shap_row_metadata.csv", index=False)

    choice_rows: list[dict[str, object]] = []
    bundle_definitions: list[dict[str, object]] = []
    for bundle_position, template in enumerate(choice_definition_templates):
        feature_family = str(template["feature_family"])
        context_column = str(template["context_column"])
        raw_source_fields = tuple(str(field) for field in template["raw_source_fields"])
        bundle_values = bundle_values_all[:, bundle_position]
        bundle_definitions.append(
            {
                key: value for key, value in template.items() if key != "context_column"
            }
            | {"mean_absolute_bundle_shap_pce_points": float(np.mean(np.abs(bundle_values)))}
        )
        for position, row_index in enumerate(sample_idx):
            state, state_is_reported, state_is_valid = normalized_choice_state(
                feature_family,
                metadata.iloc[position][context_column],
            )
            choice_rows.append(
                {
                    "production_row_index": row_index,
                    "publication_group_id": publication_group_ids[position],
                    "feature_family": feature_family,
                    "state": state,
                    "bundle_shap_pce_points": float(bundle_values[position]),
                    "state_is_reported": state_is_reported,
                    "state_is_valid": state_is_valid,
                    "raw_source_fields": " | ".join(raw_source_fields),
                    "raw_source_field_count": len(raw_source_fields),
                    "encoded_input_count": int(template["encoded_input_count"]),
                }
            )
    choice_values = pd.DataFrame(choice_rows)
    choice_values.to_csv(args.out / "pce_choice_stratified_shap_values.csv", index=False)
    grouped_residuals = expected + grouped.sum(axis=1) - pred
    grouped_relative_errors = np.abs(grouped_residuals) / np.maximum(
        np.abs(pred),
        np.finfo(float).eps,
    )
    reconstruction_validation: dict[str, object] | None = None
    if shap_method == "exact-interventional":
        reconstruction_tolerance = EXACT_MAXIMUM_ABSOLUTE_ERROR
        reconstruction_validation = build_exact_interventional_reconstruction_validation(
            grouped_residuals,
            pred,
            sample_idx,
            publication_group_ids,
            shap_version=shap_result["shap_version"],
        )
        if shap_result["encoded_reconstruction_mae"] > EXACT_COHORT_MAE_LIMIT:
            raise RuntimeError("Encoded-input exact SHAP reconstruction MAE exceeds the governed limit")
        if shap_result["encoded_reconstruction_max_abs_error"] > EXACT_MAXIMUM_ABSOLUTE_ERROR:
            raise RuntimeError("Encoded-input exact SHAP maximum reconstruction error exceeds the governed limit")
    else:
        reconstruction_tolerance = 1e-8
        if shap_result["encoded_reconstruction_max_abs_error"] > reconstruction_tolerance:
            raise RuntimeError("Encoded-input SHAP values do not reconstruct model predictions")
        if shap_result["grouped_reconstruction_max_abs_error"] > reconstruction_tolerance:
            raise RuntimeError("Grouped SHAP values do not reconstruct model predictions")
    if len(grouped_values) != len(df) or not np.array_equal(grouped_values["production_row_index"], sample_idx):
        raise RuntimeError("Full-cohort SHAP output lost rows or changed fitted-population order")
    population_fingerprint = compute_population_fingerprint(
        sample_idx,
        publication_group_ids,
        sample[target].to_numpy(dtype=float),
    )
    if shap_method == "exact-interventional":
        method_label = (
            "exact TreeExplainer TreeSHAP (feature_perturbation='interventional', model_output='raw')"
        )
    elif shap_method == "exact-path-dependent":
        method_label = (
            "exact TreeExplainer TreeSHAP (feature_perturbation='tree_path_dependent')"
        )
    else:
        method_label = (
            "TreeExplainer path approximation (approximate=True; not exact Shapley TreeSHAP)"
        )
    background_validation = shap_result["background"]
    if background_validation is not None and background_positions is not None:
        background_validation = dict(background_validation)
        background_validation.update(
            {
                "requested_rows": int(args.interventional_background_rows),
                "selection_method": (
                    "one deterministic row per publication, then evenly spaced across observed fitted-cohort PCE"
                ),
                "distinct_publications": int(
                    pd.Series(publication_group_ids).iloc[background_positions].nunique()
                ),
                "production_row_indices": sample_idx[background_positions].tolist(),
                "observed_pce_min": float(sample[target].iloc[background_positions].min()),
                "observed_pce_max": float(sample[target].iloc[background_positions].max()),
            }
        )
    encoded_mapping_sha256 = hashlib.sha256(
        mapping.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    prediction_sha256 = hashlib.sha256(
        np.asarray(pred, dtype="<f8").tobytes()
    ).hexdigest()
    contract_provenance.update(
        {
            "status": "PASS",
            "population_fingerprint_sha256": population_fingerprint,
            "scientific_group_count": len(group_names),
            "encoded_inputs_mapped_exactly_once": True,
            "encoded_mapping_sha256": encoded_mapping_sha256,
            "prediction_sha256": prediction_sha256,
        }
    )
    validation = {
        "status": "PASS",
        "model_contract": contract_provenance,
        "model_contract_label": contract_provenance["contract_name"],
        "figure4_contract_aligned": bool(args.final_harmonized_contract),
        "categorical_schema_harmonization": bool(args.final_harmonized_contract),
        "raw_feature_count": len(features),
        "numeric_raw_feature_count": len(num),
        "categorical_raw_feature_count": len(cat),
        "boolean_categorical_raw_feature_count": len(boolean_raws),
        "analysis_population": "complete final fitted-training cohort",
        "explanation_scope": "in-sample explanation of the final model refit on every eligible row",
        "out_of_fold_explanation": False,
        "training_rows": len(df),
        "fitted_training_rows": len(df),
        "population_rows": len(df),
        "aggregation_method": (
            "sum signed encoded-input SHAP contributions within each source-level scientific group "
            "for every explained row, then rank groups by mean absolute contribution"
        ),
        "explained_rows": len(grouped_values),
        "explanation_coverage_fraction": len(grouped_values) / len(df),
        "sampling_method": "none",
        "publication_group_count": int(publication_group_series.nunique()),
        "population_fingerprint_sha256": population_fingerprint,
        "scientific_group_count": len(group_names),
        "encoded_feature_count": len(names),
        "encoded_features_mapped_exactly_once": True,
        "encoded_feature_names_sha256": contract_provenance[
            "encoded_feature_names_sha256"
        ],
        "encoded_mapping_sha256": encoded_mapping_sha256,
        "model_prediction_sha256": prediction_sha256,
        "shap_method": method_label,
        "shap_version": shap_result["shap_version"],
        "shap_batch_size": shap_result["batch_size"],
        "shap_batch_count": shap_result["batch_count"],
        "shap_checkpoint": shap_result["checkpoint"],
        "exact_tree_shap": shap_result["exact_tree_shap"],
        "feature_perturbation": (
            "interventional" if shap_method == "exact-interventional" else "tree_path_dependent"
        ),
        "interventional_background": background_validation,
        "reconstruction_validation_tolerance": reconstruction_tolerance,
        "base_value_pce_points": expected,
        "prediction_reconstruction_mae": shap_result["encoded_reconstruction_mae"],
        "prediction_reconstruction_max_abs_error": shap_result["encoded_reconstruction_max_abs_error"],
        "prediction_reconstruction_max_relative_error": float(grouped_relative_errors.max()),
        "grouped_prediction_reconstruction_mae": shap_result["grouped_reconstruction_mae"],
        "grouped_prediction_reconstruction_max_abs_error": shap_result["grouped_reconstruction_max_abs_error"],
        "grouped_prediction_reconstruction_max_relative_error": float(grouped_relative_errors.max()),
        "reconstruction_mae": shap_result["grouped_reconstruction_mae"],
        "importance_display_group_count": 20,
        "importance_full_table_group_count": len(group_names),
        "doi_balanced_importance": {
            "publication_weighting": "equal weight per publication after averaging absolute row contributions within publication",
            "rank_spearman_vs_device_weighted": rank_spearman,
            "rank_kendall_vs_device_weighted": rank_kendall,
            "top20_overlap_count": top20_overlap,
            "top20_overlap_fraction": top20_overlap / 20,
        },
        "choice_bundle_count": len(bundle_definitions),
        "choice_value_rows": len(choice_values),
        "choice_state_selection": (
            "within each predefined feature family, select the three most prevalent reported valid states "
            "with at least five explained rows; selection does not use SHAP magnitude"
        ),
        "choice_bundle_definitions": bundle_definitions,
        "interpretation_note": (
            "These are in-sample attributions for the final model fitted on the complete eligible cohort. "
            "They explain model predictions, are not out-of-fold performance evidence, and do not prove causality."
        ),
        "approximation_disclosure": (
            "The fast path approximation was used for complete-cohort feasibility; reconstruction verifies additivity "
            "but does not make the approximation exact."
            if shap_result["approximate"]
            else (
                "Exact interventional TreeSHAP was used with a deterministic DOI/publication-group-diverse background "
                "spanning the observed fitted-cohort PCE range."
                if shap_method == "exact-interventional"
                else "Exact path-dependent TreeSHAP was used."
            )
        ),
    }
    if reconstruction_validation is not None:
        validation["reconstruction_validation"] = reconstruction_validation
    if exact_agreement is not None:
        validation["approximate_vs_exact_agreement_audit"] = exact_agreement
        if exact_agreement["status"] != "PASS":
            validation["status"] = "REVIEW_REQUIRED"
    elif shap_result["exact_tree_shap"]:
        (args.out / "pce_shap_exact_agreement_by_group.csv").unlink(missing_ok=True)
    (args.out / "pce_grouped_shap_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    if args.final_harmonized_contract:
        reconstruction = validation.get("reconstruction_validation", {})
        report = f"""# Final harmonized PCE grouped-SHAP audit

## Model contract

- Contract: {contract_provenance['contract_name']}.
- Relationship to Figure 4: {contract_provenance['figure4_alignment']}.
- Model script: `{contract_provenance['model_script']}` (`{contract_provenance['model_script_sha256']}`).
- Frozen typing inventory: `{contract_provenance['feature_typing_inventory']}` (`{contract_provenance['feature_typing_inventory_sha256']}`).
- Integrated data: `{contract_provenance['data']}` (`{contract_provenance['data_sha256']}`).
- Predictor manifest: `{contract_provenance['predictor_manifest']}` (`{contract_provenance['predictor_manifest_sha256']}`).
- Population: {len(df):,} fitted records in {int(publication_group_series.nunique()):,} DOI/publication groups.
- Representation: {len(features)} raw predictors ({len(num)} numeric; {len(cat)} categorical), {len(names):,} encoded inputs, and {len(group_names)} source-level scientific groups.
- Estimator: Extra Trees with 300 trees, minimum leaf size 2, square-root feature subsampling, unlimited depth, and random seed 42.
- Categorical-schema harmonization: enabled before and within the frozen preprocessing pipeline.

## Attribution contract

- {method_label}.
- Every fitted-training record was explained ({len(grouped_values):,}/{len(df):,}); no explanation sampling or row exclusion was used.
- Signed encoded-input contributions were summed within their source-level scientific groups before importance ranking.
- The deterministic interventional background contains {int(background_validation['background_rows']):,} rows from {int(background_validation['distinct_publications']):,} distinct DOI/publication groups.
- Reconstruction status: {reconstruction.get('status', 'not applicable')}; cohort MAE {float(shap_result['grouped_reconstruction_mae']):.12g} PCE points; maximum absolute residual {float(shap_result['grouped_reconstruction_max_abs_error']):.12g} PCE points.
- These are in-sample fitted-model attributions, not out-of-fold performance estimates, substitution effects, or causal effects.

## Current outputs

- `pce_raw_predictor_inventory.csv`
- `pce_encoded_feature_dictionary.csv`
- `pce_grouped_shap_values.csv`
- `pce_grouped_shap_importance.csv`
- `pce_grouped_shap_importance_doi_balanced.csv`
- `pce_grouped_shap_row_metadata.csv`
- `pce_choice_stratified_shap_values.csv`
- `pce_grouped_shap_validation.json`
"""
        (args.out / "pce_final_harmonized_shap_audit.md").write_text(
            report, encoding="utf-8"
        )
    cleanup_stale_plot_outputs(args.out)
    if args.figure5_only:
        print(
            json.dumps(
                {
                    "output": str(args.out),
                    "raw": len(features),
                    "encoded": len(names),
                    "choice_bundles": len(bundle_definitions),
                    "choice_rows": len(choice_values),
                    "shap_reconstruction_mae": recon,
                    "explained_rows": len(grouped_values),
                    "sampling_method": "none",
                    "shap_method": method_label,
                    "scope": "figure5-only",
                },
                indent=2,
            )
        )
        return

    enc_importance = shap_result["encoded_mean_absolute_shap"]; prevalence_rows = []
    for _, r in dictionary.iterrows():
        p = r.prevalence_in_fitted_population
        if pd.notna(p) and (p >= .80 or p <= .01):
            if r.required_context_field:
                role = "redundant/conditional representation requiring its paired identity field"
            elif p >= .80:
                role = "common baseline whose absence may identify an alternative group"
            else:
                role = "rare or near-constant encoded level; independent information uncertain"
            prevalence_rows.append({**r.to_dict(), "variance_frequency": p*(1-p), "mean_absolute_shap": enc_importance[int(r.encoded_feature_index)],
                                    "related_fields": r.required_context_field, "feature_role_assessment": role,
                                    "recommendation": "retain for model pending ablation; suppress standalone headline and interpret within scientific group"})
    pd.DataFrame(prevalence_rows).to_csv(args.out / "pce_feature_prevalence_audit.csv", index=False)

    revised_df, revised_features, parse_audit = revise_representation(df, features); parse_audit.to_csv(args.out / "pce_structured_parse_audit.csv", index=False)
    revised_pipe, revised_names, *_ = fit_frozen_pipeline(mod, revised_df, target, revised_features, cfg)
    base_metrics = {} if args.skip_cv else cv_metrics(mod, df, target, features, group_col, cfg)
    rev_metrics = {} if args.skip_cv else cv_metrics(mod, revised_df, target, revised_features, group_col, cfg)
    groups_n = int(df[group_col].fillna("NO_GROUP").astype(str).nunique()) if group_col else len(df)
    ablation = pd.DataFrame([{"representation":"production baseline", "raw_feature_count":len(features), "encoded_feature_count":len(names), "training_rows":len(df), "doi_groups":groups_n, **base_metrics},
                             {"representation":"scientifically coupled candidate", "raw_feature_count":len(revised_features), "encoded_feature_count":len(revised_names), "training_rows":len(df), "doi_groups":groups_n, **rev_metrics}])
    ablation.to_csv(args.out / "pce_encoding_ablation_comparison.csv", index=False)
    metric_table = "\n".join(["| " + " | ".join(ablation.columns) + " |",
                              "| " + " | ".join(["---"] * len(ablation.columns)) + " |"] +
                             ["| " + " | ".join(str(v) for v in row) + " |" for row in ablation.itertuples(index=False, name=None)])

    back_raw = df["Backcontact_deposition_procedure"]
    back = back_raw.astype(str).str.casefold().str.contains("evaporation", na=False).mean()
    back_nonmissing = back_raw.dropna().astype(str).str.casefold().str.contains("evaporation", na=False).mean()
    back_exact = back_raw.astype(str).str.strip().str.casefold().eq("evaporation").mean()
    back_exact_nonmissing = back_raw.dropna().astype(str).str.strip().str.casefold().eq("evaporation").mean()
    inorganic_raw = df["Perovskite_composition_inorganic"]
    inorganic_false = inorganic_raw.astype(str).str.casefold().eq("false").mean()
    inorganic_false_nonmissing = inorganic_raw.dropna().astype(str).str.casefold().eq("false").mean()
    coef1 = df["Perovskite_composition_a_ions_coefficients"].astype(str).str.strip().isin(["1","1.0"])
    chem = df.loc[coef1, "Perovskite_composition_a_ions"].value_counts().head(20)
    report = f"""# PCE feature interpretability audit

## Reproduction

- Production script: `{MODEL_SCRIPT}`
- Integrated fitted dataset: `{args.data}`
- Frozen leakage-audited manifest: `{args.manifest}`
- Population: {len(df):,} rows; {groups_n:,} DOI/groups; target `{target}`.
- Exact production inventory: {len(features)} raw predictors ({len(num)} numerical; {len(cat)} categorical), {len(names):,} encoded inputs.
- The fitted pipeline resolves 113 numeric and 165 categorical columns. The manuscript's 117/161 split is a semantic reconstruction rather than the fitted preprocessor split and is not reproducible from the archived estimator path; this factual classification should be reconciled before publication.
- Estimator: Extra Trees, 300 estimators, min_samples_leaf=2, max_features=sqrt, seed 42; five-fold DOI-aware GroupKFold.
- Reproduced complete-cohort SHAP attributions for all {len(df):,} fitted training rows in deterministic batches; no explanation sampling was used.
- Attribution method: {method_label}. Grouped prediction reconstruction MAE = {recon:.3g}.
- The SHAP cohort is the final model's in-sample fitted population. DOI-aware grouped cross-validation remains the performance evaluation; these attributions are not out-of-fold predictions.

## Findings and corrections

- Back-contact deposition exactly `Evaporation`: {back_exact:.6%} of all fitted rows and {back_exact_nonmissing:.6%} of non-missing rows. Any category string containing evaporation (including multilayer strings): {back:.6%} of all rows and {back_nonmissing:.6%} of non-missing rows. The fitted one-hot `Evaporation` level is {float(dictionary.loc[dictionary.exact_encoded_model_feature_name.eq('Backcontact_deposition_procedure_Evaporation'), 'prevalence_in_fitted_population'].iloc[0]):.6%} after production most-frequent imputation. It is retained as a common baseline/alternative-group discriminator, not presented as a mechanistic standalone conclusion.
- Fully inorganic = false: {inorganic_false:.6%} of all exact fitted rows and {inorganic_false_nonmissing:.6%} of non-missing rows. Its two one-hot complements are combined in the main grouped interpretation; the candidate encoding uses one 0/1 value plus an explicit missingness flag.
- Boolean metadata is derived from the exact fitted values: {len(boolean_raws)} categorical raw predictors map to {int(dictionary['is_boolean_complement'].eq('yes').sum())} complementary encoded levels. Those levels remain available to the validated model but are marked non-standalone and aggregated at source level for interpretation.
- A-site coefficient exactly 1 or 1.0: {coef1.mean():.6%} of fitted rows. Chemistry distribution (top values): {json.dumps(chem.to_dict(), ensure_ascii=False)}. Coefficients are never presented alone: baseline SHAP is grouped with ion identity, while the candidate representation parses only unambiguous aligned semicolon lists into normalized ion-specific fractions.
- Solvent ratio 1: represented as `<solvent> only` only when exactly one aligned solvent is present. Multi-solvent systems require equal-length solvent and ratio lists; layered (`|`) and sequential (`>>`) strings are left unknown rather than guessed.
- Original raw strings remain unchanged in the fitted dataframe and are named in `pce_structured_parse_audit.csv` for provenance.
- No production feature was removed on prevalence alone. The production model remains the manuscript model unless the ablation table demonstrates non-material degradation and the candidate representation is separately approved.

## Model comparison

{metric_table}

Historical-domain and LiteratureAgent-domain frozen-holdout identities were not available as row-level manifests in the archived production package; no substitute split was invented. The grouped-CV comparison uses identical population, DOI assignments, target, preprocessing safeguards, estimator parameters, and seed.

## Outputs

- `pce_raw_predictor_inventory.csv`
- `pce_encoded_feature_dictionary.csv`
- `pce_feature_prevalence_audit.csv`
- `pce_grouped_shap_values.csv`
- `pce_grouped_shap_importance.csv`
- `pce_grouped_shap_importance_doi_balanced.csv`
- `pce_grouped_shap_row_metadata.csv`
- `pce_choice_stratified_shap_values.csv`
- `pce_choice_stratified_shap_summary.csv`
- `pce_grouped_shap_validation.json`
- `pce_solvent_formulation_shap_values.csv`
- `pce_solvent_formulation_shap_summary.csv`
- `pce_grouped_shap_local_random_selection.json`
- `pce_encoding_ablation_comparison.csv`
- Publication figures are rendered once by `build_curated_publication_figures.py`
  from the audited source tables above.

## Manuscript changes required (not applied automatically)

- Methods: distinguish the complete 218-group audit from the six narrow, predefined choice bundles used for the main figure.
- Results: discuss actual prevalent material and process states without treating fitted-model attribution as a substitution effect.
- Figure 5 caption: state that all {len(df):,} final-model training records were explained, the display shows the top 20 of {len(group_names)} source groups, and ETL/HTL bundles contain only stack sequence and deposition procedure.
- Caption/methods: identify the attributions as in-sample, disclose `{method_label}`, distinguish device-weighted from DOI-balanced importance, and avoid causal language.
"""
    (args.out / "pce_feature_interpretability_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.out), "raw":len(features), "encoded":len(names), "revised_encoded":len(revised_names), "shap_reconstruction_mae":recon}, indent=2))


if __name__ == "__main__": main()
