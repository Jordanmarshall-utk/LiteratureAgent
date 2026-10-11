#!/usr/bin/env python3
"""Render publication-ready grouped-SHAP figures from frozen source tables."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from manuscript_plot_style import (
    BLUE,
    GRID,
    INK,
    NEUTRAL as PALETTE_NEUTRAL,
    ORANGE,
    PanelSpec,
    PRINT_RCPARAMS,
    export_panel,
    mm_to_inches,
)


NEGATIVE = BLUE
POSITIVE = ORANGE
NEUTRAL = PALETTE_NEUTRAL
BLACK = INK
SOLVENT_GROUP = "Perovskite precursor solvent formulation"
CHOICE_STEM = "figure_pce_choice_stratified_shap"
GLOBAL_STEM = "figure_pce_grouped_shap_beeswarm"
SOLVENT_STEM = "figure_pce_solvent_formulation_shap"
LOCAL_STEM = "figure_pce_grouped_shap_local_random"
DEFAULT_TOP_COUNT = 20
STRICT_RECONSTRUCTION_THRESHOLD = 1e-5
MAXIMUM_RECONSTRUCTION_MAE = 1e-6
MINIMUM_STRICT_PASS_FRACTION = 0.9999
MAXIMUM_RECONSTRUCTION_ERROR = 0.005
MAXIMUM_RELATIVE_RECONSTRUCTION_ERROR = 2e-4
FINAL_HARMONIZED_CONTRACT_NAME = (
    "final Figure 4 harmonized integrated full-cohort interpretability refit"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FROZEN_FINAL_CONTRACT: dict[str, object] = {
    "contract_name": FINAL_HARMONIZED_CONTRACT_NAME,
    "training_rows": 21_434,
    "publication_group_count": 4_085,
    "raw_feature_count": 278,
    "numeric_raw_feature_count": 113,
    "categorical_raw_feature_count": 165,
    "encoded_feature_count": 6_263,
    "scientific_group_count": 218,
    "choice_bundle_count": 6,
    "importance_display_group_count": 20,
    "background_rows": 100,
    "background_distinct_publications": 100,
    "population_fingerprint_sha256": (
        "a932e882ecb4f37c015e4e06de66856ebeda18eaee4915d3fcd14eed30538f9e"
    ),
    "model_script_sha256": (
        "19f58acdfb6143a75df5d034be08f1de070d43713b716cba6c77d4b422898115"
    ),
    "feature_typing_inventory_sha256": (
        "f2a193a2818279bb889145d91ebe438cf7ae173e39e0dba71c6d28145219ebdc"
    ),
    "external_harmonization_script_sha256": (
        "1c7373ed8a85448afc9ba7edb30c1f314347bc1168121dc9d539adb33386df60"
    ),
    "data_sha256": (
        "88d26d9c46c5142f546dd640db6e091713a7dc38b19399bb7887cdd751b599db"
    ),
    "predictor_manifest_sha256": (
        "56a7c34a370df4aee5cd83d2bea2e56dc3d3d5b4c1d46e7518dacd108afc9524"
    ),
    "encoded_feature_names_sha256": (
        "82ae1df9c7bde2ba0dd82c006e67fdd8a3438f60d1cb5052f4c3377772fe5f7f"
    ),
    "encoded_mapping_sha256": (
        "ecaf496b6ad06e0006eec8702a8963299e119c1b82d5adfffa15cf82c98844ca"
    ),
}
CHOICE_FAMILY_ORDER = (
    "Precursor solvent formulation",
    "ETL stack and deposition",
    "HTL stack and deposition",
    "Absorber composition identity",
    "Quenching-induced crystallization",
    "Back-contact material",
)
CHOICE_FAMILY_PREFIX = {
    "Precursor solvent formulation": "Solvent",
    "ETL stack and deposition": "ETL",
    "HTL stack and deposition": "HTL",
    "Absorber composition identity": "Absorber",
    "Quenching-induced crystallization": "Quenching",
    "Back-contact material": "Back contact",
}

COMMON_FORMULAS = re.compile(
    r"(?<![A-Za-z0-9])(?:MAPbI3|CsPbBr3|CsPbI3|TiO2|SnO2|MoO3|C60)(?![A-Za-z0-9])"
)
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/", re.IGNORECASE)
FORMULA_MATHTEXT = {
    "MAPbI3": r"MAPbI$_3$",
    "CsPbBr3": r"CsPbBr$_3$",
    "CsPbI3": r"CsPbI$_3$",
    "TiO2": r"TiO$_2$",
    "SnO2": r"SnO$_2$",
    "MoO3": r"MoO$_3$",
    "C60": r"C$_{60}$",
}

PANEL_SPECS = {
    CHOICE_STEM: PanelSpec(
        panel_id="figure5",
        stem=CHOICE_STEM,
        title="Recorded states and fitted-model PCE attribution",
        width_mm=180,
        height_mm=155,
        figure="Figure5",
        renderer="choice_shap",
    ),
    GLOBAL_STEM: PanelSpec(
        panel_id="figureS5_global",
        stem=GLOBAL_STEM,
        title="Global grouped SHAP audit",
        width_mm=180,
        height_mm=160,
        figure="FigureS1",
        role="supplementary",
        renderer="global_shap",
    ),
    SOLVENT_STEM: PanelSpec(
        panel_id="figureS5_solvent",
        stem=SOLVENT_STEM,
        title="Solvent-formulation attribution",
        width_mm=180,
        height_mm=130,
        figure="FigureS2",
        role="supplementary",
        renderer="solvent_shap",
    ),
    LOCAL_STEM: PanelSpec(
        panel_id="figureS5_local",
        stem=LOCAL_STEM,
        title="Fixed-seed single-record explanation",
        width_mm=180,
        height_mm=125,
        figure="FigureS3",
        role="supplementary",
        renderer="local_shap",
    ),
}


def validate_reconstruction_governance_contract(
    validation: dict[str, object],
    *,
    explained_rows: int,
) -> dict[str, object]:
    """Validate the declared exact-interventional additivity governance."""

    governance = validation.get("reconstruction_validation")
    if not isinstance(governance, dict):
        raise ValueError("Exact-interventional SHAP lacks nested reconstruction governance")
    required = {
        "status",
        "strict_absolute_threshold_pce_points",
        "cohort_mae_limit_pce_points",
        "minimum_strict_pass_fraction",
        "maximum_absolute_error_limit_pce_points",
        "maximum_relative_error_limit",
        "relative_error_denominator",
        "cohort_mae_pce_points",
        "maximum_absolute_error_pce_points",
        "maximum_relative_error",
        "strict_pass_count",
        "strict_pass_fraction",
        "exception_count",
        "exception_fraction",
        "exception_production_row_indices",
        "exception_publication_group_ids",
        "rows_excluded",
        "contributions_modified",
        "numerical_issue_disclosure",
    }
    missing = sorted(required.difference(governance))
    if missing:
        raise ValueError(f"Exact-interventional reconstruction governance is incomplete: {missing}")
    expected_limits = {
        "strict_absolute_threshold_pce_points": STRICT_RECONSTRUCTION_THRESHOLD,
        "cohort_mae_limit_pce_points": MAXIMUM_RECONSTRUCTION_MAE,
        "minimum_strict_pass_fraction": MINIMUM_STRICT_PASS_FRACTION,
        "maximum_absolute_error_limit_pce_points": MAXIMUM_RECONSTRUCTION_ERROR,
        "maximum_relative_error_limit": MAXIMUM_RELATIVE_RECONSTRUCTION_ERROR,
    }
    for field, expected_value in expected_limits.items():
        if not np.isclose(float(governance[field]), expected_value, rtol=0, atol=1e-15):
            raise ValueError(f"Exact-interventional reconstruction governance changed {field}")
    if str(governance["relative_error_denominator"]) != "max(abs(predicted_pce), machine_epsilon)":
        raise ValueError("Exact-interventional relative-error denominator is not governed")

    exception_rows = list(governance["exception_production_row_indices"])
    exception_publications = list(governance["exception_publication_group_ids"])
    exception_count = int(governance["exception_count"])
    strict_pass_count = int(governance["strict_pass_count"])
    if explained_rows < 1 or strict_pass_count + exception_count != explained_rows:
        raise ValueError("Exact-interventional strict-pass and exception counts are inconsistent")
    if exception_count != len(exception_rows) or len(set(exception_rows)) != exception_count:
        raise ValueError("Exact-interventional exception-row lineage is incomplete or duplicated")
    if len(set(exception_publications)) != len(exception_publications):
        raise ValueError("Exact-interventional exception-publication lineage is duplicated")
    if exception_count == 0 and exception_publications:
        raise ValueError("Exact-interventional exception publications exist without exception rows")
    if exception_count > 0 and not exception_publications:
        raise ValueError("Exact-interventional exception rows lack publication lineage")

    strict_fraction = strict_pass_count / explained_rows
    exception_fraction = exception_count / explained_rows
    if not np.isclose(float(governance["strict_pass_fraction"]), strict_fraction, rtol=0, atol=1e-12):
        raise ValueError("Exact-interventional strict-pass fraction is inconsistent")
    if not np.isclose(float(governance["exception_fraction"]), exception_fraction, rtol=0, atol=1e-12):
        raise ValueError("Exact-interventional exception fraction is inconsistent")
    expected_status = (
        "PASS" if exception_count == 0 else "PASS_WITH_DOCUMENTED_NUMERICAL_TOLERANCE"
    )
    if str(governance["status"]) != expected_status:
        raise ValueError("Exact-interventional reconstruction status is inconsistent")
    if int(governance["rows_excluded"]) != 0 or governance["contributions_modified"] is not False:
        raise ValueError("Exact-interventional SHAP rows or contributions were altered")
    disclosure = str(governance["numerical_issue_disclosure"]).casefold()
    if not all(token in disclosure for token in ("float32", "tree", "boundary", "not redistributed")):
        raise ValueError("Exact-interventional reconstruction numerical disclosure is incomplete")

    mae = float(governance["cohort_mae_pce_points"])
    maximum_absolute = float(governance["maximum_absolute_error_pce_points"])
    maximum_relative = float(governance["maximum_relative_error"])
    if not all(np.isfinite(value) and value >= 0 for value in (mae, maximum_absolute, maximum_relative)):
        raise ValueError("Exact-interventional reconstruction metrics must be finite and nonnegative")
    if not (
        mae <= MAXIMUM_RECONSTRUCTION_MAE
        and strict_fraction >= MINIMUM_STRICT_PASS_FRACTION
        and maximum_absolute <= MAXIMUM_RECONSTRUCTION_ERROR
        and maximum_relative <= MAXIMUM_RELATIVE_RECONSTRUCTION_ERROR
    ):
        raise ValueError("Exact-interventional SHAP failed governed reconstruction limits")
    tolerance = float(validation.get("reconstruction_validation_tolerance", np.nan))
    if not np.isclose(tolerance, MAXIMUM_RECONSTRUCTION_ERROR, rtol=0, atol=1e-15):
        raise ValueError("Exact-interventional reconstruction ceiling is inconsistent")
    return governance


def validate_final_harmonized_model_contract(
    validation: dict[str, object],
    *,
    _expected_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Require Figure 5 to use the final harmonized Figure 4 model contract.

    The encoded-input count is deliberately read from the fitted validation
    record and checked against the frozen preprocessing result.  The private
    expectation injection exists only so focused tests can use miniature
    tables; production callers must omit it.
    """

    expected = FROZEN_FINAL_CONTRACT if _expected_contract is None else dict(_expected_contract)
    required_expectations = {
        "contract_name",
        "training_rows",
        "publication_group_count",
        "raw_feature_count",
        "numeric_raw_feature_count",
        "categorical_raw_feature_count",
        "encoded_feature_count",
        "scientific_group_count",
        "choice_bundle_count",
        "importance_display_group_count",
        "background_rows",
        "background_distinct_publications",
        "population_fingerprint_sha256",
        "model_script_sha256",
        "feature_typing_inventory_sha256",
        "external_harmonization_script_sha256",
        "data_sha256",
        "predictor_manifest_sha256",
        "encoded_feature_names_sha256",
        "encoded_mapping_sha256",
    }
    missing_expectations = sorted(required_expectations.difference(expected))
    if missing_expectations:
        raise ValueError(
            f"Frozen Figure 5 expectation inventory is incomplete: {missing_expectations}"
        )

    required_validation = {
        "model_contract",
        "model_contract_label",
        "figure4_contract_aligned",
        "categorical_schema_harmonization",
        "training_rows",
        "publication_group_count",
        "raw_feature_count",
        "numeric_raw_feature_count",
        "categorical_raw_feature_count",
        "encoded_feature_count",
        "scientific_group_count",
        "population_fingerprint_sha256",
        "encoded_mapping_sha256",
        "model_prediction_sha256",
        "choice_bundle_count",
        "importance_display_group_count",
    }
    missing_validation = sorted(required_validation.difference(validation))
    if missing_validation:
        raise ValueError(
            "Final harmonized Figure 5 validation is missing fields: "
            f"{missing_validation}"
        )

    contract = validation["model_contract"]
    if not isinstance(contract, dict):
        raise ValueError("Final harmonized Figure 5 model_contract must be an object")
    required_contract = {
        "contract_name",
        "figure4_alignment",
        "is_one_of_figure4_evaluation_estimators",
        "full_cohort_post_evaluation_refit",
        "categorical_schema_harmonized",
        "model_script",
        "model_script_sha256",
        "feature_typing_inventory",
        "feature_typing_inventory_sha256",
        "external_harmonization_script",
        "external_harmonization_script_sha256",
        "data",
        "data_sha256",
        "predictor_manifest",
        "predictor_manifest_sha256",
        "raw_feature_count",
        "numeric_raw_feature_count",
        "categorical_raw_feature_count",
        "encoded_feature_count",
        "training_rows",
        "publication_group_count",
        "estimator",
        "encoded_feature_names_sha256",
        "status",
        "population_fingerprint_sha256",
        "scientific_group_count",
        "encoded_inputs_mapped_exactly_once",
        "encoded_mapping_sha256",
        "prediction_sha256",
    }
    missing_contract = sorted(required_contract.difference(contract))
    if missing_contract:
        raise ValueError(
            "Final harmonized Figure 5 model contract is missing fields: "
            f"{missing_contract}"
        )

    contract_name = str(contract["contract_name"]).strip()
    if contract_name != str(expected["contract_name"]):
        raise ValueError("Figure 5 does not identify the frozen final harmonized contract")
    if str(validation["model_contract_label"]).strip() != contract_name:
        raise ValueError("Figure 5 model-contract label disagrees with its nested provenance")
    if validation["figure4_contract_aligned"] is not True:
        raise ValueError("Figure 5 is not marked as aligned to the final Figure 4 contract")
    if validation["categorical_schema_harmonization"] is not True:
        raise ValueError("Figure 5 did not enable the final categorical-schema harmonizer")
    if contract["categorical_schema_harmonized"] is not True:
        raise ValueError("Nested Figure 5 provenance does not confirm schema harmonization")
    if contract["full_cohort_post_evaluation_refit"] is not True:
        raise ValueError("Figure 5 must be a post-evaluation full-cohort refit")
    if contract["is_one_of_figure4_evaluation_estimators"] is not False:
        raise ValueError(
            "Figure 5 must distinguish its all-row refit from the held-out Figure 4 estimators"
        )
    alignment = str(contract["figure4_alignment"]).strip().casefold()
    if not all(
        token in alignment
        for token in ("figure 4", "113/165", "categorical harmonizer", "refitted")
    ):
        raise ValueError("Figure 5 does not fully state how its refit aligns with Figure 4")
    if str(contract["status"]).strip().upper() != "PASS":
        raise ValueError("Figure 5 model contract provenance did not pass")
    if contract["encoded_inputs_mapped_exactly_once"] is not True:
        raise ValueError("Figure 5 model contract does not confirm complete encoded mapping")

    raw_count = int(validation["raw_feature_count"])
    numeric_count = int(validation["numeric_raw_feature_count"])
    categorical_count = int(validation["categorical_raw_feature_count"])
    encoded_count = int(validation["encoded_feature_count"])
    expected_dimensions = (
        int(expected["raw_feature_count"]),
        int(expected["numeric_raw_feature_count"]),
        int(expected["categorical_raw_feature_count"]),
        int(expected["encoded_feature_count"]),
        int(expected["training_rows"]),
        int(expected["publication_group_count"]),
        int(expected["scientific_group_count"]),
    )
    observed_dimensions = (
        raw_count,
        numeric_count,
        categorical_count,
        encoded_count,
        int(validation["training_rows"]),
        int(validation["publication_group_count"]),
        int(validation["scientific_group_count"]),
    )
    if observed_dimensions != expected_dimensions:
        raise ValueError(
            "Figure 5 dimensions differ from the frozen final contract: "
            f"expected={expected_dimensions}, observed={observed_dimensions}"
        )
    if (raw_count, numeric_count, categorical_count) != expected_dimensions[:3]:
        raise ValueError("Figure 5 does not use the frozen 278-feature, 113/165 typing contract")
    if numeric_count + categorical_count != raw_count or encoded_count < raw_count:
        raise ValueError("Figure 5 raw/encoded feature accounting is invalid")
    for field, expected_value in (
        ("raw_feature_count", raw_count),
        ("numeric_raw_feature_count", numeric_count),
        ("categorical_raw_feature_count", categorical_count),
        ("encoded_feature_count", encoded_count),
        ("training_rows", int(validation["training_rows"])),
        ("publication_group_count", int(validation["publication_group_count"])),
        ("scientific_group_count", int(validation["scientific_group_count"])),
    ):
        if int(contract[field]) != expected_value:
            raise ValueError(f"Figure 5 nested model contract disagrees on {field}")

    estimator = contract["estimator"]
    if not isinstance(estimator, dict):
        raise ValueError("Figure 5 model contract lacks estimator settings")
    required_estimator = {
        "class",
        "n_estimators",
        "min_samples_leaf",
        "max_features",
        "max_depth",
        "random_state",
    }
    if not required_estimator <= set(estimator):
        raise ValueError("Figure 5 model contract has incomplete estimator settings")
    if (
        str(estimator["class"]) != "ExtraTreesRegressor"
        or int(estimator["n_estimators"]) != 300
        or int(estimator["min_samples_leaf"]) != 2
        or str(estimator["max_features"]).casefold() != "sqrt"
        or estimator["max_depth"] is not None
        or int(estimator["random_state"]) != 42
    ):
        raise ValueError("Figure 5 estimator settings differ from the final Figure 4 contract")

    for field in (
        "model_script",
        "feature_typing_inventory",
        "external_harmonization_script",
        "data",
        "predictor_manifest",
    ):
        if not str(contract[field]).strip():
            raise ValueError(f"Figure 5 model contract has an empty source path: {field}")
    for field in (
        "model_script_sha256",
        "feature_typing_inventory_sha256",
        "external_harmonization_script_sha256",
        "data_sha256",
        "predictor_manifest_sha256",
        "encoded_feature_names_sha256",
        "population_fingerprint_sha256",
        "encoded_mapping_sha256",
        "prediction_sha256",
    ):
        if SHA256_PATTERN.fullmatch(str(contract[field]).strip().casefold()) is None:
            raise ValueError(f"Figure 5 model contract has an invalid SHA-256 value: {field}")

    pinned_hash_fields = (
        "model_script_sha256",
        "feature_typing_inventory_sha256",
        "external_harmonization_script_sha256",
        "data_sha256",
        "predictor_manifest_sha256",
        "encoded_feature_names_sha256",
        "population_fingerprint_sha256",
        "encoded_mapping_sha256",
    )
    for field in pinned_hash_fields:
        if str(contract[field]).strip().casefold() != str(expected[field]).strip().casefold():
            raise ValueError(f"Figure 5 frozen provenance hash changed: {field}")

    if str(contract["population_fingerprint_sha256"]) != str(
        validation["population_fingerprint_sha256"]
    ):
        raise ValueError("Figure 5 population fingerprint disagrees with its model contract")
    if str(contract["encoded_mapping_sha256"]) != str(validation["encoded_mapping_sha256"]):
        raise ValueError("Figure 5 encoded mapping fingerprint disagrees with its model contract")
    if str(contract["prediction_sha256"]) != str(validation["model_prediction_sha256"]):
        raise ValueError("Figure 5 prediction fingerprint disagrees with its model contract")
    if int(validation["choice_bundle_count"]) != int(expected["choice_bundle_count"]):
        raise ValueError("Figure 5 choice-bundle count differs from the frozen contract")
    if int(validation["importance_display_group_count"]) != int(
        expected["importance_display_group_count"]
    ):
        raise ValueError("Figure 5 display-group count differs from the frozen contract")

    repeat_difference = contract.get("repeat_fit_maximum_prediction_difference")
    if repeat_difference is not None:
        repeat_difference = float(repeat_difference)
        repeat_tolerance = float(contract.get("repeat_fit_prediction_tolerance", np.nan))
        if (
            not np.isfinite(repeat_difference)
            or repeat_difference < 0
            or not np.isfinite(repeat_tolerance)
            or repeat_tolerance <= 0
            or repeat_difference > repeat_tolerance
            or repeat_tolerance > 1e-12
        ):
            raise ValueError("Figure 5 repeat-fit predictions exceed the numerical equivalence gate")
    return contract


def validate_canonical_shap_contract(
    validation: dict[str, object],
    *,
    explained_rows: int | None = None,
    top_count: int | None = None,
    _expected_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Reject noncanonical Figure 5 provenance before any figure is written."""

    expected = FROZEN_FINAL_CONTRACT if _expected_contract is None else dict(_expected_contract)

    required = {
        "status",
        "training_rows",
        "explained_rows",
        "explanation_coverage_fraction",
        "sampling_method",
        "explanation_scope",
        "out_of_fold_explanation",
        "shap_method",
        "exact_tree_shap",
        "feature_perturbation",
        "interventional_background",
        "prediction_reconstruction_mae",
        "prediction_reconstruction_max_abs_error",
        "prediction_reconstruction_max_relative_error",
        "grouped_prediction_reconstruction_mae",
        "grouped_prediction_reconstruction_max_abs_error",
        "grouped_prediction_reconstruction_max_relative_error",
        "importance_display_group_count",
        "importance_full_table_group_count",
        "choice_bundle_count",
        "choice_value_rows",
        "choice_bundle_definitions",
    }
    missing = sorted(required.difference(validation))
    if missing:
        raise ValueError(f"Canonical grouped-SHAP validation is missing fields: {missing}")

    model_contract = validate_final_harmonized_model_contract(
        validation,
        _expected_contract=expected,
    )
    training_rows = int(validation["training_rows"])
    validated_rows = int(validation["explained_rows"])
    if str(validation["status"]).strip().upper() != "PASS":
        raise ValueError("Canonical grouped-SHAP validation status is not PASS")
    if training_rows < 1 or validated_rows != training_rows:
        raise ValueError("Canonical Figure 5 must explain every fitted-training row")
    if explained_rows is not None and validated_rows != int(explained_rows):
        raise ValueError("Grouped-SHAP table length does not match the canonical explained-row count")
    if not np.isclose(
        float(validation["explanation_coverage_fraction"]),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Canonical Figure 5 requires complete explanation coverage")
    if str(validation["sampling_method"]).strip().casefold() != "none":
        raise ValueError("Canonical Figure 5 does not permit a sampled explanation cohort")
    if validation["out_of_fold_explanation"] is not False or "in-sample" not in str(
        validation["explanation_scope"]
    ).casefold():
        raise ValueError("Canonical Figure 5 must identify final-refit attribution as in-sample")
    if validation["exact_tree_shap"] is not True:
        raise ValueError("Canonical Figure 5 requires exact TreeSHAP")
    method = str(validation["shap_method"]).strip().casefold()
    if "exact" not in method or "interventional" not in method:
        raise ValueError("Canonical Figure 5 requires exact interventional TreeSHAP provenance")
    if str(validation["feature_perturbation"]).strip().casefold() != "interventional":
        raise ValueError("Canonical Figure 5 requires interventional feature perturbation")
    reconstruction_governance = validate_reconstruction_governance_contract(
        validation,
        explained_rows=validated_rows,
    )
    bounded_top_level_metrics = {
        "prediction_reconstruction_mae": MAXIMUM_RECONSTRUCTION_MAE,
        "prediction_reconstruction_max_abs_error": MAXIMUM_RECONSTRUCTION_ERROR,
        "prediction_reconstruction_max_relative_error": MAXIMUM_RELATIVE_RECONSTRUCTION_ERROR,
        "grouped_prediction_reconstruction_mae": MAXIMUM_RECONSTRUCTION_MAE,
        "grouped_prediction_reconstruction_max_abs_error": MAXIMUM_RECONSTRUCTION_ERROR,
        "grouped_prediction_reconstruction_max_relative_error": MAXIMUM_RELATIVE_RECONSTRUCTION_ERROR,
    }
    for field, limit in bounded_top_level_metrics.items():
        value = float(validation[field])
        if not np.isfinite(value) or value < 0 or value > limit:
            raise ValueError(f"Canonical Figure 5 reconstruction metric failed: {field}")
    for field, governance_field in (
        ("grouped_prediction_reconstruction_mae", "cohort_mae_pce_points"),
        ("grouped_prediction_reconstruction_max_abs_error", "maximum_absolute_error_pce_points"),
        ("grouped_prediction_reconstruction_max_relative_error", "maximum_relative_error"),
    ):
        if not np.isclose(
            float(validation[field]),
            float(reconstruction_governance[governance_field]),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"Canonical Figure 5 nested governance disagrees with {field}")

    background = validation["interventional_background"]
    if not isinstance(background, dict):
        raise ValueError("Canonical Figure 5 lacks a governed interventional background")
    required_background = {
        "background_rows",
        "requested_rows",
        "background_positions",
        "production_row_indices",
        "distinct_publications",
        "selection_method",
        "feature_perturbation",
        "observed_pce_min",
        "observed_pce_max",
    }
    missing_background = sorted(required_background.difference(background))
    if missing_background:
        raise ValueError(
            f"Governed interventional background is missing fields: {missing_background}"
        )
    background_rows = int(background["background_rows"])
    positions = list(background["background_positions"])
    production_rows = list(background["production_row_indices"])
    expected_background_rows = int(expected["background_rows"])
    if (
        background_rows != expected_background_rows
        or int(background["requested_rows"]) != expected_background_rows
    ):
        raise ValueError("Governed interventional background row count is invalid")
    if len(positions) != background_rows or len(set(positions)) != background_rows:
        raise ValueError("Governed interventional background positions are incomplete or duplicated")
    if len(production_rows) != background_rows or len(set(production_rows)) != background_rows:
        raise ValueError("Governed interventional background production rows are incomplete or duplicated")
    distinct_publications = int(background["distinct_publications"])
    if distinct_publications != int(expected["background_distinct_publications"]):
        raise ValueError("Governed interventional background publication coverage is invalid")
    if not str(background["selection_method"]).strip():
        raise ValueError("Governed interventional background selection method is empty")
    if str(background["feature_perturbation"]).strip().casefold() != "interventional":
        raise ValueError("Governed background does not record interventional perturbation")

    display_count = int(validation["importance_display_group_count"])
    expected_display_count = int(expected["importance_display_group_count"])
    if display_count != expected_display_count:
        raise ValueError(
            f"Canonical Figure 5 must display the top {expected_display_count} scientific groups"
        )
    if top_count is not None and int(top_count) != display_count:
        raise ValueError("Requested display count differs from the canonical validated display count")
    return {
        "model_contract": model_contract,
        "training_rows": training_rows,
        "encoded_feature_count": int(validation["encoded_feature_count"]),
        "explained_rows": validated_rows,
        "background_rows": background_rows,
        "display_group_count": display_count,
        "reconstruction_status": str(reconstruction_governance["status"]),
        "reconstruction_exception_count": int(reconstruction_governance["exception_count"]),
    }


def validate_preflight_against_final(
    preflight: dict[str, object],
    validation: dict[str, object],
    *,
    _expected_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Cross-check the frozen-model preflight against final SHAP provenance.

    Parallel tree prediction can differ in the last floating-point bits, so the
    two prediction hashes are each required to be valid but are intentionally
    not required to be identical.  All structural, source, population,
    mapping, estimator, and background-lineage fields must match exactly.
    """

    expected = FROZEN_FINAL_CONTRACT if _expected_contract is None else dict(_expected_contract)
    contract = validate_final_harmonized_model_contract(
        validation,
        _expected_contract=expected,
    )
    required_preflight = {
        "contract_name",
        "figure4_alignment",
        "is_one_of_figure4_evaluation_estimators",
        "full_cohort_post_evaluation_refit",
        "categorical_schema_harmonized",
        "model_script",
        "model_script_sha256",
        "feature_typing_inventory",
        "feature_typing_inventory_sha256",
        "external_harmonization_script",
        "external_harmonization_script_sha256",
        "data",
        "data_sha256",
        "predictor_manifest",
        "predictor_manifest_sha256",
        "raw_feature_count",
        "numeric_raw_feature_count",
        "categorical_raw_feature_count",
        "encoded_feature_count",
        "training_rows",
        "publication_group_count",
        "estimator",
        "encoded_feature_names_sha256",
        "status",
        "population_fingerprint_sha256",
        "scientific_group_count",
        "encoded_inputs_mapped_exactly_once",
        "choice_bundle_count",
        "prediction_sha256",
        "repeat_fit_maximum_prediction_difference",
        "repeat_fit_prediction_tolerance",
        "encoded_mapping_sha256",
        "interventional_background",
    }
    missing = sorted(required_preflight.difference(preflight))
    if missing:
        raise ValueError(f"Final Figure 5 preflight is incomplete: {missing}")

    exact_common_fields = (
        "contract_name",
        "figure4_alignment",
        "is_one_of_figure4_evaluation_estimators",
        "full_cohort_post_evaluation_refit",
        "categorical_schema_harmonized",
        "model_script",
        "model_script_sha256",
        "feature_typing_inventory",
        "feature_typing_inventory_sha256",
        "external_harmonization_script",
        "external_harmonization_script_sha256",
        "data",
        "data_sha256",
        "predictor_manifest",
        "predictor_manifest_sha256",
        "raw_feature_count",
        "numeric_raw_feature_count",
        "categorical_raw_feature_count",
        "encoded_feature_count",
        "training_rows",
        "publication_group_count",
        "estimator",
        "encoded_feature_names_sha256",
        "status",
        "population_fingerprint_sha256",
        "scientific_group_count",
        "encoded_inputs_mapped_exactly_once",
        "encoded_mapping_sha256",
    )
    changed = [field for field in exact_common_fields if preflight[field] != contract[field]]
    if changed:
        raise ValueError(
            f"Final Figure 5 preflight and SHAP validation disagree: {changed}"
        )
    if int(preflight["choice_bundle_count"]) != int(validation["choice_bundle_count"]):
        raise ValueError("Final Figure 5 preflight choice-bundle count changed")

    preflight_prediction_hash = str(preflight["prediction_sha256"]).strip().casefold()
    final_prediction_hash = str(contract["prediction_sha256"]).strip().casefold()
    if (
        SHA256_PATTERN.fullmatch(preflight_prediction_hash) is None
        or SHA256_PATTERN.fullmatch(final_prediction_hash) is None
    ):
        raise ValueError("Final Figure 5 preflight/final prediction hash is invalid")
    repeat_difference = float(preflight["repeat_fit_maximum_prediction_difference"])
    repeat_tolerance = float(preflight["repeat_fit_prediction_tolerance"])
    if (
        not np.isfinite(repeat_difference)
        or repeat_difference < 0
        or not np.isfinite(repeat_tolerance)
        or repeat_tolerance <= 0
        or repeat_tolerance > 1e-12
        or repeat_difference > repeat_tolerance
    ):
        raise ValueError("Final Figure 5 preflight repeat-fit gate did not pass")

    preflight_background = preflight["interventional_background"]
    final_background = validation.get("interventional_background")
    if not isinstance(preflight_background, dict) or not isinstance(final_background, dict):
        raise ValueError("Final Figure 5 preflight/final background provenance is missing")
    exact_background_fields = (
        "requested_rows",
        "background_rows",
        "distinct_publications",
        "background_positions",
        "production_row_indices",
    )
    missing_background = sorted(
        set(exact_background_fields + ("observed_pce_min", "observed_pce_max"))
        .difference(preflight_background)
    )
    if missing_background:
        raise ValueError(
            f"Final Figure 5 preflight background is incomplete: {missing_background}"
        )
    changed_background = [
        field
        for field in exact_background_fields
        if preflight_background[field] != final_background.get(field)
    ]
    for field in ("observed_pce_min", "observed_pce_max"):
        final_value = final_background.get(field)
        if final_value is None or not np.isclose(
            float(preflight_background[field]), float(final_value), rtol=0, atol=1e-12
        ):
            changed_background.append(field)
    if changed_background:
        raise ValueError(
            "Final Figure 5 preflight and SHAP background lineage disagree: "
            f"{changed_background}"
        )
    return {
        "status": "PASS",
        "prediction_hashes_identical": preflight_prediction_hash == final_prediction_hash,
        "repeat_fit_maximum_prediction_difference": repeat_difference,
        "background_rows": int(preflight_background["background_rows"]),
    }


def configure_style() -> None:
    parameters = dict(PRINT_RCPARAMS)
    parameters.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )
    plt.rcParams.update(parameters)


def fitted_cohort_scope(
    validation: dict[str, object],
    explained_rows: int,
    *,
    _expected_contract: Mapping[str, object] | None = None,
) -> tuple[str, int | None, bool]:
    """Return the canonical final-refit scope after enforcing its provenance."""

    contract = validate_canonical_shap_contract(
        validation,
        explained_rows=explained_rows,
        _expected_contract=_expected_contract,
    )
    return (
        "Final model refit on the full training cohort; in-sample attribution (not out-of-fold)",
        int(contract["training_rows"]),
        True,
    )


def _population_fingerprint(metadata: pd.DataFrame) -> str:
    frame = pd.DataFrame(
        {
            "production_row_index": metadata["production_row_index"].astype(str).to_numpy(),
            "publication_group_id": metadata["publication_group_id"].astype(str).to_numpy(),
            "observed_pce": metadata["observed_pce"].to_numpy(dtype=float),
        }
    )
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    ).hexdigest()


def _prediction_fingerprint(predictions: pd.Series | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(predictions, dtype="<f8").tobytes()).hexdigest()


def _metadata_lineage_from_csv(path: Path) -> pd.DataFrame:
    """Parse lineage floats with Python's exact decimal round-trip parser.

    Pandas' fast CSV parser may choose an adjacent float for some shortest
    round-trip decimal strings.  Reading these four fields with ``float``
    reproduces the exact arrays hashed before the audit wrote its CSV.
    """

    required = {
        "production_row_index",
        "publication_group_id",
        "observed_pce",
        "predicted_pce",
    }
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Grouped SHAP metadata CSV lacks lineage fields: {missing}")
        for row in reader:
            rows.append(
                {
                    "production_row_index": str(row["production_row_index"]),
                    "publication_group_id": str(row["publication_group_id"]),
                    "observed_pce": float(row["observed_pce"]),
                    "predicted_pce": float(row["predicted_pce"]),
                }
            )
    return pd.DataFrame(rows)


def _deterministic_background_positions(
    publication_group_ids: pd.Series,
    observed_pce: pd.Series,
    requested_rows: int,
) -> list[int]:
    candidates = pd.DataFrame(
        {
            "position": np.arange(len(observed_pce), dtype=int),
            "publication_group_id": publication_group_ids.astype(str).to_numpy(),
            "observed_pce": observed_pce.to_numpy(dtype=float),
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
    return candidates.iloc[selected]["position"].astype(int).tolist()


def validate_tables(
    importance: pd.DataFrame,
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    validation: dict[str, object],
    *,
    metadata_source_path: Path | None = None,
) -> dict[str, object]:
    required_importance = {
        "scientific_group",
        "mean_absolute_grouped_shap_pce_points",
        "signed_mean_grouped_shap_pce_points",
        "encoded_input_count",
    }
    missing = required_importance.difference(importance.columns)
    if missing:
        raise ValueError(f"Importance table is missing columns: {sorted(missing)}")
    if "production_row_index" not in values or "production_row_index" not in metadata:
        raise ValueError("Values and metadata must contain production_row_index")
    if values["production_row_index"].duplicated().any() or metadata["production_row_index"].duplicated().any():
        raise ValueError("production_row_index must be unique")
    value_rows = values["production_row_index"].astype(int).tolist()
    metadata_rows = metadata["production_row_index"].astype(int).tolist()
    if value_rows != metadata_rows:
        raise ValueError("Grouped values and row metadata are not aligned")
    groups = [column for column in values.columns if column != "production_row_index"]
    reported_groups = importance["scientific_group"].astype(str).tolist()
    if set(groups) != set(reported_groups):
        raise ValueError("Grouped values and importance tables describe different groups")
    indexed = importance.set_index("scientific_group")
    numeric_group_values = values[groups].to_numpy(dtype=float)
    if not np.isfinite(numeric_group_values).all():
        raise ValueError("Grouped SHAP values contain non-finite contributions")
    importance_numeric_columns = (
        "mean_absolute_grouped_shap_pce_points",
        "signed_mean_grouped_shap_pce_points",
        "encoded_input_count",
    )
    if not np.isfinite(
        importance.loc[:, importance_numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Grouped SHAP importance contains non-finite values")
    recomputed_absolute = values[groups].abs().mean(axis=0).loc[reported_groups]
    recomputed_signed = values[groups].mean(axis=0).loc[reported_groups]
    maximum_absolute_difference = float(
        np.max(
            np.abs(
                recomputed_absolute.to_numpy()
                - indexed.loc[reported_groups, "mean_absolute_grouped_shap_pce_points"].to_numpy()
            )
        )
    )
    maximum_signed_difference = float(
        np.max(
            np.abs(
                recomputed_signed.to_numpy()
                - indexed.loc[reported_groups, "signed_mean_grouped_shap_pce_points"].to_numpy()
            )
        )
    )
    if maximum_absolute_difference > 1e-12 or maximum_signed_difference > 1e-12:
        raise ValueError("Grouped importance does not reproduce from per-row contributions")
    if "explained_rows" not in validation:
        raise ValueError(
            "Validation must record explained_rows; legacy sample_rows-only SHAP tables are not supported"
        )
    explained_rows = int(validation["explained_rows"])
    if explained_rows != len(values):
        raise ValueError("Validation row count does not match grouped values")
    if SOLVENT_GROUP not in groups:
        raise ValueError(f"Required scientific group is missing: {SOLVENT_GROUP}")
    if int(validation.get("scientific_group_count", -1)) != len(groups):
        raise ValueError("Validation scientific-group count does not match grouped values")
    if int(validation.get("importance_full_table_group_count", -1)) != len(groups):
        raise ValueError("Validation full importance-table count does not match grouped values")
    encoded_input_counts = pd.to_numeric(importance["encoded_input_count"], errors="raise")
    if (
        not encoded_input_counts.gt(0).all()
        or int(encoded_input_counts.sum()) != int(validation["encoded_feature_count"])
    ):
        raise ValueError("Grouped encoded-input counts do not cover the frozen encoded inventory")
    required_metadata = {
        "observed_pce",
        "predicted_pce",
        "base_value",
        "publication_group_id",
        "solvent_formulation",
        "context_fully_inorganic",
        "context_perovskite_long_form",
        "context_add_desc_text_length",
        "context_b_site_composition",
        "context_substrate_stack_sequence",
        "context_lead_free",
        "context_perovskite_deposition_procedure",
        "context_perovskite_additives",
        "context_perovskite_synthesis_atmosphere",
    }
    missing_metadata = required_metadata.difference(metadata.columns)
    if missing_metadata:
        raise ValueError(f"Row metadata is missing columns: {sorted(missing_metadata)}")
    predictions = metadata["predicted_pce"].to_numpy(dtype=float)
    metadata_numeric = metadata[["observed_pce", "predicted_pce", "base_value"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(metadata_numeric).all():
        raise ValueError("Grouped SHAP row metadata contain non-finite model values")
    lineage_metadata = (
        _metadata_lineage_from_csv(metadata_source_path)
        if metadata_source_path is not None
        else metadata[
            [
                "production_row_index",
                "publication_group_id",
                "observed_pce",
                "predicted_pce",
            ]
        ].copy()
    )
    if len(lineage_metadata) != len(metadata):
        raise ValueError("Grouped SHAP metadata CSV lineage row count changed")
    if lineage_metadata["production_row_index"].astype(str).tolist() != metadata[
        "production_row_index"
    ].astype(str).tolist():
        raise ValueError("Grouped SHAP metadata CSV production-row lineage changed")
    if lineage_metadata["publication_group_id"].astype(str).tolist() != metadata[
        "publication_group_id"
    ].astype(str).tolist():
        raise ValueError("Grouped SHAP metadata CSV publication lineage changed")
    population_fingerprint = _population_fingerprint(lineage_metadata)
    if population_fingerprint != str(validation["population_fingerprint_sha256"]):
        raise ValueError("Grouped SHAP population fingerprint does not reproduce from row metadata")
    prediction_fingerprint = _prediction_fingerprint(lineage_metadata["predicted_pce"])
    if prediction_fingerprint != str(validation["model_prediction_sha256"]):
        raise ValueError("Grouped SHAP prediction fingerprint does not reproduce from row metadata")

    background = validation["interventional_background"]
    positions = [int(value) for value in background["background_positions"]]
    if any(position < 0 or position >= explained_rows for position in positions):
        raise ValueError("Interventional background position lies outside the explained cohort")
    background_metadata = lineage_metadata.iloc[positions]
    recorded_background_rows = [str(value) for value in background["production_row_indices"]]
    actual_background_rows = background_metadata["production_row_index"].astype(str).tolist()
    if recorded_background_rows != actual_background_rows:
        raise ValueError("Interventional background production-row lineage does not reproduce")
    distinct_background_publications = int(
        background_metadata["publication_group_id"].astype(str).nunique()
    )
    if distinct_background_publications != int(background["distinct_publications"]):
        raise ValueError("Interventional background publication lineage does not reproduce")
    expected_positions = _deterministic_background_positions(
        lineage_metadata["publication_group_id"],
        lineage_metadata["observed_pce"],
        int(background["requested_rows"]),
    )
    if positions != expected_positions:
        raise ValueError("Interventional background positions do not reproduce deterministic selection")
    for field, actual_value in (
        ("observed_pce_min", float(background_metadata["observed_pce"].min())),
        ("observed_pce_max", float(background_metadata["observed_pce"].max())),
    ):
        if not np.isclose(float(background[field]), actual_value, rtol=0, atol=1e-12):
            raise ValueError(f"Interventional background {field} does not reproduce")
    reconstructed = metadata["base_value"].to_numpy(dtype=float) + values[groups].sum(axis=1).to_numpy(dtype=float)
    residuals = reconstructed - predictions
    absolute_errors = np.abs(residuals)
    relative_errors = absolute_errors / np.maximum(np.abs(predictions), np.finfo(float).eps)
    reconstruction_mae = float(absolute_errors.mean())
    reconstruction_difference = float(absolute_errors.max())
    maximum_relative_error = float(relative_errors.max())
    reconstruction_tolerance = float(validation.get("reconstruction_validation_tolerance", 1e-9))
    exact_interventional = bool(validation.get("exact_tree_shap")) and (
        str(validation.get("feature_perturbation", "")).strip().casefold() == "interventional"
    )
    if exact_interventional:
        governance = validate_reconstruction_governance_contract(
            validation,
            explained_rows=explained_rows,
        )
        strict_pass = absolute_errors <= STRICT_RECONSTRUCTION_THRESHOLD
        strict_pass_count = int(strict_pass.sum())
        strict_pass_fraction = float(strict_pass.mean())
        exception_positions = np.flatnonzero(~strict_pass)
        exception_rows = [
            int(value) for value in metadata.iloc[exception_positions]["production_row_index"].tolist()
        ]
        exception_publications = list(
            dict.fromkeys(
                metadata.iloc[exception_positions]["publication_group_id"].astype(str).tolist()
            )
        )
        reproduced = {
            "cohort_mae_pce_points": reconstruction_mae,
            "maximum_absolute_error_pce_points": reconstruction_difference,
            "maximum_relative_error": maximum_relative_error,
            "strict_pass_count": strict_pass_count,
            "strict_pass_fraction": strict_pass_fraction,
            "exception_count": len(exception_positions),
            "exception_fraction": float(len(exception_positions) / len(values)),
        }
        for field, actual_value in reproduced.items():
            reported_value = governance.get(field)
            if reported_value is None or not np.isclose(
                float(reported_value), float(actual_value), rtol=0, atol=1e-12
            ):
                raise ValueError(f"Exact-interventional reconstruction governance does not reproduce {field}")
        if list(governance.get("exception_production_row_indices", [])) != exception_rows:
            raise ValueError("Exact-interventional reconstruction exception-row lineage changed")
        if list(governance.get("exception_publication_group_ids", [])) != exception_publications:
            raise ValueError("Exact-interventional reconstruction exception-publication lineage changed")
    elif not 0 < reconstruction_tolerance <= 1e-4:
        raise ValueError("Reconstruction tolerance is absent or scientifically too permissive")
    if reconstruction_difference > reconstruction_tolerance:
        raise ValueError(f"Grouped contributions do not reconstruct predictions ({reconstruction_difference:.3g})")
    return {
        "explained_rows": explained_rows,
        "scientific_groups": len(groups),
        "maximum_importance_difference": maximum_absolute_difference,
        "maximum_signed_mean_difference": maximum_signed_difference,
        "maximum_prediction_reconstruction_difference": reconstruction_difference,
        "prediction_reconstruction_mae": reconstruction_mae,
        "maximum_prediction_reconstruction_relative_error": maximum_relative_error,
        "reconstruction_tolerance": reconstruction_tolerance,
        "population_fingerprint_sha256": population_fingerprint,
        "model_prediction_sha256": prediction_fingerprint,
        "background_lineage_reproduced": True,
    }


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "yes", "1"})


def validate_choice_values(
    choice_values: pd.DataFrame,
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    validation: dict[str, object],
) -> dict[str, int]:
    explained_rows = len(values)
    required = {
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
    missing = required.difference(choice_values.columns)
    if missing:
        raise ValueError(f"Choice-level table is missing columns: {sorted(missing)}")
    families = set(choice_values["feature_family"].astype(str))
    if families != set(CHOICE_FAMILY_ORDER):
        raise ValueError("Choice-level table does not contain the predefined feature families")
    expected_rows = explained_rows * len(CHOICE_FAMILY_ORDER)
    if len(choice_values) != expected_rows:
        raise ValueError(f"Choice-level table has {len(choice_values)} rows; expected {expected_rows}")
    if choice_values.duplicated(["production_row_index", "feature_family"]).any():
        raise ValueError("Choice-level table must contain one row per explained record and feature family")
    counts = choice_values.groupby("feature_family")["production_row_index"].nunique()
    if not counts.eq(explained_rows).all():
        raise ValueError("Every choice-level feature family must cover all explained rows")
    expected_row_ids = values["production_row_index"].astype(str).tolist()
    expected_publications = metadata["publication_group_id"].astype(str).tolist()
    for family in CHOICE_FAMILY_ORDER:
        family_rows = choice_values.loc[
            choice_values["feature_family"].astype(str).eq(family)
        ]
        if family_rows["production_row_index"].astype(str).tolist() != expected_row_ids:
            raise ValueError(f"Choice family {family} does not match grouped-row lineage")
        if family_rows["publication_group_id"].astype(str).tolist() != expected_publications:
            raise ValueError(f"Choice family {family} does not match publication lineage")
    contributions = pd.to_numeric(choice_values["bundle_shap_pce_points"], errors="coerce")
    if not np.isfinite(contributions).all():
        raise ValueError("Choice-level contributions must be finite")

    definitions = validation.get("choice_bundle_definitions")
    if not isinstance(definitions, list) or len(definitions) != len(CHOICE_FAMILY_ORDER):
        raise ValueError("Choice-bundle validation definitions are incomplete")
    definitions_by_family = {
        str(definition.get("feature_family")): definition
        for definition in definitions
        if isinstance(definition, dict)
    }
    if set(definitions_by_family) != set(CHOICE_FAMILY_ORDER):
        raise ValueError("Choice-bundle validation definitions do not match the governed families")
    for family in CHOICE_FAMILY_ORDER:
        definition = definitions_by_family[family]
        required_definition_fields = {
            "raw_source_fields",
            "raw_source_field_count",
            "encoded_input_count",
        }
        missing_definition_fields = sorted(required_definition_fields.difference(definition))
        if missing_definition_fields:
            raise ValueError(
                f"Choice family {family} definition is incomplete: {missing_definition_fields}"
            )
        family_rows = choice_values.loc[
            choice_values["feature_family"].astype(str).eq(family)
        ]
        expected_fields = " | ".join(str(field) for field in definition["raw_source_fields"])
        checks = {
            "raw_source_fields": set(family_rows["raw_source_fields"].astype(str))
            == {expected_fields},
            "raw_source_field_count": set(
                pd.to_numeric(family_rows["raw_source_field_count"], errors="raise").astype(int)
            )
            == {int(definition["raw_source_field_count"])},
            "encoded_input_count": set(
                pd.to_numeric(family_rows["encoded_input_count"], errors="raise").astype(int)
            )
            == {int(definition["encoded_input_count"])},
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Choice family {family} definition changed: {failed}")
    if int(validation["choice_bundle_count"]) != len(CHOICE_FAMILY_ORDER):
        raise ValueError("Choice-bundle validation count changed")
    if int(validation["choice_value_rows"]) != expected_rows:
        raise ValueError("Choice-bundle validation row count changed")
    return {
        "choice_families": len(families),
        "choice_value_rows": len(choice_values),
    }


def swarm_offsets(values: np.ndarray, row_ids: np.ndarray, width: float = 0.32) -> np.ndarray:
    """Return deterministic, density-preserving vertical offsets.

    The vertical extent of each horizontal bin is scaled against the densest
    bin in the row.  Sparse bins therefore remain narrow instead of being
    stretched to the same height as dense bins, which would give a misleading
    beeswarm silhouette.
    """
    values = np.asarray(values, dtype=float)
    row_ids = np.asarray(row_ids)
    offsets = np.zeros(len(values), dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return offsets
    finite_values = values[finite]
    low, high = float(finite_values.min()), float(finite_values.max())
    if np.isclose(low, high):
        bin_ids = np.zeros(finite.sum(), dtype=int)
    else:
        bin_count = max(12, min(42, int(np.sqrt(finite.sum()) * 2.8)))
        bin_ids = np.floor((finite_values - low) / (high - low) * (bin_count - 1)).astype(int)
    finite_indices = np.flatnonzero(finite)
    members_by_bin = [finite_indices[bin_ids == bin_id] for bin_id in np.unique(bin_ids)]
    maximum_level = max(1.0, max(float(len(members) // 2) for members in members_by_bin))
    for members in members_by_bin:
        members = members[np.argsort(row_ids[members], kind="stable")]
        levels = np.empty(len(members), dtype=float)
        levels[0::2] = np.arange((len(members) + 1) // 2)
        levels[1::2] = -np.arange(1, len(members) // 2 + 1)
        offsets[members] = width * levels / maximum_level
    return offsets


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    source_files: Sequence[Path] = (),
) -> list[Path]:
    if stem not in PANEL_SPECS:
        raise KeyError(f"No physical panel specification is registered for {stem}")
    export_panel(
        fig,
        out_dir,
        PANEL_SPECS[stem],
        source_files=[Path(path) for path in source_files],
    )
    return [out_dir / f"{stem}.{extension}" for extension in ("png", "svg", "pdf")]


def copy_aliases(out_dir: Path, source_stem: str, aliases: tuple[str, ...]) -> None:
    for alias in aliases:
        for extension in ("png", "svg", "pdf"):
            shutil.copyfile(out_dir / f"{source_stem}.{extension}", out_dir / f"{alias}.{extension}")


def typeset_common_formulas(text: str) -> str:
    """Use mathtext subscripts in display text without altering source values."""

    return COMMON_FORMULAS.sub(lambda match: FORMULA_MATHTEXT[match.group(0)], text)


def typeset_absorber_formula(text: str) -> str:
    """Typeset common formulas; leave complex mixed compositions unmodified."""

    return typeset_common_formulas(text)


def solvent_display_label(value: object) -> str:
    """Disclose model imputation for solvent metadata absent from the source."""

    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "not reported"}:
        return "Not reported in source (imputed for model)"
    return text


def friendly_choice_state(feature_family: str, state: object) -> str:
    text = str(state).strip()
    exact = {
        "TiO2-c | TiO2-mp; Spin-coating | Spin-coating": "compact/mesoporous TiO2; spin-coated",
        "TiO2-c; Spin-coating": "compact TiO2; spin-coated",
        "TiO2-c | TiO2-mp; Spray-pyrolys | Spin-coating": (
            "compact/mesoporous TiO2; spray/spin coat"
        ),
        "SnO2-np; Spin-coating": "SnO2 nanoparticles; spin-coated",
        "PCBM-60 | BCP; Spin-coating | Spin-coating": "PCBM-60/BCP; spin-coated",
        "Spiro-MeOTAD; Spin-coating": "Spiro-OMeTAD; spin-coated",
        "PEDOT:PSS; Spin-coating": "PEDOT:PSS; spin-coated",
        "PTAA; Spin-coating": "PTAA; spin-coated",
    }
    text = exact.get(text, text)
    if feature_family in {"ETL stack and deposition", "HTL stack and deposition"}:
        text = text.replace("Spin-coating", "spin-coated")
    if feature_family == "Absorber composition identity":
        return typeset_absorber_formula(text)
    return typeset_common_formulas(text)


def choice_state_summary(
    choice_values: pd.DataFrame,
    minimum_count: int = 5,
    maximum_states_per_family: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = choice_values.copy()
    explained_rows = float(rows["production_row_index"].nunique())
    rows["state_is_reported"] = truthy(rows["state_is_reported"])
    rows["state_is_valid"] = truthy(rows["state_is_valid"])
    rows["bundle_shap_pce_points"] = pd.to_numeric(rows["bundle_shap_pce_points"], errors="raise")
    summary = (
        rows.groupby(["feature_family", "state"], sort=False, dropna=False)
        .agg(
            sample_count=("production_row_index", "size"),
            publication_group_count=("publication_group_id", "nunique"),
            all_publication_ids_are_dois=(
                "publication_group_id",
                lambda series: series.astype(str).str.match(DOI_PATTERN).all(),
            ),
            mean_bundle_shap_pce_points=("bundle_shap_pce_points", "mean"),
            median_bundle_shap_pce_points=("bundle_shap_pce_points", "median"),
            q25_bundle_shap_pce_points=("bundle_shap_pce_points", lambda series: series.quantile(0.25)),
            q75_bundle_shap_pce_points=("bundle_shap_pce_points", lambda series: series.quantile(0.75)),
            mean_absolute_bundle_shap_pce_points=("bundle_shap_pce_points", lambda series: series.abs().mean()),
            global_absolute_attribution_mass=(
                "bundle_shap_pce_points",
                lambda series: series.abs().sum() / explained_rows,
            ),
            state_is_reported=("state_is_reported", "all"),
            state_is_valid=("state_is_valid", "all"),
            raw_source_fields=("raw_source_fields", "first"),
            raw_source_field_count=("raw_source_field_count", "first"),
            encoded_input_count=("encoded_input_count", "first"),
        )
        .reset_index()
    )
    summary["family_order"] = summary["feature_family"].map(
        {family: position for position, family in enumerate(CHOICE_FAMILY_ORDER)}
    )
    # Preserve historical columns while exposing count units explicitly for
    # manuscript tables and labels.
    summary["device_count"] = summary["sample_count"].astype(int)
    summary["doi_count"] = summary["publication_group_count"].where(
        summary["all_publication_ids_are_dois"]
    ).astype("Int64")
    summary["prevalence_rank_within_family"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary["selected_for_main"] = False
    for family in CHOICE_FAMILY_ORDER:
        eligible = summary.loc[
            summary["feature_family"].eq(family)
            & summary["state_is_reported"]
            & summary["state_is_valid"]
            & summary["sample_count"].ge(minimum_count)
        ].sort_values(
            ["sample_count", "publication_group_count", "state"],
            ascending=[False, False, True],
            kind="stable",
        )
        ranks = pd.Series(np.arange(1, len(eligible) + 1), index=eligible.index, dtype="Int64")
        summary.loc[eligible.index, "prevalence_rank_within_family"] = ranks
        summary.loc[eligible.head(maximum_states_per_family).index, "selected_for_main"] = True
    summary["selection_rule"] = (
        f"reported valid state; at least {minimum_count} explained rows; top "
        f"{maximum_states_per_family} within family by prevalence, independent of SHAP magnitude"
    )
    summary["selection_count_unit"] = "fitted-training records"
    summary = summary.sort_values(
        ["family_order", "selected_for_main", "prevalence_rank_within_family", "sample_count", "state"],
        ascending=[True, False, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    selected = summary.loc[summary["selected_for_main"]].copy()
    return rows, summary, selected


def plot_choice_stratified_shap(
    choice_values: pd.DataFrame,
    out_dir: Path,
    minimum_count: int = 5,
    maximum_states_per_family: int = 3,
    source_files: Sequence[Path] = (),
    scope_label: str = "Final fitted-model in-sample attribution table (not out-of-fold)",
) -> pd.DataFrame:
    rows, full_summary, selected = choice_state_summary(
        choice_values,
        minimum_count=minimum_count,
        maximum_states_per_family=maximum_states_per_family,
    )
    if selected.empty:
        raise ValueError("No concrete feature state meets the main-figure support rule")

    positions: dict[tuple[str, str], float] = {}
    heading_positions: dict[str, float] = {}
    current_position = 0.0
    for family in CHOICE_FAMILY_ORDER:
        heading_positions[family] = current_position
        current_position += 0.85
        family_rows = selected.loc[selected["feature_family"].eq(family)].sort_values(
            "prevalence_rank_within_family"
        )
        for row in family_rows.itertuples(index=False):
            positions[(str(row.feature_family), str(row.state))] = current_position
            current_position += 1.0
        current_position += 0.35

    fig, ax = plt.subplots(figsize=PANEL_SPECS[CHOICE_STEM].figsize)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    heading_tick_indices: list[int] = []
    for family in CHOICE_FAMILY_ORDER:
        heading_tick_indices.append(len(tick_positions))
        tick_positions.append(heading_positions[family])
        tick_labels.append(CHOICE_FAMILY_PREFIX[family])
        ax.axhline(
            heading_positions[family] - 0.34,
            color=GRID,
            linewidth=0.75,
            zorder=0,
        )
    for row in selected.sort_values(["family_order", "prevalence_rank_within_family"]).itertuples(index=False):
        family = str(row.feature_family)
        state = str(row.state)
        position = positions[(family, state)]
        subset = rows.loc[rows["feature_family"].eq(family) & rows["state"].eq(state)]
        contributions = subset["bundle_shap_pce_points"].to_numpy(dtype=float)
        row_ids = subset["production_row_index"].to_numpy()
        offsets = swarm_offsets(contributions, row_ids, width=0.27)
        colors = np.where(contributions < 0, NEGATIVE, POSITIVE)
        ax.scatter(
            contributions,
            position + offsets,
            c=colors,
            s=5.0,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        ax.plot(
            [row.q25_bundle_shap_pce_points, row.q75_bundle_shap_pce_points],
            [position, position],
            color=BLACK,
            linewidth=2.0,
            zorder=4,
        )
        ax.scatter(
            row.median_bundle_shap_pce_points,
            position,
            marker="D",
            c=BLACK,
            s=28,
            zorder=5,
        )
        display_state = friendly_choice_state(family, state)
        tick_positions.append(position)
        tick_labels.append(
            f"  {display_state}  (n={int(row.device_count):,}; "
            f"{int(row.publication_group_count):,} groups)"
        )

    ax.axvline(0, color=BLACK, linewidth=0.9, zorder=2)
    ax.set_yticks(tick_positions, tick_labels)
    for index, label in enumerate(ax.get_yticklabels()):
        if index in heading_tick_indices:
            label.set_fontweight("bold")
            label.set_fontsize(8.5)
        else:
            label.set_fontsize(7.5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(current_position - 0.10, -0.55)
    ax.set_xlabel("Bundle SHAP contribution to predicted PCE (percentage points)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=NEGATIVE, markeredgewidth=0, label="Lowers fitted prediction"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=POSITIVE, markeredgewidth=0, label="Raises fitted prediction"),
            Line2D([0], [0], marker="D", linestyle="-", color=BLACK, markersize=4, label="Median and IQR"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.47, 1.015),
        frameon=False,
        ncol=3,
        fontsize=7.5,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.51, right=0.985, bottom=0.115, top=0.93)
    save_figure(fig, out_dir, CHOICE_STEM, source_files)

    full_summary.to_csv(out_dir / "pce_choice_stratified_shap_summary.csv", index=False)
    choice_values.to_csv(out_dir / "pce_choice_stratified_shap_values.csv", index=False)
    return selected


def plot_global_beeswarm(
    importance: pd.DataFrame,
    values: pd.DataFrame,
    out_dir: Path,
    top_count: int = DEFAULT_TOP_COUNT,
    source_files: Sequence[Path] = (),
    scope_label: str = "Final fitted-model in-sample attribution table (not out-of-fold)",
) -> pd.DataFrame:
    if top_count < 1:
        raise ValueError("top_count must be at least one")
    groups = [column for column in values.columns if column != "production_row_index"]
    # Rank from every supplied device row. Validation checks that the source
    # importance table matches these values, and this recomputation prevents a
    # stale preselected table from controlling the displayed top groups.
    full_cohort_importance = values[groups].abs().mean(axis=0)
    ranked = importance.set_index("scientific_group").loc[groups].copy()
    ranked["mean_absolute_grouped_shap_pce_points"] = full_cohort_importance.loc[groups]
    ranked["ranking_device_count"] = len(values)
    ranked = ranked.reset_index().sort_values(
        ["mean_absolute_grouped_shap_pce_points", "scientific_group"],
        ascending=[False, True],
        kind="stable",
    )
    top = ranked.head(top_count).copy()
    row_ids = values["production_row_index"].to_numpy()
    fig, ax = plt.subplots(figsize=PANEL_SPECS[GLOBAL_STEM].figsize)
    for position, group in enumerate(top["scientific_group"].astype(str)):
        contributions = values[group].to_numpy(dtype=float)
        offsets = swarm_offsets(contributions, row_ids)
        colors = np.where(contributions < 0, NEGATIVE, POSITIVE)
        ax.scatter(
            contributions,
            position + offsets,
            c=colors,
            s=4.0,
            alpha=0.58,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    ax.axvline(0, color=BLACK, linewidth=0.9, zorder=2)
    wrapped_groups = [
        textwrap.fill(str(group), width=38, break_long_words=False, break_on_hyphens=False)
        for group in top["scientific_group"]
    ]
    ax.set_yticks(np.arange(len(top)), wrapped_groups)
    ax.invert_yaxis()
    ax.set_xlabel("Grouped SHAP contribution to predicted PCE\n(percentage points)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=NEGATIVE, markeredgewidth=0, label="Lowers predicted PCE"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=POSITIVE, markeredgewidth=0, label="Raises predicted PCE"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.39, right=0.985, bottom=0.12, top=0.93)
    save_figure(fig, out_dir, GLOBAL_STEM, source_files)
    return top


def solvent_summary(values: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = ["production_row_index", "solvent_formulation"]
    if "publication_group_id" in metadata.columns:
        metadata_columns.append("publication_group_id")
    rows = metadata[metadata_columns].merge(
        values[["production_row_index", SOLVENT_GROUP]],
        on="production_row_index",
        how="inner",
        validate="one_to_one",
    )
    rows = rows.rename(columns={SOLVENT_GROUP: "grouped_shap_pce_points"})
    rows["solvent_formulation"] = rows["solvent_formulation"].fillna("Not reported").astype(str)
    aggregations: dict[str, tuple[str, object]] = {
        "sample_count": ("grouped_shap_pce_points", "size"),
        "mean_grouped_shap_pce_points": ("grouped_shap_pce_points", "mean"),
        "median_grouped_shap_pce_points": ("grouped_shap_pce_points", "median"),
        "q25_grouped_shap_pce_points": (
            "grouped_shap_pce_points",
            lambda series: series.quantile(0.25),
        ),
        "q75_grouped_shap_pce_points": (
            "grouped_shap_pce_points",
            lambda series: series.quantile(0.75),
        ),
        "mean_absolute_grouped_shap_pce_points": (
            "grouped_shap_pce_points",
            lambda series: series.abs().mean(),
        ),
    }
    if "publication_group_id" in rows.columns:
        aggregations["publication_group_count"] = ("publication_group_id", "nunique")
        aggregations["all_publication_ids_are_dois"] = (
            "publication_group_id",
            lambda series: series.astype(str).str.match(DOI_PATTERN).all(),
        )
    summary = (
        rows.groupby("solvent_formulation", sort=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["sample_count", "solvent_formulation"], ascending=[False, True])
    )
    summary["device_count"] = summary["sample_count"].astype(int)
    if "publication_group_count" in summary.columns:
        summary["doi_count"] = summary["publication_group_count"].where(
            summary["all_publication_ids_are_dois"]
        ).astype("Int64")
    return rows, summary


def plot_solvent_formulations(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
    minimum_count: int = 5,
    maximum_categories: int = 10,
    source_files: Sequence[Path] = (),
    scope_label: str = "Final fitted-model in-sample attribution table (not out-of-fold)",
) -> pd.DataFrame:
    rows, summary = solvent_summary(values, metadata)
    eligible = summary.loc[summary["sample_count"] >= minimum_count].head(maximum_categories).copy()
    if eligible.empty:
        raise ValueError("No solvent formulation meets the minimum record count")
    eligible = eligible.sort_values("median_grouped_shap_pce_points", ascending=True)
    fig, ax = plt.subplots(figsize=PANEL_SPECS[SOLVENT_STEM].figsize)
    row_ids = rows["production_row_index"].to_numpy()
    for position, formulation in enumerate(eligible["solvent_formulation"]):
        mask = rows["solvent_formulation"].eq(formulation).to_numpy()
        contributions = rows.loc[mask, "grouped_shap_pce_points"].to_numpy(dtype=float)
        offsets = swarm_offsets(contributions, row_ids[mask], width=0.28)
        colors = np.where(contributions < 0, NEGATIVE, POSITIVE)
        ax.scatter(
            contributions,
            position + offsets,
            c=colors,
            s=4.0,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        row = eligible.loc[eligible["solvent_formulation"].eq(formulation)].iloc[0]
        ax.plot(
            [row["q25_grouped_shap_pce_points"], row["q75_grouped_shap_pce_points"]],
            [position, position],
            color=BLACK,
            linewidth=2.0,
            zorder=4,
        )
        ax.scatter(row["median_grouped_shap_pce_points"], position, marker="D", c=BLACK, s=17, zorder=5)
    labels = []
    for row in eligible.itertuples(index=False):
        counts = f"{int(row.device_count):,} records"
        if hasattr(row, "publication_group_count"):
            counts += f"; {int(row.publication_group_count):,} DOI/publication groups"
        formulation = textwrap.fill(
            solvent_display_label(row.solvent_formulation),
            width=36,
            break_long_words=False,
            break_on_hyphens=False,
        )
        labels.append(f"{formulation}\n({counts})")
    ax.set_yticks(np.arange(len(eligible)), labels)
    ax.axvline(0, color=BLACK, linewidth=0.9, zorder=2)
    ax.set_xlabel("Solvent-formulation SHAP contribution to predicted PCE\n(percentage points)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=NEGATIVE, markeredgewidth=0, label="Record: lowers PCE"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=POSITIVE, markeredgewidth=0, label="Record: raises PCE"),
            Line2D([0], [0], marker="D", linestyle="none", color=BLACK, markersize=4, label="Median"),
            Line2D([0], [0], linestyle="-", color=BLACK, linewidth=2.0, label="IQR"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncol=4,
        fontsize=7.5,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.36, right=0.985, bottom=0.19, top=0.92)
    save_figure(fig, out_dir, SOLVENT_STEM, source_files)
    summary.to_csv(out_dir / "pce_solvent_formulation_shap_summary.csv", index=False)
    rows.to_csv(out_dir / "pce_solvent_formulation_shap_values.csv", index=False)
    return eligible


def shortened(value: object, limit: int = 52) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "not reported"}:
        return ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def plot_random_local_explanation(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
    top_count: int = 10,
    source_files: Sequence[Path] = (),
    scope_label: str = "Final fitted-model in-sample attribution table (not out-of-fold)",
    base_value_label: str = "SHAP expected value",
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    selected_position = int(rng.integers(0, len(values)))
    value_row = values.iloc[selected_position]
    metadata_row = metadata.iloc[selected_position]
    groups = [column for column in values.columns if column != "production_row_index"]
    contributions = value_row[groups].astype(float).sort_values(key=np.abs, ascending=False)
    shown = contributions.head(top_count).copy()
    remainder = float(contributions.iloc[top_count:].sum())
    remainder_label = "All other scientific groups\n(signed net remainder; cancellation possible)"
    shown.loc[remainder_label] = remainder
    context_columns = {
        SOLVENT_GROUP: "context_solvent_formulation",
        "A-site composition": "context_a_site_composition",
        "X-site/halide composition": "context_x_site_composition",
        "ETL material/stack and processing": "context_etl",
        "HTL material/stack and processing": "context_htl",
        "Quenching-induced crystallization": "context_quenching",
        "Cell stack sequence": "context_cell_stack",
        "Back-contact material": "context_back_contact",
        "Back-contact deposition method": "context_back_contact_deposition",
        "Perovskite composition short form": "context_perovskite_short_form",
        "Perovskite composition long form": "context_perovskite_long_form",
        "Fully inorganic absorber": "context_fully_inorganic",
        "B-site composition": "context_b_site_composition",
        "Substrate stack sequence": "context_substrate_stack_sequence",
        "Perovskite composition leadfree": "context_lead_free",
        "Perovskite deposition procedure": "context_perovskite_deposition_procedure",
        "Perovskite additives compounds": "context_perovskite_additives",
        "Perovskite deposition synthesis atmosphere": "context_perovskite_synthesis_atmosphere",
        "ADD DESC text length": "context_add_desc_text_length",
        "Perovskite deposition thermal annealing time": "context_annealing_time",
        "Perovskite deposition thermal annealing temperature": "context_annealing_temperature",
    }
    labels: list[str] = []
    displayed_context: list[dict[str, object]] = []
    for group in shown.index:
        context_column = context_columns.get(group)
        context = shortened(metadata_row.get(context_column, "")) if context_column else ""
        if group == "Quenching-induced crystallization":
            context = {"True": "Yes", "False": "No"}.get(context, context)
        context = context.replace("Spin-coating", "spin-coated")
        if group == "X-site/halide composition":
            context = re.sub(r"\b([A-Za-z][A-Za-z0-9]*)\s+(\d+(?:\.\d+)?)\b", r"\1: \2", context)
        if group == "ADD DESC text length":
            context = re.sub(r"^metadata proxy;\s*", "", context, flags=re.IGNORECASE)
        context = typeset_common_formulas(context)
        group_label = (
            "Additive-text length (metadata proxy)"
            if group == "ADD DESC text length"
            else str(group)
        )
        labels.append(f"{group_label}\n{context}" if context else group_label)
        if group != remainder_label:
            displayed_context.append(
                {
                    "scientific_group": str(group),
                    "context_source_column": context_column,
                    "displayed_context": context,
                    "grouped_shap_pce_points": float(shown.loc[group]),
                    "metadata_proxy": group == "ADD DESC text length",
                }
            )
    colors = np.where(shown.to_numpy() < 0, NEGATIVE, POSITIVE)
    fig, ax = plt.subplots(figsize=PANEL_SPECS[LOCAL_STEM].figsize)
    positions = np.arange(len(shown))
    ax.barh(positions, shown.to_numpy(), color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=BLACK, linewidth=0.9)
    ax.set_xlabel("Grouped SHAP contribution to predicted PCE\n(percentage points)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    row_index = int(metadata_row["production_row_index"])
    base = float(metadata_row["base_value"])
    predicted = float(metadata_row["predicted_pce"])
    observed = float(metadata_row["observed_pce"])
    publication_group_id = str(metadata_row["publication_group_id"])
    publication_group_display = shortened(publication_group_id, 60) or "not reported"
    publication_group_device_count = int(
        metadata["publication_group_id"].astype(str).eq(publication_group_id).sum()
    )
    ax.legend(
        handles=[
            Patch(facecolor=NEGATIVE, edgecolor="none", label="Lowers this prediction"),
            Patch(facecolor=POSITIVE, edgecolor="none", label="Raises this prediction"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncol=2,
    )
    fig.subplots_adjust(left=0.40, right=0.985, bottom=0.19, top=0.92)
    save_figure(fig, out_dir, LOCAL_STEM, source_files)
    selection = {
        "selection_method": (
            f"uniform random draw from the ordered {len(values)}-row fitted-training-cohort "
            "in-sample explanation table"
        ),
        "explanation_scope": scope_label,
        "explanation_table_rows": len(values),
        "random_seed": seed,
        "selected_position_zero_based": selected_position,
        "production_row_index": row_index,
        "publication_group_id": publication_group_id,
        "publication_group_device_count": publication_group_device_count,
        "publication_group_count_unit": (
            "explained records sharing the selected publication_group_id"
        ),
        "base_value_pce_points": base,
        "base_value_semantics": base_value_label,
        "predicted_pce": predicted,
        "observed_pce": observed,
        "reconstruction_target": "predicted_pce",
        "observed_pce_role": (
            "reference outcome shown for comparison; not reconstructed by SHAP"
        ),
        "prediction_reconstruction_difference": float(base + contributions.sum() - predicted),
        "top_group_count": top_count,
        "other_group_count": len(groups) - top_count,
        "other_group_signed_net_remainder_pce_points": remainder,
        "other_group_aggregation": (
            "signed net sum of all undisplayed scientific-group contributions; "
            "positive and negative contributions can cancel"
        ),
        "displayed_group_context": displayed_context,
    }
    (out_dir / "pce_grouped_shap_local_random_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    return selection


def render_figure_set(
    importance: pd.DataFrame,
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    choice_values: pd.DataFrame,
    validation: dict[str, object],
    out_dir: Path,
    top_count: int = DEFAULT_TOP_COUNT,
    local_seed: int = 42,
    compatibility_aliases: tuple[str, ...] = (),
    source_files: Sequence[Path] = (),
    metadata_source_path: Path | None = None,
    _expected_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    configure_style()
    if _expected_contract is None and metadata_source_path is None:
        raise ValueError(
            "Canonical production rendering requires the metadata CSV path for exact lineage hashing"
        )
    validate_canonical_shap_contract(
        validation,
        explained_rows=len(values),
        top_count=top_count,
        _expected_contract=_expected_contract,
    )
    checks = validate_tables(
        importance,
        values,
        metadata,
        validation,
        metadata_source_path=metadata_source_path,
    )
    choice_checks = validate_choice_values(choice_values, values, metadata, validation)
    scope_label, fitted_training_rows, full_training_cohort_explained = fitted_cohort_scope(
        validation,
        int(checks["explained_rows"]),
        _expected_contract=_expected_contract,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_sources = tuple(Path(path) for path in source_files)
    selected_choices = plot_choice_stratified_shap(
        choice_values,
        out_dir,
        source_files=canonical_sources,
        scope_label=scope_label,
    )
    top = plot_global_beeswarm(
        importance,
        values,
        out_dir,
        top_count=top_count,
        source_files=canonical_sources,
        scope_label=scope_label,
    )
    top_table = out_dir / f"pce_grouped_shap_top{top_count}.csv"
    top.to_csv(top_table, index=False)
    dynamic_global_stem = f"{GLOBAL_STEM}_top{top_count}"
    copy_aliases(out_dir, GLOBAL_STEM, compatibility_aliases)
    if dynamic_global_stem not in compatibility_aliases:
        for suffix in (".png", ".pdf", ".svg"):
            (out_dir / f"{dynamic_global_stem}{suffix}").unlink(missing_ok=True)
    solvent = plot_solvent_formulations(
        values,
        metadata,
        out_dir,
        source_files=canonical_sources,
        scope_label=scope_label,
    )
    local = plot_random_local_explanation(
        values,
        metadata,
        out_dir,
        seed=local_seed,
        source_files=canonical_sources,
        scope_label=scope_label,
        base_value_label=(
            "Interventional-background expectation"
            if str(validation.get("feature_perturbation", "")).strip().casefold() == "interventional"
            else "SHAP expected value"
        ),
    )
    global_png = out_dir / f"{GLOBAL_STEM}.png"
    choice_png = out_dir / f"{CHOICE_STEM}.png"
    return {
        "main_choice_figure": str(choice_png),
        "main_choice_figure_sha256": hashlib.sha256(choice_png.read_bytes()).hexdigest(),
        "global_figure": str(global_png),
        "global_figure_sha256": hashlib.sha256(global_png.read_bytes()).hexdigest(),
        "top_groups_table": str(top_table),
        "solvent_figure": str(out_dir / f"{SOLVENT_STEM}.png"),
        "local_figure": str(out_dir / f"{LOCAL_STEM}.png"),
        "source_files": [str(path.resolve()) for path in canonical_sources],
        "top_groups": top["scientific_group"].astype(str).tolist(),
        "explanation_scope": scope_label,
        "fitted_training_rows": fitted_training_rows,
        "full_training_cohort_explained": full_training_cohort_explained,
        "selected_choice_states": int(len(selected_choices)),
        "solvent_formulations": int(len(solvent)),
        "local_selection": local,
        **choice_checks,
        **checks,
    }
