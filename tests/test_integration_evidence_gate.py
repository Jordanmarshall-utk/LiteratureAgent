import importlib.util
from pathlib import Path
import sys

import pandas as pd

import literature_agent_full_end_to_end_v21_3_english_sanitizer as controller


def _load_integration_module(tmp_path: Path):
    _, integration_path = controller._write_embedded_modules(tmp_path)
    spec = importlib.util.spec_from_file_location("test_integration_runtime", integration_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unsupported_targets_are_withheld_but_supported_targets_remain(tmp_path):
    module = _load_integration_module(tmp_path)
    records = pd.DataFrame([
        {
            "Ref_DOI_number": "10.1000/a",
            "JV_default_PCE": 21.2,
            "Stability_PCE_T80": 1500,
        }
    ])
    evidence = pd.DataFrame([
        {
            "lit_row_index": 0,
            "field": "JV_default_PCE",
            "evidence": "The champion device reached a PCE of 21.2%.",
        }
    ])

    gated, report = module._v21_12_withhold_targets_without_field_evidence(records, evidence)

    assert gated.loc[0, "JV_default_PCE"] == 21.2
    assert pd.isna(gated.loc[0, "Stability_PCE_T80"])
    assert "Stability_PCE_T80" in report.loc[0, "backfill_changes"]
    assert report.loc[0, "source"] == "v21.12_field_evidence_model_gate"


def test_integration_preserves_distinct_devices_with_shared_sample_id(tmp_path):
    module = _load_integration_module(tmp_path)
    records = pd.DataFrame([
        {
            "Ref_DOI_number": "10.1002/example",
            "Ref_internal_sample_id": "paper_device_1",
            "JV_default_PCE": 21.61,
            "Perovskite_composition_short_form": "CsFA MASnPbI3",
            "_evidence_JV_default_PCE": "The target PSC reached 21.61% PCE.",
        },
        {
            "Ref_DOI_number": "10.1002/example",
            "Ref_internal_sample_id": "paper_device_1",
            "JV_default_PCE": 21.13,
            "Perovskite_composition_short_form": "CsFA MASnPbI3",
            "_evidence_JV_default_PCE": "The control PSC reached 21.13% PCE.",
        },
        {
            "Ref_DOI_number": "10.1002/example",
            "Ref_internal_sample_id": "paper_device_1",
            "JV_default_PCE": 19.12,
            "Perovskite_composition_short_form": "CsFA MASnPbI3",
            "_evidence_JV_default_PCE": "The certified PCE was 19.12%.",
        },
    ])

    preserved, report = module._v21_17_split_conflicting_device_groups(records)

    assert len(preserved) == 3
    assert preserved["Ref_internal_sample_id"].nunique() == 3
    assert set(preserved["JV_default_PCE"]) == {21.61, 21.13, 19.12}
    assert set(report["source"]) == {"v21.17_conflict_aware_device_identity"}


def test_integration_still_coalesces_complementary_partial_rows(tmp_path):
    module = _load_integration_module(tmp_path)
    records = pd.DataFrame([
        {
            "Ref_DOI_number": "10.1002/example",
            "Ref_internal_sample_id": "paper_device_1",
            "JV_default_PCE": 21.61,
            "Cell_stack_sequence": None,
        },
        {
            "Ref_DOI_number": "10.1002/example",
            "Ref_internal_sample_id": "paper_device_1",
            "JV_default_PCE": None,
            "Cell_stack_sequence": "ITO/HTL/Perovskite/ETL/Ag",
        },
    ])

    preserved, report = module._v21_17_split_conflicting_device_groups(records)

    assert preserved["Ref_internal_sample_id"].nunique() == 1
    assert report.empty
