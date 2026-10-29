from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import literature_agent_pipeline as controller
from literature_agent.domain_pack import domain_column_name, load_structured_output_contract


class _FakeExtractor:
    def __init__(self, response: dict):
        self.response = response
        self.payload: dict | None = None

    def _call(self, _system: str, payload: dict, **_kwargs) -> dict:
        self.payload = payload
        return self.response


def _write_pack(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "id": "battery_materials",
                "version": "v2026.1",
                "structured_output": {
                    "domain_instruction": "Extract battery chemistry and cycling results.",
                    "fields": {
                        "electrode_material": None,
                        "cycle_metrics": [
                            {
                                "name": None,
                                "value": None,
                                "unit": None,
                                "conditions": None,
                                "evidence": None,
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_domain_pack_fields_become_grounded_csv_columns(monkeypatch, tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "battery.json")
    source = (
        "Lithium iron phosphate was used as the cathode. "
        "Capacity retention was 92 percent after 500 cycles at 1 C."
    )
    extractor = _FakeExtractor(
        {
            "domain_records": [
                {
                    "system_or_subject": "lithium-ion cell",
                    "electrode_material": "lithium iron phosphate",
                    "cycle_metrics": [
                        {
                            "name": "capacity retention",
                            "value": 92,
                            "unit": "%",
                            "conditions": "500 cycles at 1 C",
                            "evidence": "Capacity retention was 92 percent after 500 cycles at 1 C.",
                        }
                    ],
                    "field_evidence": {
                        "system_or_subject": "Lithium iron phosphate was used as the cathode.",
                        "electrode_material": "Lithium iron phosphate was used as the cathode.",
                    },
                }
            ]
        }
    )
    monkeypatch.setenv("PEROVSKITE_ONTOLOGY_PATH", str(pack))
    monkeypatch.setenv("LITERATURE_AGENT_DOMAIN_PROFILE", "general_science")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    embedded_source, _ = controller._write_embedded_modules(tmp_path / "embedded")
    runtime = controller._load_module_from_path(embedded_source, "_test_custom_domain_runtime")
    runtime.RAW_LLM_DIR = tmp_path / "raw"
    runtime.RAW_LLM_DIR.mkdir()

    records = runtime._v22_grounded_domain_records(
        {"title": "Battery study", "doi": "10.0000/battery"}, source, extractor, "battery_study"
    )

    assert extractor.payload is not None
    schema = extractor.payload["output_schema"]["domain_records"][0]
    assert "electrode_material" in schema
    assert "cycle_metrics" in schema
    assert records[0]["Domain_electrode_material"] == "lithium iron phosphate"
    assert records[0]["Domain_cycle_metrics"][0]["value"] == 92
    assert records[0]["_evidence_Domain_cycle_metrics"].startswith("Capacity retention")
    assert records[0]["_domain_pack_id"] == "battery_materials"
    assert records[0]["_domain_pack_version"] == "v2026.1"
    assert records[0]["_domain_pack_file"] == "battery.json"

    csv_path = tmp_path / "domain.csv"
    runtime.json_to_csv(records, csv_path)
    frame = pd.read_csv(csv_path, low_memory=False)
    assert "Domain_electrode_material" in frame.columns
    assert "Domain_cycle_metrics" in frame.columns
    assert frame.loc[0, "_domain_pack_id"] == "battery_materials"
    assert frame.loc[0, "_domain_pack_version"] == "v2026.1"


def test_domain_pack_validation_rejects_database_hostile_field_names(tmp_path: Path) -> None:
    pack = tmp_path / "invalid.json"
    pack.write_text(
        json.dumps({"structured_output": {"fields": {"Bad Field Name": None}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lower-case snake_case"):
        load_structured_output_contract(pack)
    with pytest.raises(ValueError):
        domain_column_name("field with spaces")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"structured_output": []}, "structured_output must be a JSON object"),
        ({"structured_output": {"fields": []}}, "fields must be a JSON object"),
        ({"structured_output": {"domain_instruction": 0}}, "domain_instruction must be text"),
        ({"id": []}, "Domain-pack id must be text or a number"),
    ],
)
def test_domain_pack_validation_rejects_falsy_wrong_types(
    tmp_path: Path, payload: dict, message: str
) -> None:
    pack = tmp_path / "invalid_type.json"
    pack.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_structured_output_contract(pack)
