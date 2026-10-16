import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_literature_knowledge_graph import GraphBuilder, norm, ontology_index


def test_oxide_record_builds_domain_graph_without_fake_pv_device(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "paper_summaries_json").mkdir(parents=True)
    summary = {
        "_domain_profile": "oxide_perovskite",
        "paper_summary": {
            "system_or_subject": "La0.7Sr0.3MnO3",
            "primary_findings": ["Oxygen annealing increased conductivity."],
            "characterization_and_evidence": ["XRD showed a single rhombohedral phase."],
        },
    }
    (work / "paper_summaries_json" / "oxide_paper_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    row = pd.Series(
        {
            "Ref_internal_sample_id": "oxide_paper_device_1",
            "Ref_DOI_number": "10.0000/example",
            "_domain_profile": "oxide_perovskite",
            "Domain_system_or_subject": "La0.7Sr0.3MnO3",
            "Domain_composition_formula": "La0.7Sr0.3MnO3",
            "Domain_a_site_species": json.dumps(["La", "Sr"]),
            "Domain_b_site_species": json.dumps(["Mn"]),
            "Domain_crystal_structure": "rhombohedral",
            "Domain_measurements": json.dumps(
                [{"name": "conductivity", "value": 120, "unit": "S/cm", "context": "300 K"}]
            ),
            "_evidence_Domain_measurements": "The conductivity was 120 S/cm at 300 K.",
        }
    )
    ontology = Path(__file__).resolve().parents[1] / "config" / "domain_packs" / "oxide_perovskite.json"
    builder = GraphBuilder(ontology_index(ontology), tmp_path / "records.csv", work)
    builder.add_record(row, 0)

    labels = {label for node in builder.nodes.values() for label in node["labels"]}
    relationships = {edge["type"] for edge in builder.edges.values()}
    assert "OxidePerovskite" in labels
    assert "DomainMeasurement" in labels
    assert "Device" not in labels
    assert "HAS_MEASUREMENT" in relationships
    assert "SUPPORTS_ENTITY_OR_MEASUREMENT" in relationships


def _write_battery_ontology(tmp_path: Path) -> Path:
    ontology = tmp_path / "battery_domain.json"
    ontology.write_text(
        json.dumps(
            {
                "id": "battery_cells",
                "canonical_terms": [
                    {
                        "canonical": "lithium iron phosphate",
                        "aliases": ["LFP", "LiFePO4"],
                        "category": "material",
                    },
                    {
                        "canonical": "galvanostatic cycling",
                        "aliases": ["constant-current cycling"],
                        "category": "method",
                    },
                ],
                "concepts": {
                    "study_design": ["experiment", "simulation"],
                    "cell_format": ["coin cell", "pouch cell"],
                },
                "structured_output": {
                    "fields": [
                        {
                            "canonical": "specific capacity",
                            "aliases": ["capacity"],
                            "category": "output_field",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return ontology


def test_explicit_canonical_terms_ignore_broad_lists_and_structured_output(tmp_path: Path) -> None:
    index = ontology_index(_write_battery_ontology(tmp_path))

    assert index[norm("LFP")] == {
        "canonical": "lithium iron phosphate",
        "category": "material",
    }
    assert index[norm("LiFePO4")]["canonical"] == "lithium iron phosphate"
    assert index[norm("constant-current cycling")]["canonical"] == "galvanostatic cycling"
    assert norm("coin cell") not in index
    assert norm("capacity") not in index


def test_custom_domain_fields_aliases_evidence_and_neo4j_export(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "paper_summaries_json").mkdir(parents=True)
    row = pd.Series(
        {
            "Ref_internal_sample_id": "battery_paper_sample_1",
            "Ref_DOI_number": "10.0000/battery-example",
            "_domain_profile": "battery_cells",
            "_domain_pack_id": "battery_materials",
            "_domain_pack_version": "2026.1",
            "_domain_pack_file": "battery.json",
            "Domain_system_or_subject": "LFP",
            "Domain_materials_or_entities": json.dumps(["LiFePO4", "graphite"]),
            "Domain_methods": "constant-current cycling",
            "Domain_measurements": json.dumps(
                [{"name": "specific capacity", "value": 155, "unit": "mAh/g", "context": "0.1 C"}]
            ),
            "_evidence_Domain_measurements": "The specific capacity was 155 mAh/g at 0.1 C.",
            "Domain_cathode_material": "LFP",
            "_evidence_Domain_cathode_material": "The cathode active material was LiFePO4 (LFP).",
            "Domain_cell_format": json.dumps(["CR2032", "pouch cell"]),
            "_evidence_Domain_cell_format": "CR2032 and pouch-cell formats were evaluated.",
            "Domain_specific_capacity_target": 160,
            "_evidence_Domain_specific_capacity_target": "The target specific capacity was 160 mAh/g.",
        }
    )
    builder = GraphBuilder(
        ontology_index(_write_battery_ontology(tmp_path)),
        tmp_path / "records.csv",
        work,
    )
    builder.add_record(row, 0)

    labels = {label for node in builder.nodes.values() for label in node["labels"]}
    relationships = {edge["type"] for edge in builder.edges.values()}
    assert labels.isdisjoint({"Device", "Perovskite", "PerformanceMeasurement", "StabilityTest"})
    assert {"StudiedSystem", "DomainAttribute", "DomainMeasurement", "DomainEvidence"} <= labels
    assert {"STUDIES_SYSTEM", "HAS_MEASUREMENT", "HAS_DOMAIN_ATTRIBUTE", "SUPPORTS_ENTITY_OR_MEASUREMENT"} <= relationships

    paper = next(node for node in builder.nodes.values() if "Paper" in node["labels"])
    system = next(node for node in builder.nodes.values() if "StudiedSystem" in node["labels"])
    assert paper["domain_pack_id"] == "battery_materials"
    assert system["domain_pack_version"] == "2026.1"

    cathode = next(
        node for node in builder.nodes.values()
        if node.get("domain_field") == "Domain_cathode_material"
    )
    assert cathode["name"] == "lithium iron phosphate"
    assert cathode["raw_name"] == "LFP"
    assert cathode["ontology_category"] == "material"
    assert any(
        edge["type"] == "SUPPORTS_ENTITY_OR_MEASUREMENT" and edge["end_id"] == cathode["node_id"]
        for edge in builder.edges.values()
    )

    out = tmp_path / "knowledge_graph"
    builder.write(out)
    nodes = pd.read_csv(out / "neo4j_import" / "nodes.csv")
    edges = pd.read_csv(out / "neo4j_import" / "relationships.csv")
    assert any("DomainAttribute" in value.split(";") for value in nodes[":LABEL"].astype(str))
    assert not any("Device" in value.split(";") for value in nodes[":LABEL"].astype(str))
    assert "HAS_DOMAIN_ATTRIBUTE" in set(edges[":TYPE"])
    assert "SUPPORTS_ENTITY_OR_MEASUREMENT" in set(edges[":TYPE"])
    assert (out / "reports" / "graph_summary.json").is_file()


def test_non_halide_summary_cannot_create_pv_performance_nodes(tmp_path: Path) -> None:
    work = tmp_path / "work"
    summaries = work / "paper_summaries_json"
    summaries.mkdir(parents=True)
    (summaries / "generic_paper_summary.json").write_text(
        json.dumps(
            {
                "paper_summary": {
                    "primary_findings": ["The treatment increased the measured response."],
                    "best_reported_performance": {"pce": 24.1, "voc": 1.18},
                }
            }
        ),
        encoding="utf-8",
    )
    row = pd.Series(
        {
            "Ref_internal_sample_id": "generic_paper_device_1",
            "Ref_DOI_number": "10.0000/generic-paper",
            "_domain_profile": "general_science",
            "Domain_system_or_subject": "enzyme assay",
            "Domain_measurements": json.dumps(
                [{"name": "response", "value": 2.4, "unit": "a.u."}]
            ),
        }
    )
    builder = GraphBuilder({}, tmp_path / "records.csv", work)

    builder.add_record(row, 0)

    labels = {label for node in builder.nodes.values() for label in node["labels"]}
    names = {str(node.get("name")) for node in builder.nodes.values()}
    relationships = {edge["type"] for edge in builder.edges.values()}
    assert "PerformanceMeasurement" not in labels
    assert "PCE" not in names
    assert "REPORTS_SUMMARY_PERFORMANCE" not in relationships


@pytest.mark.parametrize(
    ("policy", "expected_relationship"),
    [
        ("cartesian_decomposition", "HAS_DECOMPOSITION_STEP"),
        ("kantian_constraints", "POLICY_CHECKS_SCHEMA_CONSTRAINTS"),
        ("aristotelian_classification", "BELONGS_TO_REASONING_CATEGORY"),
    ],
)
def test_reasoning_policy_lenses_operate_on_generic_domain_graphs(
    tmp_path: Path, policy: str, expected_relationship: str
) -> None:
    work = tmp_path / policy / "work"
    (work / "paper_summaries_json").mkdir(parents=True)
    row = pd.Series(
        {
            "Ref_internal_sample_id": f"{policy}_paper",
            "_domain_profile": "battery_cells",
            "Domain_system_or_subject": "LFP cell",
            "Domain_materials_or_entities": json.dumps(["LFP", "graphite"]),
            "Domain_methods": "galvanostatic cycling",
            "_evidence_Domain_methods": "Galvanostatic cycling was used for the cell tests.",
            "Domain_measurements": json.dumps(
                [{"name": "capacity", "value": 155, "unit": "mAh/g"}]
            ),
            "_evidence_Domain_measurements": "The capacity was 155 mAh/g.",
        }
    )
    builder = GraphBuilder(
        ontology_index(_write_battery_ontology(tmp_path / policy)),
        tmp_path / "records.csv",
        work,
    )
    builder.add_record(row, 0)

    builder.apply_reasoning_policy_lens(policy)

    relationships = {edge["type"] for edge in builder.edges.values()}
    labels = {label for node in builder.nodes.values() for label in node["labels"]}
    assert expected_relationship in relationships
    assert "ReasoningPolicy" in labels
    assert "Device" not in labels
