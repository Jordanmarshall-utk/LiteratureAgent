import json
from pathlib import Path

import pandas as pd

from scripts.build_literature_knowledge_graph import GraphBuilder, ontology_index


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
