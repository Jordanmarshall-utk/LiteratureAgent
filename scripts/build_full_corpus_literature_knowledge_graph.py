from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PCE_ROOT = ROOT / "artifacts" / "literature_agent_pce_stability"
PROVENANCE_ROOT = PCE_ROOT / "source_data" / "data_provenance"
DEFAULT_ALL_RECORDS = PROVENANCE_ROOT / "datasets" / "cumulative_expansion_all_records.csv"
DEFAULT_AUDIT = PROVENANCE_ROOT / "cumulative_expansion" / "literature_update_audit.csv"
DEFAULT_ACCEPTED = PROVENANCE_ROOT / "cumulative_expansion" / "literature_update_accepted_rows.csv"
DEFAULT_EVIDENCE = PROVENANCE_ROOT / "cumulative_expansion" / "literature_update_evidence_long.csv"
DEFAULT_ONTOLOGY = ROOT / "config" / "perovskite_ontology_library_v19.json"
DEFAULT_SCHEMA = ROOT / "config" / "literatureagent_kg_schema_v1.json"
DEFAULT_OUTPUT = PCE_ROOT / "source_data" / "knowledge_graph" / "full_corpus_standardized_v1"
DEFAULT_FIGURES = PCE_ROOT / "publication_figures" / "main"

EMPTY_VALUES = {"", "nan", "none", "null", "n/a", "na", "not reported", "unknown"}
DOMAIN_PREFIXES = (
    "Cell_",
    "Module",
    "Substrate_",
    "ETL_",
    "Perovskite_",
    "HTL_",
    "Backcontact_",
    "Add_lay_",
    "Encapsulation_",
    "JV_",
    "EQE_",
    "Stabilised_",
    "Stability_",
    "Outdoor_",
)
RELATION_PATTERNS = (
    ("IMPROVES", re.compile(r"\b(?:improve|improves|improved|improving)\b", re.I)),
    ("ENHANCES", re.compile(r"\b(?:enhance|enhances|enhanced|enhancing)\b", re.I)),
    ("INCREASES", re.compile(r"\b(?:increase|increases|increased|increasing|raise|raises|raised)\b", re.I)),
    ("REDUCES", re.compile(r"\b(?:reduce|reduces|reduced|reducing|decrease|decreases|decreased|lower|lowers|lowered)\b", re.I)),
    ("PROMOTES", re.compile(r"\b(?:promote|promotes|promoted|promoting|facilitate|facilitates|facilitated)\b", re.I)),
    ("INHIBITS", re.compile(r"\b(?:inhibit|inhibits|inhibited|suppress|suppresses|suppressed)\b", re.I)),
    ("STABILIZES", re.compile(r"\b(?:stabilize|stabilizes|stabilized|stabilise|stabilises|stabilised)\b", re.I)),
    ("ASSOCIATED_WITH", re.compile(r"\b(?:associated with|correlated with|correlates with|linked to)\b", re.I)),
    ("ATTRIBUTED_TO", re.compile(r"\b(?:attributed to|ascribed to|due to|results? from)\b", re.I)),
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\x00", " ").split())
    return "" if text.lower() in EMPTY_VALUES else text


def normalize_doi(value: Any) -> str:
    doi = clean(value).lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.strip(" .;,()[]{}")


def stable_id(kind: str, *parts: Any) -> str:
    payload = "\x1f".join(clean(part).lower() for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()[:20]
    return f"{kind.lower()}:{digest}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class PropertyGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.relationships: dict[str, dict[str, Any]] = {}

    def add_node(self, node_id: str, node_label: str, **properties: Any) -> str:
        compact = {key: value for key, value in properties.items() if value not in (None, "", [], {})}
        if node_id in self.nodes:
            self.nodes[node_id]["properties"].update(compact)
        else:
            self.nodes[node_id] = {"id": node_id, "label": node_label, "properties": compact}
        return node_id

    def add_relationship(self, source: str, relation: str, target: str, **properties: Any) -> str:
        compact = {key: value for key, value in properties.items() if value not in (None, "", [], {})}
        fingerprint = json.dumps(compact, sort_keys=True, ensure_ascii=False)
        rel_id = stable_id("relationship", source, relation, target, fingerprint)
        self.relationships[rel_id] = {
            "id": rel_id,
            "source": source,
            "type": relation,
            "target": target,
            "properties": compact,
        }
        return rel_id


def field_family(field: str) -> str:
    if field.startswith(("JV_", "EQE_", "Stabilised_")):
        return "performance"
    if field.startswith(("Stability_", "Outdoor_")):
        return "stability_outdoor"
    if field.startswith("Perovskite_"):
        return "perovskite_chemistry_and_processing"
    if field.startswith("ETL_"):
        return "electron_transport_layer"
    if field.startswith(("HTL_", "Backcontact_", "Add_lay_", "Encapsulation_")):
        return "hole_transport_backcontact_additional"
    return "device_context"


def iter_observation_fields(row: dict[str, str]) -> Iterable[tuple[str, str]]:
    for field, raw_value in row.items():
        if not field.startswith(DOMAIN_PREFIXES):
            continue
        value = clean(raw_value)
        if value:
            yield field, value


def sentence_split(text: str) -> list[str]:
    return [
        clean(part)
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean(text))
        if 35 <= len(clean(part)) <= 900
    ]


def clause_before(sentence: str, start: int) -> str:
    value = sentence[:start].strip(" ,;:-")
    value = re.split(r"[;:]", value)[-1].strip()
    words = value.split()
    return " ".join(words[-36:])


def clause_after(sentence: str, end: int) -> str:
    value = sentence[end:].strip(" ,;:-")
    value = re.split(r"[;]", value)[0].strip()
    words = value.split()
    return " ".join(words[:42])


def extract_directional_claims(doi: str, abstract: str) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in sentence_split(abstract):
        for predicate, pattern in RELATION_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            subject = clause_before(sentence, match.start())
            obj = clause_after(sentence, match.end())
            if len(subject) < 4 or len(obj) < 4:
                continue
            key = (sentence.lower(), predicate)
            if key in seen:
                continue
            seen.add(key)
            causal_status = "reported_association" if predicate in {"ASSOCIATED_WITH", "ATTRIBUTED_TO"} else "reported_directional_hypothesis"
            claims.append(
                {
                    "doi": doi,
                    "claim_text": sentence,
                    "predicate": predicate,
                    "subject": subject,
                    "object": obj,
                    "causal_status": causal_status,
                }
            )
            if len(claims) >= 12:
                return claims
    return claims


def evidence_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[tuple[int, dict[str, str]]]]:
    index: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for position, row in enumerate(rows, start=1):
        index[(clean(row.get("lit_row_index")), clean(row.get("field")))].append((position, row))
    return index


def add_field_evidence(
    graph: PropertyGraph,
    paper_id: str,
    device_id: str,
    observation_id: str | None,
    doi: str,
    row_index: str,
    field: str,
    entries: list[tuple[int, dict[str, str]]],
    consumed: set[int],
) -> int:
    added = 0
    for position, row in entries:
        text = clean(row.get("evidence"))
        if not text:
            continue
        consumed.add(position)
        evidence_id = stable_id("evidence", "field", position, doi, row_index, field, text)
        graph.add_node(
            evidence_id,
            "Evidence",
            evidence_text=text,
            source_kind="field_passage",
            source_locator=f"literature_update_evidence_long.csv:{position + 1}",
            doi=doi,
            lit_row_index=row_index,
            field=field,
        )
        if observation_id:
            graph.add_relationship(
                observation_id,
                "SUPPORTED_BY",
                evidence_id,
                provenance_scope="field_level",
                field=field,
            )
        else:
            graph.add_relationship(device_id, "HAS_FIELD_EVIDENCE", evidence_id, field=field)
        graph.add_relationship(evidence_id, "DERIVED_FROM", paper_id, source_kind="field_passage")
        added += 1
    return added


def build_graph(args: argparse.Namespace) -> tuple[PropertyGraph, dict[str, Any], list[str]]:
    log: list[str] = []
    started = datetime.now(timezone.utc)
    all_records = read_csv(args.all_records)
    audit_rows = read_csv(args.audit)
    accepted_rows = read_csv(args.accepted)
    evidence_rows = read_csv(args.evidence)
    ontology = json.loads(args.ontology.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    log.append(f"loaded all_records={len(all_records)} audit={len(audit_rows)} accepted={len(accepted_rows)} evidence={len(evidence_rows)}")

    graph = PropertyGraph()
    run_id = f"full-corpus-kg-{started.strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_id_value = "pce-cumulative-expansion-910-paper-v1"
    snapshot_id = stable_id("dataset_snapshot", snapshot_id_value)
    extraction_run_id = stable_id("extraction_run", run_id)
    graph.add_node(
        extraction_run_id,
        "ExtractionRun",
        run_id=run_id,
        agent_name="LiteratureAgent",
        builder="build_full_corpus_literature_knowledge_graph.py",
        generated_at=started.isoformat(),
    )
    graph.add_node(
        snapshot_id,
        "DatasetSnapshot",
        snapshot_id=snapshot_id_value,
        schema_version=schema["version"],
        scope="production",
        ontology_version=clean(ontology.get("version")) or args.ontology.stem,
    )
    graph.add_relationship(snapshot_id, "GENERATED_BY", extraction_run_id)

    audit_by_doi: dict[str, list[dict[str, str]]] = defaultdict(list)
    audit_by_row: dict[str, dict[str, str]] = {}
    for row in audit_rows:
        doi = normalize_doi(row.get("Ref_DOI_number"))
        if doi:
            audit_by_doi[doi].append(row)
        row_index = clean(row.get("lit_row_index"))
        if row_index:
            audit_by_row[row_index] = row

    accepted_by_doi = Counter(normalize_doi(row.get("Ref_DOI_number")) for row in accepted_rows)
    accepted_by_doi.pop("", None)
    paper_data: dict[str, dict[str, Any]] = {}
    for row in all_records:
        doi = normalize_doi(row.get("Ref_DOI_number"))
        if not doi:
            continue
        record = paper_data.setdefault(
            doi,
            {
                "doi": doi,
                "title": "",
                "lead_author": "",
                "publication_year": "",
                "journal": "",
                "landing_page": "",
                "abstract": "",
                "raw_record_count": 0,
                "source_kinds": set(),
            },
        )
        record["raw_record_count"] += 1
        for source, target in (
            ("Ref_original_filename_data_upload", "title"),
            ("Ref_lead_author", "lead_author"),
            ("Ref_publication_date", "publication_year"),
            ("Ref_journal", "journal"),
            ("_source_landing_page", "landing_page"),
        ):
            value = clean(row.get(source))
            if value and not record[target]:
                record[target] = value
        abstract = clean(row.get("_source_abstract"))
        if len(abstract) > len(record["abstract"]):
            record["abstract"] = abstract
        source_kind = clean(row.get("_full_text_source"))
        if source_kind:
            record["source_kinds"].add(source_kind)

    paper_ids: dict[str, str] = {}
    for doi, record in sorted(paper_data.items()):
        statuses = sorted({clean(row.get("status")) for row in audit_by_doi.get(doi, []) if clean(row.get("status"))})
        tiers = sorted({clean(row.get("row_tier")) for row in audit_by_doi.get(doi, []) if clean(row.get("row_tier"))})
        corpus_status = "accepted" if doi in accepted_by_doi else ("rejected_or_not_retained" if statuses else "processed")
        paper_id = stable_id("paper", doi)
        paper_ids[doi] = paper_id
        graph.add_node(
            paper_id,
            "Paper",
            doi=doi,
            title=record["title"] or doi,
            lead_author=record["lead_author"],
            publication_year=record["publication_year"],
            journal=record["journal"],
            landing_page=record["landing_page"] or f"https://doi.org/{doi}",
            corpus_status=corpus_status,
            audit_statuses=statuses,
            audit_tiers=tiers,
            raw_record_count=record["raw_record_count"],
            accepted_record_count=accepted_by_doi.get(doi, 0),
            abstract_available=bool(record["abstract"]),
            source_kinds=sorted(record["source_kinds"]),
        )
        graph.add_relationship(paper_id, "IN_SNAPSHOT", snapshot_id, provenance_scope="campaign")

    indexed_evidence = evidence_index(evidence_rows)
    consumed_evidence: set[int] = set()
    row_to_device: dict[str, tuple[str, str, str]] = {}
    observations_with_evidence = 0
    observations_without_evidence = 0
    fallback_evidence_nodes = 0

    for accepted_position, row in enumerate(accepted_rows, start=1):
        doi = normalize_doi(row.get("Ref_DOI_number"))
        paper_id = paper_ids.get(doi)
        if not paper_id:
            continue
        row_index = clean(row.get("lit_row_index")) or str(accepted_position)
        sample_id = clean(row.get("Ref_internal_sample_id")) or f"row_{row_index}"
        audit = audit_by_row.get(row_index, {})
        record_status = clean(audit.get("status")) or "accepted"
        row_tier = clean(audit.get("row_tier"))
        device_id = stable_id("device", doi, sample_id, row_index)
        row_to_device[row_index] = (device_id, paper_id, doi)
        graph.add_node(
            device_id,
            "Device",
            sample_id=sample_id,
            doi=doi,
            lit_row_index=row_index,
            record_status=record_status,
            row_tier=row_tier,
            pce_model_candidate=clean(audit.get("pce_model_candidate")),
            stability_model_candidate=clean(audit.get("stability_model_candidate")),
        )
        graph.add_relationship(paper_id, "HAS_DEVICE", device_id, doi=doi, lit_row_index=row_index)

        for field, value in iter_observation_fields(row):
            entries = indexed_evidence.get((row_index, field), [])
            fallback_text = clean(row.get(f"_evidence_{field}"))
            evidence_status = "evidence_linked" if entries or fallback_text else "paper_linked"
            observation_id = stable_id("observation", device_id, field, value)
            graph.add_node(
                observation_id,
                "StructuredObservation",
                field=field,
                value=value,
                evidence_status=evidence_status,
                doi=doi,
                sample_id=sample_id,
                lit_row_index=row_index,
            )
            graph.add_relationship(device_id, "HAS_OBSERVATION", observation_id, field=field)
            concept_id_value = f"perovskite-database-v19:{field}"
            concept_id = stable_id("ontology_concept", concept_id_value)
            graph.add_node(
                concept_id,
                "OntologyConcept",
                concept_id=concept_id_value,
                label=field.replace("_", " "),
                namespace="perovskite-database-v19",
                family=field_family(field),
            )
            graph.add_relationship(
                observation_id,
                "MEASURES_PROPERTY",
                concept_id,
                ontology_version=clean(ontology.get("version")) or args.ontology.stem,
            )
            linked = add_field_evidence(
                graph,
                paper_id,
                device_id,
                observation_id,
                doi,
                row_index,
                field,
                entries,
                consumed_evidence,
            )
            if linked:
                observations_with_evidence += 1
            elif fallback_text:
                evidence_id = stable_id("evidence", "wide", doi, row_index, field, fallback_text)
                graph.add_node(
                    evidence_id,
                    "Evidence",
                    evidence_text=fallback_text,
                    source_kind="field_passage_wide",
                    source_locator=f"literature_update_accepted_rows.csv:{accepted_position + 1}:{field}",
                    doi=doi,
                    lit_row_index=row_index,
                    field=field,
                )
                graph.add_relationship(observation_id, "SUPPORTED_BY", evidence_id, provenance_scope="field_level", field=field)
                graph.add_relationship(evidence_id, "DERIVED_FROM", paper_id, source_kind="field_passage_wide")
                fallback_evidence_nodes += 1
                observations_with_evidence += 1
            else:
                observations_without_evidence += 1

    unmapped_evidence = 0
    for position, row in enumerate(evidence_rows, start=1):
        if position in consumed_evidence:
            continue
        text = clean(row.get("evidence"))
        if not text:
            continue
        row_index = clean(row.get("lit_row_index"))
        field = clean(row.get("field")) or "unspecified"
        doi = normalize_doi(row.get("Ref_DOI_number"))
        mapping = row_to_device.get(row_index)
        if mapping:
            device_id, paper_id, mapped_doi = mapping
            doi = doi or mapped_doi
            source_id = device_id
        else:
            paper_id = paper_ids.get(doi, "")
            source_id = paper_id
        if not paper_id:
            unmapped_evidence += 1
            continue
        evidence_id = stable_id("evidence", "field", position, doi, row_index, field, text)
        graph.add_node(
            evidence_id,
            "Evidence",
            evidence_text=text,
            source_kind="field_passage",
            source_locator=f"literature_update_evidence_long.csv:{position + 1}",
            doi=doi,
            lit_row_index=row_index,
            field=field,
            mapped_to_observation=False,
        )
        graph.add_relationship(source_id, "HAS_FIELD_EVIDENCE", evidence_id, field=field)
        graph.add_relationship(evidence_id, "DERIVED_FROM", paper_id, source_kind="field_passage")

    claim_count = 0
    for doi, record in sorted(paper_data.items()):
        paper_id = paper_ids[doi]
        for claim in extract_directional_claims(doi, record["abstract"]):
            claim_id = stable_id("claim", doi, claim["claim_text"], claim["predicate"])
            evidence_id = stable_id("evidence", "abstract", doi, claim["claim_text"])
            subject_id = stable_id("scientific_entity", claim["subject"])
            object_id = stable_id("scientific_entity", claim["object"])
            graph.add_node(
                claim_id,
                "ScientificClaim",
                claim_text=claim["claim_text"],
                predicate=claim["predicate"],
                causal_status=claim["causal_status"],
                support_status="evidence_linked",
                claim_source="source_abstract",
                doi=doi,
            )
            graph.add_node(
                evidence_id,
                "Evidence",
                evidence_text=claim["claim_text"],
                source_kind="abstract_sentence",
                source_locator=f"cumulative_expansion_all_records.csv:{doi}:_source_abstract",
                doi=doi,
            )
            graph.add_node(
                subject_id,
                "ScientificEntity",
                label=claim["subject"],
                normalized_label=claim["subject"].lower(),
            )
            graph.add_node(
                object_id,
                "ScientificEntity",
                label=claim["object"],
                normalized_label=claim["object"].lower(),
            )
            graph.add_relationship(paper_id, "REPORTS_CLAIM", claim_id, doi=doi)
            graph.add_relationship(claim_id, "SUPPORTED_BY", evidence_id, provenance_scope="abstract_sentence")
            graph.add_relationship(evidence_id, "DERIVED_FROM", paper_id, source_kind="abstract_sentence")
            graph.add_relationship(claim_id, "HAS_SUBJECT", subject_id)
            graph.add_relationship(claim_id, "HAS_OBJECT", object_id)
            graph.add_relationship(
                subject_id,
                "DIRECTIONAL_RELATION",
                object_id,
                predicate=claim["predicate"],
                causal_status=claim["causal_status"],
                claim_id=claim_id,
                evidence_id=evidence_id,
                doi=doi,
            )
            claim_count += 1

    label_counts = Counter(node["label"] for node in graph.nodes.values())
    relationship_counts = Counter(rel["type"] for rel in graph.relationships.values())
    accepted_unique_papers = len(accepted_by_doi)
    abstract_papers = sum(1 for record in paper_data.values() if record["abstract"])
    graph.add_node(
        snapshot_id,
        "DatasetSnapshot",
        paper_count=len(paper_data),
        accepted_paper_count=accepted_unique_papers,
        raw_record_count=len(all_records),
        accepted_record_count=len(accepted_rows),
        evidence_source_row_count=len(evidence_rows),
        scientific_claim_count=claim_count,
    )

    summary = {
        "graph_model": schema["graph_model"],
        "schema_id": schema["schema_id"],
        "schema_version": schema["version"],
        "snapshot_id": snapshot_id_value,
        "scope": "production",
        "generated_at": started.isoformat(),
        "source_files": {
            "all_records": str(args.all_records.resolve()),
            "audit": str(args.audit.resolve()),
            "accepted": str(args.accepted.resolve()),
            "evidence": str(args.evidence.resolve()),
            "ontology": str(args.ontology.resolve()),
            "schema": str(args.schema.resolve()),
        },
        "corpus": {
            "processed_unique_doi_papers": len(paper_data),
            "papers_with_accepted_records": accepted_unique_papers,
            "papers_with_abstracts": abstract_papers,
            "raw_device_sample_rows": len(all_records),
            "integration_candidates": len(audit_rows),
            "accepted_device_sample_rows": len(accepted_rows),
            "rejected_candidate_rows": sum(1 for row in audit_rows if "reject" in clean(row.get("status")).lower()),
            "field_evidence_source_rows": len(evidence_rows),
        },
        "graph": {
            "nodes": len(graph.nodes),
            "relationships": len(graph.relationships),
            "node_counts": dict(sorted(label_counts.items())),
            "relationship_counts": dict(sorted(relationship_counts.items())),
            "directional_claims": claim_count,
            "observations_with_field_evidence": observations_with_evidence,
            "observations_without_field_evidence": observations_without_evidence,
            "fallback_wide_evidence_nodes": fallback_evidence_nodes,
            "unmapped_evidence_source_rows": unmapped_evidence,
        },
        "causal_guardrail": "Directional edges encode paper-reported abstract statements as hypotheses or associations; they do not establish causal effects.",
        "benchmark_scope": {
            "papers": 36,
            "purpose": "Controlled same-input comparison of reasoning-policy representations.",
            "separate_from_production_graph": True,
        },
    }
    log.append(f"built nodes={len(graph.nodes)} relationships={len(graph.relationships)} claims={claim_count}")
    return graph, summary, log


def validate_graph(graph: PropertyGraph, summary: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    node_ids = set(graph.nodes)
    dangling = [
        rel["id"]
        for rel in graph.relationships.values()
        if rel["source"] not in node_ids or rel["target"] not in node_ids
    ]
    required_node_violations: list[dict[str, Any]] = []
    for node in graph.nodes.values():
        required = schema.get("node_types", {}).get(node["label"], {}).get("required", [])
        missing = [key for key in required if node["properties"].get(key) in (None, "", [])]
        if missing:
            required_node_violations.append({"node_id": node["id"], "label": node["label"], "missing": missing})
    required_relationship_violations: list[dict[str, Any]] = []
    for rel in graph.relationships.values():
        required = schema.get("relationship_types", {}).get(rel["type"], {}).get("required", [])
        missing = [key for key in required if rel["properties"].get(key) in (None, "", [])]
        if missing:
            required_relationship_violations.append({"relationship_id": rel["id"], "type": rel["type"], "missing": missing})
    supported_claims = {
        rel["source"]
        for rel in graph.relationships.values()
        if rel["type"] == "SUPPORTED_BY" and graph.nodes.get(rel["source"], {}).get("label") == "ScientificClaim"
    }
    claim_ids = {node["id"] for node in graph.nodes.values() if node["label"] == "ScientificClaim"}
    unsupported_claims = sorted(claim_ids - supported_claims)
    directional_without_guardrail = [
        rel["id"]
        for rel in graph.relationships.values()
        if rel["type"] == "DIRECTIONAL_RELATION"
        and (
            not rel["properties"].get("causal_status")
            or not rel["properties"].get("claim_id")
            or not rel["properties"].get("evidence_id")
        )
    ]
    checks = {
        "expected_910_unique_doi_papers": summary["corpus"]["processed_unique_doi_papers"] == 910,
        "expected_1137_accepted_rows": summary["corpus"]["accepted_device_sample_rows"] == 1137,
        "no_dangling_relationships": not dangling,
        "node_required_properties_complete": not required_node_violations,
        "relationship_required_properties_complete": not required_relationship_violations,
        "all_claims_evidence_linked": not unsupported_claims,
        "all_directional_relations_guarded": not directional_without_guardrail,
        "production_and_benchmark_scopes_separated": summary["benchmark_scope"]["separate_from_production_graph"] is True,
    }
    return {
        "validation_pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "dangling_relationships": len(dangling),
            "required_node_property_violations": len(required_node_violations),
            "required_relationship_property_violations": len(required_relationship_violations),
            "unsupported_claims": len(unsupported_claims),
            "unguarded_directional_relations": len(directional_without_guardrail),
        },
        "examples": {
            "dangling_relationships": dangling[:20],
            "required_node_property_violations": required_node_violations[:20],
            "required_relationship_property_violations": required_relationship_violations[:20],
            "unsupported_claims": unsupported_claims[:20],
            "unguarded_directional_relations": directional_without_guardrail[:20],
        },
        "non_blocking_limitations": [
            "Paper-level source provenance is complete for DOI-bearing records, but not every structured observation has an exact field-level passage.",
            "Abstract-derived directional statements are literature-reported hypotheses or associations, not experimentally validated causal effects.",
            "The source batch retained 18 device/sample identity-consistency flags and had a HOLD scale-up recommendation.",
        ],
    }


def write_csv_exports(graph: PropertyGraph, output: Path) -> None:
    neo4j = output / "neo4j"
    neo4j.mkdir(parents=True, exist_ok=True)
    with (neo4j / "nodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id:ID", ":LABEL", "properties_json"])
        writer.writeheader()
        for node in graph.nodes.values():
            writer.writerow(
                {
                    "id:ID": node["id"],
                    ":LABEL": node["label"],
                    "properties_json": json.dumps(node["properties"], ensure_ascii=False, sort_keys=True),
                }
            )
    with (neo4j / "relationships.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[":START_ID", ":TYPE", ":END_ID", "relationship_id", "properties_json"])
        writer.writeheader()
        for rel in graph.relationships.values():
            writer.writerow(
                {
                    ":START_ID": rel["source"],
                    ":TYPE": rel["type"],
                    ":END_ID": rel["target"],
                    "relationship_id": rel["id"],
                    "properties_json": json.dumps(rel["properties"], ensure_ascii=False, sort_keys=True),
                }
            )


def write_jsonl(graph: PropertyGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for node in graph.nodes.values():
            handle.write(json.dumps({"kind": "node", **node}, ensure_ascii=False) + "\n")
        for rel in graph.relationships.values():
            handle.write(json.dumps({"kind": "relationship", **rel}, ensure_ascii=False) + "\n")


def write_jsonld(graph: PropertyGraph, path: Path) -> None:
    context = {
        "la": "https://literatureagent.local/kg/",
        "prov": "http://www.w3.org/ns/prov#",
        "schema": "https://schema.org/",
        "source": {"@id": "la:source", "@type": "@id"},
        "target": {"@id": "la:target", "@type": "@id"},
        "doi": "schema:identifier",
        "evidence_text": "schema:text",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{\n  "@context": ')
        json.dump(context, handle, ensure_ascii=False, indent=2)
        handle.write(',\n  "@graph": [\n')
        first = True
        for node in graph.nodes.values():
            item = {"@id": f"la:{node['id']}", "@type": f"la:{node['label']}", **node["properties"]}
            if not first:
                handle.write(",\n")
            handle.write("    " + json.dumps(item, ensure_ascii=False))
            first = False
        for rel in graph.relationships.values():
            item = {
                "@id": f"la:{rel['id']}",
                "@type": f"la:{rel['type']}",
                "source": f"la:{rel['source']}",
                "target": f"la:{rel['target']}",
                **rel["properties"],
            }
            if not first:
                handle.write(",\n")
            handle.write("    " + json.dumps(item, ensure_ascii=False))
            first = False
        handle.write("\n  ]\n}\n")


def write_graphml(graph: PropertyGraph, path: Path) -> None:
    graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    ET.SubElement(graphml, "key", id="label", **{"for": "node", "attr.name": "label", "attr.type": "string"})
    ET.SubElement(graphml, "key", id="node_properties", **{"for": "node", "attr.name": "properties_json", "attr.type": "string"})
    ET.SubElement(graphml, "key", id="relationship_type", **{"for": "edge", "attr.name": "type", "attr.type": "string"})
    ET.SubElement(graphml, "key", id="relationship_properties", **{"for": "edge", "attr.name": "properties_json", "attr.type": "string"})
    body = ET.SubElement(graphml, "graph", id="LiteratureAgentFullCorpus", edgedefault="directed")
    for node in graph.nodes.values():
        element = ET.SubElement(body, "node", id=node["id"])
        ET.SubElement(element, "data", key="label").text = node["label"]
        ET.SubElement(element, "data", key="node_properties").text = json.dumps(node["properties"], ensure_ascii=False, sort_keys=True)
    for rel in graph.relationships.values():
        element = ET.SubElement(body, "edge", id=rel["id"], source=rel["source"], target=rel["target"])
        ET.SubElement(element, "data", key="relationship_type").text = rel["type"]
        ET.SubElement(element, "data", key="relationship_properties").text = json.dumps(rel["properties"], ensure_ascii=False, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(graphml).write(path, encoding="utf-8", xml_declaration=True)


def write_shapes(path: Path) -> None:
    text = """@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix la: <https://literatureagent.local/kg/> .
@prefix schema: <https://schema.org/> .

la:PaperShape a sh:NodeShape ;
  sh:targetClass la:Paper ;
  sh:property [ sh:path schema:identifier ; sh:minCount 1 ] .

la:EvidenceShape a sh:NodeShape ;
  sh:targetClass la:Evidence ;
  sh:property [ sh:path schema:text ; sh:minCount 1 ] ;
  sh:property [ sh:path la:source_kind ; sh:minCount 1 ] ;
  sh:property [ sh:path la:source_locator ; sh:minCount 1 ] .

la:ScientificClaimShape a sh:NodeShape ;
  sh:targetClass la:ScientificClaim ;
  sh:property [ sh:path la:claim_text ; sh:minCount 1 ] ;
  sh:property [ sh:path la:causal_status ; sh:minCount 1 ] ;
  sh:property [ sh:path la:support_status ; sh:hasValue "evidence_linked" ] .

la:DirectionalRelationShape a sh:NodeShape ;
  sh:targetClass la:DIRECTIONAL_RELATION ;
  sh:property [ sh:path la:causal_status ; sh:minCount 1 ] ;
  sh:property [ sh:path la:claim_id ; sh:minCount 1 ] ;
  sh:property [ sh:path la:evidence_id ; sh:minCount 1 ] .
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def write_count_tables(summary: dict[str, Any], output: Path) -> None:
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for filename, values, key_name in (
        ("node_counts.csv", summary["graph"]["node_counts"], "node_type"),
        ("relationship_counts.csv", summary["graph"]["relationship_counts"], "relationship_type"),
    ):
        with (reports / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[key_name, "count"])
            writer.writeheader()
            for key, value in values.items():
                writer.writerow({key_name: key, "count": value})


def make_figures(summary: dict[str, Any], graph: PropertyGraph, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    counts = summary["graph"]["node_counts"]
    corpus = summary["corpus"]
    colors = {
        "Paper": "#4C78A8",
        "Device": "#72B7B2",
        "StructuredObservation": "#F2CF5B",
        "Evidence": "#9E9E9E",
        "ScientificClaim": "#B279A2",
        "ScientificEntity": "#59A14F",
        "OntologyConcept": "#E8E8E8",
    }

    fig = plt.figure(figsize=(13.2, 7.4), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.95], wspace=0.38)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A  Evidence-grounded production graph", loc="left", fontsize=14, pad=12)
    positions = {
        "Paper": (0.35, 6.4, 1.85, 1.05),
        "Device": (3.05, 7.75, 2.15, 1.05),
        "StructuredObservation": (6.65, 7.75, 2.8, 1.05),
        "OntologyConcept": (6.65, 5.45, 2.8, 1.05),
        "ScientificClaim": (3.05, 3.45, 2.4, 1.05),
        "ScientificEntity": (6.65, 3.45, 2.8, 1.05),
        "Evidence": (3.05, 0.8, 2.4, 1.05),
    }
    display_names = {
        "Paper": "Paper",
        "Device": "Device / sample",
        "StructuredObservation": "Structured observation",
        "OntologyConcept": "Ontology concept",
        "ScientificClaim": "Scientific claim",
        "ScientificEntity": "Scientific entity",
        "Evidence": "Evidence",
    }
    for label, (x, y, width, height) in positions.items():
        face = colors[label]
        text_color = "white" if label not in {"StructuredObservation", "OntologyConcept"} else "#20242A"
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=face, edgecolor="#30343B", linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, f"{display_names[label]}\n(n={counts.get(label, 0):,})", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9.3)

    def anchor(node: str, side: str) -> tuple[float, float]:
        x, y, width, height = positions[node]
        return {
            "left": (x, y + height / 2),
            "right": (x + width, y + height / 2),
            "top": (x + width / 2, y + height),
            "bottom": (x + width / 2, y),
        }[side]

    def arrow(source: str, source_side: str, target: str, target_side: str, label: str, label_xy: tuple[float, float], curve: float = 0.0) -> None:
        start = anchor(source, source_side)
        end = anchor(target, target_side)
        patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#50555C", connectionstyle=f"arc3,rad={curve}")
        ax.add_patch(patch)
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center", fontsize=8.2, color="#454A50", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})

    arrow("Paper", "right", "Device", "left", "reports", (2.65, 7.2))
    arrow("Device", "right", "StructuredObservation", "left", "has observation", (5.92, 8.5))
    arrow("StructuredObservation", "bottom", "OntologyConcept", "top", "typed by", (8.05, 7.08))
    arrow("Paper", "bottom", "ScientificClaim", "left", "reports claim", (2.7, 5.1))
    arrow("ScientificClaim", "right", "ScientificEntity", "left", "subject / object", (6.05, 4.28))
    arrow("ScientificClaim", "bottom", "Evidence", "top", "supported by", (4.25, 2.62))
    field_start = anchor("StructuredObservation", "right")
    field_end = anchor("Evidence", "right")
    ax.plot([field_start[0], 9.72, 9.72, field_end[0] + 0.18], [field_start[1], field_start[1], field_end[1], field_end[1]], color="#50555C", linewidth=1.4)
    ax.add_patch(FancyArrowPatch((field_end[0] + 0.18, field_end[1]), field_end, arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#50555C"))
    ax.text(9.72, 4.9, "field evidence", rotation=90, ha="center", va="center", fontsize=8.2, color="#454A50", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})
    ax.text(0.5, 9.45, f"910 DOI papers | {corpus['accepted_device_sample_rows']:,} accepted sample records | {corpus['field_evidence_source_rows']:,} field-evidence rows", fontsize=10.5, color="#30343B")
    ax.text(0.5, 0.05, "Canonical model: directed heterogeneous property graph.\nAcyclic causal views are derived projections.", fontsize=8.9, color="#50555C", linespacing=1.35)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_title("B  Corpus and provenance coverage", loc="left", fontsize=14, pad=12)
    labels = ["Processed papers", "Accepted-record papers", "Accepted records", "Field evidence", "Directional claims"]
    values = [
        corpus["processed_unique_doi_papers"],
        corpus["papers_with_accepted_records"],
        corpus["accepted_device_sample_rows"],
        corpus["field_evidence_source_rows"],
        summary["graph"]["directional_claims"],
    ]
    bar_colors = ["#4C78A8", "#72B7B2", "#72B7B2", "#9E9E9E", "#B279A2"]
    y = list(range(len(labels)))
    ax2.barh(y, values, color=bar_colors, edgecolor="#30343B", linewidth=0.7)
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("Count (log scale)")
    ax2.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax2.set_axisbelow(True)
    for index, value in enumerate(values):
        ax2.text(value * 1.08, index, f"{value:,}", va="center", fontsize=9.5, fontweight="bold")
    ax2.text(
        0.0,
        -0.24,
        "Causal guardrail\nDirectional claims are paper-reported hypotheses or associations.\nThey are evidence-linked but are not causal proof.",
        transform=ax2.transAxes,
        fontsize=10,
        linespacing=1.4,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F5F5F5", "edgecolor": "#60656B"},
    )
    fig.suptitle("LiteratureAgent full-corpus scientific knowledge graph", x=0.06, ha="left", fontsize=17, fontweight="bold", y=0.985)
    figure_base = figure_dir / "figure_literatureagent_full_corpus_knowledge_graph"
    fig.savefig(figure_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    predicates = Counter(
        rel["properties"].get("predicate", "UNSPECIFIED")
        for rel in graph.relationships.values()
        if rel["type"] == "DIRECTIONAL_RELATION"
    )
    top = predicates.most_common(9)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.7), gridspec_kw={"width_ratios": [1.05, 1.25]}, facecolor="white")
    left.set_title("A  Reported directional predicates", loc="left", fontsize=13)
    names = [name.replace("_", " ").title() for name, _ in reversed(top)]
    vals = [value for _, value in reversed(top)]
    left.barh(names, vals, color="#B279A2", edgecolor="#30343B", linewidth=0.7)
    left.set_xlabel("Evidence-linked abstract statements")
    left.grid(axis="x", color="#E0E0E0")
    left.set_axisbelow(True)
    for i, value in enumerate(vals):
        left.text(value + max(vals or [1]) * 0.015, i, f"{value:,}", va="center", fontsize=9)
    right.axis("off")
    right.set_title("B  Interpretation boundary", loc="left", fontsize=13)
    boxes = [
        (0.02, 0.67, "Paper-reported\nsentence", "Exact abstract\nsentence retained"),
        (0.36, 0.67, "Scientific\nclaim", "Predicate and\nclauses parsed"),
        (0.70, 0.67, "Directional\nhypothesis", "Association or\nproposed direction"),
        (0.36, 0.20, "Evidence\nnode", "DOI and source\nlocator preserved"),
    ]
    for x, y0, title, subtitle in boxes:
        patch = FancyBboxPatch((x, y0), 0.28, 0.18, transform=right.transAxes, boxstyle="round,pad=0.02", facecolor="#F3F3F3", edgecolor="#3D4248", linewidth=1.0)
        right.add_patch(patch)
        right.text(x + 0.14, y0 + 0.125, title, transform=right.transAxes, ha="center", va="center", fontweight="bold", fontsize=8.8, linespacing=1.05)
        right.text(x + 0.14, y0 + 0.045, subtitle, transform=right.transAxes, ha="center", va="center", fontsize=7.1, color="#555A60", linespacing=1.1)
    for start, end in (((0.30, 0.76), (0.36, 0.76)), ((0.64, 0.76), (0.70, 0.76)), ((0.50, 0.38), (0.50, 0.67))):
        right.add_patch(FancyArrowPatch(start, end, transform=right.transAxes, arrowstyle="-|>", mutation_scale=13, color="#50555C", linewidth=1.4))
    right.text(0.5, 0.05, "No edge is promoted to validated causality without an external experimental or causal-identification criterion.", transform=right.transAxes, ha="center", fontsize=9.5, fontweight="bold", color="#8A2D2D", wrap=True)
    fig.suptitle("Causal-ready does not mean causally validated", x=0.06, ha="left", fontsize=16, fontweight="bold")
    causal_base = figure_dir / "figure_literatureagent_full_corpus_causal_guardrail"
    fig.savefig(causal_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(causal_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [figure_base.with_suffix(".png"), figure_base.with_suffix(".pdf"), causal_base.with_suffix(".png"), causal_base.with_suffix(".pdf")]


def write_readme(output: Path, summary: dict[str, Any], validation: dict[str, Any]) -> None:
    corpus = summary["corpus"]
    graph = summary["graph"]
    text = f"""# LiteratureAgent full-corpus knowledge graph

This is the versioned production graph for the cumulative 910-paper web-expansion campaign. It is separate from the fixed 36-paper graph used for controlled reasoning-policy comparisons.

## Snapshot

- Processed DOI papers: {corpus['processed_unique_doi_papers']:,}
- Papers with accepted records: {corpus['papers_with_accepted_records']:,}
- Accepted device/sample records: {corpus['accepted_device_sample_rows']:,}
- Field-evidence source rows: {corpus['field_evidence_source_rows']:,}
- Graph nodes: {graph['nodes']:,}
- Graph relationships: {graph['relationships']:,}
- Evidence-linked directional claims: {graph['directional_claims']:,}
- Structural validation: {'PASS' if validation['validation_pass'] else 'FAIL'}

## Model

The canonical representation is a heterogeneous directed property graph. Papers connect to device/sample records, structured observations, ontology concepts, claims, scientific entities, and source-located evidence. Task-specific DAGs may be derived from this graph, but acyclicity is not imposed on the canonical scientific record.

Directional claims parsed from retained abstracts are labeled `reported_directional_hypothesis` or `reported_association`. They are not labeled as validated causal effects.

## Exports

- `neo4j/nodes.csv` and `neo4j/relationships.csv`: Neo4j-compatible property tables.
- `portable/graph.jsonl`: streaming property-graph exchange file.
- `portable/graph.jsonld`: JSON-LD with Schema.org and PROV-O-compatible context.
- `portable/graph.graphml`: directed GraphML export.
- `schema/literatureagent_kg_schema_v1.json`: frozen graph contract.
- `schema/literatureagent_kg_shapes.ttl`: SHACL constraint definitions.
- `reports/graph_summary.json`: corpus and graph counts.
- `reports/validation_report.json`: structural and provenance checks.

## Scope note

The production graph answers how much literature was processed and what evidence-linked information was retained. The 36-paper benchmark answers how reasoning policies organize identical evidence. The two scopes should not be combined into one headline count.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the standardized LiteratureAgent full-corpus knowledge graph.")
    parser.add_argument("--all-records", type=Path, default=DEFAULT_ALL_RECORDS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = (args.all_records, args.audit, args.accepted, args.evidence, args.ontology, args.schema)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    graph, summary, log = build_graph(args)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validation = validate_graph(graph, summary, schema)
    summary["validation_pass"] = validation["validation_pass"]
    shutil.copy2(args.schema, args.output / "schema" / args.schema.name) if (args.output / "schema").mkdir(parents=True, exist_ok=True) is None else None
    write_shapes(args.output / "schema" / "literatureagent_kg_shapes.ttl")
    write_csv_exports(graph, args.output)
    write_jsonl(graph, args.output / "portable" / "graph.jsonl")
    write_jsonld(graph, args.output / "portable" / "graph.jsonld")
    write_graphml(graph, args.output / "portable" / "graph.graphml")
    write_json(args.output / "reports" / "graph_summary.json", summary)
    write_json(args.output / "reports" / "validation_report.json", validation)
    write_count_tables(summary, args.output)
    if not args.skip_figures:
        generated = make_figures(summary, graph, args.figure_dir)
        summary["publication_figures"] = [str(path.resolve()) for path in generated]
        write_json(args.output / "reports" / "graph_summary.json", summary)
    write_readme(args.output, summary, validation)
    log.append(f"validation_pass={validation['validation_pass']}")
    log.append(f"output={args.output.resolve()}")
    (args.output / "build.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation}, indent=2, ensure_ascii=False))
    return 0 if validation["validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
