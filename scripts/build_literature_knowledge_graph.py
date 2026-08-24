#!/usr/bin/env python3
"""Build an evidence-grounded perovskite literature property graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from literature_agent.reasoning.scientific_reasoning_policy_layer import POLICIES as REASONING_POLICIES
except Exception:
    REASONING_POLICIES = {}


MISSING = {"", "nan", "none", "null", "unknown", "not reported", "n/a", "na", "-", "--"}
PAPER_TYPE_RE = re.compile(r"paper_type=([a-z_]+)", re.I)
DEVICE_SUFFIX_RE = re.compile(r"_device_\d+$", re.I)

CHEMICAL_FIELDS = {
    "Perovskite_composition_a_ions": "HAS_A_SITE",
    "Perovskite_composition_b_ions": "HAS_B_SITE",
    "Perovskite_composition_c_ions": "HAS_X_SITE",
    "Perovskite_additives_compounds": "USES_ADDITIVE",
}
LAYER_FIELDS = {
    "Substrate_stack_sequence": "HAS_SUBSTRATE_LAYER",
    "ETL_stack_sequence": "HAS_ETL_LAYER",
    "HTL_stack_sequence": "HAS_HTL_LAYER",
    "Backcontact_stack_sequence": "HAS_BACKCONTACT_LAYER",
    "Encapsulation_stack_sequence": "HAS_ENCAPSULATION_LAYER",
    "Add_lay_front_stack_sequence": "HAS_FRONT_ADDITIONAL_LAYER",
    "Add_lay_back_stack_sequence": "HAS_BACK_ADDITIONAL_LAYER",
}
PROCESS_FIELDS = [
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_quenching_media",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_deposition_thermal_annealing_atmosphere",
    "Perovskite_deposition_after_treatment_of_formed_perovskite",
]
PERFORMANCE_FIELDS = {
    "JV_default_PCE": ("PCE", "%"),
    "JV_default_Voc": ("Voc", "V"),
    "JV_default_Jsc": ("Jsc", "mA cm-2"),
    "JV_default_FF": ("FF", "%"),
    "JV_hysteresis_index": ("hysteresis_index", ""),
    "Stabilised_performance_PCE": ("stabilised_PCE", "%"),
    "EQE_integrated_Jsc": ("EQE_integrated_Jsc", "mA cm-2"),
}
STABILITY_FIELDS = {
    "Stability_time_total_exposure": ("total_exposure", "h"),
    "Stability_PCE_T80": ("T80", "h"),
    "Stability_PCE_T95": ("T95", "h"),
    "Stability_PCE_end_of_experiment": ("end_PCE_or_retention", ""),
    "Stability_PCE_after_1000_h": ("PCE_or_retention_after_1000h", ""),
}
CHARACTERIZATION_HINTS = {
    "jv_curve": "J-V",
    "eqe_curve": "EQE",
    "stability_curve": "stability",
    "xrd_pattern": "XRD",
    "pl_uvvis": "PL/UV-vis",
    "kpfm": "KPFM",
    "npfm": "nPFM",
    "stem_edx": "STEM-EDX",
    "device_schematic": "device schematic",
}

SUMMARY_CLAIM_FIELDS = {
    "one_sentence_summary": "Finding",
    "fabrication_highlights": "ProcessFinding",
    "synthesis_or_processing_insights": "ProcessFinding",
    "stability_summary": "StabilityFinding",
    "structure_composition_insights": "StructureCompositionFinding",
    "mechanism_or_interpretation": "Mechanism",
    "main_contribution": "Contribution",
    "interpretive_takeaways": "Finding",
    "data_quality_notes": "DataQualityClaim",
}

SUMMARY_EVIDENCE_FIELDS = {
    "characterization_and_evidence": "CharacterizationEvidence",
    "figure_or_visual_evidence": "SummaryFigureEvidence",
}

DIRECTIONAL_PATTERNS = [
    (re.compile(r"^(.+?)\s+(?:significantly\s+)?(?:improves?|enhances?|boosts?)\s+(.+)$", re.I), "IMPROVES"),
    (re.compile(r"^(.+?)\s+(?:significantly\s+)?(?:reduces?|decreases?|lowers?|suppresses?|mitigates?)\s+(.+)$", re.I), "REDUCES"),
    (re.compile(r"^(.+?)\s+(?:significantly\s+)?(?:increases?|raises?|extends?|broadens?)\s+(.+)$", re.I), "INCREASES"),
    (re.compile(r"^(.+?)\s+(?:leads? to|results? in|causes?|drives?|induces?|enables?|facilitates?|promotes?)\s+(.+)$", re.I), "PROMOTES"),
    (re.compile(r"^(.+?)\s+(?:prevents?|inhibits?|hinders?)\s+(.+)$", re.I), "INHIBITS"),
    (re.compile(r"^(.+?)\s+(?:is|are|was|were)\s+(?:strongly\s+)?(?:associated|correlated)\s+with\s+(.+)$", re.I), "ASSOCIATED_WITH"),
    (re.compile(r"^(.+?)\s+(?:is|are|was|were)\s+attributed\s+to\s+(.+)$", re.I), "ATTRIBUTED_TO"),
]

LEARNED_RELATION_TYPES = {rel for _, rel in DIRECTIONAL_PATTERNS}
KG_REASONING_POLICIES = {
    "current": {
        "short_name": "Current",
        "school": "Current evidence-grounded LiteratureAgent graph",
        "meaning": "Baseline graph without an added reasoning-policy lens.",
    },
    "humean_evidence": {
        "short_name": "Humean",
        "school": "Evidence grounding and cautious induction",
        "meaning": "Emphasizes direct evidence, evidence-linked claims, and weakly supported claims.",
    },
    "kantian_constraints": {
        "short_name": "Kantian",
        "school": "Rules, schema constraints, units, and consistency checking",
        "meaning": "Emphasizes schema-validity, missing target fields, unit consistency, and extraction constraints.",
    },
    "socratic_uncertainty": {
        "short_name": "Socratic",
        "school": "Questioning, uncertainty, assumptions, and missing evidence",
        "meaning": "Emphasizes uncertainty, missing evidence, research gaps, and next evidence needed.",
    },
    "cartesian_decomposition": {
        "short_name": "Cartesian",
        "school": "Stepwise decomposition, deduction, and calculation",
        "meaning": "Emphasizes process chains, measurable steps, and device-to-metric decomposition.",
    },
    "aristotelian_classification": {
        "short_name": "Aristotelian",
        "school": "Classification, categories, definitions, and causes",
        "meaning": "Emphasizes paper, material, device, characterization, and claim categories.",
    },
    "hegelian_conflict_resolution": {
        "short_name": "Hegelian",
        "school": "Contradiction, tension, and synthesis",
        "meaning": "Emphasizes conflicting claims, mixed evidence, and tensions across papers.",
    },
    "platonic_abstraction": {
        "short_name": "Platonic",
        "school": "Abstraction, ideal forms, and general patterns",
        "meaning": "Emphasizes general scientific patterns abstracted from specific claims.",
    },
}

CLAIM_PREFIX_RE = re.compile(
    r"^(?:the\s+(?:study|work|paper|authors?)\s+(?:shows?|demonstrates?|reveals?|finds?|reports?|suggests?|"
    r"indicates?|highlights?|concludes?|interprets?)\s+(?:that\s+)?|results?\s+(?:show|indicate)\s+(?:that\s+)?)",
    re.I,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "with", "which", "while", "through", "than", "due", "using",
}


def as_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if clean(item)]
    if isinstance(value, str) and clean(value):
        return [value.strip()]
    return []


def statement_tokens(value: Any) -> set[str]:
    text = clean(value) or ""
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9+\-]{1,}", text.lower())
        if token not in STOPWORDS and not token.isdigit()
    }


def overlap_score(left: Any, right: Any) -> float:
    left_tokens = statement_tokens(left)
    right_tokens = statement_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / max(1, min(len(left_tokens), len(right_tokens)))


def clean_concept_phrase(value: str) -> str:
    text = CLAIM_PREFIX_RE.sub("", value.strip(" .;:-"))
    text = re.split(r"\s*,\s*(?:as|which|while|because|due to|with)\b", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"^(?:that|the)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(?:can|may|could)$", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" .;:-")
    return text[:180]


def concept_labels(value: str) -> List[str]:
    lowered = value.lower()
    labels = ["ScientificConcept"]
    if any(term in lowered for term in ["doping", "additive", "treatment", "passivation", "anneal", "spacer", "cation", "ligand"]):
        labels.append("Intervention")
    if any(term in lowered for term in ["transport", "transfer", "trap", "recombination", "segregation", "formation", "dissociation", "mechanism"]):
        labels.append("MechanismConcept")
    if any(term in lowered for term in ["pce", "efficiency", "stability", "band gap", "lifetime", "absorption", "luminescence", "performance", "defect"]):
        labels.append("Outcome")
    return labels


def directional_statement(statement: str) -> Optional[Tuple[str, str, str]]:
    candidate = CLAIM_PREFIX_RE.sub("", statement.strip(" .;:-"))
    for pattern, relation in DIRECTIONAL_PATTERNS:
        match = pattern.match(candidate)
        if not match:
            continue
        subject = clean_concept_phrase(match.group(1))
        object_ = clean_concept_phrase(match.group(2))
        if 2 <= len(subject.split()) <= 24 and 2 <= len(object_.split()) <= 30:
            return subject, relation, object_
    return None


def recursive_evidence_strings(data: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[str, str]]:
    if isinstance(data, dict):
        for key, value in data.items():
            key_path = path + (str(key),)
            if isinstance(value, str) and "evidence" in str(key).lower() and clean(value):
                yield ".".join(key_path), value
            else:
                yield from recursive_evidence_strings(value, key_path)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from recursive_evidence_strings(value, path + (str(index),))


def recursive_structured_observations(data: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(data, dict):
        scalar = {
            str(key): value for key, value in data.items()
            if not isinstance(value, (dict, list)) and clean(value) is not None
        }
        if scalar and path:
            yield ".".join(path), scalar
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                yield from recursive_structured_observations(value, path + (str(key),))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from recursive_structured_observations(value, path + (str(index),))


def clean(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING else text


def norm(value: Any) -> str:
    text = clean(value) or ""
    text = text.lower().replace("â€“", "-").replace("â€”", "-")
    return re.sub(r"[^a-z0-9+]+", "", text)


def stable_id(label: str, value: Any) -> str:
    key = f"{label}|{norm(value)}"
    return f"{label.lower()}:{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


def split_values(value: Any, stack: bool = False) -> List[str]:
    text = clean(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if clean(x)]
        except Exception:
            pass
    pattern = r"\s*(?:/|>|→|\||;)\s*" if stack else r"\s*(?:;|\||\n)\s*"
    values = [x.strip(" []'\"") for x in re.split(pattern, text)]
    return list(dict.fromkeys(x for x in values if x and x.lower() not in MISSING))


def numeric_value(value: Any) -> Optional[float]:
    text = clean(value)
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def xml_safe(value: Any) -> str:
    """Remove characters forbidden by XML 1.0 while preserving CSV/JSON evidence."""
    text = str(value)
    return "".join(
        char for char in text
        if char in {"\t", "\n", "\r"}
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
    )


def flatten_ontology(data: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[str, str, str]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"version", "description"}:
                continue
            if isinstance(value, list):
                canonical = path[-1] if key in {"terms", "keywords", "synonyms", "aliases"} and path else str(key)
                category = "/".join(path[:-1] if key in {"terms", "keywords", "synonyms", "aliases"} else path)
                for synonym in value:
                    yield str(synonym), canonical, category
            else:
                yield from flatten_ontology(value, path + (str(key),))


def ontology_index(path: Path) -> Dict[str, Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    index: Dict[str, Dict[str, str]] = {}
    for synonym, canonical, category in flatten_ontology(data):
        root = category.split("/", 1)[0]
        # Family gates and figure-task terms are retrieval keywords, not
        # canonical scientific entities.
        if root in {"family_gate_terms", "figure_task_ontology", "family_ontology_map", "device_layers"}:
            continue
        key = norm(synonym)
        if key and key not in index:
            index[key] = {"canonical": canonical, "category": category}
    return index


class GraphBuilder:
    def __init__(self, ontology: Dict[str, Dict[str, str]], source_csv: Path, work_dir: Path):
        self.ontology = ontology
        self.source_csv = source_csv
        self.work_dir = work_dir
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self.ontology_usage = Counter()
        self.processed_papers: set[str] = set()
        self.claim_support = Counter()
        self.reasoning_policy = "current"
        self.reasoning_report_rows: List[Dict[str, Any]] = []

    def canonical(self, name: str) -> Tuple[str, str]:
        hit = self.ontology.get(norm(name))
        root = hit["category"].split("/", 1)[0] if hit else ""
        if hit and root in {"a_site_cations", "b_site_cations", "x_site_anions", "additives_passivators", "solvents"}:
            self.ontology_usage[(hit["category"], hit["canonical"])] += 1
            return hit["canonical"], hit["category"]
        return name, ""

    def add_node(self, node_id: str, labels: Sequence[str], name: str, **props: Any) -> str:
        existing = self.nodes.get(node_id, {"node_id": node_id, "labels": set(), "name": name})
        existing["labels"].update(labels)
        if not existing.get("name"):
            existing["name"] = name
        for key, value in props.items():
            if clean(value) is not None and clean(existing.get(key)) is None:
                existing[key] = value
        self.nodes[node_id] = existing
        return node_id

    def add_edge(self, start: str, end: str, rel_type: str, source_field: str, record_id: str, **props: Any) -> str:
        raw = f"{start}|{end}|{rel_type}|{source_field}|{record_id}|{json_text(props)}"
        edge_id = f"rel:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"
        if edge_id not in self.edges:
            self.edges[edge_id] = {
                "relationship_id": edge_id,
                "start_id": start,
                "end_id": end,
                "type": rel_type,
                "source_field": source_field,
                "source_record_id": record_id,
                "provenance_source": str(self.source_csv),
                **props,
            }
        return edge_id

    def apply_reasoning_policy_lens(self, policy_id: str) -> None:
        """Add an optional reasoning-school overlay without changing the baseline graph."""
        if policy_id in {"", "none", "current"}:
            self.reasoning_policy = "current"
            return
        if policy_id not in KG_REASONING_POLICIES:
            raise ValueError(f"Unsupported reasoning policy for KG view: {policy_id}")

        self.reasoning_policy = policy_id
        local = KG_REASONING_POLICIES[policy_id]
        policy = REASONING_POLICIES.get(policy_id)
        policy_node = stable_id("ReasoningPolicy", policy_id)
        self.add_node(
            policy_node,
            ["ReasoningPolicy", "PolicyLens"],
            local["short_name"],
            policy_id=policy_id,
            school_of_thought=getattr(policy, "school_of_thought", local["school"]),
            simple_meaning=getattr(policy, "simple_meaning", local["meaning"]),
            prompt_instruction=getattr(policy, "prompt_instruction", ""),
            output_checks=" | ".join(getattr(policy, "output_checks", [])),
        )

        claim_nodes = [node for node in self.nodes.values() if "Claim" in node["labels"]]
        paper_nodes = [node for node in self.nodes.values() if "Paper" in node["labels"]]
        device_nodes = [node for node in self.nodes.values() if "Device" in node["labels"]]
        measurement_nodes = [node for node in self.nodes.values() if "Measurement" in node["labels"]]
        material_nodes = [node for node in self.nodes.values() if "Material" in node["labels"]]

        if policy_id == "humean_evidence":
            for claim in claim_nodes:
                support = clean(claim.get("support_status")) or "unknown"
                claim["reasoning_policy"] = policy_id
                claim["evidence_grounding_status"] = support
                rel = "POLICY_PRIORITIZES_EVIDENCE_LINKED_CLAIM" if support == "evidence_linked" else "POLICY_FLAGS_WEAK_EVIDENCE"
                self.add_edge(
                    policy_node,
                    claim["node_id"],
                    rel,
                    "reasoning_policy",
                    policy_id,
                    evidence_support_count=claim.get("evidence_support_count", 0),
                    visual_support_count=claim.get("visual_support_count", 0),
                    direct_text_support_count=claim.get("direct_text_support_count", 0),
                )
                self.reasoning_report_rows.append({
                    "policy_id": policy_id,
                    "node_id": claim["node_id"],
                    "node_type": "Claim",
                    "status": support,
                    "reason": "Direct evidence linked" if support == "evidence_linked" else "Claim currently depends on summary text only",
                })

        elif policy_id == "kantian_constraints":
            for node in [*device_nodes, *measurement_nodes, *material_nodes]:
                node["reasoning_policy"] = policy_id
                node["constraint_checked"] = True
                self.add_edge(policy_node, node["node_id"], "POLICY_CHECKS_SCHEMA_CONSTRAINTS", "reasoning_policy", policy_id)
            for device in device_nodes:
                outgoing = [edge for edge in self.edges.values() if edge["start_id"] == device["node_id"]]
                has_performance = any(edge["type"] == "HAS_PERFORMANCE_MEASUREMENT" for edge in outgoing)
                has_stability = any(edge["type"] == "HAS_STABILITY_TEST" for edge in outgoing)
                if not has_performance or not has_stability:
                    issue_text = "missing_performance_or_stability_target"
                    issue_id = stable_id("ConstraintIssue", f"{device['node_id']}|{issue_text}")
                    self.add_node(
                        issue_id,
                        ["ReasoningFinding", "ConstraintIssue"],
                        issue_text,
                        policy_id=policy_id,
                        missing_performance=not has_performance,
                        missing_stability=not has_stability,
                    )
                    self.add_edge(device["node_id"], issue_id, "HAS_SCHEMA_CONSTRAINT_WARNING", "reasoning_policy", policy_id)
                    self.reasoning_report_rows.append({
                        "policy_id": policy_id,
                        "node_id": device["node_id"],
                        "node_type": "Device",
                        "status": "warning",
                        "reason": issue_text,
                    })

        elif policy_id == "socratic_uncertainty":
            for claim in claim_nodes:
                support = clean(claim.get("support_status")) or "unknown"
                if support == "evidence_linked":
                    continue
                gap_text = f"Need stronger evidence for: {claim.get('statement', '')[:140]}"
                gap_id = stable_id("UncertaintyGap", f"{claim['node_id']}|{gap_text}")
                self.add_node(
                    gap_id,
                    ["ReasoningFinding", "UncertaintyGap", "ResearchGap"],
                    gap_text,
                    policy_id=policy_id,
                    source_claim_id=claim["node_id"],
                    next_evidence_needed="Link claim to direct text, figure, table, or extracted measurement evidence.",
                )
                claim["reasoning_policy"] = policy_id
                claim["uncertainty_flag"] = "needs_direct_evidence"
                self.add_edge(claim["node_id"], gap_id, "HAS_UNCERTAINTY_GAP", "reasoning_policy", policy_id)
                self.add_edge(policy_node, gap_id, "POLICY_IDENTIFIES_GAP", "reasoning_policy", policy_id)
                self.reasoning_report_rows.append({
                    "policy_id": policy_id,
                    "node_id": claim["node_id"],
                    "node_type": "Claim",
                    "status": "needs_evidence",
                    "reason": "Summary-only claim needs direct support before being treated as strong knowledge",
                })
            for paper in paper_nodes:
                outgoing = [edge for edge in self.edges.values() if edge["start_id"] == paper["node_id"]]
                if not any(edge["type"] in {"HAS_TEXT_EVIDENCE", "HAS_FIGURE_EVIDENCE", "ASSERTS_CLAIM"} for edge in outgoing):
                    gap_id = stable_id("UncertaintyGap", f"{paper['node_id']}|limited_extracted_knowledge")
                    self.add_node(
                        gap_id,
                        ["ReasoningFinding", "UncertaintyGap"],
                        "Limited extracted knowledge for paper",
                        policy_id=policy_id,
                    )
                    self.add_edge(paper["node_id"], gap_id, "HAS_UNCERTAINTY_GAP", "reasoning_policy", policy_id)
                    self.reasoning_report_rows.append({
                        "policy_id": policy_id,
                        "node_id": paper["node_id"],
                        "node_type": "Paper",
                        "status": "limited_knowledge",
                        "reason": "Paper has little linked claim/evidence structure in current KG",
                    })

        elif policy_id == "cartesian_decomposition":
            for device in device_nodes:
                related_edges = [edge for edge in self.edges.values() if edge["start_id"] == device["node_id"]]
                ordered_types = [
                    ("USES_ABSORBER", "absorber material"),
                    ("FABRICATED_BY", "fabrication/process"),
                    ("HAS_PERFORMANCE_MEASUREMENT", "performance metric"),
                    ("HAS_STABILITY_TEST", "stability test"),
                ]
                previous_step = None
                for step_number, (rel_type, step_name) in enumerate(ordered_types, start=1):
                    if not any(edge["type"] == rel_type for edge in related_edges):
                        continue
                    step_id = stable_id("ReasoningStep", f"{device['node_id']}|{policy_id}|{step_number}|{rel_type}")
                    self.add_node(
                        step_id,
                        ["ReasoningFinding", "DecompositionStep"],
                        f"{step_number}. {step_name}",
                        policy_id=policy_id,
                        step_number=step_number,
                        relationship_type=rel_type,
                    )
                    self.add_edge(policy_node, step_id, "POLICY_DECOMPOSES_STEP", "reasoning_policy", policy_id)
                    self.add_edge(device["node_id"], step_id, "HAS_DECOMPOSITION_STEP", "reasoning_policy", policy_id)
                    if previous_step:
                        self.add_edge(previous_step, step_id, "NEXT_REASONING_STEP", "reasoning_policy", policy_id)
                    previous_step = step_id
                    self.reasoning_report_rows.append({
                        "policy_id": policy_id,
                        "node_id": device["node_id"],
                        "node_type": "Device",
                        "status": "decomposed",
                        "reason": f"Device construction decomposed through {step_name}",
                    })

        elif policy_id == "aristotelian_classification":
            for node in [*paper_nodes, *device_nodes, *material_nodes, *measurement_nodes, *claim_nodes]:
                primary = sorted(node["labels"])[0] if node.get("labels") else "Entity"
                if "Claim" in node.get("labels", set()):
                    primary = clean(node.get("claim_type")) or "Claim"
                category_id = stable_id("ReasoningCategory", f"{policy_id}|{primary}")
                self.add_node(
                    category_id,
                    ["ReasoningFinding", "ReasoningCategory"],
                    primary,
                    policy_id=policy_id,
                    category_basis="node_label_or_claim_type",
                )
                self.add_edge(node["node_id"], category_id, "BELONGS_TO_REASONING_CATEGORY", "reasoning_policy", policy_id)
            self.reasoning_report_rows.append({
                "policy_id": policy_id,
                "node_id": policy_node,
                "node_type": "ReasoningPolicy",
                "status": "classified",
                "reason": "Graph entities grouped by definitions, paper/material/device roles, and claim types",
            })

        elif policy_id == "hegelian_conflict_resolution":
            claim_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
            opposite = {
                "IMPROVES": {"REDUCES", "INHIBITS"},
                "PROMOTES": {"REDUCES", "INHIBITS"},
                "INCREASES": {"REDUCES"},
                "REDUCES": {"IMPROVES", "PROMOTES", "INCREASES"},
                "INHIBITS": {"IMPROVES", "PROMOTES"},
            }
            for edge in self.edges.values():
                if edge["type"] in LEARNED_RELATION_TYPES:
                    claim_by_pair[(edge["start_id"], edge["end_id"])].append(edge)
            for (subject_id, object_id), rels in claim_by_pair.items():
                rel_types = {edge["type"] for edge in rels}
                has_tension = any(bool(opposite.get(rel, set()) & rel_types) for rel in rel_types)
                mixed_support = len({edge.get("assertion_status", "") for edge in rels}) > 1 or len(rels) > 1
                if not has_tension and not mixed_support:
                    continue
                tension_id = stable_id("ReasoningTension", f"{subject_id}|{object_id}|{sorted(rel_types)}")
                self.add_node(
                    tension_id,
                    ["ReasoningFinding", "Tension", "ConflictOrSynthesis"],
                    "Tension between reported relationships",
                    policy_id=policy_id,
                    relationship_types=";".join(sorted(rel_types)),
                    relation_count=len(rels),
                    has_direct_opposition=has_tension,
                )
                self.add_edge(policy_node, tension_id, "POLICY_IDENTIFIES_TENSION", "reasoning_policy", policy_id)
                self.add_edge(subject_id, tension_id, "HAS_RELATIONSHIP_TENSION", "reasoning_policy", policy_id)
                self.add_edge(tension_id, object_id, "TENSION_CONCERNS", "reasoning_policy", policy_id)
                self.reasoning_report_rows.append({
                    "policy_id": policy_id,
                    "node_id": tension_id,
                    "node_type": "Tension",
                    "status": "tension",
                    "reason": f"Relationship set contains {', '.join(sorted(rel_types))}",
                })

        elif policy_id == "platonic_abstraction":
            abstraction_rules = [
                ("performance_improvement", ["pce", "efficiency", "performance", "voc", "jsc", "fill factor"]),
                ("stability_retention", ["stability", "retention", "t80", "t95", "lifetime", "aging"]),
                ("interface_passivation", ["interface", "passivation", "defect", "trap", "recombination"]),
                ("composition_dimensionality", ["composition", "cation", "halide", "2d", "3d", "dimensional"]),
                ("processing_control", ["anneal", "solvent", "deposition", "processing", "crystallization"]),
            ]
            for claim in claim_nodes:
                text = f"{claim.get('statement', '')} {claim.get('claim_type', '')}".lower()
                matched = [name for name, terms in abstraction_rules if any(term in text for term in terms)]
                if not matched:
                    matched = ["general_literature_pattern"]
                for abstraction in matched:
                    abs_id = stable_id("AbstractPrinciple", f"{policy_id}|{abstraction}")
                    self.add_node(
                        abs_id,
                        ["ReasoningFinding", "AbstractPrinciple"],
                        abstraction,
                        policy_id=policy_id,
                        abstraction_basis="claim_text_keyword_pattern",
                    )
                    self.add_edge(claim["node_id"], abs_id, "INSTANTIATES_ABSTRACT_PATTERN", "reasoning_policy", policy_id)
                    self.add_edge(policy_node, abs_id, "POLICY_ABSTRACTS_PATTERN", "reasoning_policy", policy_id)
                self.reasoning_report_rows.append({
                    "policy_id": policy_id,
                    "node_id": claim["node_id"],
                    "node_type": "Claim",
                    "status": "abstracted",
                    "reason": "Claim grouped into one or more abstract scientific principles",
                })

    def paper_id(self, row: pd.Series) -> str:
        doi = clean(row.get("Ref_DOI_number"))
        sample = clean(row.get("Ref_internal_sample_id")) or clean(row.get("Ref_original_filename_data_upload")) or "unknown"
        return stable_id("Paper", doi or DEVICE_SUFFIX_RE.sub("", sample))

    def artifact_paths(self, paper_slug: str) -> Dict[str, str]:
        candidates = {
            "summary_path": self.work_dir / "paper_summaries_json" / f"{paper_slug}_summary.json",
            "figure_evidence_path": self.work_dir / "figure_evidence" / f"{paper_slug}_figure_evidence_v18.json",
            "paper_json_path": self.work_dir / "json" / f"{paper_slug}.json",
        }
        return {k: str(v) for k, v in candidates.items() if v.exists()}

    def add_figure_evidence(self, paper_id: str, paper_slug: str, record_id: str) -> List[Tuple[str, str, str]]:
        evidence_nodes: List[Tuple[str, str, str]] = []
        path = self.work_dir / "figure_evidence" / f"{paper_slug}_figure_evidence_v18.json"
        if not path.exists():
            return evidence_nodes
        try:
            figures = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return evidence_nodes
        for figure in figures if isinstance(figures, list) else []:
            figure_name = clean(figure.get("figure_id")) or "figure"
            figure_type = clean(figure.get("figure_type")) or "unknown"
            evidence_text = clean(figure.get("text")) or ""
            node_id = stable_id("FigureEvidence", f"{paper_slug}|{figure_name}")
            self.add_node(
                node_id,
                ["Evidence", "FigureEvidence"],
                figure_name,
                figure_type=figure_type,
                priority=figure.get("priority"),
                score=figure.get("score"),
                evidence_text=evidence_text,
                source=figure.get("source"),
                evidence_path=str(path),
                needs_human_check=figure.get("needs_human_check"),
            )
            self.add_edge(paper_id, node_id, "HAS_FIGURE_EVIDENCE", "figure_evidence", record_id, evidence_path=str(path))
            char_name = CHARACTERIZATION_HINTS.get(figure_type, figure_type)
            char_id = stable_id("Characterization", char_name)
            self.add_node(char_id, ["Characterization"], char_name, ontology_category="characterization")
            self.add_edge(paper_id, char_id, "USES_CHARACTERIZATION", "figure_type", record_id, supported_by=node_id)
            self.add_edge(node_id, char_id, "EVIDENCE_FOR", "figure_type", record_id)
            evidence_nodes.append((node_id, evidence_text, "visual"))
        return evidence_nodes

    def add_raw_text_evidence(self, paper_id: str, paper_slug: str, record_id: str, limit: int = 30) -> List[Tuple[str, str, str]]:
        evidence_nodes: List[Tuple[str, str, str]] = []
        seen: set[str] = set()
        paths = sorted((self.work_dir / "raw_llm_json").glob(f"{paper_slug}_*.json"))
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for field_path, evidence_text in recursive_evidence_strings(data):
                normalized = norm(evidence_text)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                node_id = stable_id("TextEvidence", f"{paper_slug}|{evidence_text}")
                self.add_node(
                    node_id,
                    ["Evidence", "TextEvidence", "RawExtractionEvidence"],
                    evidence_text[:120],
                    evidence_text=evidence_text,
                    evidence_path=str(path),
                    evidence_field=field_path,
                    source="raw_llm_json",
                )
                self.add_edge(paper_id, node_id, "HAS_TEXT_EVIDENCE", field_path, record_id, evidence_path=str(path))
                evidence_nodes.append((node_id, evidence_text, "text"))
                if len(evidence_nodes) >= limit:
                    return evidence_nodes
        return evidence_nodes

    def add_summary_evidence(
        self,
        paper_id: str,
        paper_slug: str,
        record_id: str,
        summary: Dict[str, Any],
        summary_path: Path,
    ) -> List[Tuple[str, str, str]]:
        evidence_nodes: List[Tuple[str, str, str]] = []
        for field, evidence_type in SUMMARY_EVIDENCE_FIELDS.items():
            for evidence_text in as_text_list(summary.get(field)):
                node_id = stable_id("SummaryEvidence", f"{paper_slug}|{field}|{evidence_text}")
                self.add_node(
                    node_id,
                    ["Evidence", "TextEvidence", "SummaryEvidence", evidence_type],
                    evidence_text[:120],
                    evidence_text=evidence_text,
                    evidence_path=str(summary_path),
                    evidence_field=field,
                    source="paper_summary",
                )
                self.add_edge(paper_id, node_id, "HAS_SUMMARY_EVIDENCE", field, record_id, evidence_path=str(summary_path))
                evidence_nodes.append((node_id, evidence_text, "summary"))
        return evidence_nodes

    def add_summary_material_system(
        self,
        paper_id: str,
        paper_slug: str,
        record_id: str,
        summary: Dict[str, Any],
        summary_path: Path,
    ) -> None:
        system = summary.get("perovskite_system")
        if not system:
            return
        if isinstance(system, dict):
            composition = clean(system.get("composition")) or clean(summary.get("materials_focus")) or "Summary-extracted material system"
            properties = {
                "composition": clean(system.get("composition")),
                "dimensionality": clean(system.get("dimensionality")),
                "spacer_additive": clean(system.get("spacer/additive")),
                "chemistry": clean(system.get("A/B/halide chemistry")),
                "study_scope": clean(system.get("experimental/review scope")),
            }
        else:
            composition = clean(system) or "Summary-extracted material system"
            properties = {"description": clean(system)}
        system_id = stable_id("MaterialSystem", f"{paper_slug}|{composition}")
        self.add_node(
            system_id,
            ["MaterialSystem", "SummaryExtractedEntity"],
            composition,
            source="paper_summary",
            source_path=str(summary_path),
            **properties,
        )
        self.add_edge(paper_id, system_id, "STUDIES_MATERIAL_SYSTEM", "perovskite_system", record_id, source_path=str(summary_path))
        for key, relation in [
            ("dimensionality", "HAS_SUMMARY_DIMENSIONALITY"),
            ("spacer_additive", "HAS_SUMMARY_SPACER_ADDITIVE"),
            ("chemistry", "HAS_SUMMARY_CHEMISTRY"),
        ]:
            value = clean(properties.get(key))
            if value:
                concept_id = stable_id("ScientificConcept", value)
                self.add_node(concept_id, concept_labels(value) + ["SummaryExtractedEntity"], value)
                self.add_edge(system_id, concept_id, relation, f"perovskite_system.{key}", record_id, source_path=str(summary_path))

    def add_summary_performance(
        self,
        paper_id: str,
        paper_slug: str,
        record_id: str,
        summary: Dict[str, Any],
        summary_path: Path,
    ) -> None:
        performance = summary.get("best_reported_performance")
        if not isinstance(performance, dict):
            return
        units = {"pce": "%", "voc": "V", "jsc": "mA cm-2", "ff": "%"}
        context = clean(performance.get("context"))
        for metric in ["pce", "voc", "jsc", "ff"]:
            value = numeric_value(performance.get(metric))
            if value is None:
                continue
            measurement_id = stable_id("SummaryMeasurement", f"{paper_slug}|{metric}|{value}|{context}")
            self.add_node(
                measurement_id,
                ["Measurement", "PerformanceMeasurement", "SummaryMeasurement", "SummaryExtractedEntity"],
                metric.upper(),
                metric=metric.upper(),
                value=value,
                unit=units[metric],
                context=context,
                validation_status="summary_extracted_candidate",
                source_path=str(summary_path),
            )
            self.add_edge(
                paper_id,
                measurement_id,
                "REPORTS_SUMMARY_PERFORMANCE",
                f"best_reported_performance.{metric}",
                record_id,
                source_path=str(summary_path),
                validation_status="summary_extracted_candidate",
            )

    def add_structured_observations(self, paper_id: str, paper_slug: str, record_id: str, limit: int = 160) -> None:
        observation_count = 0
        paths = sorted((self.work_dir / "raw_llm_json").glob(f"{paper_slug}_*_consolidated.json"))
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for field_path, payload in recursive_structured_observations(data):
                if observation_count >= limit:
                    return
                observation_type = re.sub(r"\.\d+(?=\.|$)", "", field_path)
                property_name = observation_type.split(".")[-1].replace("_", " ")
                payload_text = "; ".join(f"{key}={value}" for key, value in payload.items())
                observation_id = stable_id("StructuredObservation", f"{paper_slug}|{observation_type}|{json_text(payload)}")
                self.add_node(
                    observation_id,
                    ["Observation", "StructuredObservation", "SummaryExtractedEntity"],
                    f"{property_name}: {payload_text}"[:180],
                    observation_type=observation_type,
                    payload_json=json_text(payload),
                    source_path=str(path),
                    validation_status="llm_structured_extraction",
                )
                self.add_edge(
                    paper_id,
                    observation_id,
                    "REPORTS_STRUCTURED_OBSERVATION",
                    observation_type,
                    record_id,
                    source_path=str(path),
                    validation_status="llm_structured_extraction",
                )
                property_id = stable_id("ScientificProperty", property_name)
                self.add_node(property_id, ["ScientificConcept", "ScientificProperty"], property_name)
                self.add_edge(observation_id, property_id, "OBSERVES_PROPERTY", observation_type, record_id)
                for entity_key in ["material", "composition", "sample", "device", "phase", "additive", "treatment"]:
                    entity_value = clean(payload.get(entity_key))
                    if not entity_value:
                        continue
                    entity_id = stable_id("ScientificConcept", entity_value)
                    self.add_node(entity_id, concept_labels(entity_value) + ["StructuredExtractedEntity"], entity_value)
                    self.add_edge(observation_id, entity_id, "OBSERVED_FOR", f"{observation_type}.{entity_key}", record_id)
                observation_count += 1

    def add_claim(
        self,
        paper_id: str,
        paper_slug: str,
        record_id: str,
        claim_type: str,
        source_field: str,
        statement: str,
        summary_path: Path,
        evidence_nodes: Sequence[Tuple[str, str, str]],
    ) -> str:
        claim_id = stable_id("Claim", f"{paper_slug}|{claim_type}|{statement}")
        self.add_node(
            claim_id,
            ["Claim", claim_type],
            statement[:140],
            statement=statement,
            claim_type=claim_type,
            source_field=source_field,
            source_path=str(summary_path),
            extraction_status="summary_derived_claim",
        )
        self.add_edge(paper_id, claim_id, "ASSERTS_CLAIM", source_field, record_id, source_path=str(summary_path))

        directional = directional_statement(statement)
        if directional:
            subject, relation, object_ = directional
            subject_id = stable_id("ScientificConcept", subject)
            object_id = stable_id("ScientificConcept", object_)
            self.add_node(subject_id, concept_labels(subject), subject)
            self.add_node(object_id, concept_labels(object_), object_)
            self.add_edge(claim_id, subject_id, "HAS_SUBJECT", source_field, record_id)
            self.add_edge(claim_id, object_id, "HAS_OBJECT", source_field, record_id)
            self.add_edge(
                subject_id,
                object_id,
                relation,
                source_field,
                record_id,
                claim_id=claim_id,
                paper_id=paper_id,
                statement=statement,
                source_path=str(summary_path),
                assertion_status="reported_claim",
            )

        supported = 0
        visual = 0
        direct_text = 0
        for evidence_id, evidence_text, evidence_kind in evidence_nodes:
            score = overlap_score(statement, evidence_text)
            threshold = 0.18 if evidence_kind == "visual" else 0.24
            if len(statement_tokens(statement) & statement_tokens(evidence_text)) < 2 or score < threshold:
                continue
            self.add_edge(
                evidence_id,
                claim_id,
                "SUPPORTS_CLAIM",
                source_field,
                record_id,
                overlap_score=round(score, 4),
                evidence_kind=evidence_kind,
            )
            supported += 1
            visual += int(evidence_kind == "visual")
            direct_text += int(evidence_kind == "text")
        claim = self.nodes[claim_id]
        claim["evidence_support_count"] = supported
        claim["visual_support_count"] = visual
        claim["direct_text_support_count"] = direct_text
        claim["support_status"] = "evidence_linked" if supported else "summary_only"
        self.claim_support[claim["support_status"]] += 1
        return claim_id

    def add_paper_knowledge(self, paper_id: str, paper_slug: str, record_id: str) -> None:
        summary_path = self.work_dir / "paper_summaries_json" / f"{paper_slug}_summary.json"
        if not summary_path.exists():
            self.add_figure_evidence(paper_id, paper_slug, record_id)
            return
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = payload.get("paper_summary", payload)
        except Exception:
            self.add_figure_evidence(paper_id, paper_slug, record_id)
            return
        if not isinstance(summary, dict):
            return

        evidence_nodes: List[Tuple[str, str, str]] = []
        evidence_nodes.extend(self.add_figure_evidence(paper_id, paper_slug, record_id))
        evidence_nodes.extend(self.add_summary_evidence(paper_id, paper_slug, record_id, summary, summary_path))
        evidence_nodes.extend(self.add_raw_text_evidence(paper_id, paper_slug, record_id))
        self.add_summary_material_system(paper_id, paper_slug, record_id, summary, summary_path)
        self.add_summary_performance(paper_id, paper_slug, record_id, summary, summary_path)
        self.add_structured_observations(paper_id, paper_slug, record_id)

        for field, claim_type in SUMMARY_CLAIM_FIELDS.items():
            for statement in as_text_list(summary.get(field)):
                self.add_claim(
                    paper_id,
                    paper_slug,
                    record_id,
                    claim_type,
                    field,
                    statement,
                    summary_path,
                    evidence_nodes,
                )

    def add_record(self, row: pd.Series, row_index: int) -> None:
        record_id = clean(row.get("Ref_internal_sample_id")) or f"row_{row_index}"
        paper_slug = DEVICE_SUFFIX_RE.sub("", record_id)
        paper_id = self.paper_id(row)
        artifacts = self.artifact_paths(paper_slug)
        paper_type_match = PAPER_TYPE_RE.search(clean(row.get("Ref_free_text_comment")) or "")
        paper_type = paper_type_match.group(1) if paper_type_match else ""
        paper_name = clean(row.get("Ref_DOI_number")) or clean(row.get("Ref_original_filename_data_upload")) or paper_slug
        self.add_node(
            paper_id,
            ["Paper"],
            paper_name,
            doi=clean(row.get("Ref_DOI_number")),
            lead_author=clean(row.get("Ref_lead_author")),
            publication_date=clean(row.get("Ref_publication_date")),
            journal=clean(row.get("Ref_journal")),
            paper_type=paper_type,
            source_pdf_url=clean(row.get("_source_pdf_url")),
            source_landing_page=clean(row.get("_source_landing_page")),
            **artifacts,
        )
        if paper_type:
            type_id = stable_id("PaperType", paper_type)
            self.add_node(type_id, ["PaperType"], paper_type)
            self.add_edge(paper_id, type_id, "CLASSIFIED_AS", "Ref_free_text_comment", record_id)

        device_id = stable_id("Device", record_id)
        self.add_node(
            device_id,
            ["Device"],
            record_id,
            architecture=clean(row.get("Cell_architecture")),
            stack_sequence=clean(row.get("Cell_stack_sequence")),
            flexible=clean(row.get("Cell_flexible")),
            semitransparent=clean(row.get("Cell_semitransparent")),
        )
        self.add_edge(paper_id, device_id, "REPORTS_DEVICE", "Ref_internal_sample_id", record_id)

        composition = clean(row.get("Perovskite_composition_short_form")) or clean(row.get("Perovskite_composition_long_form"))
        if composition:
            material_id = stable_id("Material", composition)
            self.add_node(
                material_id,
                ["Material", "Perovskite"],
                composition,
                short_form=clean(row.get("Perovskite_composition_short_form")),
                long_form=clean(row.get("Perovskite_composition_long_form")),
                band_gap=numeric_value(row.get("Perovskite_band_gap")),
                pl_max=numeric_value(row.get("Perovskite_pl_max")),
                thickness=numeric_value(row.get("Perovskite_thickness")),
            )
            self.add_edge(device_id, material_id, "USES_ABSORBER", "Perovskite_composition_short_form", record_id)
            for field, rel in CHEMICAL_FIELDS.items():
                for value in split_values(row.get(field)):
                    canonical, category = self.canonical(value)
                    chemical_id = stable_id("Chemical", canonical)
                    self.add_node(chemical_id, ["Chemical"], canonical, raw_name=value, ontology_category=category)
                    self.add_edge(material_id, chemical_id, rel, field, record_id)
            for field, dimension in [
                ("Perovskite_dimension_0D", "0D"),
                ("Perovskite_dimension_2D", "2D"),
                ("Perovskite_dimension_2D3D_mixture", "2D/3D mixture"),
                ("Perovskite_dimension_3D", "3D"),
                ("Perovskite_dimension_3D_with_2D_capping_layer", "3D with 2D capping layer"),
            ]:
                value = clean(row.get(field))
                if value and value.lower() not in {"false", "0", "no"}:
                    dim_id = stable_id("Dimensionality", dimension)
                    self.add_node(dim_id, ["Dimensionality"], dimension)
                    self.add_edge(material_id, dim_id, "HAS_DIMENSIONALITY", field, record_id)

        for field, rel in LAYER_FIELDS.items():
            previous: Optional[str] = None
            for position, layer in enumerate(split_values(row.get(field), stack=True), start=1):
                canonical, category = layer, ""
                layer_id = stable_id("LayerMaterial", canonical)
                self.add_node(layer_id, ["LayerMaterial"], canonical, raw_name=layer, ontology_category=category)
                self.add_edge(device_id, layer_id, rel, field, record_id, position=position)
                if previous:
                    self.add_edge(previous, layer_id, "NEXT_LAYER", field, record_id, device_id=device_id, position=position)
                previous = layer_id

        for field in PROCESS_FIELDS:
            value = clean(row.get(field))
            if value:
                process_id = stable_id("Process", f"{field}|{value}")
                self.add_node(process_id, ["Process"], value, process_field=field)
                self.add_edge(device_id, process_id, "FABRICATED_BY", field, record_id)

        for field, (metric, unit) in PERFORMANCE_FIELDS.items():
            value = numeric_value(row.get(field))
            if value is not None:
                measurement_id = stable_id("Measurement", f"{record_id}|{field}|{value}")
                self.add_node(measurement_id, ["Measurement", "PerformanceMeasurement"], metric, metric=metric, value=value, unit=unit)
                self.add_edge(device_id, measurement_id, "HAS_PERFORMANCE_MEASUREMENT", field, record_id)

        stability_values = {field: numeric_value(row.get(field)) for field in STABILITY_FIELDS}
        stability_meta = any(clean(row.get(field)) for field in ["Stability_protocol", "Stability_temperature_range", "Stability_relative_humidity_average_value", "Stability_light_intensity"])
        if any(v is not None for v in stability_values.values()) or stability_meta:
            test_id = stable_id("StabilityTest", record_id)
            self.add_node(
                test_id,
                ["StabilityTest"],
                f"Stability test for {record_id}",
                protocol=clean(row.get("Stability_protocol")),
                temperature=clean(row.get("Stability_temperature_range")),
                relative_humidity=clean(row.get("Stability_relative_humidity_average_value")),
                light_intensity=clean(row.get("Stability_light_intensity")),
                atmosphere=clean(row.get("Stability_atmosphere")),
            )
            self.add_edge(device_id, test_id, "HAS_STABILITY_TEST", "Stability_protocol", record_id)
            for field, (metric, unit) in STABILITY_FIELDS.items():
                value = stability_values[field]
                if value is not None:
                    measurement_id = stable_id("Measurement", f"{record_id}|{field}|{value}")
                    self.add_node(measurement_id, ["Measurement", "StabilityMeasurement"], metric, metric=metric, value=value, unit=unit)
                    self.add_edge(test_id, measurement_id, "HAS_STABILITY_MEASUREMENT", field, record_id)

        if paper_id not in self.processed_papers:
            self.add_paper_knowledge(paper_id, paper_slug, record_id)
            self.processed_papers.add(paper_id)

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        neo4j_dir = out_dir / "neo4j_import"
        portable_dir = out_dir / "portable"
        reports_dir = out_dir / "reports"
        views_dir = out_dir / "derived_views"
        plots_dir = out_dir / "plots"
        for path in [neo4j_dir, portable_dir, reports_dir, views_dir, plots_dir]:
            path.mkdir(parents=True, exist_ok=True)

        node_rows = []
        for node in self.nodes.values():
            row = {
                "node_id:ID": node["node_id"],
                "name": node.get("name", ""),
                ":LABEL": ";".join(sorted(node["labels"])),
                "properties_json": json_text({k: v for k, v in node.items() if k not in {"node_id", "name", "labels"}}),
            }
            node_rows.append(row)
        edge_rows = []
        for edge in self.edges.values():
            edge_rows.append({
                ":START_ID": edge["start_id"],
                ":END_ID": edge["end_id"],
                ":TYPE": edge["type"],
                "relationship_id": edge["relationship_id"],
                "source_field": edge.get("source_field", ""),
                "source_record_id": edge.get("source_record_id", ""),
                "provenance_source": edge.get("provenance_source", ""),
                "properties_json": json_text({k: v for k, v in edge.items() if k not in {"start_id", "end_id", "type", "relationship_id", "source_field", "source_record_id", "provenance_source"}}),
            })
        pd.DataFrame(node_rows).to_csv(neo4j_dir / "nodes.csv", index=False)
        pd.DataFrame(edge_rows).to_csv(neo4j_dir / "relationships.csv", index=False)

        with (portable_dir / "graph.jsonl").open("w", encoding="utf-8") as handle:
            for node in self.nodes.values():
                serial = {**node, "labels": sorted(node["labels"])}
                handle.write(json_text({"kind": "node", **serial}) + "\n")
            for edge in self.edges.values():
                handle.write(json_text({"kind": "relationship", **edge}) + "\n")

        graph = nx.MultiDiGraph()
        for node in self.nodes.values():
            graph.add_node(
                node["node_id"],
                labels=xml_safe(";".join(sorted(node["labels"]))),
                name=xml_safe(node.get("name", "")),
                properties_json=xml_safe(json_text({k: v for k, v in node.items() if k not in {"node_id", "labels", "name"}})),
            )
        for edge in self.edges.values():
            graph.add_edge(
                edge["start_id"],
                edge["end_id"],
                key=edge["relationship_id"],
                type=xml_safe(edge["type"]),
                source_field=xml_safe(edge.get("source_field", "")),
                source_record_id=xml_safe(edge.get("source_record_id", "")),
            )
        nx.write_graphml(graph, portable_dir / "literature_knowledge_graph.graphml")

        node_counts = Counter(label for node in self.nodes.values() for label in node["labels"])
        rel_counts = Counter(edge["type"] for edge in self.edges.values())
        dangling = [edge["relationship_id"] for edge in self.edges.values() if edge["start_id"] not in self.nodes or edge["end_id"] not in self.nodes]
        claim_nodes = [node for node in self.nodes.values() if "Claim" in node["labels"]]
        claim_type_counts = Counter(node.get("claim_type", "Claim") for node in claim_nodes)
        support_counts = Counter(node.get("support_status", "unknown") for node in claim_nodes)
        learned_edges = [edge for edge in self.edges.values() if edge["type"] in LEARNED_RELATION_TYPES]
        summary = {
            "graph_model": "evidence-grounded claim-centered scientific property graph",
            "reasoning_policy": self.reasoning_policy,
            "nodes": len(self.nodes),
            "relationships": len(self.edges),
            "scientific_claims": len(claim_nodes),
            "learned_directional_relationships": len(learned_edges),
            "claim_counts_by_type": dict(claim_type_counts),
            "claim_support_status": dict(support_counts),
            "node_counts_by_label": dict(node_counts),
            "relationship_counts_by_type": dict(rel_counts),
            "dangling_relationships": len(dangling),
            "source_csv": str(self.source_csv),
            "work_dir": str(self.work_dir),
        }
        (reports_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame([{"label": k, "count": v} for k, v in node_counts.most_common()]).to_csv(reports_dir / "node_counts_by_label.csv", index=False)
        pd.DataFrame([{"relationship_type": k, "count": v} for k, v in rel_counts.most_common()]).to_csv(reports_dir / "relationship_counts_by_type.csv", index=False)
        pd.DataFrame([{"ontology_category": k[0], "canonical_term": k[1], "usage_count": v} for k, v in self.ontology_usage.most_common()]).to_csv(reports_dir / "ontology_term_usage.csv", index=False)
        pd.DataFrame([{"relationship_id": x} for x in dangling]).to_csv(reports_dir / "dangling_relationships.csv", index=False)
        pd.DataFrame(self.reasoning_report_rows).to_csv(reports_dir / "reasoning_policy_findings.csv", index=False)
        pd.DataFrame([{
            "reasoning_policy": self.reasoning_policy,
            "nodes": len(self.nodes),
            "relationships": len(self.edges),
            "scientific_claims": len(claim_nodes),
            "learned_directional_relationships": len(learned_edges),
            "evidence_linked_claims": support_counts.get("evidence_linked", 0),
            "summary_only_claims": support_counts.get("summary_only", 0),
            "reasoning_findings": len(self.reasoning_report_rows),
        }]).to_csv(reports_dir / "reasoning_policy_summary.csv", index=False)
        pd.DataFrame([
            {
                "claim_id": node["node_id"],
                "claim_type": node.get("claim_type", ""),
                "statement": node.get("statement", ""),
                "source_field": node.get("source_field", ""),
                "source_path": node.get("source_path", ""),
                "support_status": node.get("support_status", ""),
                "evidence_support_count": node.get("evidence_support_count", 0),
                "visual_support_count": node.get("visual_support_count", 0),
                "direct_text_support_count": node.get("direct_text_support_count", 0),
            }
            for node in claim_nodes
        ]).to_csv(reports_dir / "claim_catalog.csv", index=False)
        pd.DataFrame([
            {
                "claim_id": node["node_id"],
                "claim_type": node.get("claim_type", ""),
                "statement": node.get("statement", ""),
                "source_path": node.get("source_path", ""),
            }
            for node in claim_nodes if node.get("support_status") != "evidence_linked"
        ]).to_csv(reports_dir / "claims_without_linked_evidence.csv", index=False)
        pd.DataFrame([
            {"claim_type": key, "count": value}
            for key, value in claim_type_counts.most_common()
        ]).to_csv(reports_dir / "claim_counts_by_type.csv", index=False)
        pd.DataFrame([
            {"support_status": key, "count": value}
            for key, value in support_counts.most_common()
        ]).to_csv(reports_dir / "claim_support_status.csv", index=False)

        learned_rows = []
        for edge in learned_edges:
            learned_rows.append({
                "relationship_id": edge["relationship_id"],
                "subject_id": edge["start_id"],
                "subject": self.nodes[edge["start_id"]].get("name", ""),
                "relationship": edge["type"],
                "object_id": edge["end_id"],
                "object": self.nodes[edge["end_id"]].get("name", ""),
                "claim_id": edge.get("claim_id", ""),
                "paper_id": edge.get("paper_id", ""),
                "statement": edge.get("statement", ""),
                "source_field": edge.get("source_field", ""),
                "source_record_id": edge.get("source_record_id", ""),
                "source_path": edge.get("source_path", ""),
                "assertion_status": edge.get("assertion_status", ""),
            })
        learned_df = pd.DataFrame(learned_rows)
        learned_df.to_csv(views_dir / "learned_scientific_relationships.csv", index=False)
        if not learned_df.empty:
            aggregated = (
                learned_df.groupby(["subject", "relationship", "object"], dropna=False)
                .agg(
                    paper_count=("paper_id", "nunique"),
                    claim_count=("claim_id", "nunique"),
                    example_statement=("statement", "first"),
                )
                .reset_index()
                .sort_values(["paper_count", "claim_count"], ascending=False)
            )
            aggregated.to_csv(views_dir / "cross_paper_relationship_patterns.csv", index=False)

        claims_by_paper: Dict[str, List[str]] = defaultdict(list)
        evidence_by_claim: Dict[str, List[str]] = defaultdict(list)
        for edge in self.edges.values():
            if edge["type"] == "ASSERTS_CLAIM":
                claims_by_paper[edge["start_id"]].append(edge["end_id"])
            elif edge["type"] == "SUPPORTS_CLAIM":
                evidence_by_claim[edge["end_id"]].append(edge["start_id"])
        with (views_dir / "paper_knowledge_cards.jsonl").open("w", encoding="utf-8") as handle:
            for paper_id, claim_ids in claims_by_paper.items():
                paper = self.nodes[paper_id]
                card = {
                    "paper_id": paper_id,
                    "paper_name": paper.get("name", ""),
                    "paper_type": paper.get("paper_type", ""),
                    "claims": [
                        {
                            "claim_id": claim_id,
                            "claim_type": self.nodes[claim_id].get("claim_type", ""),
                            "statement": self.nodes[claim_id].get("statement", ""),
                            "support_status": self.nodes[claim_id].get("support_status", ""),
                            "evidence_ids": evidence_by_claim.get(claim_id, []),
                        }
                        for claim_id in claim_ids
                    ],
                }
                handle.write(json_text(card) + "\n")

        flow = pd.DataFrame([{"source_type": sorted(self.nodes[e["start_id"]]["labels"])[0], "relationship": e["type"], "target_type": sorted(self.nodes[e["end_id"]]["labels"])[0]} for e in self.edges.values()])
        if not flow.empty:
            schema_flows = flow.value_counts().reset_index(name="value")
            schema_flows.to_csv(views_dir / "sankey_schema_flows.csv", index=False)
            schema_graph = nx.DiGraph()
            for _, item in schema_flows.iterrows():
                source = item["source_type"]
                target = item["target_type"]
                schema_graph.add_node(source, count=node_counts.get(source, 0))
                schema_graph.add_node(target, count=node_counts.get(target, 0))
                schema_graph.add_edge(source, target, weight=int(item["value"]))
            fig, ax = plt.subplots(figsize=(13, 8), dpi=180)
            positions = nx.spring_layout(schema_graph, seed=42, k=1.5)
            sizes = [700 + 80 * math.sqrt(max(schema_graph.nodes[n].get("count", 1), 1)) for n in schema_graph.nodes]
            widths = [0.6 + math.log1p(schema_graph[u][v].get("weight", 1)) for u, v in schema_graph.edges]
            nx.draw_networkx_nodes(schema_graph, positions, node_size=sizes, node_color="#8fb9a8", edgecolors="#354f52", ax=ax)
            nx.draw_networkx_labels(schema_graph, positions, font_size=8, ax=ax)
            nx.draw_networkx_edges(schema_graph, positions, width=widths, alpha=0.45, arrows=True, arrowsize=14, ax=ax)
            ax.set_title("LiteratureAgent knowledge graph schema and relationship flow")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(plots_dir / "knowledge_graph_schema.png", bbox_inches="tight")
            plt.close(fig)
        dag_types = {"USES_ABSORBER", "HAS_A_SITE", "HAS_B_SITE", "HAS_X_SITE", "USES_ADDITIVE", "FABRICATED_BY", "HAS_PERFORMANCE_MEASUREMENT", "HAS_STABILITY_TEST", "HAS_STABILITY_MEASUREMENT"}
        pd.DataFrame([e for e in edge_rows if e[":TYPE"] in dag_types]).to_csv(views_dir / "material_process_performance_dag_edges.csv", index=False)

        if node_counts:
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
            labels, values = zip(*node_counts.most_common())
            ax.barh(range(len(labels)), values)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Node count")
            ax.set_title("Literature knowledge graph node coverage")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(plots_dir / "node_coverage_by_type.png", bbox_inches="tight")
            plt.close(fig)
        if rel_counts:
            fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(rel_counts))), dpi=180)
            labels, values = zip(*rel_counts.most_common())
            ax.barh(range(len(labels)), values)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Relationship count")
            ax.set_title("Literature knowledge graph relationship coverage")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(plots_dir / "relationship_coverage_by_type.png", bbox_inches="tight")
            plt.close(fig)
        if claim_type_counts:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
            labels, values = zip(*claim_type_counts.most_common())
            ax.barh(range(len(labels)), values, color="#4c956c")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Extracted claim count")
            ax.set_title("Scientific knowledge extracted from paper summaries")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(plots_dir / "scientific_claim_coverage_by_type.png", bbox_inches="tight")
            plt.close(fig)
        if support_counts:
            fig, ax = plt.subplots(figsize=(7.5, 5), dpi=180)
            labels, values = zip(*support_counts.most_common())
            ax.bar(labels, values, color=["#2a9d8f" if label == "evidence_linked" else "#e9c46a" for label in labels])
            ax.set_ylabel("Claim count")
            ax.set_title("Claim-to-evidence linkage status")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(plots_dir / "claim_evidence_linkage.png", bbox_inches="tight")
            plt.close(fig)
        learned_rel_counts = Counter(edge["type"] for edge in learned_edges)
        if learned_rel_counts:
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
            labels, values = zip(*learned_rel_counts.most_common())
            ax.barh(range(len(labels)), values, color="#577590")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Evidence-grounded relationship count")
            ax.set_title("Directional scientific relationships learned from literature")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(plots_dir / "learned_scientific_relationships.png", bbox_inches="tight")
            plt.close(fig)

        readme = f"""LiteratureAgent Knowledge Graph
================================

Canonical model: evidence-grounded claim-centered scientific property graph.

Why this model:
- Scientific relationships are not naturally acyclic, so a DAG is only a derived view.
- Sankey diagrams are useful flow visualizations, not a canonical knowledge representation.
- Property-graph relationships can carry source fields, record IDs, and evidence paths.

Graph summary:
- Reasoning policy: {self.reasoning_policy}
- Nodes: {len(self.nodes)}
- Relationships: {len(self.edges)}
- Scientific claims: {len(claim_nodes)}
- Learned directional relationships: {len(learned_edges)}
- Dangling relationships: {len(dangling)}

Open first:
- reports/graph_summary.json
- plots/node_coverage_by_type.png
- plots/relationship_coverage_by_type.png
- plots/knowledge_graph_schema.png
- plots/scientific_claim_coverage_by_type.png
- plots/claim_evidence_linkage.png
- plots/learned_scientific_relationships.png
- reports/claim_catalog.csv
- reports/claims_without_linked_evidence.csv
- reports/reasoning_policy_summary.csv
- reports/reasoning_policy_findings.csv
- neo4j_import/nodes.csv
- neo4j_import/relationships.csv
- portable/literature_knowledge_graph.graphml
- derived_views/sankey_schema_flows.csv
- derived_views/material_process_performance_dag_edges.csv
- derived_views/learned_scientific_relationships.csv
- derived_views/cross_paper_relationship_patterns.csv
- derived_views/paper_knowledge_cards.jsonl

Neo4j import:
The CSV files use Neo4j bulk-import headers (:ID, :LABEL, :START_ID, :END_ID, :TYPE).
Relationship properties preserve source_field, source_record_id, and provenance_source.
Directional scientific relationships are reported paper claims, not assumed universal facts.
"""
        (out_dir / "README.txt").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence-grounded perovskite literature knowledge graph.")
    parser.add_argument("--records", required=True, help="LiteratureAgent all_records.csv")
    parser.add_argument("--ontology", required=True, help="Perovskite ontology JSON")
    parser.add_argument("--work-dir", required=True, help="LiteratureAgent output directory containing evidence artifacts")
    parser.add_argument("--out", required=True, help="Knowledge-graph output directory")
    parser.add_argument("--max-records", type=int, default=0, help="Optional test limit; 0 processes all records")
    parser.add_argument(
        "--reasoning-policy",
        default="current",
        choices=sorted(KG_REASONING_POLICIES),
        help="Optional reasoning-school overlay for local POLARIS policy testing",
    )
    args = parser.parse_args()

    records_path = Path(args.records)
    ontology_path = Path(args.ontology)
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out)
    df = pd.read_csv(records_path, low_memory=False)
    if args.max_records > 0:
        df = df.head(args.max_records)
    builder = GraphBuilder(ontology_index(ontology_path), records_path, work_dir)
    for row_index, row in df.iterrows():
        builder.add_record(row, int(row_index))
    builder.apply_reasoning_policy_lens(args.reasoning_policy)
    builder.write(out_dir)
    print(f"Knowledge graph written to: {out_dir.resolve()}")
    print(f"Reasoning policy: {builder.reasoning_policy}")
    print(f"Nodes: {len(builder.nodes):,} | Relationships: {len(builder.edges):,}")


if __name__ == "__main__":
    main()
