"""POLARIS-facing adapter for the Scientific Reasoning Policy Layer.

This file is intentionally small and dependency-free. It gives POLARIS a stable
contract to use later, while LiteratureAgent can test the same interface locally.
The policy layer remains a pool: callers can request an explicit policy, or let
the adapter select policies from task/context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .scientific_reasoning_policy_layer import (
    POLICIES,
    TASK_TYPE_POLICY_PRIORS,
    prepare_reasoned_prompt,
    select_reasoning_policy,
)


LITERATURE_MULTI_POLICY_DEFAULTS: Dict[str, List[str]] = {
    "summary": ["humean_evidence", "aristotelian_classification"],
    "summarization": ["humean_evidence", "aristotelian_classification"],
    "extraction": ["humean_evidence", "kantian_constraints"],
    "structured_extraction": ["kantian_constraints", "humean_evidence"],
    "schema_validation": ["kantian_constraints", "humean_evidence"],
    "classification": ["aristotelian_classification", "kantian_constraints"],
    "paper_type_gating": ["aristotelian_classification", "kantian_constraints"],
    "figure_vision": ["humean_evidence", "socratic_uncertainty"],
    "gap_analysis": ["socratic_uncertainty", "platonic_abstraction"],
    "hypothesis": ["socratic_uncertainty", "platonic_abstraction"],
    "pattern_mining": ["platonic_abstraction", "humean_evidence"],
    "evidence_comparison": ["humean_evidence", "hegelian_conflict_resolution"],
    "conflict_resolution": ["hegelian_conflict_resolution", "humean_evidence"],
    "workflow_planning": ["cartesian_decomposition", "socratic_uncertainty"],
}


@dataclass
class ReasoningRequest:
    agent_name: str
    task: str
    base_prompt: str
    task_type: str = "auto"
    context: str = ""
    output_schema: str = ""
    constraints: Optional[List[str]] = None
    policy_mode: str = "auto"
    requested_policy: str = "auto"
    metadata_destination: str = "sidecar"


@dataclass
class ReasoningEnvelope:
    prompt: str
    metadata: Dict[str, Any]


def _normalize_mode(mode: str) -> str:
    mode = str(mode or "auto").strip().lower()
    return mode if mode in {"off", "single", "multi", "auto"} else "auto"


def _infer_task_type(task: str, base_prompt: str = "", output_schema: str = "") -> str:
    task_text = "\n".join([task or "", base_prompt or ""]).lower()
    schema_text = str(output_schema or "").lower()
    text = "\n".join([task_text, schema_text]).lower()

    # Prefer the scientific task intent over generic compatibility/schema hints.
    # Literature summaries may still be serialized as JSON, but they should not
    # become Kantian schema-checking tasks just because the output wrapper says JSON.
    if any(k in task_text for k in ["hypothesis", "future work", "gap", "uncertainty", "next step"]):
        return "gap_analysis"
    if any(k in task_text for k in ["vision", "figure", "caption", "ocr", "image"]):
        return "figure_vision"
    if any(k in task_text for k in ["summarize", "summary", "overview", "literature summary"]):
        return "summarization"
    if any(k in task_text for k in ["paper type", "gate", "classify", "classification"]):
        return "paper_type_gating"
    if any(k in task_text for k in ["conflict", "contradict", "disagree", "mixed evidence"]):
        return "conflict_resolution"
    if any(k in task_text for k in ["pattern", "cross-paper", "knowledge graph", "generalize"]):
        return "pattern_mining"
    if any(k in task_text for k in ["plan", "workflow", "campaign", "batch"]):
        return "workflow_planning"
    if any(k in task_text for k in ["structured extraction", "schema", "json", "ontology", "required field", "validate", "validation"]):
        return "structured_extraction"
    if any(k in task_text for k in ["extract", "extraction", "evidence", "claim", "reported"]):
        return "extraction"
    if any(k in text for k in ["schema", "json", "ontology", "required field", "validate", "validation"]):
        return "structured_extraction"
    return "summary"


def select_policy_set(
    task: str,
    agent_name: str = "Literature Agent",
    task_type: str = "auto",
    policy_mode: str = "auto",
    requested_policy: str = "auto",
    context: str = "",
) -> Dict[str, Any]:
    """Select primary and secondary policies for POLARIS-compatible calls."""

    mode = _normalize_mode(policy_mode)
    inferred_task_type = _infer_task_type(task, output_schema="") if task_type in {"", "auto", None} else str(task_type)
    if mode == "off":
        return {
            "reasoning_layer_enabled": False,
            "policy_mode": mode,
            "task_type": inferred_task_type,
            "primary_policy": None,
            "secondary_policies": [],
            "policy_set": [],
            "selection_reason": "Reasoning policy mode is off.",
        }

    selection = select_reasoning_policy(
        task=task,
        agent_name=agent_name,
        task_type=inferred_task_type,
        requested_policy=requested_policy,
        context=context,
    )
    defaults = LITERATURE_MULTI_POLICY_DEFAULTS.get(inferred_task_type) or TASK_TYPE_POLICY_PRIORS.get(inferred_task_type, [])
    primary_source = "selector"
    primary = selection.policy_id
    if mode in {"auto", "multi"} and requested_policy in {"", "auto", None} and defaults:
        primary = defaults[0]
        primary_source = "task_default"

    if mode == "single":
        secondaries: List[str] = []
    else:
        secondaries = []
        for policy_id in [*defaults, *selection.secondary_policy_ids]:
            if policy_id != primary and policy_id in POLICIES and policy_id not in secondaries:
                secondaries.append(policy_id)
        if mode == "auto":
            secondaries = secondaries[:2]

    return {
        "reasoning_layer_enabled": True,
        "policy_mode": mode,
        "task_type": inferred_task_type,
        "primary_policy": primary,
        "primary_policy_source": primary_source,
        "secondary_policies": secondaries,
        "policy_set": [primary, *secondaries],
        "selection": selection.to_dict(),
        "selection_reason": selection.reason,
    }


def build_reasoning_envelope(request: ReasoningRequest) -> ReasoningEnvelope:
    """Build a prompt and metadata envelope without executing an LLM call."""

    task_type = (
        _infer_task_type(request.task, request.base_prompt, request.output_schema)
        if request.task_type in {"", "auto", None}
        else request.task_type
    )
    selection = select_policy_set(
        task=request.task,
        agent_name=request.agent_name,
        task_type=task_type,
        policy_mode=request.policy_mode,
        requested_policy=request.requested_policy,
        context=request.context,
    )
    if not selection["reasoning_layer_enabled"]:
        return ReasoningEnvelope(prompt=request.base_prompt, metadata=selection)

    primary = selection["primary_policy"] or request.requested_policy
    constraints = list(request.constraints or [])
    if selection["secondary_policies"]:
        readable = ", ".join(POLICIES[pid].short_name for pid in selection["secondary_policies"])
        constraints.append(
            "Use the primary policy to structure the answer. Consider these secondary policies "
            f"only where useful: {readable}."
        )

    prepared = prepare_reasoned_prompt(
        base_prompt=request.base_prompt,
        task=request.task,
        agent_name=request.agent_name,
        task_type=selection["task_type"],
        context=request.context,
        constraints=constraints,
        output_schema=request.output_schema,
        policy=primary,
    )
    metadata = dict(prepared.get("reasoning_metadata") or {})
    metadata.update({
        **selection,
        "metadata_destination": request.metadata_destination,
        "output_format_preserved": request.metadata_destination == "sidecar",
    })
    return ReasoningEnvelope(prompt=str(prepared.get("prompt") or request.base_prompt), metadata=metadata)


def build_reasoning_prompt_for_polaris(**kwargs: Any) -> Dict[str, Any]:
    """Plain-dict entry point for POLARIS orchestrators and JSON configs."""

    request = ReasoningRequest(**kwargs)
    envelope = build_reasoning_envelope(request)
    return {"prompt": envelope.prompt, "reasoning_metadata": envelope.metadata}


def available_reasoning_policies() -> List[Dict[str, Any]]:
    """Return serializable policy metadata for config UIs and POLARIS health checks."""

    return [
        {
            "policy_id": policy_id,
            **asdict(policy),
        }
        for policy_id, policy in POLICIES.items()
    ]
