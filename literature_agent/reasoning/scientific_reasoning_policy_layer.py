"""
Scientific Reasoning Policy Layer for POLARIS
============================================

Purpose
-------
This is a standalone, dependency-free reasoning-policy layer that can be copied
into any POLARIS project folder. It does not replace existing agents. Instead,
it gives any POLARIS agent a shared pool of scientific reasoning policies to
use before sending a prompt to an LLM.

Core idea
---------
Domain agent = what task is being done
Reasoning policy = how the agent should think through that task

Recommended first integration target
------------------------------------
Use the Literature Agent first. The Literature Agent should mainly use:
- Humean evidence grounding for literature extraction and summary
- Kantian rule/schema checking for structured extraction and validation
- Socratic uncertainty checking for gaps, assumptions, and future work

Minimal usage
-------------
from scientific_reasoning_policy_layer import generate_with_reasoning

result = generate_with_reasoning(
    generate_text_fn=my_existing_llm_function,
    agent_name="Literature Agent",
    task="Summarize evidence for MACl effects in FAPbI3 crystallization.",
    base_prompt="Use the provided paper text and extract evidence.",
    context="<paper text or evidence packet>",
    constraints=["Separate direct evidence from interpretation."],
    output_schema="Return: evidence, interpretation, uncertainty, next_gap."
)

print(result["reasoning_metadata"])
print(result["response"])

Codex integration instruction
-----------------------------
Tell Codex:
"Copy scientific_reasoning_policy_layer.py into the POLARIS project. Do not
refactor the whole project. Locate the Literature Agent LLM calls and wrap them
with generate_with_reasoning. Keep the layer optional and preserve existing
outputs. If adding metadata breaks downstream parsing, save reasoning_metadata
to a sidecar log instead."
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import json
import re


@dataclass(frozen=True)
class ReasoningPolicy:
    """A reusable scientific reasoning mode that any POLARIS agent can use."""

    policy_id: str
    short_name: str
    school_of_thought: str
    simple_meaning: str
    best_for: List[str]
    prompt_instruction: str
    output_checks: List[str]
    keywords: List[str]


@dataclass(frozen=True)
class PolicySelection:
    """Metadata describing why a reasoning policy was selected."""

    policy_id: str
    short_name: str
    confidence: float
    reason: str
    matched_keywords: List[str]
    secondary_policy_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


POLICIES: Dict[str, ReasoningPolicy] = {
    "socratic_uncertainty": ReasoningPolicy(
        policy_id="socratic_uncertainty",
        short_name="Socratic",
        school_of_thought="Questioning, assumption testing, and hypothesis elimination",
        simple_meaning="Ask what assumptions, gaps, and uncertainties remain.",
        best_for=[
            "hypothesis generation",
            "uncertainty analysis",
            "gap finding",
            "future work",
            "experimental next steps",
        ],
        prompt_instruction=(
            "Use Socratic reasoning. Identify assumptions, missing evidence, uncertain steps, "
            "and alternative explanations. Do not just give an answer. Make clear what would need "
            "to be checked experimentally or with additional data."
        ),
        output_checks=[
            "Assumptions are stated explicitly.",
            "Uncertainty is separated from conclusions.",
            "The next evidence needed is identified.",
        ],
        keywords=[
            "hypothesis", "assumption", "uncertainty", "unknown", "gap", "future work",
            "next step", "why", "mechanism", "test", "question", "ambiguous",
        ],
    ),
    "cartesian_decomposition": ReasoningPolicy(
        policy_id="cartesian_decomposition",
        short_name="Cartesian",
        school_of_thought="Stepwise decomposition, deduction, and calculation",
        simple_meaning="Break the problem into small logical steps.",
        best_for=[
            "calculations",
            "curve fitting",
            "spectral decomposition",
            "peak fitting",
            "workflow planning",
        ],
        prompt_instruction=(
            "Use Cartesian reasoning. Decompose the task into clear steps. For numerical or fitting "
            "tasks, define inputs, intermediate operations, assumptions, and outputs. Check each step "
            "before making the final interpretation."
        ),
        output_checks=[
            "The task is decomposed into steps.",
            "Inputs and outputs are clear.",
            "Calculations or fitting choices are justified.",
        ],
        keywords=[
            "calculate", "calculation", "fit", "fitting", "peak", "spectrum", "spectra",
            "xrd", "pl", "absorbance", "decompose", "decomposition", "steps", "workflow",
            "equation", "numeric", "quantitative", "model training", "feature", "metric",
        ],
    ),
    "kantian_constraints": ReasoningPolicy(
        policy_id="kantian_constraints",
        short_name="Kantian",
        school_of_thought="Rules, constraints, premises, and consistency checking",
        simple_meaning="Check whether the output follows rules and stays within limits.",
        best_for=[
            "schema validation",
            "physical constraints",
            "chemical constraints",
            "rule checks",
            "data quality control",
        ],
        prompt_instruction=(
            "Use Kantian constraint checking. Identify the rules, premises, schema requirements, "
            "units, categories, and physical or chemical limits that govern the task. Flag any output "
            "that violates those constraints or lacks required information."
        ),
        output_checks=[
            "Required schema fields are checked.",
            "Units and categories are consistent.",
            "Physical or chemical constraints are not violated.",
        ],
        keywords=[
            "schema", "validate", "validation", "constraint", "rule", "rules", "unit", "units",
            "json", "ontology", "required", "field", "fields", "category", "consistent",
            "consistency", "qc", "quality", "allowed", "invalid", "physical constraint",
        ],
    ),
    "humean_evidence": ReasoningPolicy(
        policy_id="humean_evidence",
        short_name="Humean",
        school_of_thought="Evidence grounding, observation, comparison, and cautious induction",
        simple_meaning="Follow the evidence and avoid overclaiming.",
        best_for=[
            "literature summarization",
            "evidence extraction",
            "claim comparison",
            "experimental interpretation",
            "multimodal evidence comparison",
        ],
        prompt_instruction=(
            "Use Humean evidence grounding. Base conclusions on observable evidence from the provided "
            "text, data, or figures. Separate direct evidence from interpretation. Avoid unsupported "
            "mechanistic claims. When evidence is weak, say so clearly."
        ),
        output_checks=[
            "Direct evidence is separated from interpretation.",
            "Unsupported claims are flagged.",
            "The confidence level matches the strength of evidence.",
        ],
        keywords=[
            "literature", "paper", "evidence", "extract", "extraction", "summary", "summarize",
            "claim", "claims", "support", "reported", "observed", "compare", "comparison",
            "trend", "figure", "citation", "source", "mechanism", "experimental evidence",
        ],
    ),
    "aristotelian_classification": ReasoningPolicy(
        policy_id="aristotelian_classification",
        short_name="Aristotelian",
        school_of_thought="Classification, categories, definitions, and causes",
        simple_meaning="Define what type of thing this is and why it belongs there.",
        best_for=[
            "classification",
            "paper-type gating",
            "phase labeling",
            "material-family grouping",
            "causal explanation",
        ],
        prompt_instruction=(
            "Use Aristotelian classification. Define the relevant categories, decide which category "
            "the item belongs to, explain the causal or defining features, and note borderline cases."
        ),
        output_checks=[
            "Categories are defined.",
            "The classification decision is justified.",
            "Borderline or mixed cases are flagged.",
        ],
        keywords=[
            "classify", "classification", "category", "categorize", "type", "family", "phase",
            "label", "gating", "group", "definition", "define", "cause", "causal",
        ],
    ),
    "hegelian_conflict_resolution": ReasoningPolicy(
        policy_id="hegelian_conflict_resolution",
        short_name="Hegelian",
        school_of_thought="Contradiction, tension, and synthesis",
        simple_meaning="Compare conflicting explanations and resolve the tension.",
        best_for=[
            "conflicting literature",
            "contradictory data",
            "competing hypotheses",
            "reconciling mechanisms",
        ],
        prompt_instruction=(
            "Use Hegelian conflict resolution. Identify the competing claims or explanations, state "
            "the tension between them, and propose a synthesis that explains what may be true in each "
            "case. Do not force agreement if the evidence remains contradictory."
        ),
        output_checks=[
            "Competing explanations are stated fairly.",
            "The contradiction is clearly identified.",
            "Any synthesis remains evidence-based.",
        ],
        keywords=[
            "conflict", "contradiction", "contradictory", "opposite", "disagree", "inconsistent",
            "competing", "reconcile", "synthesis", "tension", "mixed evidence",
        ],
    ),
    "platonic_abstraction": ReasoningPolicy(
        policy_id="platonic_abstraction",
        short_name="Platonic",
        school_of_thought="Abstraction, ideal forms, and general patterns",
        simple_meaning="Look for the general pattern behind specific examples.",
        best_for=[
            "conceptual framing",
            "theory-level interpretation",
            "generalization",
            "high-level model explanation",
        ],
        prompt_instruction=(
            "Use Platonic abstraction. Identify the general pattern, idealized structure, or conceptual "
            "principle behind the examples. Keep the abstraction connected to the actual evidence."
        ),
        output_checks=[
            "The general pattern is stated clearly.",
            "The abstraction remains connected to evidence.",
            "Limits of generalization are stated.",
        ],
        keywords=[
            "concept", "conceptual", "abstract", "abstraction", "general pattern", "principle",
            "theory", "framework", "ideal", "generalize", "high-level", "big picture",
        ],
    ),
}


POLICY_ALIASES: Dict[str, str] = {
    "socrates": "socratic_uncertainty",
    "socratic": "socratic_uncertainty",
    "uncertainty": "socratic_uncertainty",
    "descartes": "cartesian_decomposition",
    "cartesian": "cartesian_decomposition",
    "decomposition": "cartesian_decomposition",
    "kant": "kantian_constraints",
    "kantian": "kantian_constraints",
    "constraints": "kantian_constraints",
    "schema": "kantian_constraints",
    "hume": "humean_evidence",
    "humean": "humean_evidence",
    "evidence": "humean_evidence",
    "aristotle": "aristotelian_classification",
    "aristotelian": "aristotelian_classification",
    "classification": "aristotelian_classification",
    "hegel": "hegelian_conflict_resolution",
    "hegelian": "hegelian_conflict_resolution",
    "conflict": "hegelian_conflict_resolution",
    "plato": "platonic_abstraction",
    "platonic": "platonic_abstraction",
    "abstraction": "platonic_abstraction",
}


# Agent-level priors. These are intentionally soft defaults, not hard-coded rules.
AGENT_POLICY_PRIORS: Dict[str, List[str]] = {
    "literature": ["humean_evidence", "kantian_constraints", "socratic_uncertainty"],
    "literature_agent": ["humean_evidence", "kantian_constraints", "socratic_uncertainty"],
    "hypothesis": ["socratic_uncertainty", "humean_evidence", "kantian_constraints"],
    "hypothesis_agent": ["socratic_uncertainty", "humean_evidence", "kantian_constraints"],
    "experiment": ["cartesian_decomposition", "kantian_constraints", "socratic_uncertainty"],
    "experiment_agent": ["cartesian_decomposition", "kantian_constraints", "socratic_uncertainty"],
    "curve_fitting": ["cartesian_decomposition", "kantian_constraints", "humean_evidence"],
    "curve_fitting_agent": ["cartesian_decomposition", "kantian_constraints", "humean_evidence"],
    "xrd": ["cartesian_decomposition", "kantian_constraints", "humean_evidence"],
    "pl": ["cartesian_decomposition", "kantian_constraints", "humean_evidence"],
    "xrd_pl_analysis": ["cartesian_decomposition", "kantian_constraints", "humean_evidence"],
    "analysis": ["humean_evidence", "kantian_constraints", "socratic_uncertainty"],
    "analysis_agent": ["humean_evidence", "kantian_constraints", "socratic_uncertainty"],
    "active_learning": ["socratic_uncertainty", "humean_evidence", "kantian_constraints"],
    "active_learning_agent": ["socratic_uncertainty", "humean_evidence", "kantian_constraints"],
    "schema_validator": ["kantian_constraints", "aristotelian_classification", "humean_evidence"],
}


TASK_TYPE_POLICY_PRIORS: Dict[str, List[str]] = {
    "summary": ["humean_evidence", "socratic_uncertainty"],
    "summarization": ["humean_evidence", "socratic_uncertainty"],
    "extraction": ["humean_evidence", "kantian_constraints"],
    "structured_extraction": ["kantian_constraints", "humean_evidence"],
    "schema_validation": ["kantian_constraints"],
    "classification": ["aristotelian_classification", "kantian_constraints"],
    "hypothesis": ["socratic_uncertainty", "humean_evidence"],
    "gap_analysis": ["socratic_uncertainty", "humean_evidence"],
    "calculation": ["cartesian_decomposition", "kantian_constraints"],
    "curve_fitting": ["cartesian_decomposition", "kantian_constraints"],
    "evidence_comparison": ["humean_evidence", "hegelian_conflict_resolution"],
    "conflict_resolution": ["hegelian_conflict_resolution", "humean_evidence"],
    "conceptual_framing": ["platonic_abstraction", "humean_evidence"],
}


CONFIG: Dict[str, Any] = {
    "enabled": True,
    "default_policy": "humean_evidence",
    "include_metadata_in_prompt": True,
    "include_output_contract": True,
    "allow_secondary_policy_notes": True,
}


def set_reasoning_layer_enabled(enabled: bool) -> None:
    """Globally enable or disable the layer without changing downstream code."""

    CONFIG["enabled"] = bool(enabled)


def normalize_key(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def resolve_policy_id(policy: Optional[str]) -> Optional[str]:
    """Resolve a policy id or alias. Returns None if policy is auto/empty."""

    if not policy or normalize_key(policy) in {"", "auto", "none"}:
        return None
    key = normalize_key(policy)
    if key in POLICIES:
        return key
    return POLICY_ALIASES.get(key)


def _keyword_hits(text: str, keywords: Iterable[str]) -> List[str]:
    text_low = text.lower()
    hits = []
    for kw in keywords:
        kw_low = kw.lower()
        # Use substring matching because scientific phrases vary a lot.
        if kw_low in text_low:
            hits.append(kw)
    return hits


def select_reasoning_policy(
    task: str,
    agent_name: Optional[str] = None,
    task_type: Optional[str] = None,
    requested_policy: str = "auto",
    context: Optional[str] = None,
) -> PolicySelection:
    """
    Select the most appropriate reasoning policy for a task.

    This uses simple transparent heuristics so it remains easy to audit and easy
    to port into different POLARIS versions.
    """

    direct_policy_id = resolve_policy_id(requested_policy)
    if direct_policy_id:
        policy = POLICIES[direct_policy_id]
        return PolicySelection(
            policy_id=direct_policy_id,
            short_name=policy.short_name,
            confidence=1.0,
            reason=f"Requested explicit policy: {policy.short_name}.",
            matched_keywords=[],
            secondary_policy_ids=[],
        )

    combined_text = "\n".join(part for part in [task or "", task_type or "", context or ""] if part)
    agent_key = normalize_key(agent_name)
    task_type_key = normalize_key(task_type)

    scores: Dict[str, float] = {policy_id: 0.0 for policy_id in POLICIES}
    matched: Dict[str, List[str]] = {policy_id: [] for policy_id in POLICIES}
    reasons: List[str] = []

    # Soft prior from agent identity.
    if agent_key in AGENT_POLICY_PRIORS:
        priors = AGENT_POLICY_PRIORS[agent_key]
        for rank, policy_id in enumerate(priors):
            scores[policy_id] += max(0.25, 0.60 - 0.15 * rank)
        reasons.append(f"agent prior={agent_key}")
    else:
        # Try partial matching for names like "POLARIS Literature Agent".
        for known_agent_key, priors in AGENT_POLICY_PRIORS.items():
            if known_agent_key and known_agent_key in agent_key:
                for rank, policy_id in enumerate(priors):
                    scores[policy_id] += max(0.25, 0.55 - 0.15 * rank)
                reasons.append(f"agent prior matched={known_agent_key}")
                break

    # Soft prior from task type.
    if task_type_key in TASK_TYPE_POLICY_PRIORS:
        priors = TASK_TYPE_POLICY_PRIORS[task_type_key]
        for rank, policy_id in enumerate(priors):
            scores[policy_id] += max(0.35, 0.70 - 0.20 * rank)
        reasons.append(f"task type prior={task_type_key}")

    # Keyword scoring from task/context.
    for policy_id, policy in POLICIES.items():
        hits = _keyword_hits(combined_text, policy.keywords)
        matched[policy_id] = hits
        scores[policy_id] += min(1.25, 0.18 * len(hits))

    # Fallback.
    if all(score == 0.0 for score in scores.values()):
        fallback_id = CONFIG.get("default_policy", "humean_evidence")
        scores[fallback_id] = 0.25
        reasons.append("fallback default policy")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ranked[0]
    secondaries = [pid for pid, score in ranked[1:4] if score > 0]

    # Heuristic confidence scaled to 0.50-0.95 for auto selection.
    total_score = sum(score for _, score in ranked if score > 0)
    if total_score <= 0:
        confidence = 0.50
    else:
        confidence = 0.50 + 0.45 * min(1.0, best_score / total_score)
        confidence = round(confidence, 3)

    policy = POLICIES[best_id]
    reason_bits = reasons[:]
    if matched[best_id]:
        reason_bits.append("matched keywords=" + ", ".join(matched[best_id][:8]))
    if not reason_bits:
        reason_bits.append("selected by default heuristic")

    return PolicySelection(
        policy_id=best_id,
        short_name=policy.short_name,
        confidence=confidence,
        reason="; ".join(reason_bits),
        matched_keywords=matched[best_id],
        secondary_policy_ids=secondaries,
    )


def _format_optional_section(title: str, content: Optional[Any]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        content_str = content.strip()
    else:
        content_str = json.dumps(content, indent=2, ensure_ascii=False)
    if not content_str:
        return ""
    return f"\n\n## {title}\n{content_str}"


def build_reasoning_prompt(
    base_prompt: str,
    task: str,
    agent_name: Optional[str] = None,
    task_type: Optional[str] = None,
    context: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    output_schema: Optional[str] = None,
    policy: str = "auto",
    include_reasoning_metadata: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a prompt with the selected reasoning policy.

    Returns:
        (reasoned_prompt, reasoning_metadata)
    """

    if not CONFIG.get("enabled", True):
        metadata = {
            "reasoning_layer_enabled": False,
            "reasoning_policy": None,
            "reasoning_policy_reason": "Reasoning layer disabled.",
        }
        return base_prompt, metadata

    selection = select_reasoning_policy(
        task=task,
        agent_name=agent_name,
        task_type=task_type,
        requested_policy=policy,
        context=context,
    )
    selected_policy = POLICIES[selection.policy_id]

    secondary_notes = ""
    if CONFIG.get("allow_secondary_policy_notes", True) and selection.secondary_policy_ids:
        secondary_policy_names = [POLICIES[pid].short_name for pid in selection.secondary_policy_ids]
        secondary_notes = (
            "\nSecondary policies that may be useful if needed: "
            + ", ".join(secondary_policy_names)
            + ". Do not force them if they are not relevant."
        )

    constraints_text = ""
    if constraints:
        constraints_text = "\n".join(f"- {item}" for item in constraints)

    output_checks_text = "\n".join(f"- {item}" for item in selected_policy.output_checks)

    output_contract = ""
    if CONFIG.get("include_output_contract", True):
        output_contract = f"""

## Required scientific output behavior
- Separate direct evidence from interpretation.
- State uncertainty and missing evidence when relevant.
- Do not invent sources, measurements, or mechanisms.
- Do not overclaim beyond the provided context.
- Check whether the output satisfies these policy checks:
{output_checks_text}
"""

    metadata = {
        "reasoning_layer_enabled": True,
        "reasoning_policy": selection.policy_id,
        "reasoning_policy_short_name": selection.short_name,
        "reasoning_policy_confidence": selection.confidence,
        "reasoning_policy_reason": selection.reason,
        "matched_keywords": selection.matched_keywords,
        "secondary_policy_ids": selection.secondary_policy_ids,
        "agent_name": agent_name,
        "task_type": task_type,
    }

    metadata_text = ""
    if include_reasoning_metadata and CONFIG.get("include_metadata_in_prompt", True):
        metadata_text = _format_optional_section("Reasoning metadata", metadata)

    prompt = f"""You are working inside the POLARIS scientific agentic pipeline.

## Domain agent
{agent_name or "Unspecified POLARIS agent"}

## Task
{task}

## Selected scientific reasoning policy
{selected_policy.short_name}: {selected_policy.simple_meaning}

{selected_policy.prompt_instruction}{secondary_notes}

## Original agent prompt
{base_prompt}
"""

    prompt += _format_optional_section("Available context", context)
    prompt += _format_optional_section("Task constraints", constraints_text)
    prompt += _format_optional_section("Expected output schema or format", output_schema)
    prompt += output_contract
    prompt += metadata_text
    prompt += "\n\nNow complete the task using the selected scientific reasoning policy."

    return prompt, metadata


def generate_with_reasoning(
    generate_text_fn: Callable[[str], str],
    base_prompt: str,
    task: str,
    agent_name: Optional[str] = None,
    task_type: Optional[str] = None,
    context: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    output_schema: Optional[str] = None,
    policy: str = "auto",
    include_prompt: bool = False,
) -> Dict[str, Any]:
    """
    Wrap an existing POLARIS LLM call with a reasoning policy.

    Args:
        generate_text_fn: existing function that accepts a prompt string and returns text.
        base_prompt: original prompt the agent would have sent.
        task: short task description used for policy selection.
        agent_name: e.g., "Literature Agent".
        task_type: optional task label, e.g., "extraction", "schema_validation".
        context: optional paper text, evidence packet, data summary, etc.
        constraints: optional list of constraints.
        output_schema: optional output format/schema description.
        policy: "auto" or explicit policy/alias such as "hume", "kant", "socratic".
        include_prompt: if True, include final prompt in result for debugging.

    Returns:
        dict with response and reasoning_metadata.
    """

    prompt, metadata = build_reasoning_prompt(
        base_prompt=base_prompt,
        task=task,
        agent_name=agent_name,
        task_type=task_type,
        context=context,
        constraints=constraints,
        output_schema=output_schema,
        policy=policy,
    )

    response = generate_text_fn(prompt)
    result = {
        "response": response,
        "reasoning_metadata": metadata,
    }
    if include_prompt:
        result["reasoned_prompt"] = prompt
    return result


def prepare_reasoned_prompt(
    base_prompt: str,
    task: str,
    agent_name: Optional[str] = None,
    task_type: Optional[str] = None,
    context: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    output_schema: Optional[str] = None,
    policy: str = "auto",
) -> Dict[str, Any]:
    """
    Build a reasoned prompt without calling an LLM.

    Useful when the project has a custom LLM client and Codex only needs to
    insert prompt-building before the existing call.
    """

    prompt, metadata = build_reasoning_prompt(
        base_prompt=base_prompt,
        task=task,
        agent_name=agent_name,
        task_type=task_type,
        context=context,
        constraints=constraints,
        output_schema=output_schema,
        policy=policy,
    )
    return {
        "prompt": prompt,
        "reasoning_metadata": metadata,
    }


def apply_literature_reasoning_policy(
    task: str,
    base_prompt: str,
    context: Optional[str] = None,
    output_schema: Optional[str] = None,
    policy: str = "auto",
    constraints: Optional[List[str]] = None,
    generate_text_fn: Optional[Callable[[str], str]] = None,
    include_prompt: bool = False,
) -> Dict[str, Any]:
    """
    Literature Agent adapter and first integration target.

    If generate_text_fn is supplied, this executes the LLM call.
    If generate_text_fn is not supplied, this only returns the reasoned prompt
    and metadata so it can be used with any existing POLARIS LLM client.
    """

    default_constraints = [
        "Separate direct evidence from interpretation.",
        "Flag unsupported mechanistic claims.",
        "Preserve the existing Literature Agent output format unless metadata can be added safely.",
        "If structured output is requested, check required fields and units.",
    ]
    merged_constraints = default_constraints + list(constraints or [])

    task_type = infer_literature_task_type(task, base_prompt, output_schema)

    if generate_text_fn is None:
        return prepare_reasoned_prompt(
            base_prompt=base_prompt,
            task=task,
            agent_name="Literature Agent",
            task_type=task_type,
            context=context,
            constraints=merged_constraints,
            output_schema=output_schema,
            policy=policy,
        )

    return generate_with_reasoning(
        generate_text_fn=generate_text_fn,
        base_prompt=base_prompt,
        task=task,
        agent_name="Literature Agent",
        task_type=task_type,
        context=context,
        constraints=merged_constraints,
        output_schema=output_schema,
        policy=policy,
        include_prompt=include_prompt,
    )


def infer_literature_task_type(
    task: str,
    base_prompt: str = "",
    output_schema: Optional[str] = None,
) -> str:
    """Infer a task type for Literature Agent routing."""

    text = "\n".join([task or "", base_prompt or "", output_schema or ""]).lower()

    if any(k in text for k in ["schema", "json", "ontology", "required field", "validate", "validation"]):
        return "structured_extraction"
    if any(k in text for k in ["hypothesis", "future work", "gap", "open question", "uncertainty", "next step"]):
        return "gap_analysis"
    if any(k in text for k in ["compare", "conflict", "contradict", "disagree", "mixed evidence"]):
        return "evidence_comparison"
    if any(k in text for k in ["extract", "extraction", "evidence", "claim", "reported", "figure"]):
        return "extraction"
    if any(k in text for k in ["summarize", "summary", "overview"]):
        return "summarization"
    return "summary"


def attach_reasoning_metadata_to_json_output(
    output: Any,
    reasoning_metadata: Dict[str, Any],
    key: str = "reasoning_metadata",
) -> Any:
    """
    Safely attach metadata to dict/list JSON-like output.

    If output is a dict, metadata is added under key.
    If output is a list, returns {"items": output, key: metadata}.
    Otherwise, returns output unchanged. Use a sidecar log if unchanged output is required.
    """

    if isinstance(output, dict):
        copied = dict(output)
        copied[key] = reasoning_metadata
        return copied
    if isinstance(output, list):
        return {"items": output, key: reasoning_metadata}
    return output


def validate_reasoning_output(output_text: str, policy_id: str) -> Dict[str, Any]:
    """
    Lightweight post-check. This is intentionally simple and should not replace
    schema validation. It only flags likely missing reasoning sections.
    """

    policy_id = resolve_policy_id(policy_id) or policy_id
    text = output_text.lower() if isinstance(output_text, str) else ""

    flags: List[str] = []
    suggestions: List[str] = []

    if policy_id == "humean_evidence":
        if "evidence" not in text and "reported" not in text and "observed" not in text:
            flags.append("missing_explicit_evidence_grounding")
            suggestions.append("Add direct evidence from the paper, data, or figure.")
        if "interpret" not in text and "suggest" not in text:
            flags.append("direct_evidence_and_interpretation_not_separated")
            suggestions.append("Separate direct evidence from interpretation.")

    if policy_id == "kantian_constraints":
        if "constraint" not in text and "schema" not in text and "required" not in text:
            flags.append("missing_constraint_or_schema_check")
            suggestions.append("State which schema, unit, category, or physical rule was checked.")

    if policy_id == "socratic_uncertainty":
        if "uncertain" not in text and "assumption" not in text and "missing" not in text:
            flags.append("missing_uncertainty_or_assumption_check")
            suggestions.append("Add assumptions, uncertainty, and what evidence is still needed.")

    if policy_id == "cartesian_decomposition":
        if not re.search(r"\b(step|first|second|third|input|output|calculate|fit)\b", text):
            flags.append("missing_stepwise_decomposition")
            suggestions.append("Break the reasoning into clear steps or inputs/outputs.")

    return {
        "policy_id": policy_id,
        "flags": flags,
        "suggestions": suggestions,
        "passes_basic_check": len(flags) == 0,
    }


def dump_reasoning_metadata_json(metadata: Dict[str, Any], path: str) -> None:
    """Save reasoning metadata to a sidecar JSON file."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def list_available_policies() -> List[Dict[str, Any]]:
    """Return simple descriptions for UI, docs, or debugging."""

    rows = []
    for policy in POLICIES.values():
        rows.append(
            {
                "policy_id": policy.policy_id,
                "short_name": policy.short_name,
                "simple_meaning": policy.simple_meaning,
                "best_for": policy.best_for,
            }
        )
    return rows


# Tiny built-in smoke test. Run with:
# python scientific_reasoning_policy_layer.py
if __name__ == "__main__":
    examples = [
        ("Literature Agent", "Extract claims and evidence from this paper about MACl in FAPbI3."),
        ("Literature Agent", "Validate this JSON schema for extracted perovskite stability data."),
        ("Hypothesis Agent", "Identify assumptions and missing evidence for this degradation mechanism."),
        ("XRD PL Analysis Agent", "Fit PL peaks and decompose spectra into phase components."),
    ]

    for agent, task in examples:
        selected = select_reasoning_policy(task=task, agent_name=agent)
        print(f"{agent}: {selected.short_name} ({selected.policy_id}) confidence={selected.confidence}")
        print(f"  reason: {selected.reason}\n")
