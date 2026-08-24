#!/usr/bin/env python3
"""Smoke test for the optional LiteratureAgent scientific reasoning layer.

This does not call Ollama or process a PDF. It only verifies that the copied
policy module can route LiteratureAgent-style tasks to the intended policies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REASONING_DIR = PROJECT_ROOT / "literature_agent" / "reasoning"
sys.path.insert(0, str(REASONING_DIR))

from scientific_reasoning_policy_layer import apply_literature_reasoning_policy  # noqa: E402


EXAMPLES = [
    {
        "name": "summary/evidence",
        "task": "Literature summary or evidence extraction",
        "base_prompt": "Summarize this perovskite paper using evidence from text, captions, tables, and figures.",
        "policy": "humean_evidence",
    },
    {
        "name": "structured/json",
        "task": "Structured extraction, ontology mapping, JSON output, or schema validation",
        "base_prompt": "Extract Perovskite Database fields and return valid JSON only.",
        "policy": "kantian_constraints",
    },
    {
        "name": "gaps/uncertainty",
        "task": "Research gaps, future work, hypothesis generation, or uncertainty analysis",
        "base_prompt": "Identify uncertainty and next experiments after reading this literature batch.",
        "policy": "socratic_uncertainty",
    },
]


def main() -> None:
    for example in EXAMPLES:
        result = apply_literature_reasoning_policy(
            task=example["task"],
            base_prompt=example["base_prompt"],
            context="Sample halide perovskite device/evidence packet.",
            output_schema="Preserve the existing LiteratureAgent output format.",
            policy=example["policy"],
            generate_text_fn=None,
        )
        meta = result["reasoning_metadata"]
        print("=" * 80)
        print(example["name"])
        print(json.dumps({
            "selected_policy": meta.get("reasoning_policy"),
            "task_type": meta.get("task_type"),
            "confidence": meta.get("reasoning_policy_confidence"),
            "reason": meta.get("reasoning_policy_reason"),
        }, indent=2))
        print("prompt_preview:")
        print(result["prompt"][:700].strip())


if __name__ == "__main__":
    main()
