from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class StructuredOutputContract:
    """Optional editable extraction contract stored inside a domain-pack JSON file."""

    domain_instruction: str
    fields: dict[str, Any]
    pack_id: str
    pack_version: str
    source_path: Path


def domain_column_name(field_name: str) -> str:
    """Return the stable CSV column used for a configured structured field."""

    if not FIELD_NAME_RE.fullmatch(field_name):
        raise ValueError(
            f"Invalid structured-output field {field_name!r}; use lower-case snake_case "
            "letters, numbers, and underscores (maximum 64 characters)."
        )
    return f"Domain_{field_name}"


def load_domain_pack(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Domain pack must contain one JSON object: {source}")
    return payload


def load_structured_output_contract(path: str | Path) -> StructuredOutputContract:
    """Load the optional ``structured_output`` section from a domain pack.

    Domain packs without this section keep the built-in general-science contract.
    Configured fields extend (or deliberately override) that contract and are
    exported as ``Domain_<field_name>`` plus matching evidence columns.
    """

    source = Path(path).expanduser()
    payload = load_domain_pack(source)
    configured = payload.get("structured_output")
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise ValueError(f"structured_output must be a JSON object: {source}")

    instruction = configured.get("domain_instruction")
    if instruction is None:
        instruction = ""
    if not isinstance(instruction, str):
        raise ValueError(f"structured_output.domain_instruction must be text: {source}")

    fields = configured.get("fields")
    if fields is None:
        fields = {}
    if not isinstance(fields, dict):
        raise ValueError(f"structured_output.fields must be a JSON object: {source}")

    validated: dict[str, Any] = {}
    for raw_name, schema_example in fields.items():
        name = str(raw_name)
        domain_column_name(name)
        if name == "field_evidence":
            raise ValueError(
                "field_evidence is managed by LiteratureAgent; configure scientific fields only."
            )
        validated[name] = copy.deepcopy(schema_example)

    raw_pack_id = payload.get("id")
    raw_pack_version = payload.get("version")
    for label, value in (("id", raw_pack_id), ("version", raw_pack_version)):
        if value is not None and not isinstance(value, (str, int, float)):
            raise ValueError(f"Domain-pack {label} must be text or a number: {source}")
    pack_id = str(raw_pack_id).strip() if raw_pack_id is not None else source.stem
    pack_version = str(raw_pack_version).strip() if raw_pack_version is not None else "unversioned"
    if not pack_id:
        pack_id = source.stem
    if not pack_version:
        pack_version = "unversioned"

    return StructuredOutputContract(
        domain_instruction=instruction.strip(),
        fields=validated,
        pack_id=pack_id,
        pack_version=pack_version,
        source_path=source.resolve(),
    )
