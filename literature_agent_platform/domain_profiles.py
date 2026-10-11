from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DomainProfile:
    id: str
    label: str
    description: str
    ontology: Path
    supports_pce_modeling: bool = False


DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "general_science": DomainProfile(
        id="general_science",
        label="General science",
        description="Domain-neutral summaries, methods, measurements, findings, evidence, and limitations.",
        ontology=PROJECT_ROOT / "config" / "domain_packs" / "general_science.json",
    ),
    "oxide_perovskite": DomainProfile(
        id="oxide_perovskite",
        label="Oxide perovskites",
        description="ABO3 chemistry, substitutions, oxygen defects, structure, processing, and measured properties.",
        ontology=PROJECT_ROOT / "config" / "domain_packs" / "oxide_perovskite.json",
    ),
    "halide_pv": DomainProfile(
        id="halide_pv",
        label="Halide-perovskite photovoltaics",
        description="Existing device-record extraction, PCE/stability integration, and modeling workflow.",
        ontology=PROJECT_ROOT / "config" / "perovskite_ontology_library_v19.json",
        supports_pce_modeling=True,
    ),
}


def get_domain_profile(profile_id: str) -> DomainProfile:
    try:
        return DOMAIN_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown domain profile: {profile_id}") from exc


def command_option(command: list[str], flag: str, default: str = "") -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(command):
        return default
    return str(command[index + 1])
