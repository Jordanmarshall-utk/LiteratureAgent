from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .run_manager import CONTROLLER, DEFAULT_MODEL, load_settings


def _ollama_models(url: str = "http://localhost:11434/api/tags") -> tuple[bool, list[str], str]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [str(item.get("name", "")) for item in payload.get("models", [])]
        return True, models, "Connected"
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return False, [], str(exc)


def _model_present(configured: str, installed: list[str]) -> bool:
    wanted = configured.strip().casefold()
    if not wanted:
        return False
    aliases = {wanted, wanted.removesuffix(":latest")}
    for name in installed:
        normalized = name.strip().casefold()
        if normalized in aliases or normalized.removesuffix(":latest") in aliases:
            return True
    return False


def platform_health() -> dict[str, Any]:
    settings = load_settings()
    configured_base = Path(settings["base_csv"])
    configured_ontology = Path(settings["ontology"])
    ollama_ok, models, ollama_message = _ollama_models()
    checks = {
        "Python": {"ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]},
        "Controller": {"ok": CONTROLLER.exists(), "detail": str(CONTROLLER)},
        "Baseline database": {"ok": configured_base.exists(), "detail": str(configured_base)},
        "Ontology": {"ok": configured_ontology.exists(), "detail": str(configured_ontology)},
        "Model script": {"ok": DEFAULT_MODEL.exists(), "detail": str(DEFAULT_MODEL)},
        "Ollama": {"ok": ollama_ok, "detail": ollama_message},
        "Text model": {"ok": _model_present(settings["llm_model"], models), "detail": f"Configured: {settings['llm_model']} | Installed: {', '.join(models) or 'none'}"},
        "Vision model": {"ok": _model_present(settings["vision_model"], models), "detail": f"Configured: {settings['vision_model']} | Installed: {', '.join(models) or 'none'}"},
    }
    return {"ready": all(item["ok"] for item in checks.values()), "checks": checks}
