from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .run_manager import CONTROLLER, DEFAULT_BASE_CSV, DEFAULT_MODEL, DEFAULT_ONTOLOGY


def _ollama_models(url: str = "http://localhost:11434/api/tags") -> tuple[bool, list[str], str]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [str(item.get("name", "")) for item in payload.get("models", [])]
        return True, models, "Connected"
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return False, [], str(exc)


def platform_health() -> dict[str, Any]:
    ollama_ok, models, ollama_message = _ollama_models()
    checks = {
        "Python": {"ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]},
        "Controller": {"ok": CONTROLLER.exists(), "detail": str(CONTROLLER)},
        "Baseline database": {"ok": DEFAULT_BASE_CSV.exists(), "detail": str(DEFAULT_BASE_CSV)},
        "Ontology": {"ok": DEFAULT_ONTOLOGY.exists(), "detail": str(DEFAULT_ONTOLOGY)},
        "Model script": {"ok": DEFAULT_MODEL.exists(), "detail": str(DEFAULT_MODEL)},
        "Ollama": {"ok": ollama_ok, "detail": ollama_message},
        "Text model": {"ok": any(name.startswith("qwen2.5:7b") for name in models), "detail": ", ".join(models) or "No models found"},
        "Vision model": {"ok": any("qwen2.5vl:7b" in name for name in models), "detail": ", ".join(models) or "No models found"},
    }
    return {"ready": all(item["ok"] for key, item in checks.items() if key != "Vision model"), "checks": checks}
