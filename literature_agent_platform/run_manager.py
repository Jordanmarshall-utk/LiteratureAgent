from __future__ import annotations

import csv
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = PROJECT_ROOT / "literature_agent_full_end_to_end_v21_3_english_sanitizer.py"
CAMPAIGN_RUNNER = PROJECT_ROOT / "scripts" / "run_expansion_campaign.py"
DEFAULT_BASE_CSV = PROJECT_ROOT / "data" / "Perovskite_database_content_all_data.csv"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "config" / "perovskite_ontology_library_v19.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pce_then_stability_same_approach.py"
PLATFORM_ROOT = PROJECT_ROOT / "platform_workspace"
RUNS_DIR = PLATFORM_ROOT / "runs"
STATE_FILE = PLATFORM_ROOT / "platform_state.json"
SETTINGS_FILE = PLATFORM_ROOT / "settings.json"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_settings() -> dict[str, str]:
    defaults = {
        "base_csv": str(DEFAULT_BASE_CSV),
        "ontology": str(DEFAULT_ONTOLOGY),
        "llm_api_url": "http://localhost:11434/v1/chat/completions",
        "llm_model": "qwen2.5:7b",
        "vision_api_url": "http://localhost:11434/v1/chat/completions",
        "vision_model": "qwen2.5vl:7b-q4_K_M",
        "oauth_client": str(PROJECT_ROOT / "secrets" / "google_drive_oauth_client.json"),
        "oauth_token": str(PROJECT_ROOT / "secrets" / "google_drive_token.json"),
    }
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            defaults.update({key: str(value) for key, value in saved.items() if key in defaults})
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def save_settings(settings: dict[str, str]) -> None:
    _json_dump(SETTINGS_FILE, settings)


def _process_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False


@dataclass
class RunRecord:
    id: str
    name: str
    workflow: str
    status: str
    created_at: str
    command: list[str]
    run_dir: str
    stdout_path: str
    stderr_path: str
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None


class RunStore:
    def __init__(self, state_file: Path = STATE_FILE) -> None:
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {"runs": []}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"runs": []}

    def _save(self, state: dict[str, Any]) -> None:
        _json_dump(self.state_file, state)

    def list_runs(self) -> list[RunRecord]:
        state = self._load()
        runs = [RunRecord(**item) for item in state.get("runs", [])]
        changed = False
        for run in runs:
            if run.status in {"queued", "running"} and not _process_running(run.pid):
                run.status = "finished" if run.return_code in {None, 0} else "failed"
                run.finished_at = run.finished_at or datetime.now().isoformat(timespec="seconds")
                changed = True
        if changed:
            state["runs"] = [asdict(run) for run in runs]
            self._save(state)
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def get(self, run_id: str) -> RunRecord | None:
        return next((run for run in self.list_runs() if run.id == run_id), None)

    def launch(self, name: str, workflow: str, command: list[str], run_dir: Path) -> RunRecord:
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "platform_stdout.log"
        stderr_path = run_dir / "platform_stderr.log"
        run = RunRecord(
            id=f"{_timestamp()}_{uuid.uuid4().hex[:8]}",
            name=name.strip() or f"{workflow}_{_timestamp()}",
            workflow=workflow,
            status="queued",
            created_at=datetime.now().isoformat(timespec="seconds"),
            command=[str(value) for value in command],
            run_dir=str(run_dir),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                run.command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        run.pid = process.pid
        run.status = "running"
        run.started_at = datetime.now().isoformat(timespec="seconds")
        state = self._load()
        state.setdefault("runs", []).append(asdict(run))
        self._save(state)
        _json_dump(run_dir / "platform_run.json", asdict(run))
        return run

    def register_existing(self, name: str, output_dir: Path) -> RunRecord:
        output_dir = output_dir.resolve()
        if not output_dir.exists():
            raise FileNotFoundError(output_dir)
        existing = next((run for run in self.list_runs() if Path(run.run_dir) == output_dir), None)
        if existing:
            return existing
        run = RunRecord(
            id=f"imported_{_timestamp()}_{uuid.uuid4().hex[:8]}",
            name=name.strip() or output_dir.name,
            workflow="existing_collection",
            status="finished",
            created_at=datetime.now().isoformat(timespec="seconds"),
            command=[],
            run_dir=str(output_dir),
            stdout_path=str(output_dir / "platform_stdout.log"),
            stderr_path=str(output_dir / "platform_stderr.log"),
            finished_at=datetime.now().isoformat(timespec="seconds"),
            return_code=0,
        )
        state = self._load()
        state.setdefault("runs", []).append(asdict(run))
        self._save(state)
        return run

    def stop(self, run_id: str) -> bool:
        state = self._load()
        stopped = False
        for item in state.get("runs", []):
            if item.get("id") != run_id or not _process_running(item.get("pid")):
                continue
            pid = int(item["pid"])
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T"], capture_output=True, check=False)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            item["status"] = "stopped"
            item["finished_at"] = datetime.now().isoformat(timespec="seconds")
            stopped = True
        self._save(state)
        return stopped


def controller_command(*args: str) -> list[str]:
    return [sys.executable, "-u", str(CONTROLLER), *[str(arg) for arg in args]]


def campaign_command(*args: str) -> list[str]:
    return [sys.executable, "-u", str(CAMPAIGN_RUNNER), *[str(arg) for arg in args]]


def common_controller_args(
    *, base_csv: str, ontology: str, work_dir: str, integration_dir: str,
    model_dir: str, reasoning_mode: str = "auto", llm_api_url: str = "",
    llm_model: str = "", vision_api_url: str = "", vision_model: str = "",
) -> list[str]:
    args = [
        "--base_csv", base_csv,
        "--ontology_path", ontology,
        "--work_dir", work_dir,
        "--integration_out_dir", integration_dir,
        "--model_out_dir", model_dir,
        "--model_script", str(DEFAULT_MODEL),
        "--llm_cache_enable", "1",
        "--use_reasoning_layer", "0" if reasoning_mode == "off" else "1",
        "--reasoning_policy_mode", reasoning_mode,
        "--inline_vision", "0",
    ]
    if llm_api_url:
        args += ["--llm_api_url", llm_api_url]
    if llm_model:
        args += ["--llm_model", llm_model]
    if vision_api_url:
        args += ["--vision_api_url", vision_api_url]
    if vision_model:
        args += ["--vision_model", vision_model]
    return args


def tail_text(path: str | Path, lines: int = 200) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])


def discover_outputs(run_dir: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    root = Path(run_dir)
    if not root.exists():
        return []
    allowed = {".csv", ".json", ".txt", ".md", ".png", ".svg", ".xlsx", ".pdf"}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [
        {
            "name": path.name,
            "path": str(path),
            "type": path.suffix.lower().lstrip("."),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
        for path in files[:limit]
    ]


def csv_preview(path: str | Path, limit: int = 100) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append(dict(row))
        return list(reader.fieldnames or []), rows
