from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = PROJECT_ROOT / "deliverables" / "distribution"
VERSION = "2.1.0"
PACKAGE_NAME = f"LiteratureAgent_Standalone_Windows_{VERSION}"
STAGING_ROOT = DIST_ROOT / PACKAGE_NAME
ZIP_PATH = DIST_ROOT / f"{PACKAGE_NAME}.zip"
PYTHON_RUNTIME = PROJECT_ROOT / "polaris_full_stack" / ".tools" / "Python312"
OLLAMA_INSTALLER = PROJECT_ROOT / "third_party" / "installers" / "OllamaSetup.exe"

IGNORED_NAMES = {
    ".git", ".github", ".idea", ".pytest_cache", ".ruff_cache", ".venv",
    ".vscode", "__pycache__", "platform_workspace", "runs",
}
FORBIDDEN_SUFFIXES = {".db", ".log", ".pyc", ".pyo", ".sqlite", ".sqlite3", ".tmp"}
TEXT_SUFFIXES = {".bat", ".cmd", ".ini", ".json", ".md", ".ps1", ".py", ".toml", ".txt", ".yaml", ".yml"}
LOCAL_MARKERS = (
    "E:" + "\\LiteratureAgentProject",
    "C:" + "\\Users\\jorda",
    ".codex" + "\\attachments",
    "Jmarsh39" + "@vols.utk.edu",
    "client_secret_" + "298117362376",
)


def _assert_distribution_path(path: Path) -> None:
    root = DIST_ROOT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing to modify a path outside {root}: {resolved}")


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name for name in names
        if name.casefold() in {item.casefold() for item in IGNORED_NAMES}
        or name.casefold() in {".env", ".env.local", ".env.production", ".env.development"}
        or Path(name).suffix.casefold() in FORBIDDEN_SUFFIXES
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, ignore=_ignore)


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copy_payload() -> None:
    _copy_tree(PYTHON_RUNTIME, STAGING_ROOT / ".tools" / "Python312")
    _copy_tree(PROJECT_ROOT / "literature_agent_platform", STAGING_ROOT / "literature_agent_platform")
    _copy_tree(PROJECT_ROOT / "literature_agent", STAGING_ROOT / "literature_agent")
    _copy_tree(PROJECT_ROOT / "config", STAGING_ROOT / "config")

    for name in (
        "literature_agent_full_end_to_end_v21_3_english_sanitizer.py",
        "requirements.txt",
        "requirements-platform.txt",
        "START_LITERATURE_AGENT.cmd",
        "Setup_LiteratureAgent.cmd",
        "Setup_Quality_Models.cmd",
        "Launch_LiteratureAgent.cmd",
    ):
        _copy(PROJECT_ROOT / name, STAGING_ROOT / name)

    for name in (
        "install_and_launch_platform.ps1",
        "run_expansion_campaign.py",
        "build_literature_knowledge_graph.py",
        "build_reasoning_policy_knowledge_graph_variants.py",
    ):
        _copy(PROJECT_ROOT / "scripts" / name, STAGING_ROOT / "scripts" / name)

    _copy(
        PROJECT_ROOT / "data" / "Perovskite_database_content_all_data.csv",
        STAGING_ROOT / "data" / "Perovskite_database_content_all_data.csv",
    )
    _copy(
        PROJECT_ROOT / "models" / "pce_then_stability_same_approach.py",
        STAGING_ROOT / "models" / "pce_then_stability_same_approach.py",
    )
    _copy(OLLAMA_INSTALLER, STAGING_ROOT / "installers" / "OllamaSetup.exe")
    _copy_tree(PROJECT_ROOT / "third_party" / "licenses", STAGING_ROOT / "third_party" / "licenses")
    _copy(PROJECT_ROOT / "docs" / "STANDALONE_PLATFORM_GUIDE.md", STAGING_ROOT / "docs" / "STANDALONE_PLATFORM_GUIDE.md")
    _copy(PROJECT_ROOT / "docs" / "CUSTOMIZING_FOR_A_NEW_DOMAIN.md", STAGING_ROOT / "docs" / "CUSTOMIZING_FOR_A_NEW_DOMAIN.md")
    (STAGING_ROOT / "secrets").mkdir(parents=True, exist_ok=True)
    _write(STAGING_ROOT / "secrets" / "README.txt", "Optional Google OAuth files belong here. Never share personal credentials or tokens.")


def _write_recipient_files() -> None:
    _write(STAGING_ROOT / "VERSION.txt", f"LiteratureAgent standalone Windows distribution {VERSION}")
    _write(
        STAGING_ROOT / "README_FIRST.txt",
        r"""
LITERATUREAGENT STANDALONE FOR WINDOWS
======================================

FIRST USE
1. Extract the entire ZIP to a normal local folder. Do not run it from inside the ZIP.
2. Double-click Setup_LiteratureAgent.cmd.
3. If Ollama is not installed, complete the official Ollama installer window that opens.
4. Leave the setup window open while dependencies and the Qwen models download.
5. When setup reports success, double-click Launch_LiteratureAgent.cmd.

The default laptop model download is approximately 5.1 GB. Internet access is
required for Python packages, Qwen model downloads, and literature web searches.
Python and the official Ollama Windows installer are included in this package.

OPTIONAL QUALITY MODE
Double-click Setup_Quality_Models.cmd to use the larger 7B text and vision models
(approximately 10.7 GB total). Later launches remember the selected profile.

SCIENTIFIC WORKFLOWS
- General science: domain-neutral structured records, summaries, evidence, figures, and knowledge graphs.
- Oxide perovskites: ABO3 composition/sites, substitutions, oxygen defects, structure, processing,
  characterization, properties, evidence, and knowledge graphs.
- Halide-perovskite photovoltaics: the original detailed device/PCE/stability records, integration,
  predictive modeling, and knowledge graphs.

Only the halide-PV profile can run the PCE/stability integration and predictive model.
All three profiles support structured extraction and knowledge-graph generation.

BEFORE A NON-PEROVSKITE OR CUSTOM-DOMAIN RUN
Read docs\CUSTOMIZING_FOR_A_NEW_DOMAIN.md before processing a full collection.
Start with the General science profile and pilot it on 1-3 papers you know well.

- Copy config\domain_packs\custom_domain_template.json and edit the copy.
- In General science > Advanced options, select the copy as the domain-pack JSON.
- Decide the database row grain, identifiers, fields, types, units, and missing-value rules.
- canonical_terms controls graph names and aliases; it does not add database columns.
- structured_output.fields adds evidence-gated Domain_<field> columns without code edits.
- Custom fields receive generic graph nodes/evidence links; add typed mappings only when needed.
- Inspect csv\domain_records.csv, raw JSON, and the Neo4j import CSVs before scaling up.

The bundled Perovskite Database CSV and PCE/stability model are retained only for
the halide-photovoltaics profile. Do not use or overwrite that baseline as the
database for another scientific domain.

LATER USE
Double-click Launch_LiteratureAgent.cmd. The interface opens at http://127.0.0.1:8501.
This production package opens the working application directly; it contains no video-demo mode.

REQUIREMENTS
- 64-bit Windows 10 or 11.
- At least 8 GB RAM for the laptop profile; 16 GB or more is recommended.
- About 9 GB free for setup with laptop models, in addition to this extracted folder.
- Internet access and permission to run a per-user installer.

PRIVACY
No API key, OAuth credential, prior paper, prior run, or user setting is included.
Local Qwen requests are sent to Ollama on this computer. Literature retrieval still
contacts the selected scholarly/web sources.

TROUBLESHOOTING
- Keep the extracted folder together; do not move individual files out of it.
- Re-run Setup_LiteratureAgent.cmd after an interrupted model or dependency download.
- Logs and settings are under %LOCALAPPDATA%\LiteratureAgent.
- The original Ollama installer is in installers\OllamaSetup.exe.
""",
    )
    _write(
        STAGING_ROOT / "THIRD_PARTY_NOTICES.txt",
        """
This package includes the unchanged official Ollama Windows installer and its MIT
license in third_party/licenses/Ollama_LICENSE.txt. Qwen model weights are not in
the ZIP; Ollama downloads them on first setup. Model names and license links are
documented in third_party/licenses/QWEN_MODEL_NOTICE.txt.
""",
    )


def _audit_and_manifest() -> tuple[int, int]:
    required = (
        ".tools/Python312/python.exe",
        "installers/OllamaSetup.exe",
        "scripts/install_and_launch_platform.ps1",
        "literature_agent_platform/app.py",
        "literature_agent_platform/domain_profiles.py",
        "literature_agent_full_end_to_end_v21_3_english_sanitizer.py",
        "scripts/build_literature_knowledge_graph.py",
        "data/Perovskite_database_content_all_data.csv",
        "config/domain_packs/custom_domain_template.json",
        "config/domain_packs/general_science.json",
        "config/domain_packs/oxide_perovskite.json",
        "config/perovskite_ontology_library_v19.json",
        "docs/CUSTOMIZING_FOR_A_NEW_DOMAIN.md",
        "docs/STANDALONE_PLATFORM_GUIDE.md",
        "literature_agent/domain_pack.py",
        "third_party/licenses/Ollama_LICENSE.txt",
        "README_FIRST.txt",
    )
    missing = [name for name in required if not (STAGING_ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Missing required package files: " + ", ".join(missing))

    files = sorted(path for path in STAGING_ROOT.rglob("*") if path.is_file())
    problems: list[str] = []
    demo_module = STAGING_ROOT / "literature_agent_platform" / "video_demo.py"
    if demo_module.exists():
        problems.append(f"video-demo module in production package: {demo_module.relative_to(STAGING_ROOT)}")
    app_path = STAGING_ROOT / "literature_agent_platform" / "app.py"
    app_text = app_path.read_text(encoding="utf-8", errors="ignore").casefold()
    for marker in ("video_demo", "video walkthrough", "render_video_demo", "demo_requested"):
        if marker in app_text:
            problems.append(f"video-demo activation marker in literature_agent_platform/app.py: {marker}")
    for path in files:
        relative = path.relative_to(STAGING_ROOT)
        if path.name.casefold() in {".env", ".env.local"} or path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            problems.append(f"private/generated file: {relative}")
        if path.suffix.casefold() in TEXT_SUFFIXES and path.stat().st_size <= 5_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in LOCAL_MARKERS:
                if marker.casefold() in text.casefold():
                    problems.append(f"machine-specific path in {relative}: {marker}")
    if problems:
        raise RuntimeError("Package audit failed:\n" + "\n".join(problems[:50]))

    lines = [
        "LITERATUREAGENT STANDALONE PACKAGE MANIFEST",
        f"Version: {VERSION}",
        "Audit: PASS",
        "", "SHA256  PATH",
    ]
    for path in files:
        lines.append(f"{_sha256(path)}  {path.relative_to(STAGING_ROOT).as_posix()}")
    _write(STAGING_ROOT / "PACKAGE_MANIFEST.txt", "\n".join(lines))
    final_files = [path for path in STAGING_ROOT.rglob("*") if path.is_file()]
    return len(final_files), sum(path.stat().st_size for path in final_files)


def _zip() -> str:
    if ZIP_PATH.exists():
        _assert_distribution_path(ZIP_PATH)
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", allowZip64=True) as archive:
        for path in sorted(STAGING_ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(PACKAGE_NAME) / path.relative_to(STAGING_ROOT)
            compression = zipfile.ZIP_STORED if path.suffix.casefold() in {".exe", ".zip"} else zipfile.ZIP_DEFLATED
            archive.write(path, relative, compress_type=compression, compresslevel=6 if compression == zipfile.ZIP_DEFLATED else None)
    digest = _sha256(ZIP_PATH)
    _write(ZIP_PATH.with_suffix(".sha256.txt"), f"{digest}  {ZIP_PATH.name}")
    return digest


def main() -> int:
    for required in (PYTHON_RUNTIME / "python.exe", OLLAMA_INSTALLER):
        if not required.is_file():
            raise FileNotFoundError(required)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    if STAGING_ROOT.exists():
        _assert_distribution_path(STAGING_ROOT)
        shutil.rmtree(STAGING_ROOT)
    _copy_payload()
    _write_recipient_files()
    file_count, uncompressed_bytes = _audit_and_manifest()
    digest = _zip()
    print(f"Built: {ZIP_PATH}")
    print(f"Files: {file_count:,}")
    print(f"Uncompressed bytes: {uncompressed_bytes:,}")
    print(f"ZIP bytes: {ZIP_PATH.stat().st_size:,}")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
