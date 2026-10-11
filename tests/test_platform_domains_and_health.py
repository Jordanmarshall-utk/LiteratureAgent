from literature_agent_platform.domain_profiles import DOMAIN_PROFILES, PROJECT_ROOT, command_option
from literature_agent_platform.health import _model_present


def test_required_domain_profiles_are_shipped() -> None:
    assert {"general_science", "oxide_perovskite", "halide_pv"} <= DOMAIN_PROFILES.keys()
    assert all(profile.ontology.is_file() for profile in DOMAIN_PROFILES.values())
    assert DOMAIN_PROFILES["halide_pv"].supports_pce_modeling
    assert not DOMAIN_PROFILES["oxide_perovskite"].supports_pce_modeling


def test_command_option_preserves_run_domain() -> None:
    command = ["python", "controller.py", "--domain_profile", "oxide_perovskite"]
    assert command_option(command, "--domain_profile", "halide_pv") == "oxide_perovskite"
    assert command_option(command, "--missing", "fallback") == "fallback"


def test_health_requires_the_configured_model_not_any_qwen_variant() -> None:
    installed = ["qwen2.5:7b", "qwen2.5vl:7b-q4_K_M"]
    assert _model_present("qwen2.5:7b", installed)
    assert not _model_present("qwen2.5:3b", installed)


def test_windows_setup_handles_expected_missing_dependencies_and_writes_json_without_bom() -> None:
    script = (PROJECT_ROOT / "scripts" / "install_and_launch_platform.ps1").read_text(encoding="utf-8")
    assert "try {\n    & $venvPython -c \"import streamlit\"" in script
    assert "New-Object System.Text.UTF8Encoding($false)" in script
    assert "[System.IO.File]::WriteAllText" in script
    assert "installers\\OllamaSetup.exe" in script


def test_ollama_packaging_inputs_are_licensed_and_source_safe() -> None:
    installer = PROJECT_ROOT / "third_party" / "installers" / "OllamaSetup.exe"
    license_path = PROJECT_ROOT / "third_party" / "licenses" / "Ollama_LICENSE.txt"
    assert "MIT License" in license_path.read_text(encoding="utf-8")
    if installer.exists():
        assert installer.stat().st_size > 1_000_000_000
    else:
        assert "installers/*.exe" in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
