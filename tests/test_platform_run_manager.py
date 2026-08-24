from pathlib import Path

from literature_agent_platform.run_manager import (
    CAMPAIGN_RUNNER,
    CONTROLLER,
    campaign_command,
    common_controller_args,
    controller_command,
)


def test_controller_command_uses_argument_list() -> None:
    command = controller_command("--search_query", "perovskite stability T80")
    assert Path(command[2]) == CONTROLLER
    assert command[-1] == "perovskite stability T80"
    assert "shell=True" not in command


def test_campaign_command_preserves_query_as_one_argument() -> None:
    command = campaign_command("--search-query", "perovskite stability T80")
    assert Path(command[2]) == CAMPAIGN_RUNNER
    assert command[-1] == "perovskite stability T80"


def test_common_args_keep_vision_out_of_extraction() -> None:
    args = common_controller_args(
        base_csv="base.csv",
        ontology="ontology.json",
        work_dir="outputs",
        integration_dir="integration",
        model_dir="models",
        reasoning_mode="auto",
    )
    assert args[args.index("--inline_vision") + 1] == "0"
    assert args[args.index("--use_reasoning_layer") + 1] == "1"


def test_reasoning_can_be_disabled_without_changing_output_contract() -> None:
    args = common_controller_args(
        base_csv="base.csv",
        ontology="ontology.json",
        work_dir="outputs",
        integration_dir="integration",
        model_dir="models",
        reasoning_mode="off",
    )
    assert args[args.index("--use_reasoning_layer") + 1] == "0"
