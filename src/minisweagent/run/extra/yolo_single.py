"""Run on a single instance with no recommended workflow."""

import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.math import (
    DOCKER_IMAGE_NAME,
    DEFAULT_AGENT_SPEC_FILE_NAME,
)
from minisweagent.run.extra.extra_helpers import get_extra_environment
from minisweagent.run.extra.extra_helpers import build_config_from_specs_single
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT = global_config_dir / "last_yolo_single_run.traj.json"

OUTPUT_FILE_NAME = "yolo_single_run.json"

DEFAULT_CAPABILITY_SPEC_FILE_NAME = builtin_config_dir / "extra" / "yolo-spec.yaml"

# fmt: off
@app.command()
def main(
    subset: str = typer.Option("math500", "--subset", help="MATH dataset from custom mappings or path to a HF dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="MATH instance ID or index", rich_help_panel="Data selection"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    agent_spec: Path = typer.Option(DEFAULT_AGENT_SPEC_FILE_NAME, "-a", "--agent-spec", help="Path to the agent config file", rich_help_panel="Basic"),
    capability_spec: Path = typer.Option(DEFAULT_CAPABILITY_SPEC_FILE_NAME, "-s", "--capability-spec", help="Path to the capability config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", rich_help_panel="Advanced"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Basic"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
    task: str = typer.Option("What is the meaning of the universe?", "-t", "--task", help="Task to run", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    """Run on a single instance with no recommended workflow."""
    instance = {
        "task": task,
    }

    config = build_config_from_specs_single(agent_spec, capability_spec)
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    env = get_extra_environment(config, instance, DOCKER_IMAGE_NAME)
    agent = InteractiveAgent(
        get_model(model_name, config.get("model", {})),
        env,
        **({"mode": "yolo"} | config.get("agent", {})),
        instance_id = 0
    )

    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(instance)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance_spec}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
