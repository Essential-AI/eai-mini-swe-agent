"""Run on a single MATH instance."""

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
    DATASET_MAPPING,
    DatasetConfig,
    DOCKER_IMAGE_NAME,
    DEFAULT_AGENT_SPEC_FILE_NAME,
    DEFAULT_CAPABILITY_SPEC_FILE_NAME,
)
from minisweagent.run.extra.extra_helpers import get_extra_environment
from minisweagent.run.extra.extra_helpers import build_config_from_specs_single
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT = global_config_dir / "last_math_single_run.traj.json"

OUTPUT_FILE_NAME = "math_single_run.json"

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
) -> None:
    # fmt: on
    """Run on a single MATH instance."""
    dataset_config: DatasetConfig = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset from {dataset_config.hf_dataset_name}, split {split}...")
    instances = {
        dataset_config.id_field(inst): inst  # type: ignore
        for inst in load_dataset(dataset_config.hf_dataset_name, split=split)
    }
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

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
        instance_id = dataset_config.id_field(instance)
    )

    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(dataset_config.mk_task(instance, None))  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance_spec}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
