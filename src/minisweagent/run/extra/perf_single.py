"""Run on a single performant code generation instance."""

import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.perf import (
    DATASET_MAPPING,
    DatasetConfig,
    DOCKER_IMAGE_NAME,
    DEFAULT_AGENT_SPEC_FILE_NAME,
    DEFAULT_SPEC_FILE_NAME,
)
from minisweagent.run.extra.extra_helpers import get_extra_environment
from minisweagent.run.extra.extra_helpers import build_config_from_specs_single
from minisweagent.run.extra.dataset_mappings import get_dataset
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT = global_config_dir / "last_perf_single_run.traj.json"

OUTPUT_FILE_NAME = "perf_single_run.json"

# fmt: off
@app.command()
def main(
    subset: str = typer.Option("enamel", "--subset", help="Perf dataset from custom mappings or path to a HF dataset", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    instance_select_file: str = typer.Option("", "--instance-select-file", help="File containing instance IDs to select", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="Perf instance ID or index", rich_help_panel="Data selection"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    agent_spec: Path = typer.Option(DEFAULT_AGENT_SPEC_FILE_NAME, "-a", "--agent-spec", help="Path to the agent config file", rich_help_panel="Basic"),
    capability_spec: Path = typer.Option(DEFAULT_SPEC_FILE_NAME, "-s", "--capability-spec", help="Path to the capability config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", rich_help_panel="Advanced"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Basic"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    """Run on a single Perf instance."""

    assert subset in DATASET_MAPPING, f"Subset {subset} not found in DATASET_MAPPING: {DATASET_MAPPING.keys()}"
    dataset_config: DatasetConfig = DATASET_MAPPING[subset]

    filtered_instances = get_dataset(dataset_config=dataset_config,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        instance_select_file=instance_select_file,
        redo_existing=True,
        output_file=output / OUTPUT_FILE_NAME)

    # sort the instances by the id field
    instance = filtered_instances[int(instance_spec)]  # type: ignore

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
        task = dataset_config.mk_task(instance, None)
        exit_status, result = agent.run(task)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance_spec}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
