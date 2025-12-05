#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import json
from pathlib import Path

import typer
from datasets import load_dataset
from minisweagent.run.extra.extra_helpers import filter_instances, run_agent, DatasetConfig
from minisweagent.config import builtin_config_dir
from minisweagent.utils.log import add_file_handler, logger
from typing import Optional, Callable

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DEFAULT_SWEBENCH_AGENT_SPEC = builtin_config_dir / "extra" / "agent-spec.yaml"
DEFAULT_SWEBENCH_CAPABILITY_SPEC = builtin_config_dir / "extra" / "swebench-spec.yaml"

def default_problem_statement(instance: dict, lm_eval_task: any) -> str:
    """Default problem statement for the SWEBench dataset."""
    return instance["problem_statement"]

class SwebenchDatasetConfig(DatasetConfig):
    """Configuration for the SWEBench dataset."""
    dataset_name: str
    id_field: Callable[[dict], str] = lambda x: x["instance_id"]
    id_field_name: str = "instance_id"
    mk_task: Callable[[dict], str] = default_problem_statement
    def __init__(self, dataset_name: str, id_field: Optional[Callable[[dict], str]] = lambda x: x["instance_id"], mk_task: Optional[Callable[[dict], str]] = default_problem_statement, id_field_name: str = "instance_id"):
        self.dataset_name = dataset_name
        self.id_field = id_field
        self.mk_task = mk_task
        self.id_field_name = id_field_name

DATASET_MAPPING = {
    "full": SwebenchDatasetConfig(dataset_name="princeton-nlp/SWE-Bench"),
    "verified": SwebenchDatasetConfig(dataset_name="princeton-nlp/SWE-Bench_Verified"),
    "lite": SwebenchDatasetConfig(dataset_name="princeton-nlp/SWE-Bench_Lite"),
    "multimodal": SwebenchDatasetConfig(dataset_name="princeton-nlp/SWE-Bench_Multimodal"),
    "multilingual": SwebenchDatasetConfig(dataset_name="swe-bench/SWE-Bench_Multilingual"),
    "smith": SwebenchDatasetConfig(dataset_name="SWE-bench/SWE-smith"),
    "_test": SwebenchDatasetConfig(dataset_name="klieret/swe-bench-dummy-test-dataset"),
    "eai-smith": SwebenchDatasetConfig(dataset_name="gs://consus-dataproc/code-data/eai-smith"),
}

def load_task_dataset(dataset_config: SwebenchDatasetConfig, split: str):
    dataset_name = dataset_config.dataset_name
    if dataset_name.startswith("gs://"):
        # if the dataset is on GCS, download and cache it
        if split.find("batch") == -1:
            logger.error(f"GCS datasets only support batch splits, got '{split}'")
            logger.error("Please use a split like 'batch1' to load from a batch of tasks.")
            return []
        gcs_uri = f"{dataset_name}/{split}/swesmith_ig_v2_train.jsonl"
        from minisweagent.run.utils.gcs_cache import ensure_local_gcs_file
        cached_file_path = ensure_local_gcs_file(gcs_uri)
        with open(cached_file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return load_dataset(dataset_name, split=split)

def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name

# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    instance_select_file: str = typer.Option("", "--instance-select-file", help="File containing instance IDs to select", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    agent_spec: Path = typer.Option(DEFAULT_SWEBENCH_AGENT_SPEC, "-a", "--agent-spec", help="Path to the agent config file", rich_help_panel="Basic"),
    capability_spec: Path = typer.Option(DEFAULT_SWEBENCH_CAPABILITY_SPEC, "-s", "--capability-spec", help="Path to the capability config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    max_runtime: int | None = typer.Option(None, "--max-runtime", help="Maximum runtime for all instances in seconds", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_config = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_config.dataset_name}, split {split}...")
    instances = list(load_task_dataset(dataset_config, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle, instance_select_file=instance_select_file, dataset_config=dataset_config)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    # replace image_name in all instances
    for instance in instances:
        instance["image_name"] = get_swebench_docker_image_name(instance)

    run_agent(instances=instances,
        lm_eval_task=None,
        output_path=output_path,
        agent_spec=agent_spec,
        capability_spec=capability_spec,
        dataset_config=dataset_config,
        default_docker_image_name=None,
        output_file_name="preds.json",
        workers=workers,
        environment_class=environment_class,
        model=model, model_class=model_class,
        total_time_for_all_instances=max_runtime)

if __name__ == "__main__":
    app()
