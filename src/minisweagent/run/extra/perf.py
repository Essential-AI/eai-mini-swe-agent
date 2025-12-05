#!/usr/bin/env python3

"""Run mini-SWE-agent on Perf instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/math/  (usage docs)

import concurrent.futures
import json
from pathlib import Path
from dataclasses import dataclass
import typer
from datasets import load_dataset
from typing import Callable

from minisweagent.run.extra.extra_helpers import filter_instances, run_agent, RESULT_FIELD
from minisweagent.run.extra.extra_helpers import upload_to_gcs, clean_output_path, instance_index_str
from minisweagent.run.extra.dataset_mappings import get_dataset, CodeDatasetConfig, DATASET_MAPPING
from minisweagent.run.extra.extra_helpers import DatasetConfig

from minisweagent.run.extra.perf_eval.enamel import is_valid_enamel_name
from minisweagent.config import builtin_config_dir
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on Perf instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/perf/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

OUTPUT_FILE_NAME = "perf-preds.json"
EVAL_RESULTS_FILE_NAME = "perf-results.json"
LOG_FILE_NAME = "minisweagent-perf.log"
DEFAULT_SPEC_FILE_NAME = builtin_config_dir / "extra" / "perf-spec.yaml"
DEFAULT_AGENT_SPEC_FILE_NAME = builtin_config_dir / "extra" / "agent-spec.yaml"


DOCKER_IMAGE_NAME = "researchessential/python-3.12-slim-linux-env"

def rm_non_fns(completion: str) -> str:
    """
    Remove any toplevel print and assertion statements from the completion.
    Look for any line that starts with 'print' or 'assert' and remove it.
    """
    lines = completion.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('print') or line.startswith('assert'):
            continue
        new_lines.append(line)
    return '\n'.join(new_lines)


def check_results(output_path: Path, instances: list[dict], dataset_config: CodeDatasetConfig, print_individual_metrics: bool) -> Path:
    """Check if the results are correct."""
    results = json.loads((output_path / OUTPUT_FILE_NAME).read_text())

    # create a list of "HumanEval/k" -> [completion1, completion2, ...]
    completions: dict[str, list[str]] = {}
    for instance in instances:
        instance_id = dataset_config.id_field(instance)
        instance_id_str = instance_index_str(instance_id, lm_eval_task=None)
        if instance_id_str not in results:
            continue
        if not is_valid_enamel_name(instance_id):
            logger.warning(f'Enamel dataset only evals HumanEval/k instances. Skipping {instance_id}...')
            continue
        completion = results[instance_id_str][RESULT_FIELD]
        completions[instance_id] = [rm_non_fns(completion)] if isinstance(completion, str) else completion

    efficiency = dataset_config.eval_function(completions, print_individual_metrics)

    eval = {
        "efficiency": efficiency,
        "completions": completions,
        "instances": instances,
    }
    results_path = output_path / EVAL_RESULTS_FILE_NAME
    results_path.write_text(json.dumps(eval, indent=2))

    return results_path

# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("enamel", "--subset", help="Perf dataset from custom mappings or path to a HF dataset", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    instance_select_file: str = typer.Option("", "--instance-select-file", help="File containing instance IDs to select", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    output: str = typer.Option("agent-output", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    agent_spec: Path = typer.Option(DEFAULT_AGENT_SPEC_FILE_NAME, "-a", "--agent-spec", help="Path to the agent config file", rich_help_panel="Basic"),
    capability_spec: Path = typer.Option(DEFAULT_SPEC_FILE_NAME, "-c", "--capability-spec", help="Path to the capability config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    max_runtime: int | None = typer.Option(None, "--max-runtime", help="Maximum runtime for all instances in seconds", rich_help_panel="Advanced"),
    skip_agent_run: bool = typer.Option(False, "--skip-agent-run", help="Skip the agent run", rich_help_panel="Basic"),
    print_individual_metrics: bool = typer.Option(False, "--print-individual-metrics", help="Print individual metrics", rich_help_panel="Basic"),
) -> None:
    output_path = Path(output)
    clean_output_path(output_path, redo_existing=redo_existing)
    add_file_handler(output_path / LOG_FILE_NAME)

    assert subset in DATASET_MAPPING, f"Subset {subset} not found in DATASET_MAPPING: {DATASET_MAPPING.keys()}"
    dataset_config: DatasetConfig = DATASET_MAPPING[subset]

    filtered_instances = get_dataset(dataset_config=dataset_config,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        instance_select_file=instance_select_file,
        redo_existing=redo_existing,
        output_file=output_path / OUTPUT_FILE_NAME)

    if not skip_agent_run:
        logger.info(f"Running on {len(filtered_instances)} instances...")
        run_agent(instances=filtered_instances,
                  lm_eval_task=None,
                  output_path=output_path,
                  agent_spec=agent_spec,
                  capability_spec=capability_spec,
                  dataset_config=dataset_config,
                  default_docker_image_name=DOCKER_IMAGE_NAME,
                  output_file_name=OUTPUT_FILE_NAME,
                  workers=workers,
                  total_time_for_all_instances=max_runtime,
                  environment_class=environment_class,
                  model=model, model_class=model_class)

    # now check if the results are correct
    eval_results_path = None
    if dataset_config.eval_function is not None:
        eval_results_path = check_results(output_path, filtered_instances,
        dataset_config=dataset_config, print_individual_metrics=print_individual_metrics)

    # upload to gcs
    dir_prefix = f"perf-{subset}-{dataset_config.split if dataset_config.split is not None else 'all'}"
    gcs_location = upload_to_gcs(output_path, upload_dir_prefix=dir_prefix, overwrite=redo_existing)

    logger.info(f"[Local] Predictions saved to {output_path / OUTPUT_FILE_NAME}")
    logger.info(f"[Local] Evaluation results saved to {eval_results_path}")
    logger.info(f"[Local] Log file saved to {output_path / LOG_FILE_NAME}")


if __name__ == "__main__":
    app()
