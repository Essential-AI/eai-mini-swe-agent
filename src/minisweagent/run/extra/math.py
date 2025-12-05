#!/usr/bin/env python3

"""Run mini-SWE-agent on MATH instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/math/  (usage docs)

import json
import re
import threading
import hashlib
from typing import Callable
from pathlib import Path
from dataclasses import dataclass
import typer
from datasets import load_dataset
from minisweagent.run.extra.math_equiv import latex_extract, math_match

from minisweagent.run.extra.extra_helpers import DatasetConfig, filter_instances, clean_output_path, run_agent, RESULT_FIELD, upload_to_gcs, instance_index_str

from minisweagent.config import builtin_config_dir
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on MATH instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/math/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

OUTPUT_FILE_NAME = "math-preds.json"
EVAL_RESULTS_FILE_NAME = "math-results.json"
TRAJECTORY_SUFFIX = ".traj.json"
LOG_FILE_NAME = "minisweagent-math.log"
DEFAULT_AGENT_SPEC_FILE_NAME = builtin_config_dir / "extra" / "agent-spec.yaml"
DEFAULT_CAPABILITY_SPEC_FILE_NAME = builtin_config_dir / "extra" / "math-spec.yaml"

@dataclass
class MathDatasetConfig(DatasetConfig):
    # The name of the HF dataset
    hf_dataset_name: str
    # instance ID field
    id_field_name: str
    id_field: Callable[..., str]
    # answer field
    answer_fn: Callable[[dict], str]
    # hardness field
    hardness_field: str
    # task maker
    mk_task: Callable[[dict], str]
    # verify answer
    verify_answer: Callable[[str, str], bool]

def latex_verify(result: str, answer: str) -> bool:
    extracted_answer = latex_extract(answer)
    extracted_result = latex_extract(result)
    return extracted_result, math_match(extracted_answer, extracted_result)

def extract_aops_answer(instance: dict) -> str:
    # extract expected_answer from  'metadata': '{"expected_answer": "25", "problem_source": "aops_c4_high_school_math"}'
    metadata = instance["metadata"]
    return json.loads(metadata)["expected_answer"]

def extract_gsm_answer(instance: dict) -> str:
    # extract from #### <answer> at the end of the answer field
    answer = instance["answer"]
    if "####" not in answer:
        return answer
    else:
        # find the last occurrence of ####
        last_occurrence = answer.rfind("####")
        if last_occurrence == -1:
            return answer
        else:
            return answer[last_occurrence + len("####"):].strip()

def small_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]

DATASET_MAPPING = {
    "math500": MathDatasetConfig(
        hf_dataset_name="HuggingFaceH4/MATH-500",
        id_field=lambda instance: f'{instance["unique_id"]}',
        id_field_name="unique_id",
        answer_fn=lambda instance: instance["answer"],
        hardness_field="level",
        mk_task=lambda instance, lm_eval_task: instance["problem"],
        verify_answer=latex_verify,
    ),
    "aime25": MathDatasetConfig(
        hf_dataset_name="Maxwell-Jia/AIME_2024",
        id_field=lambda instance: f'{instance["ID"]}',
        id_field_name="ID",
        answer_fn=lambda instance: instance["Answer"],
        hardness_field="ID", # not a very good hardness metric, but it's all we have
        mk_task=lambda instance, lm_eval_task: instance["Problem"],
        verify_answer=latex_verify,
    ),
    "aime25": MathDatasetConfig(
        hf_dataset_name="math-ai/aime25",
        id_field=lambda instance: f'{instance["id"]}',
        id_field_name="id",
        answer_fn=lambda instance: instance["answer"],
        hardness_field="id", # not a very good hardness metric, but it's all we have
        mk_task=lambda instance, lm_eval_task: instance["problem"],
        verify_answer=latex_verify,
    ),
    "aime": MathDatasetConfig(
        hf_dataset_name="gneubig/aime-1983-2024",
        id_field=lambda instance: f'{instance["ID"]}',
        id_field_name="ID",
        answer_fn=lambda instance: instance["Answer"],
        hardness_field="Year", # not a very good hardness metric, but it's all we have
        mk_task=lambda instance, lm_eval_task: instance["Question"],
        verify_answer=latex_verify,
    ),
    "gsm8k": MathDatasetConfig(
        hf_dataset_name="epfl-dlab/gsm8k", # same as openai/gsm8k but with id field https://huggingface.co/datasets/epfl-dlab/gsm8k/blob/main/README.md
        id_field=lambda instance: f'{instance["id"]}',
        id_field_name="id",
        answer_fn=extract_gsm_answer,
        hardness_field="id", # no hardness designation
        mk_task=lambda instance, lm_eval_task: instance["question"] + "\n\nWrite your final at the end on a new line with #### <answer>",
        verify_answer=latex_verify,
    ),

    # Putnam Axiom dataset
    # splits=full_eval: 522 rows, variations: 500 rows
    "putnam-axiom": MathDatasetConfig(
        hf_dataset_name="Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522",
        id_field=lambda instance: f'{instance["year"]}-{instance["id"]}-{instance["variation"]}-{small_hash(instance["problem"])}',
        id_field_name="year-id-variation-prbhash",
        answer_fn=lambda instance: latex_extract(instance["solution"]),
        hardness_field="id", # no hardness designation
        mk_task=lambda instance, lm_eval_task: instance["problem"],
        verify_answer=latex_verify,
    ),
    # splits=original: 236 rows, variations: 265 rows
    "putnam-axiom-v1": MathDatasetConfig(
        hf_dataset_name="Putnam-AXIOM/putnam-axiom-dataset-v1",
        id_field=lambda instance: f'{instance["year"]}-{instance["id"]}-{instance["variation"]}',
        id_field_name="year-id-variation",
        answer_fn=lambda instance: latex_extract(instance["solution"]),
        hardness_field="id", # no hardness designation
        mk_task=lambda instance, lm_eval_task: instance["problem"],
        verify_answer=latex_verify,
    ),

}

DOCKER_IMAGE_NAME = "researchessential/python-3.12-slim-linux-env"


STATUS_EMOJI = {
    "CORRECT": "✅",
    "INCORRECT": "❌",
    "NO_ANSWER": "❓",
}

def check_results(output_path: Path, instances: list[dict], dataset_config: DatasetConfig) -> Path:
    """Check if the results are correct."""
    results = json.loads((output_path / OUTPUT_FILE_NAME).read_text())
    correct = 0
    incorrect = 0
    no_answer = 0
    total = 0
    eval_results = []
    for instance in instances:
        instance_id = dataset_config.id_field(instance)
        instance_id_str = instance_index_str(instance_id, lm_eval_task=None)
        if instance_id_str not in results:
            no_answer += 1
            continue
        answer = dataset_config.answer_fn(instance)
        result = results[instance_id_str][RESULT_FIELD]
        extracted_result, valid_answer = dataset_config.verify_answer(result, answer)
        status = "CORRECT" if valid_answer else "INCORRECT"
        if valid_answer:
            correct += 1
        else:
            incorrect += 1
        total += 1
        equal_not_equal = "==" if valid_answer else "!="
        eval_results.append({
            "instance_id": instance_id,
            "model_answer": result,
            "unboxed_model_answer": extracted_result,
            "gold": answer,
            "is_correct": valid_answer,
        })
        try:
            logger.info(f"{STATUS_EMOJI[status]} {extracted_result} {equal_not_equal} {answer} (ref) / {instance_id}")
        except:
            pass
    accuracy = round(float(correct) / total if total > 0 else 0, 4)
    logger.info(f"Total instances: {total}")
    logger.info(f"Correct instances: {correct}")
    logger.info(f"Incorrect instances: {incorrect}")
    logger.info(f"No answer instances: {no_answer}")
    logger.info(f"Accuracy: {accuracy}")

    eval_results_json = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "no_answer": no_answer,
        "eval_results": eval_results,
    }
    results_path = output_path / EVAL_RESULTS_FILE_NAME
    results_path.write_text(json.dumps(eval_results_json, indent=2))

    return results_path

# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("math500", "--subset", help="MATH dataset from custom mappings or path to a HF dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    instance_select_file: str = typer.Option("", "--instance-select-file", help="File containing instance IDs to select", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("agent-output", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    skip_agent_run: bool = typer.Option(False, "--skip-agent-run", help="Skip the agent run", rich_help_panel="Basic"),
    agent_spec: Path = typer.Option(DEFAULT_AGENT_SPEC_FILE_NAME, "-a", "--agent-spec", help="Path to the agent config file", rich_help_panel="Basic"),
    capability_spec: Path = typer.Option(DEFAULT_CAPABILITY_SPEC_FILE_NAME, "-s", "--capability-spec", help="Path to the capability config file", rich_help_panel="Basic"),
    max_runtime: int | None = typer.Option(None, "--max-runtime", help="Maximum runtime for all instances in seconds", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    clean_output_path(output_path, redo_existing=redo_existing)
    add_file_handler(output_path / LOG_FILE_NAME)

    assert subset in DATASET_MAPPING, f"Invalid subset: {subset}. Valid options are {list(DATASET_MAPPING.keys())}"

    dataset_config = DATASET_MAPPING[subset]
    dataset_path = dataset_config.hf_dataset_name
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    if dataset_path.startswith("gs://"):
        type = "parquet" if dataset_path.endswith(".parquet") else "json"
        dataset = load_dataset(type, data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split=split)
    instances = list(dataset)

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle, instance_select_file=instance_select_file, dataset_config=dataset_config)
    if not redo_existing and (output_path / OUTPUT_FILE_NAME).exists():
        existing_instances = list(json.loads((output_path / OUTPUT_FILE_NAME).read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if dataset_config.id_field(instance) not in existing_instances]

    if not skip_agent_run:
        run_agent(instances=instances,
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
    eval_results_path = check_results(output_path, instances, dataset_config=dataset_config)

    # upload to gcs
    dir_prefix = f"math-{subset}-{split}-{slice_spec}"
    gcs_location = upload_to_gcs(output_path, upload_dir_prefix=dir_prefix, overwrite=redo_existing)

    logger.info(f"Predictions saved to {output_path / OUTPUT_FILE_NAME}")
    logger.info(f"Evaluation results saved to {eval_results_path}")
    logger.info(f"Log file saved to {output_path / LOG_FILE_NAME}")


if __name__ == "__main__":
    app()
