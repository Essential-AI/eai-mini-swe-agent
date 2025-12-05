from minisweagent.run.extra.perf_eval.enamel import eval_enamel
from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import json
from minisweagent.utils.log import logger
from datasets import load_dataset


from minisweagent.run.extra.extra_helpers import filter_instances, DatasetConfig

@dataclass
class CodeDatasetConfig:
    dataset_name: str
    id_field: Callable[..., str]
    id_field_name: str
    mk_task: Callable[[dict, any], str]
    eval_function: Callable[[dict[int, list[str]], bool], dict]
    split: str

DATASET_MAPPING = {
    "he": CodeDatasetConfig(
        dataset_name="openai/openai_humaneval",
        id_field=lambda instance: instance["task_id"],
        id_field_name="task_id",
        mk_task=lambda instance, lm_eval_task: instance["prompt"],
        split="test",
        eval_function=None,
    ),
    "enamel": CodeDatasetConfig(
        dataset_name="q-rz/enamel",
        id_field=lambda instance: instance["task_id"],
        id_field_name="task_id",
        eval_function=eval_enamel,
        mk_task=lambda instance, lm_eval_task: instance["prompt"],
        split="ENAMEL_HumanEval",
    ),
}

def get_dataset(dataset_config: DatasetConfig,
    filter_spec: str,
    slice_spec: str,
    instance_select_file: str,
    redo_existing: bool,
    output_file: Path) -> list[dict]:
    """Get the dataset instances for the given subset, split, filter spec, slice spec, instance select file and redo existing."""
    logger.info(f"Loading dataset {dataset_config.dataset_name}, split {dataset_config.split}...")
    if dataset_config.dataset_name.startswith("gs://"):
        # For GCS paths, we need to download the file first or use a supported path
        # load_dataset with "json" format expects local files or HTTP URLs
        # For JSONL files, we should specify the split explicitly
        dataset = load_dataset("json", data_files=dataset_config.dataset_name, split="train")
    else:
        dataset = load_dataset(dataset_config.dataset_name, split=dataset_config.split)
    dataset_instances = list(dataset)
    # print(f"Dataset instances: {dataset_instances[0]}")
    filtered_instances = filter_instances(dataset_instances, filter_spec=filter_spec, slice_spec=slice_spec, instance_select_file=instance_select_file, dataset_config=dataset_config)
    if not redo_existing and output_file.exists():
        existing_instances = list(json.loads((output_file).read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        filtered_instances = [instance for instance in instances if instance[dataset_config.id_field] not in existing_instances]
    return filtered_instances
