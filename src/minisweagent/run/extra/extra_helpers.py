
import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from typing import Callable, Optional
from pathlib import Path
import yaml
import subprocess
import shutil
from jinja2 import StrictUndefined, Template
from rich.live import Live
import os

from minisweagent.agents.default import DefaultAgent
from minisweagent import Environment
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    id_field_name: str
    id_field: Callable[..., str]
    mk_task: Callable[..., str]

TRAJECTORY_SUFFIX = ".traj.json"
RESULT_FIELD = "model_patch"
MODEL_NAME_FIELD = "model_name_or_path"
UNSPECIFIED_TASK = "unspecified-task"

_OUTPUT_FILE_LOCK = threading.Lock()

class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()

def get_extra_docker_image_name(instance: dict, default_docker_image_name: str) -> str:
    """Get the image name for an instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        image_name = default_docker_image_name.lower()
    return image_name


def get_extra_environment(config: dict, instance: dict, default_docker_image_name: str) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_extra_docker_image_name(instance, default_docker_image_name)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str, dataset_config: DatasetConfig):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            dataset_config.id_field_name: instance_id,
            RESULT_FIELD: result,
            MODEL_NAME_FIELD: model_name,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))

def instance_index_str(instance_id: str, lm_eval_task: any) -> str:
    return str(instance_id)


def process_instance(
    instance: dict,
    lm_eval_task: any,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    dataset_config: DatasetConfig,
    default_docker_image_name: str,
    output_file_name: str,
) -> None:
    """Process a single instance."""
    instance_id = dataset_config.id_field(instance)
    index_str = instance_index_str(instance_id, lm_eval_task=lm_eval_task)
    instance_dir = output_dir / index_str
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / output_file_name, index_str)
    (instance_dir / f"{index_str}{TRAJECTORY_SUFFIX}").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = dataset_config.mk_task(instance=instance, lm_eval_task=lm_eval_task)

    progress_manager.on_instance_start(index_str)
    progress_manager.update_instance_status(index_str, "Pulling/starting docker")

    agent = None
    extra_info = None

    try:
        env = get_extra_environment(config, instance, default_docker_image_name)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=index_str,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task)
    except Exception as e:
        logger.error(f"Error processing instance {index_str}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            instance_dir / f"{instance_id}{TRAJECTORY_SUFFIX}",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=index_str,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / output_file_name, index_str, model.config.model_name, result, dataset_config)
        progress_manager.on_instance_end(index_str, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False, instance_select_file: str = "", dataset_config: DatasetConfig
) -> list[dict]:
    """Filter and slice a list of instances."""

    # reset instances to be from file ids if provided
    if instance_select_file:
        with open(instance_select_file, "r") as f:
            instance_select_file = [line.strip() for line in f.readlines()]
        instances_selected = {dataset_config.id_field(instance): instance for instance in instances if dataset_config.id_field(instance) in instance_select_file}
        # reorder instances to be in the order of the instance select file
        instances = [instances_selected[instance_id] for instance_id in instance_select_file]

    # the before instances
    before_filter = len(instances)

    # slice the instances if slice spec present
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]

    # shuffle the instances if asked
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: dataset_config.id_field(x))
        random.seed(42)
        random.shuffle(instances)

    # if a filter spec is provided, filter the instances
    print(f"Filtering instances with filter spec: {filter_spec}")
    print(f"Instances: {str(dataset_config.id_field(instances[0]))}")
    instances = [instance for instance in instances if re.match(filter_spec, str(dataset_config.id_field(instance)))]

    # report the number of instances after filtering
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
        ids = lambda instances: [dataset_config.id_field(instance) for instance in instances]
    return instances


from jinja2 import Environment, DebugUndefined, meta
class PreserveUndefined(DebugUndefined):
    undefined_vars = set()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PreserveUndefined.undefined_vars.add(self._undefined_name)

    def __str__(self):
        return f"{{{{{self._undefined_name}}}}}"
    __repr__ = __str__

def partial_substitute(template_string: str, partial_vars: dict) -> str:
    # report the variables that will not be substituted
    env = Environment()
    ast = env.parse(template_string)
    all_variables = meta.find_undeclared_variables(ast)
    unsubstituted = all_variables - set(partial_vars.keys())
    logger.info(f"Partial substitution that will leave these variables UNSUBSTITUTED: {unsubstituted}")

    env = Environment(undefined=PreserveUndefined)
    template = env.from_string(template_string)
    return template.render(**partial_vars)

def build_config_from_specs(agent_config: dict, capability_config: dict) -> dict:
    """Build the config from the agent and capability configs."""
    config = agent_config.copy()

    # for each key in the capability config:
    # if key present in agent config:
    #    - treat the value in the agent as template and replace with variables in the capability config
    # if key is NOT present in agent config, add it to the config
    for key, value in capability_config.items():
        if key in config:
            if isinstance(config[key], str):
                logger.info(f"Partial substituting `{key}` in agent config")
                # replace the values in agent from capability config, IF the value is a template string
                config[key] = partial_substitute(config[key], value)
            else:
                # if the value is not a template string, then recurse
                config[key] = build_config_from_specs(config[key], value)
        else:
            # if the key is not present in the agent config, add it to the config
            config[key] = value

    return config

def build_config_from_specs_single(agent_spec: Path, capability_spec: Path) -> dict:
    # load the agent and capability configs
    agent_config_path = get_config_path(agent_spec)
    logger.info(f"Loading agent config from '{agent_config_path}'")
    agent_config = yaml.safe_load(agent_config_path.read_text())
    capability_config_path = get_config_path(capability_spec)
    logger.info(f"Loading capability config from '{capability_config_path}'")
    capability_config = yaml.safe_load(capability_config_path.read_text())
    # replace the values in the agent config with the values in the capability config
    config = build_config_from_specs(agent_config, capability_config)

    return config

def run_agent(instances: list[dict],
              lm_eval_task: any,
              output_path: Path,
              agent_spec: Path,
              capability_spec: Path,
              dataset_config: DatasetConfig,
              default_docker_image_name: str,
              output_file_name: str,
              workers: int,
              total_time_for_all_instances: int | None = None,
              environment_class: str | None = None,
              model: str | None = None,
              model_class: str | None = None):
    """Run the agent on a list of instances."""
    logger.info(f"Running agent on {len(instances)} instances...")
    config = build_config_from_specs_single(agent_spec, capability_spec)

    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def get_resuls_from_done_futures(done: list[concurrent.futures.Future],
                                     futures: dict[concurrent.futures.Future, str],
                                     progress_manager: RunBatchProgressManager):
        for future in done:
            instance_id = futures[future]
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)
            del futures[future]
        return futures


    def process_futures(futures: dict[concurrent.futures.Future, str], start_time: float, total_timeout: int | None):
        while futures:
            # Check if total timeout has been exceeded
            if total_timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= total_timeout:
                    logger.warning(f"Total timeout of {total_timeout}s exceeded. Cancelling {len(futures)} pending futures...")
                    for future, instance_id in list(futures.items()):
                        if not future.done():
                            future.cancel()
                            progress_manager.on_uncaught_exception(instance_id, TimeoutError(f"Total timeout of {total_timeout}s exceeded"))
                    # Process any remaining completed futures before exiting
                    get_resuls_from_done_futures(list(futures.keys()), futures, progress_manager)
                    break
            
            # Use wait with a short timeout to periodically check for completion and total timeout
            done, not_done = concurrent.futures.wait(
                list(futures.keys()),
                timeout=0.5,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Process completed futures
            get_resuls_from_done_futures(done, futures, progress_manager)

    with Live(progress_manager.render_group, refresh_per_second=4):
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance,
                                instance=instance,
                                lm_eval_task=lm_eval_task,
                                output_dir=output_path,
                                config=config,
                                progress_manager=progress_manager,
                                dataset_config=dataset_config,
                                default_docker_image_name=default_docker_image_name,
                                output_file_name=output_file_name,
                ): dataset_config.id_field(instance)
                for instance in instances
            }
            try:
                process_futures(futures, start_time, total_time_for_all_instances)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures, start_time, total_time_for_all_instances)

    return config


def get_next_run_id(gcs_path_prefix: str, upload_name: str, run_id: Optional[int]):
    pattern = "[0-9]*" if not run_id else f"{run_id}"

    # get the latest run id by checking if the file already exists in the GCS bucket
    location_glob = f"{gcs_path_prefix}{pattern}/{upload_name}"
    cmd = ["gsutil", "ls", location_glob]
    ls_output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    exists = ls_output.returncode == 0

    if not exists:
        run_id = 0 if run_id is None else run_id
    else:
        try:
            # get the latest run id by splitting the output of ls, split by lines, remove gcs_path_prefix from the line, split by "-run", and get the last element
            lines = ls_output.stdout.decode("utf-8").splitlines()
            # only lines with the prefix
            lines = [line[len(gcs_path_prefix):] for line in lines if line.startswith(gcs_path_prefix) and "/" in line]
            # lines now are of the form NUMBER/subdir/FILENAME.json, get the number from the start of the line
            run_ids = [int(line.split("/")[0]) for line in lines]
            run_id = max(run_ids) + 1
        except ValueError as e:
            print(f"Error getting latest run id from ls output:")
            print(f"ls output:")
            print("-"*100)
            print(ls_output.stdout.decode('utf-8'))
            print("-"*100)
            print(lines)
            print(f"Error: {e}")
            print("-"*100)
            print(f"Using arbitrarily high run id {HIGH_ID}")
            input("Press Enter to continue...")
            run_id = HIGH_ID

    return run_id

DEFAULT_GCS_BUCKET = os.getenv("DEFAULT_GCS_BUCKET")
DEFAULT_GCS_MINI_EXTRA_DIR = f"code_data/synthetic-mini-swe-agent/mini-extra"
DEFAULT_GCS_UPLOAD_DIR = f"{DEFAULT_GCS_MINI_EXTRA_DIR}/runs"

HIGH_ID = 89898989

def upload_to_gcs(local_dir: Path,
                  upload_dir_prefix: str,
                  gcs_upload_toplevel_dir: str = DEFAULT_GCS_MINI_EXTRA_DIR,
                  upload_dir: str = DEFAULT_GCS_UPLOAD_DIR,
                  overwrite: bool = False,
                  run_id: Optional[int] = None):
    # Upload source_data to GCS; upload_dir is the directory in the GCS bucket to upload to
    # upload_prefix is the prefix of the file to upload
    # overwrite is whether to overwrite the file if it already exists
    # run_id is the run id to use

    if DEFAULT_GCS_BUCKET is None:
        # print that no gcs bucket is set, and return without uploading
        print("No GCS bucket is set in the environment variables. Skipping upload to GCS.")
        print("To upload to GCS, set the DEFAULT_GCS_BUCKET environment variable.")
        print("Example: export DEFAULT_GCS_BUCKET=your-gcs-bucket-name")
        return None

    # get the next run id
    gcs_path_prefix = f"gs://{DEFAULT_GCS_BUCKET}/{gcs_upload_toplevel_dir}/{upload_dir_prefix}-run"
    next_run_id = get_next_run_id(gcs_path_prefix, local_dir.name, run_id)

    # if next_run_id != run_id and overwrite asked, force use the provided run_id
    if next_run_id != run_id and overwrite and run_id is not None:
        next_run_id = run_id

    # upload the file to the GCS bucket
    gcs_path = f"{gcs_path_prefix}{next_run_id}/{local_dir.name}"
    cmd = ["gsutil", "-m", "cp", "-r", local_dir, gcs_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Uploaded {local_dir} to {gcs_path} for run {run_id}")
    return gcs_path

def clean_output_path(output_path: Path, redo_existing: bool = False):
    """Clean the output path."""
    if output_path.exists():
        if not redo_existing:
            raise ValueError(f"Output path {output_path} already exists. Use --redo-existing to redo existing instances.")
        logger.info(f"Cleaning output path {output_path}")
        shutil.rmtree(output_path)
    logger.info(f"Results will be saved to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
