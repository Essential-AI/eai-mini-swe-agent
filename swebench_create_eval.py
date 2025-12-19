import argparse
import sys
import os
import subprocess
import json
from datetime import datetime

# if the following import fails report to user to install the SWE-bench repo in the same environment
# https://github.com/SWE-bench/SWE-bench
try:
    from swebench.harness.run_evaluation import main as swebench_eval_main
except ImportError:
    print("Error: SWE-bench repo not found in the same environment")
    print("Please install the SWE-bench repo in the same environment and try again")
    print("clone the repo with: git clone https://github.com/SWE-bench/SWE-bench")
    print("and install with: pip install -e ./SWE-bench")
    sys.exit(1)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-bench",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "multimodal": "princeton-nlp/SWE-bench_Multimodal",
    "multilingual": "swe-bench/SWE-bench_Multilingual",
}

DEFAULT_MODEL = ""
DEFAULT_MODEL_CLASS = None
DEFAULT_WORKERS = 50
MAX_WORKERS_FOR_EVAL = 25

import argparse
import subprocess
import sys
import os
import yaml
from pathlib import Path
from typing import Optional

import multiprocessing
VCPUs = multiprocessing.cpu_count()
DEFAULT_WORKERS = min(VCPUs * 0.75, 50)

DEFAULT_BATCH_SZ = 500

# get the gcs bucket from the environment variable GCS_BUCKET
DEFAULT_GCS_BUCKET = os.environ.get("GCS_BUCKET")
DEFAULT_GCS_SWEBENCH_EVAL_DIR = f"code_data/mini-swebench-eval"
DEFAULT_GCS_UPLOAD_DIR = f"{DEFAULT_GCS_SWEBENCH_EVAL_DIR}/runs"

if DEFAULT_GCS_BUCKET is None:
    print("WARNING: GCS_BUCKET environment variable is not set")
    confirmation = input("Will skip upload to GCS. Continue? (y/n)")
    if confirmation != "y":
        print("Skipping upload to GCS")

HIGH_ID = 89898989

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

def upload_to_gcs(gcs_upload_dir: str, upload_dir: str, upload_name: str, source_data: str, overwrite: bool, run_id: Optional[int], exit_on_error: bool = False):
    # Upload source_data to GCS; upload_dir is the directory in the GCS bucket to upload to
    # upload_name is the name of the file to upload
    # source_data is the path to the file to upload

    if DEFAULT_GCS_BUCKET is None:
        print("WARNING: GCS_BUCKET environment variable is not set")
        print("Skipping upload to GCS")
        return

    # get the next run id
    gcs_path_prefix = f"gs://{DEFAULT_GCS_BUCKET}/{gcs_upload_dir}/{upload_dir}-run"
    next_run_id = get_next_run_id(gcs_path_prefix, upload_name, run_id)

    # if next_run_id != run_id and overwrite asked, force use the provided run_id
    if next_run_id != run_id and overwrite:
        next_run_id = run_id

    # upload the file to the GCS bucket
    gcs_path = f"{gcs_path_prefix}{next_run_id}/{upload_name}"
    cmd = ["gsutil", "cp", source_data, gcs_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=exit_on_error)
    print(f"Uploaded {upload_name} to {gcs_path} for run {run_id}")
    return gcs_path

def generate_mini_extra_cmd_common(model, model_class, output, subset, split, capability, capability_spec, workers, redo_existing):
    swebench_cmd = [
        "mini-extra",
        capability,
        "--workers", str(workers),
        "--subset", subset,
        "--split", split,
        "--output", output,
        "--capability-spec", capability_spec,
    ]
    if model is not None:
        swebench_cmd.append("--model")
        swebench_cmd.append(model)
    if model_class is not None:
        swebench_cmd.append("--model-class")
        swebench_cmd.append(model_class)
    if redo_existing:
        swebench_cmd.append("--redo-existing")
    
    return swebench_cmd

def generate_mini_extra_cmd(model, model_class, file_ids, start, end, output, subset, split, capability, capability_spec, workers, redo_existing):
    mini_extra_cmd = generate_mini_extra_cmd_common(model=model,
        model_class=model_class,
        output=output,
        subset=subset,
        split=split,
        capability=capability,
        capability_spec=capability_spec,
        workers=workers,
        redo_existing=redo_existing)
    if start is None and end is None:
        slice = None
    else:
        if start is None:
            start = ""
        if end is None:
            end = ""
        slice = f"{start}:{end}"

    if slice is not None:
        mini_extra_cmd.extend([
            "--slice", slice,
        ])
    if file_ids is not None:
        mini_extra_cmd.extend([
            "--instance-select-file", file_ids,
        ])
    return mini_extra_cmd


def exec_generation(mini_extra_cmd, output, run_docker_clean):
    # create output directory if it doesn't exist
    Path(output).mkdir(parents=True, exist_ok=True)

    # clean docker containers
    if run_docker_clean:
        clean_docker_containers()
    
    # the api_keys.sh file in the same directory contains the api keys for the models
    # before running the script, source the api_keys.sh file in each shell below
    env = os.environ.copy()

    if os.path.exists("api_keys.sh"):
        with open("api_keys.sh", "r") as f:
            for line in f:
                if line.startswith("export"):
                    without_export = line.split("export")[1].strip()
                    k, v = without_export.split("=")
                    # strip the value of quotes
                    assert v.startswith("\"") and v.endswith("\"")
                    v = v.strip()[1:-1]
                    env[k] = v

    try:
        result = subprocess.run(mini_extra_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running mini-extra: {e}", file=sys.stderr)
        sys.exit(1)

def get_confirmation_for_enough_completions(output):
    # read the {output}/exit_statuses_{ID1}.{ID2}.yaml file
    # first find the file with the name exit_statuses_{ID1}.{ID2}.yaml
    # count the number of instances in each category
    instances_by_exit_status = {}
    total_instances = 0
    for file in os.listdir(output):
        if file.startswith("exit_statuses_") and file.endswith(".yaml"):
            with open(os.path.join(output, file), "r") as f:
                data = yaml.safe_load(f)
                if data is not None:
                    for exit_status, instances in data["instances_by_exit_status"].items():
                        instances_by_exit_status[exit_status] = len(instances)
                        total_instances += len(instances)
                else:
                    print(f"No data in {file}")
    return instances_by_exit_status, total_instances

def clean_docker_containers():
    # run docker kill $(docker ps -q) && docker rm $(docker ps -aq)
    # and docker system prune -f
    # and sudo systemctl restart docker
    print("Cleaning docker containers")
    print("Running: docker kill $(docker ps -q)")
    subprocess.run(["docker kill $(docker ps -q)"], shell=True)
    print("Running: docker rm $(docker ps -aq)")
    subprocess.run(["docker rm $(docker ps -aq)"], shell=True)
    print("Running: docker system prune -af")
    subprocess.run(["docker system prune -af"], shell=True)
    print("Running: sudo systemctl restart docker")
    subprocess.run(["sudo systemctl restart docker"], shell=True)
    print("Docker containers cleaned")
    # print the docker ps output
    print(subprocess.run(["docker ps"], shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr))
    print("Docker ps output above; should be empty")

def run_swebench_eval(start, end, output, workers, run_docker_clean, variant, split):
    # python -m swebench.harness.run_evaluation \
	# --dataset_name princeton-nlp/SWE-bench_Verified \
	# --predictions_path ${MODEL_CONFIG}-swebench-eval/preds.json \
	# --run_id ${MODEL_CONFIG}-swebench-eval \
	# --max_workers 20
    instances_by_exit_status, total_instances = get_confirmation_for_enough_completions(output)
    percent_submitted = float(instances_by_exit_status["Submitted"]) / total_instances if "Submitted" in instances_by_exit_status else 0.0
    if percent_submitted < 0.9:
        print("="*100)
        print(f"!!!!WARNING!!!!")
        print(f"Submission rate is too low. {percent_submitted*100}% submitted, {instances_by_exit_status}")
        print("="*100)

    # clean docker containers
    if run_docker_clean:
        clean_docker_containers()

    try:
        # force max workers to 25 for evals even if generation max_workers is higher
        # we get failed docker starts with more than 25 workers; specifically the following error:
        # UnixHTTPConnectionPool(host='localhost', port=None): Read timed out. (read timeout=60)
        workers = MAX_WORKERS_FOR_EVAL if workers > MAX_WORKERS_FOR_EVAL else workers

        print(f"[SWEBENCH EVAL] Running evaluation on {output}/preds.json")
        result = swebench_eval_main(
            dataset_name=DATASET_MAPPING[variant],
            predictions_path=f"{output}/preds.json",
            run_id=f"{output}-eval",
            max_workers=workers,
            split=split,
            instance_ids=[],
            force_rebuild=False,
            namespace="swebench",
            timeout=1800, # seconds for each instance
            rewrite_reports=False, # comment in SWEBENCH help: "Doesn't run new instances, only writes reports for instances with existing test outputs"
            modal=False, # run on Modal
            cache_level="env", # none/base/env/instance -- remove images above this level
            clean=False, # clean images above cache level
            open_file_limit=4096, # open file limit
        )
        print(f"[SWEBENCH EVAL] Evaluation result: {result}")
    except Exception as e:
        print(f"[SWEBENCH EVAL] Error running evaluation: {e}", file=sys.stderr)
        sys.exit(1)

    return result

def run_swebench_generations(start, end, model, model_class, output, workers, run_docker_clean, variant, split):
    # MODEL_CONFIG="gpt5-mini-medium"
    # mini-extra swebench --config swebench-${MODEL_CONFIG}.yaml \
    # --split test \
    # --subset verified \
    # --output ${MODEL_CONFIG}-swebench-eval \
    # --workers 20

    if variant == "multilingual":
        config = f"swe-multilingual-spec.yaml"
    else:
        config = f"swebench-spec.yaml"

    print(f"[SWEBENCH EVAL] Running batch from {start} to {end} with output directory: {output}")
    swebench_cmd = generate_mini_extra_cmd(
            model=model,
            model_class=model_class,
            file_ids=None,
            start=start,
            end=end,
            output=output,
            subset=variant,
            split=split,
            capability="swebench",
            capability_spec=config,
            workers=workers,
            redo_existing=True)
    
    # Run mini-extra swebench; appending console output to logfile
    exec_generation(swebench_cmd, output, run_docker_clean)

    print(f"[SWEBENCH EVAL] Done!")
    if run_docker_clean:
        clean_docker_containers()
        print(f"[SWEBENCH EVAL] Docker containers cleaned")

def make_jsonl_from_output_dir(output_dir):
    # output_dir has structure: trajectory_id/trajectory_id.traj.json
    # read all the json files in the output_dir and return a jsonl string
    jsonl_list = []
    for trajectory_id in os.listdir(output_dir):
        # trajectory_id is a directory
        if not os.path.isdir(os.path.join(output_dir, trajectory_id)):
            continue
        traj_json = os.path.join(output_dir, trajectory_id, f"{trajectory_id}.traj.json")
        # traj_json should be a valid file
        if not os.path.isfile(traj_json):
            continue
        with open(traj_json, "r") as f:
            traj_data = json.load(f)
            jsonl_list.append(json.dumps(traj_data))
    # write the jsonl list to a temporary file with name suffix trajectories_<num_trajectories>.jsonl
    temp_file = f"trajectories_{len(jsonl_list)}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jsonl"
    with open(temp_file, "w") as f:
        for traj_data in jsonl_list:
            f.write(traj_data + "\n")
    return temp_file, len(jsonl_list)

def upload_results_and_trajectories(results_json, output, gcs_upload_dir, overwrite_previous_gcs_files, run_id):
    gcs_outdir = output
    # upload the eval results
    upload_to_gcs(gcs_upload_dir=gcs_upload_dir,
                    upload_dir=gcs_outdir,
                    upload_name="results.json",
                    source_data=results_json,
                    overwrite=overwrite_previous_gcs_files,
                    run_id=run_id)
    # upload the output trajectories
    output_jsonl, num_trajectories = make_jsonl_from_output_dir(output)
    upload_to_gcs(gcs_upload_dir=gcs_upload_dir,
                    upload_dir=gcs_outdir,
                    upload_name=f"trajectories_{num_trajectories}.jsonl",
                    source_data=output_jsonl,
                    overwrite=overwrite_previous_gcs_files,
                    run_id=run_id)
    # upload the output/{preds.json, minisweagent.log, exit_statuses_*.*.yaml}
    for file in os.listdir(output):
        if file.endswith(".json") or file.endswith(".log") or file.endswith(".yaml"):
            src = os.path.join(output, file)
            upload_to_gcs(gcs_upload_dir=gcs_upload_dir,
                            upload_dir=gcs_outdir,
                            upload_name=file,
                            source_data=src,
                            overwrite=overwrite_previous_gcs_files,
                            run_id=run_id)


def run_swebench_eval_pipeline(start, end, output, model, model_class, workers, gcs_upload_dir,
                               run_docker_clean, skip_generations, skip_upload_to_gcs, overwrite_previous_gcs_files, run_id, variant, split):
    # Set default output directory if not provided
    if output is None:
        output = f"swebench-eval-{start}-{end}"
    
    if not run_docker_clean:
        print("You asked to clean docker containers. This will remove all docker containers and images after each step. Continue?")
        confirmation = input("(y/n)")
        if confirmation != "y":
            sys.exit(1)

    if not skip_generations:
        run_swebench_generations(
                start=start,
                end=end,
                model=model,
                model_class=model_class,
                output=output,
                workers=workers,
                run_docker_clean=run_docker_clean,
                variant=variant,
                split=split)

    results_json = run_swebench_eval(
            start=start,
            end=end,
            output=output,
            workers=workers,
            run_docker_clean=run_docker_clean,
            variant=variant,
            split=split)

    if not skip_upload_to_gcs:
        upload_results_and_trajectories(results_json=results_json,
                                        output=output,
                                        gcs_upload_dir=gcs_upload_dir,
                                        overwrite_previous_gcs_files=overwrite_previous_gcs_files,
                                        run_id=run_id)

def run_swebench_eval_pipeline_in_batches(batch_size, start, end, output, model, model_class, workers, gcs_upload_dir,
                                          run_docker_clean, skip_generations, skip_upload_to_gcs, overwrite_previous_gcs_files, run_id, variant, split):
    run_pipeline = lambda batch_start, batch_end: run_swebench_eval_pipeline(start=batch_start, end=batch_end,
                                   output=f"{output}-batch{batch_start}-{batch_end}", model=model,
                                   model_class=model_class,
                                   workers=workers,
                                   gcs_upload_dir=gcs_upload_dir,
                                   run_docker_clean=run_docker_clean,
                                   skip_generations=skip_generations,
                                   skip_upload_to_gcs=skip_upload_to_gcs,
                                   overwrite_previous_gcs_files=overwrite_previous_gcs_files,
                                   run_id=f"{run_id}-batch{batch_start}-{batch_end}",
                                   variant=variant,
                                   split=split)
    start = start if start is not None else 0
    if batch_size is None:
        run_pipeline(start, end)
        return

    # batch size is provided
    if end is None:
        print(f"Error: Need to specify end index if batch size is provided. Exiting...")
        sys.exit(1)

    # batch size is provided and end is provided
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        print(f"Running batch from {batch_start} to {batch_end}")
        print(f"Output directory: {output}-batch{batch_start}-{batch_end}")
        print(f"Running batch {batch_start}-{batch_end}")
        run_pipeline(batch_start, batch_end)
        print(f"Batch {batch_start}-{batch_end} done")

def main():
    parser = argparse.ArgumentParser(
        description="Create trajectories for SWE-BENCH using mini-swe-agent and evaluate them"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default=DEFAULT_MODEL_CLASS,
        help=f"Model class to use (default: {DEFAULT_MODEL_CLASS})",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for slice (default: None)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help=f"End index for slice (default: None)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: swebench-eval-{start}:{end}-{model})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of workers by (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--skip-docker-clean",
        action="store_true",
        help="Skip cleaning docker containers",
    )
    parser.add_argument(
        "--skip-generations",
        action="store_true",
        help="Skip generations",
    )
    parser.add_argument(
        "--gcs-upload-dir",
        type=str,
        default=DEFAULT_GCS_UPLOAD_DIR,
        help="GCS upload directory",
    )
    parser.add_argument(
        "--skip-upload-to-gcs",
        action="store_true",
        help="Skip uploading to GCS",
    )
    parser.add_argument(
        "--overwrite-gcs",
        action="store_true",
        help="Overwrite previous GCS files",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=999,
        help="Run ID; if provided uploads will be suffixed with -run{run_id}; otherwise it will be -run{max_existing_run_id+1}",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="verified",
        help="Variant to use (default: verified); options: full, verified, lite, multimodal, multilingual",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to use (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: None)",
    )
    args = parser.parse_args()

    run_swebench_eval_pipeline_in_batches(args.batch_size, args.start, args.end,
                            args.output, args.model, args.model_class,
                            args.workers, args.gcs_upload_dir,
                            not args.skip_docker_clean,
                            args.skip_generations,
                            args.skip_upload_to_gcs,
                            args.overwrite_gcs,
                            args.run_id,
                            args.variant,
                            args.split)

if __name__ == "__main__":
    main()
