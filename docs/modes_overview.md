# Modes

Mini-SWE-agent supports **modes**: self-contained task configurations that specialize the agent for a particular kind of workload (e.g., math problem solving, performance optimization).

A **mode** bundles together:

- Task-specific agent behavior (reasoning loop, tools)
- Datasets and instance loading logic
- Evaluation and metrics
- Default Docker image / environment
- Default capability spec (`*-spec.yaml`)

Examples of modes are `math` and `perf`, and the system can be extended with additional modes.

---

## Quick start
The quickest way to run a model in an environment mode is to use a Together.AI endpoint. Export your Together access key as the `TOGETHERAI_API_KEY` environment variable; install the repo using `pip install -e .`, and then run:
```
mini-extra perf-single [--instance 10]
mini-extra swebench-single [--instance 10]
```

---

## Available modes

| Mode     | Purpose                                      | Single-instance command        | Batch command              | Default capability spec     |
|----------|----------------------------------------------|--------------------------------|----------------------------|-----------------------------|
| swebench | Solve software engineering tasks             | `mini-extra swebench-single`   | `mini-extra swebench`      | `extra/swebench.yaml`       |
| math     | Solve and evaluate math problems             | `mini-extra math-single`       | `mini-extra math`          | `extra/math-spec.yaml`      |
| perf     | Optimize code for runtime performance        | `mini-extra perf-single`       | `mini-extra perf`          | `extra/perf-spec.yaml`      |
| yolo     | Open-ended problem solving (user-defined)    | `mini-extra yolo-single`       | N/A                        | `extra/yolo-spec.yaml`   |

Each mode defines its own datasets, evaluation logic, and environment. For example:

- **swebench mode**
  - Focus: resolving real-world GitHub issues in open-source repositories
  - Datasets: `lite`, `verified`, `test` (SWE-bench splits)
  - Evaluation: patch correctness validated against test suites
  - Outputs: `all_preds.jsonl`, per-instance `*.traj.json`, `minisweagent.log`
  - Environment: Docker containers with repository-specific environments
  - See [detailed SWE-bench documentation](usage/swebench.md) for more information

- **math mode**
  - Focus: mathematical reasoning and solution verification
  - Datasets: `math500`, `aime`, `gsm8k`, `fn_math`, etc.
  - Evaluation: LaTeX answer extraction + mathematical equivalence checking
  - Outputs: `math-preds.json`, `math-results.json`, per-instance `*.traj.json`, `minisweagent-math.log`

- **perf mode**
  - Focus: algorithmic and runtime performance optimization
  - Datasets: `enamel`
  - Evaluation: functional correctness + ENAMEL efficiency metrics (`eff@k`)
  - Outputs: `perf-preds.json`, `perf-results.json`, per-instance `*.traj.json`, `minisweagent-perf.log`
  - Tools: Uses `kernprof` and `line_profiler` for performance analysis

- **yolo mode**
  - Focus: open-ended problem solving with agent-determined workflow
  - Outputs: trajectory file saved to global config directory
  - Usage: Agent decides its own strategy based on the problem statement

---

## Running a mode

All modes follow the same pattern:

- **Single instance**
  - `mini-extra <mode>-single`
  - Example: `mini-extra swebench-single --subset verified --instance 0`
  - Example: `mini-extra math-single --subset math500 --instance 0`
  - Example: `mini-extra perf-single --subset enamel --instance 0`

- **Batch**
  - `mini-extra <mode>`
  - Example: `mini-extra swebench --subset lite --split dev --workers 32`
  - Example: `mini-extra math --subset math500 --slice 0:10 --workers 32`
  - Example: `mini-extra perf --subset enamel --slice 0:20 --workers 32`

Common options across modes:

- `--subset`: dataset name / key defined by the mode
- `--slice`, `--filter`, `--instance-select-file`: control which instances to run
- `-m, --model`: model identifier
- `-a, --agent-spec`: agent configuration (shared across modes)
- `-c, --capability-spec`: mode-specific capability spec
- `-o, --output`: output directory
- `--skip-agent-run`: only evaluate previously generated predictions
- `--max-runtime`: global time cap for the batch
- `-w, --workers`: parallelism

Each mode can also expose extra flags specific to its evaluation:
- `swebench`: `--split`, `--environment-class`
- `math`: `--split`, `--shuffle`
- `perf`: `--print-individual-metrics`, `--environment-class`
- `yolo`: `--task` (the question/problem to solve)

---

## Extending with new modes

To add a new mode:

1. **Create a specific docker for that mode's custom environment**
   - If your mode needs specialized tools, create and push a new docker image.
     - E.g., the images `researchessential/python-3.12-slim-linux-env` is a linux container with python and numpy sympy pandas scipy line_profiler installed.

2. **Create a capability spec**
   - Add `extra/<mode>-spec.yaml` describing:
     - Tools, environment, and instructions to tell the model what to expect in that mode's environment.
     - Step and cost limits
     - Any mode-specific workflow constraints

3. **Add a mode runner**
   - Create a `<mode>_single.py` script similar to `math_single.py` or `perf_single.py`:
     - This allows running a single task in that mode
   - (Optional) Create a `<mode>.py` script similar to `math.py` or `perf.py`:
     - Load and filter instances
     - Call `run_agent(...)` with:
       - `agent_spec`
       - `capability_spec`
       - `dataset_config`
       - default Docker image
     - This batch processing can use a `DatasetConfig` for specifying:
       - How to load instances (HF dataset name / file path)
       - How to get instance IDs and task prompts
       - How to extract gold answers (if applicable)
       - How to evaluate model outputs

4. **Register the mode**
   - Register the runner in the `mini-extra` CLI, following the existing pattern:
     ```python
     ("minisweagent.run.extra.<mode>_single", [f"{mode}-single"], "Evaluate on <mode> (single instance)")
     ("minisweagent.run.extra.<mode>", [mode], "Evaluate on <mode> (batch mode)")
     ```

Once registered, the new mode behaves like `math` and `perf`: it can be run via `mini-extra <mode>` / `mini-extra <mode>-single`, with its own datasets, evaluation metrics, and capability spec.
