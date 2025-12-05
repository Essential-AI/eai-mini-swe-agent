from enam import *
import hashlib
import csv
import json
import os

def instance_id_to_name(instance_id: int) -> str:
    return f'HumanEval/{instance_id}'

def is_valid_enamel_name(name: str) -> bool:
    return name.startswith('HumanEval/')

def name_to_instance_id(name: str) -> int:
    assert is_valid_enamel_name(name), f'Invalid name: {name}'
    return int(name.split('/')[-1])

def eval_enamel(completions: dict[str, list[str]], print_individual_metrics: bool = False) -> dict:
    """Evaluate the performance of the model on the Enamel dataset."""
    id_dict = {}
    for name, completions in completions.items():
        if not is_valid_enamel_name(name):
            print(f'Enamel dataset only evals HumanEval/k instances. Skipping {name}...')
            continue
        if not isinstance(completions, list):
            print(f'Need completions list, even if singleton. Skipping {name} because it is not a list...')
            continue
        id_dict[name_to_instance_id(name)] = completions
    name = 'mini-perf-agent'
    return eval_completions(name, id_dict, print_individual_metrics)

DEFAULT_SEED = 998244353
DEFAULT_MEMORY_GIGA = 4.
DEFAULT_TIMEOUT_FACTOR = 2.
DEFAULT_TOLERENCE_SEC = 0.01
DEFAULT_EVAL_K = [1, 2, 4, 8, 10, 100]
DEFAULT_VERBOSE = False
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), 'dataset/enamel.csv')
DEFAULT_SUBSET = PROBLEMSET
DEFAULT_TESTS = None
DEFAULT_SAVE_NAME = 'cache/eval'
DEFAULT_N_TESTS = [8, 4, 4, 4]
DEFAULT_N_REPS = 6
DEFAULT_HARDNESS = [0., 3., 3., 4.]

def replace_unkillable_loops(code: str): # loads the zipped code samples from EvalPlus
    # TODO: problematic fragments are those that are infinite loops inside the try block.
    # TODO: replace them with a loop that can be killed.
    KILLABLE_LOOP_CODE = 'while True:\n    pass\n'
    # TODO: Identify such code fragments. For now, we don't have a quick way to do this.
    return code

def get_enamel_dataset_instance_ids() -> list[int]:
    # load DEFAULT_DATASET, and get the "task_id" column as a list
    with open(DEFAULT_DATASET, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip the header
        for row in reader:
            name = row[0]
            yield name_to_instance_id(name)

def create_ordered_list(labelled_completions: dict[int, list[str]], enamel_dataset_instance_ids: list[int]):
    # input is { k: [completion1, completion2, ...] }
    # output should be { enamel_ordering[instance_id]: [completion1, completion2, ...] }
    enamel_ordering = lambda k: enamel_dataset_instance_ids.index(k)
    fixed_codes = lambda completions: [replace_unkillable_loops(completion) for completion in completions]
    return { enamel_ordering(k): fixed_codes(completions) for k, completions in labelled_completions.items() }

def get_subset_instance_ids(labelled_completions: dict[int, list[str]], subset: list[int]):
    narrowed = [k for k in labelled_completions.keys() if k in subset]
    narrowed.sort()
    return narrowed

def get_pass_eff_metrics(save_name: str) -> dict:
    # save_name~}{passes,effs}.json 
    # effs: is like { "74": [[0.0, 0.4676602086438152, 0.0, 0.0]] ... }
    # passes: is like { "74": [true], ...}
    with open(f'{save_name}~passes.json', 'r') as f:
        passes = json.load(f)
    with open(f'{save_name}~effs.json', 'r') as f:
        effs = json.load(f)
    return {k: (passes[k][0], effs[k][0]) for k in passes.keys()}

def eval_completions(name: str, labelled_completions: dict[int, list[str]], print_individual_metrics: bool):
    print(f'Evaluating {name} with {len(labelled_completions)} completions', flush = True)
    subset_instance_ids = get_subset_instance_ids(labelled_completions, DEFAULT_SUBSET)
    evaluator = Evaluator(
        problems = DEFAULT_DATASET,
        subset = subset_instance_ids,
        n_tests = DEFAULT_N_TESTS,
        n_reps = DEFAULT_N_REPS,
        hardness = DEFAULT_HARDNESS,
        memory_giga = DEFAULT_MEMORY_GIGA,
        timeout_factor = DEFAULT_TIMEOUT_FACTOR,
        tolerence_sec = DEFAULT_TOLERENCE_SEC,
        seed = DEFAULT_SEED,
    )
    sorted_ids = sorted(subset_instance_ids)
    hash_from_ids = hashlib.sha256(str(sorted_ids).encode()).hexdigest()[:8]
    default_loc = f'cache/eval~{hash_from_ids}'
    default_tests = f'{default_loc}~tests.pkl'
    if not evaluator.load_tests(fname = default_tests):
        print(f'[Notice] tests not found at {default_tests}. Creating tests now.', flush = True)
        os.makedirs(os.path.dirname(default_tests), exist_ok = True)
        evaluator.make_tests(save_name = default_loc)
    print(f'Evaluating {name} with {len(labelled_completions)} completions.', flush = True)
    enamel_dataset_instance_ids: list[int] = list(get_enamel_dataset_instance_ids())
    codes = create_ordered_list(labelled_completions, enamel_dataset_instance_ids)
    save_name = f'{DEFAULT_SAVE_NAME}~{name}'
    print(f'Saving results to {save_name}', flush = True)
    metrics = evaluator.evaluate(codes, k = DEFAULT_EVAL_K, save_name = save_name, verbose = DEFAULT_VERBOSE)
    if print_individual_metrics:
        individual_metrics = get_pass_eff_metrics(save_name)
        for k, v in individual_metrics.items():
            pass_symbol = "✅" if v[0] else '❌'
            formatted_eff = [f'{x:>4.2f}' for x in v[1]]
            print(f'{k:>3}: {pass_symbol} -> {", ".join(formatted_eff)}')
    float_metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    print(f'{name} -> {float_metrics}')
    return float_metrics
