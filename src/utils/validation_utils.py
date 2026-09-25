import json
import os

import numpy as np
import torch

from src.parsedata import get_data, parse


def select_representative_files(file_names, sample_size):
    if sample_size <= 0:
        return []
    if sample_size >= len(file_names):
        return sorted(file_names)

    sorted_files = sorted(file_names)
    raw_indexes = np.linspace(0, len(sorted_files) - 1, num=sample_size)
    selected_indexes = []
    used = set()

    for idx in raw_indexes:
        rounded = int(round(float(idx)))
        if rounded not in used:
            used.add(rounded)
            selected_indexes.append(rounded)

    for idx in range(len(sorted_files)):
        if len(selected_indexes) >= sample_size:
            break
        if idx not in used:
            used.add(idx)
            selected_indexes.append(idx)

    selected_indexes.sort()
    return [sorted_files[i] for i in selected_indexes]


def extract_reference_makespan(solution_path):
    with open(solution_path, 'r') as infile:
        solution_data = json.load(infile)

    # indicators.best_objective_bound is CP-SAT's proven LOWER BOUND at the solver's
    # 15s time limit, not necessarily the objective of the stored schedule - for ~60% of
    # this val set, best_objective_bound is strictly below the stored final_schedule's
    # own makespan (see src/utils/check_validation_set.py), meaning optimality was never
    # proven and every gap computed against it was inflated by up to >100%, with no
    # policy able to reach gap=0 no matter how good. The stored final_schedule IS a real,
    # achieved, feasible solution, so its own makespan (max "end" over all operations) is
    # used as the reference instead: gap=0 means matching CP-SAT's own incumbent at that
    # time budget, and a negative gap is possible (the policy beating that incumbent).
    schedule = solution_data.get("final_schedule")
    if not schedule:
        raise ValueError(f"No final_schedule in solution file: {solution_path}")
    reference = max(entry["end"] for entry in schedule)
    if reference is None or float(reference) <= 0:
        raise ValueError(f"Invalid reference objective in solution file: {solution_path}")
    return float(reference)


def _load_entry(instance_file, instances_dir, solutions_dir, dbg_fn=None):
    instance_name = os.path.splitext(instance_file)[0]
    instance_path = os.path.join(instances_dir, instance_file)
    solution_path = os.path.join(solutions_dir, instance_name + '.json')

    if not os.path.isfile(solution_path):
        if dbg_fn is not None:
            dbg_fn(1, f"Skipping validation instance without matching solution: {instance_file}")
        return None

    with open(instance_path, 'r') as file:
        contents = file.read()
        jobs, operations, info, maximum = get_data(parse(contents))

    reference_makespan = extract_reference_makespan(solution_path)

    return {
        "name": instance_name,
        "score": reference_makespan,
        "jobs": jobs,
        "operations": operations,
        "maximum": maximum,
        "num_machines": info["machinesNb"],
    }


def _build_dataset_from_files(instance_files, instances_dir, solutions_dir, dbg_fn=None):
    dataset = []
    for instance_file in instance_files:
        entry = _load_entry(instance_file, instances_dir, solutions_dir, dbg_fn)
        if entry is not None:
            dataset.append(entry)
    return dataset


def generate_fixed_splits(instances_dir="val/instances", solutions_dir="val/solutions",
                           val_size=20, test_size=20,
                           val_path="val/validation_dataset.json", test_path="val/test_dataset.json",
                           dbg_fn=None):
    """
    Build ONE fixed, disjoint validation/test split from val/instances + val/solutions and
    persist both to disk as plain JSON (already-parsed jobs/operations + reference
    makespan, so no re-parsing or re-selection is needed on later loads).

    Why disjoint, persisted splits: the previous build_validation_dataset() re-selected a
    "representative" subset from val/instances on every call, sized by whatever
    validation_size a given run happened to pass in - two runs (or two hyperparameter
    trials, or a run today vs. one after val/instances grew) were not actually validating
    on the same instances, making their reported gaps not comparable. And train()'s
    checkpoint selection (`best_avg_gap`) and its final reported number were both computed
    from that SAME resampled set - i.e. the number a run reports at the end had already
    been used to pick which checkpoint to report, an optimistic selection bias. A fixed,
    checked-in validation split removes the first problem; a disjoint, held-out test split
    (only ever touched once, after training, by evaluate_on_test_set()) removes the second.

    Returns:
        (validation_set, test_set) - the same list-of-dict structure normally returned by
        build_validation_dataset(), now also durably written to val_path / test_path.
    """
    if not os.path.isdir(instances_dir):
        raise FileNotFoundError(f"Validation instances directory not found: {instances_dir}")
    if not os.path.isdir(solutions_dir):
        raise FileNotFoundError(f"Validation solutions directory not found: {solutions_dir}")

    # Only consider instances that actually have a matching solution file BEFORE picking
    # the representative subset, so val_size/test_size are met exactly instead of quietly
    # coming up short whenever an .fjs happens to lack a solution.
    all_instance_files = [
        f for f in os.listdir(instances_dir)
        if f.lower().endswith('.fjs')
        and os.path.isfile(os.path.join(solutions_dir, os.path.splitext(f)[0] + '.json'))
    ]
    total_needed = val_size + test_size
    if total_needed > len(all_instance_files):
        raise ValueError(
            f"Requested val_size+test_size={total_needed} but only {len(all_instance_files)} "
            f"instance file(s) with a matching solution are available in {instances_dir}"
        )

    selected = select_representative_files(all_instance_files, total_needed)
    # Interleave (even/odd indexes of the representative selection) instead of splitting
    # into two contiguous halves, so both splits span the same range of instance
    # sizes/difficulty - a contiguous split could e.g. give validation all the smaller
    # instances (sorted first) and test all the larger ones.
    val_files = selected[0::2][:val_size]
    test_files = selected[1::2][:test_size]

    validation_set = _build_dataset_from_files(val_files, instances_dir, solutions_dir, dbg_fn)
    test_set = _build_dataset_from_files(test_files, instances_dir, solutions_dir, dbg_fn)

    if len(validation_set) == 0:
        raise RuntimeError("No validation instances were loaded from val/instances + val/solutions")
    if len(test_set) == 0:
        raise RuntimeError("No test instances were loaded from val/instances + val/solutions")

    os.makedirs(os.path.dirname(val_path) or ".", exist_ok=True)
    with open(val_path, 'w') as f:
        json.dump(validation_set, f)
    os.makedirs(os.path.dirname(test_path) or ".", exist_ok=True)
    with open(test_path, 'w') as f:
        json.dump(test_set, f)

    if dbg_fn is not None:
        dbg_fn(1, f"Generated fixed splits | validation={len(validation_set)} -> {val_path} | "
                  f"test={len(test_set)} -> {test_path}")

    return validation_set, test_set


def load_fixed_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_validation_dataset(sample_size=20, instances_dir="val/instances", solutions_dir="val/solutions",
                              dbg_fn=None, val_path="val/validation_dataset.json",
                              test_path="val/test_dataset.json"):
    """
    Entry point used by train(): load the fixed validation split if val_path already
    exists on disk (guaranteeing every run validates on the exact same instances,
    regardless of the sample_size THIS run happens to pass in), otherwise generate it -
    together with its disjoint test split, see generate_fixed_splits() - once, and
    persist both for every future run to reuse.
    """
    if os.path.isfile(val_path):
        validation_set = load_fixed_dataset(val_path)
        if dbg_fn is not None:
            dbg_fn(1, f"Loaded fixed validation dataset from {val_path}: {len(validation_set)} instance(s)")
        if len(validation_set) != sample_size:
            print(f"[VALIDATION] Note: using the existing fixed validation dataset at {val_path} "
                  f"({len(validation_set)} instances) - the requested sample_size={sample_size} is "
                  f"ignored so every run stays comparable. Delete {val_path} (and its paired "
                  f"{test_path}) to regenerate both splits at a new size.")
        return validation_set

    validation_set, _ = generate_fixed_splits(
        instances_dir=instances_dir, solutions_dir=solutions_dir,
        val_size=sample_size, test_size=sample_size,
        val_path=val_path, test_path=test_path, dbg_fn=dbg_fn,
    )
    return validation_set


def get_test_dataset(sample_size=20, instances_dir="val/instances", solutions_dir="val/solutions",
                      val_path="val/validation_dataset.json", test_path="val/test_dataset.json",
                      dbg_fn=None):
    """
    Loads the held-out test split for a FINAL, one-time report after training ends (see
    train.py's evaluate_on_test_set() call). Never used for checkpoint selection or
    during-training tracking - that's what build_validation_dataset()'s set is for. If
    neither split exists yet, generates both (via build_validation_dataset(), which
    creates the pair) and returns the test one.
    """
    if os.path.isfile(test_path):
        test_set = load_fixed_dataset(test_path)
        if dbg_fn is not None:
            dbg_fn(1, f"Loaded fixed test dataset from {test_path}: {len(test_set)} instance(s)")
        return test_set

    build_validation_dataset(sample_size=sample_size, instances_dir=instances_dir,
                              solutions_dir=solutions_dir, dbg_fn=dbg_fn,
                              val_path=val_path, test_path=test_path)
    return load_fixed_dataset(test_path)


def run_validation(ppo_agent, val_env, validation_set, episode_number=None, dbg_fn=None, print_fn=print):
    all_val_results = []
    swaps, blocked = [], []
    with torch.no_grad():
        for i in range(len(validation_set)):
            v_state = val_env.reset(sel_index=i)
            for q in range(1, 10**10):
                v_action = ppo_agent.select_action(v_state, 2, q)
                v_state, _, v_done, _ = val_env.step(v_action)
                if v_done:
                    ref = float(validation_set[i]["score"])
                    gap = val_env.mk / ref - 1.0
                    all_val_results.append(float(gap))
                    if hasattr(val_env, "num_swaps"):
                        swaps.append(int(val_env.num_swaps))
                        blocked.append(int(val_env.num_blocked))
                    if dbg_fn is not None:
                        dbg_fn(1, f"  val instance {i+1}/{len(validation_set)} ({validation_set[i]['name']}) | makespan={val_env.mk} | ref={ref:.2f} | gap={gap:.4f}")
                    break

    avg_gap = float(np.mean(all_val_results))
    std_gap = float(np.std(all_val_results))
    q80_gap = float(np.percentile(all_val_results, 80))
    prefix = f"[TRAIN][VAL][ep {episode_number}]" if episode_number is not None else "[TRAIN][VAL]"
    print_fn(f"{prefix} avg_gap={avg_gap:.4f} | std_gap={std_gap:.4f} | q80_gap={q80_gap:.4f} | n={len(all_val_results)}")

    metrics = {
        "avg_gap": avg_gap,
        "std_gap": std_gap,
        "q80_gap": q80_gap,
        "all_gaps": all_val_results,
    }
    if swaps:
        # blocking env: a swap is a deadlock resolved by moving a cycle of parts at once
        # (feasible, but worth reporting: it means the policy walked into circular blocking)
        metrics["total_swaps"] = int(sum(swaps))
        metrics["instances_with_swap"] = int(sum(1 for s in swaps if s))
        metrics["total_blocked"] = int(sum(blocked))
        print_fn(f"{prefix} blocked_events={metrics['total_blocked']} | swaps={metrics['total_swaps']} "
                 f"in {metrics['instances_with_swap']} instance(s)")
    return metrics
