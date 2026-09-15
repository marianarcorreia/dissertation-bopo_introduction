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

    indicators = solution_data.get("indicators", {})
    reference = indicators.get("best_objective_bound", None)
    if reference is None or float(reference) <= 0:
        raise ValueError(f"Invalid reference objective in solution file: {solution_path}")
    return float(reference)


def build_validation_dataset(sample_size=20, instances_dir="val/instances", solutions_dir="val/solutions", dbg_fn=None):
    if not os.path.isdir(instances_dir):
        raise FileNotFoundError(f"Validation instances directory not found: {instances_dir}")
    if not os.path.isdir(solutions_dir):
        raise FileNotFoundError(f"Validation solutions directory not found: {solutions_dir}")

    all_instance_files = [f for f in os.listdir(instances_dir) if f.lower().endswith('.fjs')]
    selected_instance_files = select_representative_files(all_instance_files, sample_size)

    validation_set = []
    for instance_file in selected_instance_files:
        instance_name = os.path.splitext(instance_file)[0]
        instance_path = os.path.join(instances_dir, instance_file)
        solution_path = os.path.join(solutions_dir, instance_name + '.json')

        if not os.path.isfile(solution_path):
            if dbg_fn is not None:
                dbg_fn(1, f"Skipping validation instance without matching solution: {instance_file}")
            continue

        with open(instance_path, 'r') as file:
            contents = file.read()
            jobs, operations, info, maximum = get_data(parse(contents))

        reference_makespan = extract_reference_makespan(solution_path)

        validation_set.append({
            "name": instance_name,
            "score": reference_makespan,
            "jobs": jobs,
            "operations": operations,
            "maximum": maximum,
            "num_machines": info["machinesNb"],
        })

    if len(validation_set) == 0:
        raise RuntimeError("No validation instances were loaded from val/instances + val/solutions")

    return validation_set


def run_validation(ppo_agent, val_env, validation_set, episode_number=None, dbg_fn=None, print_fn=print):
    all_val_results = []
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
                    if dbg_fn is not None:
                        dbg_fn(1, f"  val instance {i+1}/{len(validation_set)} ({validation_set[i]['name']}) | makespan={val_env.mk} | ref={ref:.2f} | gap={gap:.4f}")
                    break

    avg_gap = float(np.mean(all_val_results))
    std_gap = float(np.std(all_val_results))
    q80_gap = float(np.percentile(all_val_results, 80))
    prefix = f"[TRAIN][VAL][ep {episode_number}]" if episode_number is not None else "[TRAIN][VAL]"
    print_fn(f"{prefix} avg_gap={avg_gap:.4f} | std_gap={std_gap:.4f} | q80_gap={q80_gap:.4f} | n={len(all_val_results)}")

    return {
        "avg_gap": avg_gap,
        "std_gap": std_gap,
        "q80_gap": q80_gap,
        "all_gaps": all_val_results,
    }
