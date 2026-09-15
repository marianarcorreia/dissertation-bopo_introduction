from src.generator import generate_instance_list
from src.parsedata import get_data, parse
from src.solver import solve_fjsp
import json
import os
import pandas as pd


def generate_val(n_cases):
    print("=" * 60)
    print(f"[VAL] Generating validation set | n_cases={n_cases}")
    if not os.path.exists("val"):
        os.makedirs("val")
    
    if not os.path.exists("candidate_models"):
        os.makedirs("candidate_models")

    val_config = {
        "n_cases": n_cases,
        "range_jobs": (5, 15),
        "range_machines": (4, 13),
        "range_op_per_job": (4, 9),
        "max_processing": 25
    }
    print(f"[VAL] Config: {val_config}")

    list_instances = generate_instance_list(**val_config)
    print(f"[VAL] Generated {len(list_instances)} raw instance(s). Solving with OR-Tools...")

    list_scores = []
    for idx, instance in enumerate(list_instances):
        jobs, operations, info, maximum = get_data(parse(instance))
        _, score, _ = solve_fjsp(jobs, operations, info)
        print(f"[VAL]   instance {idx+1}/{len(list_instances)} | jobs={len(jobs)} | machines={info['machinesNb']} | optimal_makespan={int(score)}")
        list_scores.append({"score": str(int(score)), "jobs": jobs, "operations": operations, "maximum": maximum, "num_machines": info["machinesNb"]})

    with open('val/validation_set.json', 'w') as outfile:
        json.dump(list_scores, outfile)

    val_results = [{'best': 0.4, 'name': None}]*len(list_scores)
    with open('val/validation_results.json', 'w') as outfile:
        json.dump(val_results, outfile)
    
    model_params = []
    with open('candidate_models/model_params.json', 'w') as outfile:
        json.dump(model_params, outfile)
    print(f"[VAL] Validation set saved to val/validation_set.json")
    print("=" * 60)

