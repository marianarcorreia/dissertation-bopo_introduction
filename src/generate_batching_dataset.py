"""
Build a solved dataset for the FJSP with parallel batching.

Writes, under --out (default data/batching):
  instances/bat001.json ...   the generated instances (see src/batch_generator.py)
  solutions/bat001.json ...   CP-SAT solution: status, makespan, bound, final_schedule
  batching_dataset.json       every instance + its reference makespan ("score"), in the
                              same list-of-dicts layout as val/validation_dataset.json,
                              extended with family / capacities / delta

Usage:
  python -m src.generate_batching_dataset --n-cases 40 --time-limit 30 --seed 0
"""
import argparse
import json
import os
import random

from src.batch_generator import generate_batching_instance_list
from src.batch_solver import check_schedule, solve_batching


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-cases", type=int, default=40)
    p.add_argument("--out", default="data/batching")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=30.0, help="CP-SAT seconds per instance")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--jobs", type=int, nargs=2, default=(8, 10))
    p.add_argument("--machines", type=int, nargs=2, default=(5, 10))
    p.add_argument("--ops", type=int, nargs=2, default=(5, 6))
    p.add_argument("--max-processing", type=int, default=100)
    p.add_argument("--delta", type=int, default=2)
    p.add_argument("--family-span", type=int, default=3)
    p.add_argument("--family-ratio", type=float, default=0.6)
    p.add_argument("--family-size", type=int, nargs=2, default=(2, 4))
    p.add_argument("--capacity", type=int, nargs=2, default=(2, 3))
    args = p.parse_args()

    random.seed(args.seed)
    inst_dir = os.path.join(args.out, "instances")
    sol_dir = os.path.join(args.out, "solutions")
    os.makedirs(inst_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)

    instances = generate_batching_instance_list(
        n_cases=args.n_cases,
        range_jobs=tuple(args.jobs),
        range_machines=tuple(args.machines),
        range_op_per_job=tuple(args.ops),
        max_processing=args.max_processing,
        delta=args.delta,
        family_span=args.family_span,
        family_ratio=args.family_ratio,
        family_size_range=tuple(args.family_size),
        capacity_range=tuple(args.capacity),
    )

    dataset = []
    for idx, inst in enumerate(instances):
        name = f"bat{idx + 1:03d}"
        with open(os.path.join(inst_dir, name + ".json"), "w") as f:
            json.dump(inst, f)

        sol = solve_batching(inst, time_limit=args.time_limit, workers=args.workers, seed=args.seed)
        errors = check_schedule(inst, sol["final_schedule"]) if sol["makespan"] is not None else ["no solution"]
        sol["check_errors"] = errors
        with open(os.path.join(sol_dir, name + ".json"), "w") as f:
            json.dump(sol, f, indent=2)

        n_fam = len(inst["capacities"])
        in_fam = sum(1 for x in inst["family"] if x >= 0)
        print(f"[{name}] jobs={len(inst['jobs'])} machines={inst['num_machines']} ops={len(inst['operations'])} "
              f"families={n_fam} ({in_fam} ops) | {sol['status']} Cmax={sol['makespan']} "
              f"bound={sol.get('best_bound')} batches>1={sol.get('num_multi_op_batches')} "
              f"| check={'OK' if not errors else errors}", flush=True)

        if sol["makespan"] is not None and not errors:
            dataset.append({"name": name, "score": float(sol["makespan"]), **inst})

    with open(os.path.join(args.out, "batching_dataset.json"), "w") as f:
        json.dump(dataset, f)
    print(f"Saved {len(dataset)}/{len(instances)} solved instances to {args.out}/batching_dataset.json")


if __name__ == "__main__":
    main()
