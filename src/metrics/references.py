"""Reference makespans for relative error.

Convention: the references of an instance folder live next to it -
    val/instances/x.fjs        -> val/solutions/x.json            (the existing layout)
    data/test/large/x.fjs      -> data/test/solutions/large/x.json
A reference file holds either a CP-SAT `final_schedule` (its max "end" is used, exactly as
src/utils/validation_utils.py:extract_reference_makespan does) or a plain "makespan".
Instances without a file get relative error only against the lower bound.
"""
import json
import os
from multiprocessing import Pool

from src.parsedata import get_data, parse


def reference_dir(folder):
    folder = os.path.normpath(folder)
    parent, name = os.path.split(folder)
    if name == "instances":
        return os.path.join(parent, "solutions")
    return os.path.join(parent, "solutions", name)


def load_reference(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("final_schedule"):
        return float(max(entry["end"] for entry in data["final_schedule"]))
    if data.get("makespan") is not None:
        return float(data["makespan"])
    return None


def load_folder(folder):
    """Every instance file of a folder (sorted by name), parsed, with its reference makespan
    as "score" (None when there is no reference file). Plain FJSP instances are .fjs text;
    batching instances are .json dicts with family / capacities / delta
    (src/batch_generator.py)."""
    ref_dir = reference_dir(folder)
    instances = []
    for file_name in sorted(os.listdir(folder)):
        path = os.path.join(folder, file_name)
        name, ext = os.path.splitext(file_name)
        if not os.path.isfile(path) or ext.lower() not in (".fjs", ".json"):
            continue
        with open(path, "r") as f:
            if ext.lower() == ".json":
                instance = json.load(f)
            else:
                jobs, operations, info, maximum = get_data(parse(f.read()))
                instance = {"jobs": jobs, "operations": operations, "maximum": maximum,
                            "num_machines": info["machinesNb"]}
        instance["name"] = name
        instance["score"] = load_reference(os.path.join(ref_dir, name + ".json"))
        instances.append(instance)
    return instances


def load_dataset(path):
    """A fixed split saved as JSON (e.g. val/test_dataset_blocking.json): a list of already
    parsed instances with their reference makespan in "score"."""
    with open(path, "r") as f:
        instances = json.load(f)
    for k, inst in enumerate(instances):
        inst.setdefault("name", f"{os.path.splitext(os.path.basename(path))[0]}_{k:03d}")
        inst["score"] = None if inst.get("score") is None else float(inst["score"])
    return instances


def load_instances(path):
    """A folder of .fjs files (references from the matching solutions folder) or a JSON split."""
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return load_dataset(path)
    return load_folder(path)


def _solve_one(args):
    from src.solver import solve_fjsp  # ortools only needed when solving
    path, out_path = args
    with open(path, "r") as f:
        jobs, operations, info, _ = get_data(parse(f.read()))
    status, objective, schedule = solve_fjsp(jobs, operations, info)
    if status not in ("OPTIMAL", "FEASIBLE"):
        return out_path, status, None
    final_schedule = [{
        "job": int(e["job_id"]), "operation": int(e["task_id"]), "machine": int(e["machine"]),
        # solve_fjsp returns datetime.fromtimestamp(t); .timestamp() undoes it exactly
        "start": round(e["start"].timestamp()), "end": round(e["end"].timestamp()),
    } for e in schedule]
    with open(out_path, "w") as f:
        json.dump({"solver": "cp-sat", "status": status, "makespan": float(objective),
                   "final_schedule": final_schedule}, f, indent=1)
    return out_path, status, float(objective)


def solve_missing_references(folder, workers=1):
    """CP-SAT (src/solver.py, 15 s limit per instance) for every instance of `folder` that
    has no reference file yet. Slow on large instances; the solution is the solver's best
    incumbent at the time limit, not necessarily optimal (its status says which)."""
    ref_dir = reference_dir(folder)
    os.makedirs(ref_dir, exist_ok=True)
    jobs = []
    for file_name in sorted(os.listdir(folder)):
        if not file_name.lower().endswith(".fjs"):
            continue
        out_path = os.path.join(ref_dir, os.path.splitext(file_name)[0] + ".json")
        if not os.path.isfile(out_path):
            jobs.append((os.path.join(folder, file_name), out_path))
    if not jobs:
        return []
    print(f"[REF] Solving {len(jobs)} missing reference(s) for {folder} with CP-SAT -> {ref_dir}")
    with Pool(max(1, workers)) as pool:
        results = []
        for k, res in enumerate(pool.imap_unordered(_solve_one, jobs), 1):
            print(f"[REF]   {k}/{len(jobs)} {os.path.basename(res[0])} | {res[1]} | makespan={res[2]}")
            results.append(res)
    return results
