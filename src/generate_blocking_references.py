"""Blocking validation/test sets with blocking-aware reference makespans.

Generates new instances with the blocking instance setting of src/blocking_config.py (the
default validation instances never make blocking bind, so they cannot be reused), solves each
with CP-SAT both without blocking (src/solver.py) and with buffers/blocking
(src/solver_blocking.py), and writes two disjoint splits:
    val/validation_dataset_blocking.json, val/test_dataset_blocking.json
"score" is the blocking makespan (the CP-SAT incumbent, as for the non-blocking references);
"score_nonblocking", both solver statuses, the capacities and the tightness
(score / score_nonblocking) are stored alongside.

Usage:
    python -m src.generate_blocking_references
    python -m src.generate_blocking_references --n 20 --time-limit 120 --seed 7
(the time limit applies to both CP-SAT models, so both references are equally strong)
"""
import argparse
import json
import random

from src.blocking_config import BLOCKING_CONFIG
from src.generator import generate_instance_list
from src.parsedata import get_data, parse
from src.solver import solve_fjsp
from src.solver_blocking import solve_blocking_fjsp

SPLITS = ("val/validation_dataset_blocking.json", "val/test_dataset_blocking.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20, help="instances per split")
    parser.add_argument("--time-limit", type=float, default=120.0, help="CP-SAT seconds per instance (blocking model)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    in_cap, out_cap = BLOCKING_CONFIG["in_cap"], BLOCKING_CONFIG["out_cap"]

    random.seed(args.seed)
    raw = generate_instance_list(2 * args.n, **BLOCKING_CONFIG["generator"])
    for split, path in enumerate(SPLITS):
        out = []
        for i, text in enumerate(raw[split * args.n:(split + 1) * args.n]):
            jobs, operations, info, maximum = get_data(parse(text))
            name = f"blk{split * args.n + i + 1:03d}"
            st_nb, ms_nb, _ = solve_fjsp(jobs, operations, info, args.time_limit)
            st_b, ms_b, _ = solve_blocking_fjsp(jobs, operations, in_cap, out_cap, args.time_limit)
            if ms_b is None:
                raise RuntimeError(f"{name}: CP-SAT found no blocking solution ({st_b})")
            print(f"[BLOCKING-REF] {path} {i + 1}/{args.n} {name} | jobs={len(jobs)} machines={info['machinesNb']} | "
                  f"blocking={ms_b:.0f} ({st_b}) | non-blocking={ms_nb:.0f} ({st_nb}) | +{ms_b / ms_nb - 1:.1%}", flush=True)
            out.append({"name": name, "score": float(ms_b), "jobs": jobs, "operations": operations,
                        "maximum": maximum, "num_machines": info["machinesNb"],
                        "score_nonblocking": float(ms_nb), "tightness": float(ms_b / ms_nb),
                        "blocking_status": st_b, "nonblocking_status": st_nb,
                        "in_cap": in_cap, "out_cap": out_cap, "generator": BLOCKING_CONFIG["generator"]})
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"[BLOCKING-REF] wrote {path}")


if __name__ == "__main__":
    main()
