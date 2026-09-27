"""
Calibration of the transport scale rho (docs/transport_formulation.tex, experimental grid).

For every rho, the same batching instances (the layout is drawn after everything else, see
src/batch_generator.py) are solved by CP-SAT with and without transport, and the script
reports
  increase : mean relative makespan increase caused by transport, C(rho) / C(0) - 1
  share    : mean fraction of the jobs' time spent travelling in the CP-SAT schedule,
             sum_j transport_j / sum_j completion_j (depot -> first machine included)
  gap      : mean CP-SAT optimality gap (makespan / best bound - 1), to see whether the
             time limit was enough
rho should make transport clearly present but not dominant.

Usage:
  python -m src.calibrate_transport --rho 0.1 0.3 0.5 --n-cases 8 --time-limit 20
"""
import argparse
import random

import numpy as np

from src.batch_generator import generate_batching_instance_list
from src.batch_solver import check_schedule, solve_batching
from src.transport import depot_index, transport_matrix


def transport_share(instance, schedule):
    tau = transport_matrix(instance)
    by_op = {e["op_id"]: e for e in schedule}
    travel = completion = 0.0
    for job in instance["jobs"]:
        loc = depot_index(instance)
        for o in job:
            travel += tau[loc][by_op[o]["machine"]]
            loc = by_op[o]["machine"]
        completion += by_op[job[-1]]["end"]
    return travel / completion


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rho", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--jobs", type=int, nargs=2, default=(8, 10))
    p.add_argument("--machines", type=int, nargs=2, default=(5, 10))
    p.add_argument("--ops", type=int, nargs=2, default=(5, 6))
    args = p.parse_args()

    def instances(rho):
        # one instance per call, so that every instance is identical across rho values
        out = []
        for i in range(args.n_cases):
            random.seed(args.seed * 1000 + i)
            out += generate_batching_instance_list(n_cases=1, range_jobs=tuple(args.jobs),
                                                   range_machines=tuple(args.machines),
                                                   range_op_per_job=tuple(args.ops), transport_rho=rho)
        return out

    solve = lambda inst: solve_batching(inst, time_limit=args.time_limit, workers=args.workers, seed=args.seed)
    base = [solve(inst) for inst in instances(None)]
    print(f"rho=0    | mean Cmax={np.mean([s['makespan'] for s in base]):.1f} | "
          f"optimal={sum(s['status'] == 'OPTIMAL' for s in base)}/{len(base)}", flush=True)
    for rho in args.rho:
        insts = instances(rho)
        sols = [solve(inst) for inst in insts]
        for inst, sol in zip(insts, sols):
            assert check_schedule(inst, sol["final_schedule"]) == [], "infeasible CP-SAT schedule"
        increase = [s["makespan"] / b["makespan"] - 1 for s, b in zip(sols, base)]
        share = [transport_share(i, s["final_schedule"]) for i, s in zip(insts, sols)]
        gap = [s["makespan"] / max(s["best_bound"], 1) - 1 for s in sols]
        print(f"rho={rho:<4} | increase={np.mean(increase):.3f} (min {min(increase):.3f}, max {max(increase):.3f}) "
              f"| share={np.mean(share):.3f} | gap={np.mean(gap):.3f} "
              f"| optimal={sum(s['status'] == 'OPTIMAL' for s in sols)}/{len(sols)}", flush=True)


if __name__ == "__main__":
    main()
