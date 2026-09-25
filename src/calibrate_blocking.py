"""Calibration of the blocking experiments: which instance setting and buffer capacities make
the blocking constraint actually bind (the chosen values go into src/blocking_config.py).

Stage 1 (cheap, earliest-completion-time expert rule only), for every combination of
    flexibility (max eligible machines per operation) x bottleneck factor x size x capacities:
      blocked      mean blocked events per instance (a finished part held on its machine)
      share_blocked share of instances with at least one blocked event
      swaps        mean deadlocks resolved by a swap per instance
      h_tightness  expert makespan with these buffers / expert makespan with unlimited buffers
    The same instances are used for every capacity setting, so capacities are compared fairly.

Stage 2 (CP-SAT, on the most promising stage-1 settings):
      tightness    blocking optimum / non-blocking optimum (the real measure of how much the
                   constraint changes the problem; report it in the thesis)
      optimal      share of instances CP-SAT solved to proven optimality (both models)

Selection criterion: blocking on (almost) every instance and a clear cost for a buffer-blind rule
(h_tightness), few swaps (<= --max-swaps per instance), flexibility >= 2 (so the problem stays a
*flexible* job shop), and CP-SAT mostly optimal (trustworthy references). The CP-SAT tightness
is reported next to h_tightness: when h_tightness is high but the optimum barely moves, blocking
is avoidable with buffer-aware decisions - the room a buffer representation has to show value.

Usage:
    python -m src.calibrate_blocking --stage 1
    python -m src.calibrate_blocking --stage 2 --top 5
Outputs: results/blocking_calibration/stage1.csv, stage2.csv
"""
import argparse
import itertools
import os
import random

import numpy as np
import pandas as pd

from src.env_blocking import FJSPEnvBlocking
from src.generator import generate_instance_list
from src.parsedata import get_data, parse
from src.solver import solve_fjsp
from src.solver_blocking import solve_blocking_fjsp

OUT_DIR = "results/blocking_calibration"
FLEXIBILITY = (1, 2, 3)
BOTTLENECK_FACTOR = (1.5, 2.0, 3.0)
SIZES = {"10-12j_4-6m": ((10, 12), (4, 6)), "12-15j_4-6m": ((12, 15), (4, 6))}
CAPACITIES = ((5, 2), (3, 1), (2, 1), (1, 1), (1, 0), (0, 0))  # (x, 0): pure blocking
UNLIMITED = 10 ** 6


def make_instances(flex, factor, size, n, seed):
    range_jobs, range_machines = SIZES[size]
    random.seed(seed)
    raw = generate_instance_list(n, range_jobs, range_machines, (5, 6), 100, mas_per_ope_max=flex,
                                 n_bottlenecks=1, bottleneck_prob=0.7, bottleneck_factor=factor)
    instances = []
    for text in raw:
        jobs, operations, info, maximum = get_data(parse(text))
        instances.append({"jobs": jobs, "operations": operations, "maximum": maximum,
                          "num_machines": info["machinesNb"]})
    return instances


def expert_run(instances, in_cap, out_cap):
    env = FJSPEnvBlocking(instances, mask_option=1, sel_k=100, in_cap=in_cap, out_cap=out_cap,
                          blocking_repr="none")
    rows = []
    for i in range(len(instances)):
        env.reset(sel_index=i)
        done = False
        while not done:
            _, _, done, _ = env.step(env.expert_action())
        rows.append((env.mk, env.num_blocked, env.num_swaps))
    return rows


def stage1(n, seed):
    rows = []
    for flex, factor, size in itertools.product(FLEXIBILITY, BOTTLENECK_FACTOR, SIZES):
        instances = make_instances(flex, factor, size, n, seed)
        base = [mk for mk, _, _ in expert_run(instances, UNLIMITED, UNLIMITED)]
        for in_cap, out_cap in CAPACITIES:
            res = expert_run(instances, in_cap, out_cap)
            mk = np.array([r[0] for r in res])
            blocked = np.array([r[1] for r in res])
            row = {"flexibility": flex, "bottleneck_factor": factor, "size": size,
                   "in_cap": in_cap, "out_cap": out_cap,
                   "blocked": blocked.mean(), "share_blocked": (blocked > 0).mean(),
                   "swaps": np.mean([r[2] for r in res]), "h_tightness": float(np.mean(mk / np.array(base)))}
            rows.append(row)
            print("[CAL1]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "stage1.csv"), index=False)
    return df


def stage2(n, seed, top, time_limit, max_swaps):
    df = pd.read_csv(os.path.join(OUT_DIR, "stage1.csv"))
    # stay a flexible job shop, and avoid settings where the env depends heavily on swaps
    # (every setting where blocking matters has some: the expert rule ignores the buffers)
    cand = df[(df.flexibility >= 2) & (df.swaps <= max_swaps)]
    # the best setting per capacity pair first (so capacities are compared), then the rest
    best_per_cap = cand.sort_values("h_tightness", ascending=False).groupby(["in_cap", "out_cap"]).head(1)
    rest = cand.drop(best_per_cap.index).sort_values("h_tightness", ascending=False)
    chosen = pd.concat([best_per_cap, rest]).head(top)
    rows = []
    for _, c in chosen.iterrows():
        instances = make_instances(int(c.flexibility), float(c.bottleneck_factor), c["size"], n, seed)
        ratios, optimal = [], []
        for inst in instances:
            info = {"machinesNb": inst["num_machines"]}
            st_nb, ms_nb, _ = solve_fjsp(inst["jobs"], inst["operations"], info)
            st_b, ms_b, _ = solve_blocking_fjsp(inst["jobs"], inst["operations"], int(c.in_cap), int(c.out_cap), time_limit)
            ratios.append(ms_b / ms_nb)
            optimal.append(st_nb == "OPTIMAL" and st_b == "OPTIMAL")
        row = {**c.to_dict(), "tightness": float(np.mean(ratios)), "tightness_min": float(np.min(ratios)),
               "tightness_max": float(np.max(ratios)), "optimal": float(np.mean(optimal))}
        rows.append(row)
        print("[CAL2]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "stage2.csv"), index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--n", type=int, default=10, help="instances per setting")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top", type=int, default=5, help="stage 2: settings to solve with CP-SAT")
    parser.add_argument("--time-limit", type=float, default=20.0, help="stage 2: CP-SAT seconds (blocking model)")
    parser.add_argument("--max-swaps", type=float, default=2.0, help="stage 2: max mean swaps per instance (expert rule)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage == 1:
        stage1(args.n, args.seed)
    else:
        stage2(args.n, args.seed, args.top, args.time_limit, args.max_swaps)


if __name__ == "__main__":
    main()
