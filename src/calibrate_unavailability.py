"""Calibration of the unavailability setting (src/unavailability_config.py).

On instances of the blocking setting (src/blocking_config.py), for a grid of window settings,
runs the earliest-completion-time rule
    - without windows (the blocking problem alone),
    - window-blind (ranks machines ignoring the windows) and
    - window-aware (ranks them with the information view),
on the same realisations, and records per setting (mean over the instances):
    downtime_cost   aware makespan / makespan without windows: how much the windows cost
    info_value      blind makespan / aware makespan: what knowing the windows is worth to a
                    greedy rule (> 1: the information helps). It stays close to 1: a one-step
                    rule rarely changes its choice because of a window, so the value of the
                    information shows in the gap to CP-SAT, not here
    down_share      share of machine time inside windows before the makespan
    trapped_parts, swaps, waits  (window-aware rule)
The chosen setting must make the windows bind (downtime_cost clearly > 1) with a realistic
down_share, and keep swaps / waits rare.

Usage:
    python -m src.calibrate_unavailability --mode mixed --n 10
Writes results/unavailability_calibration/<mode>.csv.
"""
import argparse
import copy
import csv
import itertools
import os
import random

import numpy as np

from src.blocking_config import BLOCKING_CONFIG
from src.env_unavailability import FJSPEnvUnavailBlind
from src.generator import generate_instance_list
from src.metrics.schedule import lower_bound
from src.metrics.unavailability import unavailability_stats
from src.parsedata import get_data, parse
from src.unavailability import sample_scenario, scenario_to_json
from src.unavailability_config import UNAVAIL_CONFIG

GRID = {
    "sched_length": ((0.05, 0.12), (0.10, 0.20)),
    "availability": ((0.85, 0.97), (0.75, 0.90)),
    "mttr": ((0.03, 0.08), (0.06, 0.12)),
}
OUT_DIR = "results/unavailability_calibration"


def run(env, rule):
    env.reset(sel_index=0)
    done = False
    while not done:
        _, _, done, _ = env.step(rule(env))
    return env


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--mode", default="mixed", choices=("scheduled", "breakdown", "mixed"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    instances = []
    for text in generate_instance_list(args.n, **BLOCKING_CONFIG["generator"]):
        jobs, operations, info, maximum = get_data(parse(text))
        instances.append({"jobs": jobs, "operations": operations})
    base = []  # the same earliest-completion-time rule on the same dynamics, without windows
    for inst in instances:
        empty = {**inst, "unavailability": {args.mode: {"windows": [[] for _ in inst["operations"][0]],
                                                         "beta": None, "eta": None, "mttr": None}}}
        base.append(run(FJSPEnvUnavailBlind([empty], 0, 100, unavail_mode=args.mode), lambda e: e.expert_action()).mk)

    rows = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        setting = dict(zip(keys, values))
        if args.mode == "scheduled" and (setting["availability"], setting["mttr"]) != (GRID["availability"][0], GRID["mttr"][0]):
            continue
        if args.mode == "breakdown" and setting["sched_length"] != GRID["sched_length"][0]:
            continue
        cfg = copy.deepcopy(UNAVAIL_CONFIG)
        cfg["scheduled"]["length"] = setting["sched_length"]
        cfg["breakdown"]["availability"] = setting["availability"]
        cfg["breakdown"]["mttr"] = setting["mttr"]
        stats = {k: [] for k in ("downtime_cost", "info_value", "down_share", "trapped_parts", "swaps", "waits")}
        for i, inst in enumerate(instances):
            scenario = sample_scenario(inst["operations"], lower_bound(inst), args.mode,
                                       np.random.default_rng([args.seed, i]), cfg)
            stored = {**inst, "unavailability": {args.mode: scenario_to_json(scenario)}}
            aware = run(FJSPEnvUnavailBlind([stored], 0, 100, unavail_mode=args.mode), lambda e: e.expert_action())
            blind_mk = run(FJSPEnvUnavailBlind([stored], 0, 100, unavail_mode=args.mode),
                           lambda e: e.blind_expert_action()).mk
            down = sum(max(0.0, min(e, aware.mk) - s) for ws in scenario["windows"] for s, e, _ in ws)
            stats["downtime_cost"].append(aware.mk / base[i])
            stats["info_value"].append(blind_mk / aware.mk)
            stats["down_share"].append(down / (aware.mk * len(inst["operations"][0])))
            stats["trapped_parts"].append(unavailability_stats(aware.schedule, aware.true_windows)["trapped_parts"])
            stats["swaps"].append(aware.num_swaps)
            stats["waits"].append(aware.num_waits)
        row = {k: str(v) for k, v in setting.items()}
        row.update({k: round(float(np.mean(v)), 4) for k, v in stats.items()})
        row["info_value_min"] = round(float(np.min(stats["info_value"])), 4)
        row["info_value_max"] = round(float(np.max(stats["info_value"])), 4)
        rows.append(row)
        print(f"[CALIB] {args.mode} {row}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.mode}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CALIB] wrote {path}")


if __name__ == "__main__":
    main()
