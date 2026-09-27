"""Unavailability validation/test sets with CP-SAT references (blocking + windows).

Generates new instances with the blocking setting of src/blocking_config.py and, for each, one
stored realisation per unavailability mode (src/unavailability.py:sample_scenario with the
settings of src/unavailability_config.py; the same seed per instance, so "scheduled" and
"mixed" share their planned maintenance). Each (instance, mode) is solved with the blocking
CP-SAT model plus the windows (src/solver_blocking.py), with the window-aware earliest-completion
rule's makespan as the horizon and its schedule as a starting solution. Writes two disjoint splits:
    val/validation_dataset_unavailability.json, val/test_dataset_unavailability.json
Per instance: "unavailability" {mode: realisation}, "score_<mode>" (CP-SAT), "status_<mode>",
"ect_<mode>" / "blind_ect_<mode>" (window-aware / window-blind earliest completion time) and
"score" = score of the configured mode (src/train.py:_select_scores picks the model's mode).

With breakdowns the solver knows the realised windows in advance: score_breakdown and
score_mixed are CLAIRVOYANT bounds (no online policy can be expected to reach them), and the
gap to them includes the unavoidable cost of not knowing the future. score_scheduled is an
ordinary reference (the windows are known to the policy too).

Every CP-SAT schedule is checked with src/metrics/unavailability.py, the same checks the env
is held to, so the solver and the env cannot silently disagree on the rules.

Usage:
    python -m src.generate_unavailability_references
    python -m src.generate_unavailability_references --n 20 --time-limit 120 --seed 11
    python -m src.generate_unavailability_references --n 2 --time-limit 10 --out-prefix val/smoke_unavailability
"""
import argparse
import json
import math
import os
import random

import numpy as np

from src.blocking_config import BLOCKING_CONFIG
from src.env_unavailability import FJSPEnvUnavailBlind
from src.generator import generate_instance_list
from src.metrics.schedule import check_schedule, lower_bound
from src.metrics.unavailability import stored_windows, unavailability_constraints
from src.parsedata import get_data, parse
from src.solver_blocking import solve_blocking_fjsp
from src.unavailability import MODES, sample_scenario, scenario_to_json
from src.unavailability_config import UNAVAIL_CONFIG


def expert_makespan(instance, mode, blind=False):
    """Makespan of the earliest-completion-time rule, window-aware or window-blind."""
    env = FJSPEnvUnavailBlind([instance], 0, 100, unavail_mode=mode)
    env.reset(sel_index=0)
    done = False
    while not done:
        _, _, done, _ = env.step(env.blind_expert_action() if blind else env.expert_action())
    return float(env.mk), env


def env_hint(env):
    """The env's finished schedule in the solver's format: a feasible starting solution."""
    return {rec["operation"]: {"machine": rec["machine"], "r": rec["routed"], "s": rec["start"],
                               "c": rec["end"], "d": rec["depart"]} for rec in env.schedule}


def solver_records(instance, schedule):
    """CP-SAT schedule -> schedule entries in the env's format (for the metric checks)."""
    records = []
    for j, job in enumerate(instance["jobs"]):
        for k, o in enumerate(job):
            rec = schedule[o]
            last = k == len(job) - 1
            records.append({"job": j, "operation": o, "machine": rec["machine"], "routed": rec["r"],
                            "start": rec["s"], "end": rec["c"], "depart": rec["d"],
                            "leave_out": rec["d"] if last else schedule[job[k + 1]]["r"]})
    return sorted(records, key=lambda r: (r["start"], r["operation"]))


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--n", type=int, default=20, help="instances per split")
    parser.add_argument("--time-limit", type=float, default=120.0, help="CP-SAT seconds per (instance, mode)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-prefix", default=None,
                        help="write <prefix>_validation.json / <prefix>_test.json instead of the default splits")
    args = parser.parse_args()
    if args.out_prefix:
        splits = (f"{args.out_prefix}_validation.json", f"{args.out_prefix}_test.json")
    else:
        splits = ("val/validation_dataset_unavailability.json", "val/test_dataset_unavailability.json")
    in_cap, out_cap = BLOCKING_CONFIG["in_cap"], BLOCKING_CONFIG["out_cap"]

    random.seed(args.seed)
    raw = generate_instance_list(2 * args.n, **BLOCKING_CONFIG["generator"])
    for split, path in enumerate(splits):
        out = []
        for i, text in enumerate(raw[split * args.n:(split + 1) * args.n]):
            idx = split * args.n + i
            jobs, operations, info, maximum = get_data(parse(text))
            inst = {"name": f"unav{idx + 1:03d}", "jobs": jobs, "operations": operations, "maximum": maximum,
                    "num_machines": info["machinesNb"], "in_cap": in_cap, "out_cap": out_cap,
                    "generator": BLOCKING_CONFIG["generator"], "unavail_config": UNAVAIL_CONFIG,
                    "unavailability": {}}
            lb = lower_bound(inst)
            for mode in args.modes:
                scenario = sample_scenario(operations, lb, mode, np.random.default_rng([args.seed, idx]), UNAVAIL_CONFIG)
                inst["unavailability"][mode] = scenario_to_json(scenario)
                ect, ect_env = expert_makespan(inst, mode)
                blind, _ = expert_makespan(inst, mode, blind=True)
                status, ms, schedule = solve_blocking_fjsp(jobs, operations, in_cap, out_cap, args.time_limit,
                                                           args.workers, windows=scenario["windows"],
                                                           horizon=math.ceil(ect), hint=env_hint(ect_env))
                if ms is None:
                    raise RuntimeError(f"{inst['name']} ({mode}): CP-SAT found no schedule within the ECT horizon ({status})")
                check = check_schedule(inst, solver_records(inst, schedule),
                                       unavailability_constraints(in_cap, out_cap, stored_windows(mode)))
                if not check["feasible"]:
                    raise RuntimeError(f"{inst['name']} ({mode}): the CP-SAT schedule breaks {check['violations']}")
                inst.update({f"score_{mode}": ms, f"status_{mode}": status,
                             f"ect_{mode}": ect, f"blind_ect_{mode}": blind})
                print(f"[UNAVAIL-REF] {path} {i + 1}/{args.n} {inst['name']} {mode:9s} | cp-sat={ms:.0f} ({status}) | "
                      f"ect={ect:.0f} (+{ect / ms - 1:.1%}) | blind ect={blind:.0f} (+{blind / ms - 1:.1%})", flush=True)
            mode = UNAVAIL_CONFIG["mode"] if UNAVAIL_CONFIG["mode"] in args.modes else args.modes[0]
            inst["score"] = inst[f"score_{mode}"]
            out.append(inst)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"[UNAVAIL-REF] wrote {path}")


if __name__ == "__main__":
    main()
