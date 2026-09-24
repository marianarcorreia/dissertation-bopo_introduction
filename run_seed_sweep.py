"""Multi-seed backbone sweep: every representation x every GNN backbone x several seeds,
with the fixed BOPO loss (see src/bopo_utils.py:bopo_group_loss), via run_diagnostic_job.py.

Why: single-seed comparisons were not reliable - ojm/gat (2 layers) scored test 0.188 /
0.104 / 0.106 with seeds 42 / 43 / 44, a spread larger than most backbone differences.

Runs are executed ONE AT A TIME by default (the machine is shared with Optuna studies and
parallel trainings have been OOM-killed). Loops seed-outermost, so after the first 9 runs
there is one complete seed for every (representation, backbone), after 18 there are two.
Resumable: a run whose results/<run_name>/run_summary.json already exists is skipped, so
the same command can simply be started again after an interruption.

Usage:
    python run_seed_sweep.py                                   # 3 reps x 3 backbones x seeds 42 43 44
    python run_seed_sweep.py --representations ojm --seeds 42 43
    python run_seed_sweep.py --dry-run                         # list what would run

Run names: seeds_<rep>_<backbone>_L<layers>_s<seed>. Afterwards:
    python rescore_checkpoints.py --pattern "seeds_*"
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--representations", nargs="+", default=["oo", "om", "ojm"])
parser.add_argument("--backbones", nargs="+", default=["gin", "gat", "transformer"])
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

jobs = [
    (rep, bb, seed, f"seeds_{rep}_{bb}_L{args.num_layers}_s{seed}")
    for seed in args.seeds
    for rep in args.representations
    for bb in args.backbones
]
os.makedirs(os.path.join(ROOT, "sweep_logs"), exist_ok=True)


def log(msg):
    print(f"[SWEEP {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


log(f"{len(jobs)} job(s) | layers={args.num_layers}")
for n, (rep, bb, seed, run_name) in enumerate(jobs, 1):
    if os.path.isfile(os.path.join(ROOT, "results", run_name, "run_summary.json")):
        log(f"[{n}/{len(jobs)}] skip {run_name} (already complete)")
        continue
    cmd = [sys.executable, "-u", os.path.join(ROOT, "run_diagnostic_job.py"),
           "--run-name", run_name, "--representation", rep, "--gnn-type", bb,
           "--num-layers", str(args.num_layers), "--seed", str(seed)]
    log(f"[{n}/{len(jobs)}] start {run_name}")
    if args.dry_run:
        continue
    t0 = time.time()
    with open(os.path.join(ROOT, "sweep_logs", f"{run_name}.log"), "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"[{n}/{len(jobs)}] {'done' if rc == 0 else f'FAILED (exit {rc})'} {run_name} in {(time.time() - t0) / 3600:.1f}h")

log("sweep finished")
