"""Batching sweep (docs/batching_runs_todo.md): the three batching representations for every
(GNN backbone, number of layers, seed), trained one at a time with run_diagnostic_job.py,
then one metrics report per (backbone, seed) on the held-out test split with run_metrics.py
(JSON + Markdown + bar chart of every metric).

Settings shared by every run: sel_k 1, mask_option 1, lr 4.7e-3, 350 BOPO steps, 3 heads,
jm_design edges, validation on 20 instances (the rest is the v2 setup of run_diagnostic_job.py).

Loops seed-outermost, so each finished seed has every backbone. Resumable: a run whose
results/<run>/run_summary.json exists is skipped, and so is a report that already exists.

Usage:
    python run_batching_sweep.py --configs gin:3 --seeds 42          # one group
    python run_batching_sweep.py --configs gat:2 gin:3 --seeds 43 44
    python run_batching_sweep.py --dry-run
    python run_batching_sweep.py --configs gat:2 --seeds 42 43 44 --step-metrics 4 --tag steps
Run names: batching_<gnn>_L<layers>_<rep>_s<seed>[_<tag>]; reports:
report_metrics_batching_<gnn>_L<layers>_s<seed>[_<tag>] (the first GAT report, seed 42, is
results/metrics/report_metrics_batching.*). --step-metrics N records every src/metrics metric
after every training step on N validation instances (<run>/step_metrics.json); --tag keeps
those runs apart from the ones trained without it.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPS = ("ojmb_node", "ojmb_edge", "ojmb_base")
TEST_SPLIT = "data/batching/batching_test_split.json"

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+", default=["gat:2", "gin:3"], help="backbone:num_layers")
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
parser.add_argument("--lr", default="4.7e-3")
parser.add_argument("--max-episodes", default="350")
parser.add_argument("--step-metrics", default="0", help="Validation instances for per-step metrics (0 = off).")
parser.add_argument("--tag", default="", help="Suffix for run and report names.")
parser.add_argument("--variant", default="v1", choices=["v1", "v2"],
                    help="v2: the batching envs with wait actions and the batch-aware mask (ojmb_*_v2).")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
suffix = f"_{args.tag}" if args.tag else ""
configs = [(c.split(":")[0], int(c.split(":")[1])) for c in args.configs]
os.makedirs(os.path.join(ROOT, "sweep_logs"), exist_ok=True)


def log(msg):
    print(f"[SWEEP {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd, log_name):
    t0 = time.time()
    with open(os.path.join(ROOT, "sweep_logs", log_name), "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, (time.time() - t0) / 3600


for seed in args.seeds:
    for gnn, layers in configs:
        best = {}
        # v2 adds Representation C (features only), which exists only in v2
        for rep in (REPS if args.variant == "v1" else tuple(r + "_v2" for r in REPS + ("ojmb_feat",))):
            name = f"batching_{gnn}_L{layers}_{rep}_s{seed}{suffix}"
            summary = os.path.join(ROOT, "results", name, "run_summary.json")
            if os.path.isfile(summary):
                log(f"skip {name} (already complete)")
            elif args.dry_run:
                log(f"would train {name}")
                continue
            else:
                log(f"start {name}")
                rc, hours = run([sys.executable, "-u", "run_diagnostic_job.py", "--run-name", name,
                                 "--representation", rep, "--gnn-type", gnn, "--num-layers", str(layers),
                                 "--sel-k", "1", "--mask-option", "1", "--lr", args.lr,
                                 "--max-episodes", args.max_episodes, "--validation-size", "20",
                                 "--jm-design", "edges", "--seed", str(seed),
                                 "--step-metrics", args.step_metrics], f"{name}.log")
                log(f"{'done' if rc == 0 else f'FAILED (exit {rc})'} {name} in {hours:.2f} h")
            if os.path.isfile(summary):
                with open(summary) as f:
                    s = json.load(f)
                log(f"  {name}: best={s.get('best_model_path')} | val_gap={s.get('best_validation_avg_gap')} "
                    f"| test_gap={s.get('test_avg_gap')}")
                if s.get("best_model_path"):
                    best[rep] = os.path.basename(s["best_model_path"])

        variant = "" if args.variant == "v1" else "_v2"
        report = "report_metrics_batching" if (gnn, layers, seed, suffix, variant) == ("gat", 2, 42, "", "") \
            else f"report_metrics_batching{variant}_{gnn}_L{layers}_s{seed}{suffix}"
        if args.dry_run or not best:
            log(f"{'would write' if args.dry_run else 'no checkpoint for'} {report}")
            continue
        if os.path.isfile(os.path.join(ROOT, "results", "metrics", f"{report}.json")):
            log(f"skip {report} (already exists)")
            continue
        log(f"start {report} on {best}")
        rc, hours = run([sys.executable, "-u", "run_metrics.py", "--run-name", report,
                         "--models-file", "candidate_models/model_params.json", "--models", *best.values(),
                         "--folders", TEST_SPLIT], f"{report}.log")
        log(f"{'done' if rc == 0 else f'FAILED (exit {rc})'} {report} in {hours * 60:.0f} min")

log("sweep finished")
