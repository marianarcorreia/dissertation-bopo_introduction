"""Write best_schedule.json for trials that already ran, from their saved best checkpoint.

Re-runs the same deterministic greedy test evaluation train() does at the end, so the
schedules are exactly what that checkpoint produced. Loading follows rescore_checkpoints.py.
The file is what optuna_dashboard.py's "Best Schedule" tab draws as a Gantt chart.

Usage:
    python backfill_best_schedules.py                       # every results/optuna_*/trial_*
    python backfill_best_schedules.py --pattern "optuna_ojm_gat_L2_*" --cpu
    python backfill_best_schedules.py --force               # overwrite existing files
"""
import argparse
import fnmatch
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", nargs="+", default=["optuna_*"],
                    help="fnmatch patterns over results/<study> folders.")
parser.add_argument("--force", action="store_true", help="Overwrite an existing best_schedule.json.")
parser.add_argument("--cpu", action="store_true", help="Hide CUDA so evaluation runs on CPU.")
args = parser.parse_args()
if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must happen before torch is imported

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src.train import _resolve_representation_modules
from src.utils.validation_utils import load_fixed_dataset, run_validation

test_set = load_fixed_dataset("val/test_dataset.json")
model_params = {m["name"]: m for m in json.load(open("candidate_models/model_params.json"))}
quiet = lambda msg: None

studies = sorted(d for d in os.listdir("results")
                 if os.path.isdir(os.path.join("results", d))
                 and any(fnmatch.fnmatch(d, p) for p in args.pattern))
print(f"[BACKFILL] {len(studies)} study folder(s) matched {args.pattern}")

for study in studies:
    for trial in sorted(os.listdir(os.path.join("results", study))):
        trial_dir = os.path.join("results", study, trial)
        summary_path = os.path.join(trial_dir, "run_summary.json")
        out_path = os.path.join(trial_dir, "best_schedule.json")
        if not trial.startswith("trial_") or not os.path.isfile(summary_path):
            continue
        if os.path.isfile(out_path) and not args.force:
            print(f"[BACKFILL] skip {study}/{trial}: best_schedule.json already exists")
            continue

        summary = json.load(open(summary_path))
        ckpt = summary.get("best_model_path")
        if not ckpt or not os.path.isfile(ckpt):
            print(f"[BACKFILL] skip {study}/{trial}: checkpoint missing ({ckpt})")
            continue
        params = model_params.get(os.path.basename(ckpt))
        if params is None:
            print(f"[BACKFILL] skip {study}/{trial}: {os.path.basename(ckpt)} not in candidate_models/model_params.json")
            continue

        rep = summary.get("representation") or params.get("representation")
        gnn_type = summary.get("gnn_type") or params.get("gnn_type", "gat")
        rep, env_cls, bopo_cls = _resolve_representation_modules(rep)
        mask_option, sel_k = params["mask_option"], params["sel_k"]
        jm_design = params.get("jm_design", "baseline")
        jm_kwargs = {} if jm_design == "baseline" else {"jm_design": jm_design}

        env = env_cls(test_set, mask_option, sel_k, **jm_kwargs)
        metadata = env.reset().metadata()
        agent = bopo_cls(1e-3, env, metadata, params["hidden_channels"], params["num_layers"], params["heads"],
                         gnn_type=gnn_type, **jm_kwargs)
        with torch.no_grad():  # materialize lazy Linear(-1) parameters before loading weights
            agent.select_action(env.reset(sel_index=0), 2, 0)
        agent.load(ckpt)

        m = run_validation(agent, env_cls(test_set, mask_option, sel_k, **jm_kwargs), test_set, print_fn=quiet)
        with open(out_path, "w") as f:
            json.dump({
                "best_model_path": ckpt,
                "split": "test",
                "best_instance": min(m["schedule"], key=lambda s: s["gap"])["name"],
                "instances": m["schedule"],
            }, f)

        reported = summary.get("test_avg_gap")
        same = reported is not None and abs(m["avg_gap"] - reported) < 1e-6
        reported_str = "n/a" if reported is None else f"{reported:.4f}"
        print(f"[BACKFILL] {study}/{trial}: rebuilt test gap={m['avg_gap']:.4f} "
              f"(reported {reported_str}) | reproduces={same}")
