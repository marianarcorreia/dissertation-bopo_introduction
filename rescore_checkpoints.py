"""Re-score saved best checkpoints with ONE code path, on the fixed validation and test splits.

Why: one table, from one code path, with each checkpoint's OWN mask settings (sel_k,
mask_option from candidate_models/model_params.json) - these were not uniform across
sweeps (v2 om/ojm used sel_k=2, everything else sel_k=1), and evaluating a checkpoint with
the wrong sel_k silently gives a different number. Also flags any run whose reported test
gap does not reproduce.

Also scores the earliest-completion-time teacher heuristic (env.expert_action, the
warm-start's behavior-cloning target) per representation, as a floor any trained policy
has to beat.

Usage:
    python rescore_checkpoints.py                         # default run-name patterns
    python rescore_checkpoints.py --pattern "seeds_*"     # only the multi-seed sweep
    python rescore_checkpoints.py --cpu                   # keep the GPU free

Writes results/rescore/rescore_summary.csv (one row per run) and results/rescore/rescore.json.
"""
import argparse
import fnmatch
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", nargs="+",
                    default=["convergence_check_*", "ablation_*", "reseed_*", "seeds_*"],
                    help="fnmatch patterns over results/<run_name> folders to re-score.")
parser.add_argument("--cpu", action="store_true", help="Hide CUDA so evaluation runs on CPU.")
parser.add_argument("--no-heuristic", action="store_true", help="Skip scoring the ECT heuristic.")
args = parser.parse_args()
if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must happen before torch is imported

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src.train import _resolve_representation_modules
from src.utils.validation_utils import load_fixed_dataset, run_validation

OUT_DIR = os.path.join("results", "rescore")
os.makedirs(OUT_DIR, exist_ok=True)

splits = {
    "val": load_fixed_dataset("val/validation_dataset.json"),
    "test": load_fixed_dataset("val/test_dataset.json"),
}
model_params = {m["name"]: m for m in json.load(open("candidate_models/model_params.json"))}
quiet = lambda msg: None


def heuristic_gaps(env_cls, dataset, mask_option, sel_k):
    env = env_cls(dataset, mask_option, sel_k)
    gaps = []
    for i, inst in enumerate(dataset):
        env.reset(sel_index=i)
        while True:
            _, _, done, _ = env.step(env.expert_action())
            if done:
                gaps.append(env.mk / float(inst["score"]) - 1.0)
                break
    return gaps


rows = []
runs = sorted(d for d in os.listdir("results")
              if any(fnmatch.fnmatch(d, p) for p in args.pattern)
              and os.path.isfile(os.path.join("results", d, "run_summary.json")))
print(f"[RESCORE] {len(runs)} run(s) matched {args.pattern}")

heuristics_done = set()
for run in runs:
    summary = json.load(open(os.path.join("results", run, "run_summary.json")))
    ckpt = summary.get("best_model_path")
    if not ckpt or not os.path.isfile(ckpt):
        print(f"[RESCORE] skip {run}: checkpoint missing ({ckpt})")
        continue
    params = model_params.get(os.path.basename(ckpt))
    if params is None:
        print(f"[RESCORE] skip {run}: {os.path.basename(ckpt)} not in candidate_models/model_params.json")
        continue

    rep = summary.get("representation") or params.get("representation")
    gnn_type = summary.get("gnn_type") or params.get("gnn_type", "gat")
    rep, env_cls, bopo_cls = _resolve_representation_modules(rep)
    mask_option, sel_k = params["mask_option"], params["sel_k"]

    if not args.no_heuristic and (rep, mask_option, sel_k) not in heuristics_done:
        heuristics_done.add((rep, mask_option, sel_k))
        row = {"run_name": f"heuristic_ect_{rep}", "representation": rep, "gnn_type": "ect_heuristic"}
        for split, data in splits.items():
            g = heuristic_gaps(env_cls, data, mask_option, sel_k)
            row[f"{split}_gap"] = float(np.mean(g))
            row[f"{split}_q80"] = float(np.percentile(g, 80))
        rows.append(row)
        print(f"[RESCORE] ECT heuristic {rep}: val={row['val_gap']:.4f} test={row['test_gap']:.4f}")

    env = env_cls(splits["val"], mask_option, sel_k)
    metadata = env.reset().metadata()
    agent = bopo_cls(1e-3, env, metadata, params["hidden_channels"], params["num_layers"], params["heads"],
                     gnn_type=gnn_type)
    with torch.no_grad():  # materialize lazy Linear(-1) parameters before loading weights
        agent.select_action(env.reset(sel_index=0), 2, 0)
    agent.load(ckpt)

    row = {
        "run_name": run,
        "representation": rep,
        "gnn_type": gnn_type,
        "num_layers": params["num_layers"],
        "sel_k": sel_k,
        "mask_option": mask_option,
        "seed": summary.get("seed"),
        "logp_norm": summary.get("logp_norm", "sum"),
        "exclude_greedy_from_loss": summary.get("exclude_greedy_from_loss", False),
        "checkpoint": ckpt,
        "checkpoint_episode": params.get("episode"),
        "reported_val_raw": params.get("avg_gap"),
        "reported_test": summary.get("test_avg_gap"),
    }
    for split, data in splits.items():
        m = run_validation(agent, env_cls(data, mask_option, sel_k), data, print_fn=quiet)
        row[f"{split}_gap"] = m["avg_gap"]
        row[f"{split}_q80"] = m["q80_gap"]
    row["reproduces"] = bool(row["reported_test"] is not None and abs(row["test_gap"] - row["reported_test"]) < 1e-6)
    rows.append(row)
    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
    print(f"[RESCORE] {run:40s} val={row['val_gap']:.4f} (reported raw {fmt(row['reported_val_raw'])}) | "
          f"test={row['test_gap']:.4f} (reported {fmt(row['reported_test'])}) | reproduces={row['reproduces']}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "rescore_summary.csv"), index=False)
json.dump(rows, open(os.path.join(OUT_DIR, "rescore.json"), "w"), indent=2, default=str)
print(f"[RESCORE] wrote {OUT_DIR}/rescore_summary.csv")
