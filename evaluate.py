"""Full metric evaluation of saved models.

For every model in --models-file and every instance folder in --folders, runs the policy
greedily and records, per instance:
  schedule quality       makespan, relative error (vs CP-SAT reference and vs lower bound),
                         scheduling score
  constraint quality     violation rate per step, feasibility, violations per constraint
  efficiency             inference time (total / per decision), memory (GPU peak, process RSS),
                         model parameters and size
  representation quality expressiveness and heterophily (skip with --no-representation)
and per (model, folder): descriptive statistics with 95% CIs (src/metrics/statistics.py).
Across models: paired Wilcoxon tests and effect sizes on the same instances. Across
folders: generalization to unseen sizes, relative to --in-distribution-folder.

Usage:
    python evaluate.py --models-file candidate_models/model_params.json --models 123.pth 456.pth
    python evaluate.py --folders val/instances data/test/medium data/test/large --limit 10
    python evaluate.py --folders data/test/large --solve-missing-references --solve-workers 4
Output: results/<run-name>/ - rows_<folder>.csv/json, summary.json, comparisons.json,
generalization.json
"""
import argparse
import json
import os
import sys
from itertools import combinations

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch

from src.metrics.evaluate import evaluate_agent, summarize
from src.metrics.references import load_folder, solve_missing_references
from src.metrics.statistics import compare, describe, required_sample_size
from src.train import _resolve_representation_modules
from src.utils import OutputManager


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved models with the full metric set.")
    p.add_argument("--run-name", default="evaluation")
    p.add_argument("--models-file", default="models/model_params.json")
    p.add_argument("--models", nargs="*", default=None,
                   help="Model file names to evaluate (default: every model in --models-file).")
    p.add_argument("--folders", nargs="+", default=["val/instances"],
                   help="Instance folders (.fjs). References are read from the matching solutions folder "
                        "(see src/metrics/references.py).")
    p.add_argument("--in-distribution-folder", default=None,
                   help="Folder whose sizes match training, the baseline for generalization (default: first folder).")
    p.add_argument("--train-jobs", nargs=2, type=int, default=[8, 10], help="Training range of jobs (min max).")
    p.add_argument("--train-machines", nargs=2, type=int, default=[5, 10], help="Training range of machines (min max).")
    p.add_argument("--limit", type=int, default=None, help="Evaluate at most this many instances per folder.")
    p.add_argument("--no-representation", action="store_true",
                   help="Skip expressiveness / heterophily (halves the evaluation time).")
    p.add_argument("--solve-missing-references", action="store_true",
                   help="Solve instances without a reference with CP-SAT (15 s each) before evaluating.")
    p.add_argument("--solve-workers", type=int, default=1)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def load_agent(param, models_dir, instances):
    rep = param.get("representation", "oo")
    rep, EnvClass, BOPOClass = _resolve_representation_modules(rep)
    jm_design = param.get("jm_design", "baseline")
    jm_kwargs = {} if jm_design == "baseline" else {"jm_design": jm_design}
    env = EnvClass(instances, param["mask_option"], param["sel_k"], **jm_kwargs)
    metadata = env.reset().metadata()
    agent = BOPOClass(0.001, env, metadata, param["hidden_channels"], param["num_layers"], param["heads"],
                      gnn_type=param.get("gnn_type", "gat"), **jm_kwargs)
    agent.load(os.path.join(models_dir, param["name"]))
    agent.policy.eval()
    return rep, env, agent


def in_training_range(row, args):
    return (args.train_jobs[0] <= row["num_jobs"] <= args.train_jobs[1]
            and args.train_machines[0] <= row["num_machines"] <= args.train_machines[1])


def _key(folder):
    return os.path.normpath(folder).replace(os.sep, "_").replace("/", "_")


def main():
    args = parse_args()
    out = OutputManager(output_dir=args.output_dir, run_name=args.run_name)
    print(f"[EVAL] Output folder: {out.run_dir}")

    with open(args.models_file) as f:
        params = json.load(f)
    if args.models:
        params = [p for p in params if p["name"] in set(args.models)]
        missing = set(args.models) - {p["name"] for p in params}
        if missing:
            raise ValueError(f"Not in {args.models_file}: {sorted(missing)}")
    models_dir = os.path.dirname(args.models_file) or "."

    folders = {}
    for folder in args.folders:
        if args.solve_missing_references:
            solve_missing_references(folder, args.solve_workers)
        instances = load_folder(folder)[: args.limit]
        n_ref = sum(i["score"] is not None for i in instances)
        print(f"[EVAL] {folder}: {len(instances)} instance(s), {n_ref} with a CP-SAT reference")
        folders[folder] = instances

    summary = {}   # model -> folder -> summary
    rows_by = {}   # (model, folder) -> rows
    with torch.no_grad():
        for param in params:
            name = param["name"]
            summary[name] = {"params": param}
            for folder, instances in folders.items():
                if not instances:
                    continue
                rep, env, agent = load_agent(param, models_dir, instances)
                print(f"[EVAL] {name} ({rep}) on {folder} ...")
                rows, repr_summary = evaluate_agent(agent, env, instances, rep,
                                                    representation_metrics=not args.no_representation)
                for r in rows:
                    r["model"] = name
                    r["representation"] = rep
                    r["folder"] = folder
                    r["in_training_range"] = in_training_range(r, args)
                s = summarize(rows, repr_summary, agent.policy.actor)
                s["in_training_range_fraction"] = sum(r["in_training_range"] for r in rows) / len(rows)
                summary[name][folder] = s
                rows_by[(name, folder)] = rows
                m = s["metrics"]
                print(f"[EVAL]   makespan={m['makespan']['mean']:.1f} | RE={m['relative_error'].get('mean')} | "
                      f"RE_lb={m['relative_error_lb']['mean']:.4f} | score={m['scheduling_score']['mean']:.4f} | "
                      f"feasible={s['feasibility_rate']:.2f} | viol/step={m['violation_rate_per_step']['mean']:.4f} | "
                      f"ms/decision={m['time_per_decision_ms']['mean']:.2f}")

                base = os.path.join(out.run_dir, f"rows_{_key(name)}_{_key(folder)}")
                out.save_json(rows, base + ".json")
                pd.json_normalize(rows).to_csv(base + ".csv", index=False)
    out.save_json(summary, os.path.join(out.run_dir, "summary.json"))

    # paired comparisons between models, per folder
    comparisons = []
    for folder in folders:
        for a, b in combinations([p["name"] for p in params], 2):
            ra, rb = rows_by.get((a, folder)), rows_by.get((b, folder))
            if not ra or not rb:
                continue
            for metric in ("makespan", "relative_error_lb", "scheduling_score"):
                c = compare([r[metric] for r in ra], [r[metric] for r in rb])
                c.update({"folder": folder, "model_a": a, "model_b": b, "metric": metric,
                          "n_needed_for_observed_effect": required_sample_size(c.get("cohens_d", 0))})
                comparisons.append(c)
    out.save_json(comparisons, os.path.join(out.run_dir, "comparisons.json"))

    # generalization to unseen sizes, relative to the in-distribution folder
    base_folder = args.in_distribution_folder or args.folders[0]
    generalization = []
    for param in params:
        name = param["name"]
        base_rows = rows_by.get((name, base_folder), [])
        base_re = describe([r["relative_error_lb"] for r in base_rows]).get("mean")
        for folder in folders:
            rows = rows_by.get((name, folder), [])
            if not rows:
                continue
            s = summary[name][folder]
            re_lb = s["metrics"]["relative_error_lb"].get("mean")
            generalization.append({
                "model": name,
                "folder": folder,
                "size_range": s["size_range"],
                "in_training_range_fraction": s["in_training_range_fraction"],
                "relative_error_lb": re_lb,
                "relative_error": s["metrics"]["relative_error"].get("mean"),
                "scheduling_score": s["metrics"]["scheduling_score"].get("mean"),
                "feasibility_rate": s["feasibility_rate"],
                # extra relative error (vs lower bound) on this folder w.r.t. training-size instances
                "generalization_gap": None if base_re is None or re_lb is None else re_lb - base_re,
                "baseline_folder": base_folder,
            })
    out.save_json(generalization, os.path.join(out.run_dir, "generalization.json"))
    if generalization:
        print(pd.DataFrame(generalization)[["model", "folder", "in_training_range_fraction", "relative_error_lb",
                                            "relative_error", "scheduling_score", "generalization_gap"]]
              .to_string(index=False))
    print(f"[EVAL] Done -> {out.run_dir}")


if __name__ == "__main__":
    main()
