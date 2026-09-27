"""Run every metric for one or more saved models and write ONE report file.

For every model and every instance folder the policy is rolled out greedily and each
instance gets: makespan, relative error (vs CP-SAT and vs lower bound), scheduling score,
constraint violation rate per step / feasibility, inference time and memory, and the
representation metrics (expressiveness, heterophily). Then, per (model, folder):
descriptive statistics with 95% CIs; across models: paired Wilcoxon tests and effect
sizes; across folders: generalization to unseen sizes. See src/metrics/ for definitions.

Output (default results/metrics/):
    <run-name>.json  everything, incl. per-instance rows, git branch/commit and the exact
                     arguments - the input of compare_metrics.py
    <run-name>.md    readable summary tables

Usage:
    python run_metrics.py --run-name main_ojm --models-file candidate_models/model_params.json \
        --models 565547736080381.pth --folders val/instances data/test/medium data/test/large
    python run_metrics.py --folders data/test/large --solve-missing-references --solve-workers 4
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import combinations

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from src.metrics.evaluate import evaluate_agent, summarize
from src.blocking_config import BLOCKING_CONFIG
from src.metrics.references import load_instances, solve_missing_references
from src.metrics.report import HEADLINE_HEADER, fmt, headline_row, markdown_table
from src.metrics.statistics import compare, required_sample_size
from src.train import BLOCKING_REPRESENTATIONS, _resolve_representation_modules


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run every metric for saved models and write one report file.")
    p.add_argument("--run-name", default="metrics", help="Report file name (without extension).")
    p.add_argument("--label", default=None,
                   help="Short name for this run in comparisons (default: '<git branch>:<run-name>').")
    p.add_argument("--models-file", default="models/model_params.json")
    p.add_argument("--models", nargs="*", default=None,
                   help="Model file names to evaluate (default: every model in --models-file).")
    p.add_argument("--folders", nargs="+", default=["val/instances"],
                   help="Instance folders (.fjs; references from the matching solutions folder, see "
                        "src/metrics/references.py) or JSON splits with references, e.g. "
                        "val/test_dataset_blocking.json for the blocking models.")
    p.add_argument("--in-distribution-folder", default=None,
                   help="Folder whose sizes match training, the baseline for generalization (default: first folder).")
    p.add_argument("--train-jobs", nargs=2, type=int, default=None,
                   help="Training range of jobs (min max). Default: 8 10, or src/blocking_config.py's "
                        "range for the blocking representations.")
    p.add_argument("--train-machines", nargs=2, type=int, default=None,
                   help="Training range of machines (min max). Default: 5 10, or the blocking config's.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate at most this many instances per folder.")
    p.add_argument("--no-representation", action="store_true",
                   help="Skip expressiveness / heterophily (halves the evaluation time).")
    p.add_argument("--solve-missing-references", action="store_true",
                   help="Solve instances without a reference with CP-SAT (15 s each) before evaluating.")
    p.add_argument("--solve-workers", type=int, default=1)
    p.add_argument("--output-dir", default="results/metrics")
    return p.parse_args(argv)


def git_info():
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def load_agent(param, models_dir, instances):
    rep, EnvClass, BOPOClass = _resolve_representation_modules(param.get("representation", "oo"))
    jm_design = param.get("jm_design", "baseline")
    jm_kwargs = {} if jm_design == "baseline" else {"jm_design": jm_design}
    # blocking models are evaluated with the buffer capacities they were trained with
    cap_kwargs = {k: param[k] for k in ("in_cap", "out_cap") if k in param}
    env = EnvClass(instances, param["mask_option"], param["sel_k"], **jm_kwargs, **cap_kwargs)
    metadata = env.reset().metadata()
    agent = BOPOClass(0.001, env, metadata, param["hidden_channels"], param["num_layers"], param["heads"],
                      gnn_type=param.get("gnn_type", "gat"), **jm_kwargs)
    agent.load(os.path.join(models_dir, param["name"]))
    agent.policy.eval()
    return rep, env, agent


def training_range(rep, args):
    """(jobs, machines) ranges the model was trained on."""
    if rep in BLOCKING_REPRESENTATIONS:
        gen = BLOCKING_CONFIG["generator"]
        default = (list(gen["range_jobs"]), list(gen["range_machines"]))
    else:
        default = ([8, 10], [5, 10])
    return args.train_jobs or default[0], args.train_machines or default[1]


def in_training_range(row, ranges):
    jobs, machines = ranges
    return jobs[0] <= row["num_jobs"] <= jobs[1] and machines[0] <= row["num_machines"] <= machines[1]


def _unique_path(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"


def run(args):
    with open(args.models_file) as f:
        params = json.load(f)
    if args.models:
        params = [p for p in params if p["name"] in set(args.models)]
        missing = set(args.models) - {p["name"] for p in params}
        if missing:
            raise ValueError(f"Not in {args.models_file}: {sorted(missing)}")
    if not params:
        raise ValueError(f"No models to evaluate in {args.models_file}")
    models_dir = os.path.dirname(args.models_file) or "."

    folders = {}
    for folder in args.folders:
        if args.solve_missing_references and os.path.isdir(folder):
            solve_missing_references(folder, args.solve_workers)
        instances = load_instances(folder)[: args.limit]
        print(f"[METRICS] {folder}: {len(instances)} instance(s), "
              f"{sum(i['score'] is not None for i in instances)} with a CP-SAT reference")
        folders[folder] = instances

    results = []
    with torch.no_grad():
        for param in params:
            for folder, instances in folders.items():
                if not instances:
                    continue
                rep, env, agent = load_agent(param, models_dir, instances)
                print(f"[METRICS] {param['name']} ({rep}) on {folder} ...")
                rows, repr_summary = evaluate_agent(agent, env, instances, rep,
                                                    representation_metrics=not args.no_representation)
                ranges = training_range(rep, args)
                for r in rows:
                    r["in_training_range"] = in_training_range(r, ranges)
                summary = summarize(rows, repr_summary, agent.policy.actor)
                summary["in_training_range_fraction"] = sum(r["in_training_range"] for r in rows) / len(rows)
                m = summary["metrics"]
                print(f"[METRICS]   makespan={fmt(m['makespan'].get('mean'))} | RE={fmt(m['relative_error'].get('mean'))} | "
                      f"RE_lb={fmt(m['relative_error_lb'].get('mean'))} | score={fmt(m['scheduling_score'].get('mean'))} | "
                      f"feasible={fmt(summary['feasibility_rate'])} | ms/decision={fmt(m['time_per_decision_ms'].get('mean'))}")
                results.append({"model": param["name"], "representation": rep, "folder": folder,
                                "summary": summary, "rows": rows})

    by = {(r["model"], r["folder"]): r for r in results}

    # paired comparisons between the models of this run, per folder (same instances)
    comparisons = []
    for folder in folders:
        for a, b in combinations([p["name"] for p in params], 2):
            ra, rb = by.get((a, folder)), by.get((b, folder))
            if not ra or not rb:
                continue
            for metric in ("makespan", "relative_error_lb", "scheduling_score"):
                c = compare([r[metric] for r in ra["rows"]], [r[metric] for r in rb["rows"]])
                c.update({"folder": folder, "model_a": a, "model_b": b, "metric": metric,
                          "n_needed_for_observed_effect": required_sample_size(c.get("cohens_d", 0))})
                comparisons.append(c)

    # generalization to unseen sizes, relative to the in-distribution folder
    base_folder = args.in_distribution_folder or args.folders[0]
    generalization = []
    for r in results:
        m = r["summary"]["metrics"]
        base = by.get((r["model"], base_folder))
        base_re = base["summary"]["metrics"]["relative_error_lb"].get("mean") if base else None
        re_lb = m["relative_error_lb"].get("mean")
        generalization.append({
            "model": r["model"], "representation": r["representation"], "folder": r["folder"],
            "size_range": r["summary"]["size_range"],
            "in_training_range_fraction": r["summary"]["in_training_range_fraction"],
            "relative_error": m["relative_error"].get("mean"),
            "relative_error_lb": re_lb,
            "scheduling_score": m["scheduling_score"].get("mean"),
            "feasibility_rate": r["summary"]["feasibility_rate"],
            # extra relative error (vs lower bound) w.r.t. the training-size folder
            "generalization_gap": None if base_re is None or re_lb is None else re_lb - base_re,
            "baseline_folder": base_folder,
        })

    git = git_info()
    return {
        "meta": {
            "run_name": args.run_name,
            "label": args.label or f"{git['branch']}:{args.run_name}",
            "created": datetime.now().isoformat(timespec="seconds"),
            "git": git,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "args": vars(args),
        },
        "models": {p["name"]: p for p in params},
        "results": results,
        "comparisons": comparisons,
        "generalization": generalization,
    }


def to_markdown(report):
    meta = report["meta"]
    git = meta["git"]
    out = [f"# Metrics report: {meta['label']}", "",
           f"- created: {meta['created']}",
           f"- git: {git['branch']} @ {git['commit']}{' (uncommitted changes)' if git['dirty'] else ''}",
           f"- device: {meta['device']}",
           f"- models file: {meta['args']['models_file']}", ""]

    for folder in dict.fromkeys(r["folder"] for r in report["results"]):
        rows = [[r["model"], r["representation"], r["summary"]["n_instances"], *headline_row(r["summary"])]
                for r in report["results"] if r["folder"] == folder]
        out += [f"## {folder}", "", "Mean ± 95% CI half-width.", "",
                markdown_table(["Model", "Repr.", "n", *HEADLINE_HEADER], rows), ""]

    if report["generalization"]:
        out += ["## Generalization to unseen sizes", "",
                f"Gap = RE (LB) on the folder minus RE (LB) on `{report['generalization'][0]['baseline_folder']}`.", "",
                markdown_table(
                    ["Model", "Folder", "Jobs", "Machines", "In training range", "RE (CP-SAT)", "RE (LB)", "Score", "Gap"],
                    [[g["model"], g["folder"], "-".join(map(str, g["size_range"]["num_jobs"])),
                      "-".join(map(str, g["size_range"]["num_machines"])), fmt(g["in_training_range_fraction"], 2),
                      fmt(g["relative_error"]), fmt(g["relative_error_lb"]), fmt(g["scheduling_score"]),
                      fmt(g["generalization_gap"])] for g in report["generalization"]]), ""]

    if report["comparisons"]:
        out += ["## Paired comparisons between models", "",
                "Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.", "",
                markdown_table(
                    ["Folder", "Metric", "Model A", "Model B", "n", "Mean diff", "p", "Significant", "Cohen's d",
                     "Rank-biserial", "n needed"],
                    [[c["folder"], c["metric"], c["model_a"], c["model_b"], c["n"], fmt(c.get("mean_diff")),
                      fmt(c.get("wilcoxon_p")), fmt(c.get("significant")), fmt(c.get("cohens_d")),
                      fmt(c.get("rank_biserial")), fmt(c.get("n_needed_for_observed_effect"))]
                     for c in report["comparisons"]]), ""]
    return "\n".join(out)


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = _unique_path(os.path.join(args.output_dir, f"{args.run_name}.json"))
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1)
    md_path = os.path.splitext(json_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(report))
    print(f"[METRICS] Report: {json_path}")
    print(f"[METRICS] Summary: {md_path}")
    return json_path


if __name__ == "__main__":
    main()
