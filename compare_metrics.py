"""Compare several run_metrics.py reports (other models, representations, runs or branches)
and write ONE comparison file.

Every (report, model) pair is a "system". For each instance folder that at least two
systems were evaluated on:
  - a headline table (mean ± 95% CI) of every system,
  - the best system per metric,
  - pairwise statistical tests per metric: paired Wilcoxon + effect sizes when the two
    systems share instances (same instance names), otherwise Mann-Whitney U + Cliff's delta
    (e.g. two constraint branches evaluated on different instance sets).
Plus the generalization rows of every report side by side.

Folders are matched by path. When the same role is played by differently named folders
(e.g. val/instances on main vs val_blocking/instances on a branch), map them to one name
with --rename-folder OLD=NEW, or use --pool-folders to compare over all instances at once.

Usage:
    python compare_metrics.py results/metrics/main_oo.json results/metrics/main_ojm.json
    python compare_metrics.py a.json b.json --labels main blocking --output results/metrics/main_vs_blocking
Output: <output>.json and <output>.md
"""
import argparse
import json
import os
import sys
from datetime import datetime
from itertools import combinations

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.metrics.evaluate import SUMMARY_METRICS
from src.metrics.report import BETTER, HEADLINE_HEADER, fmt, headline_row, markdown_table
from src.metrics.statistics import compare, compare_unpaired, describe

TEST_METRICS = ("makespan", "relative_error", "relative_error_lb", "scheduling_score", "feasible",
                "violation_rate_per_step", "time_per_decision_ms", "gpu_peak_mb")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare several run_metrics.py reports.")
    p.add_argument("reports", nargs="+", help="Report JSON files written by run_metrics.py.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="One label per report (default: the label stored in each report).")
    p.add_argument("--metrics", nargs="*", default=list(TEST_METRICS), help="Metrics to test pairwise.")
    p.add_argument("--rename-folder", nargs="*", default=[], metavar="OLD=NEW",
                   help="Treat folder OLD as NEW when matching folders across reports.")
    p.add_argument("--pool-folders", action="store_true",
                   help="Ignore folders: compare every system over all of its instances.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", default=None,
                   help="Output path without extension (default: results/metrics/comparison_<timestamp>).")
    return p.parse_args(argv)


def load_systems(args):
    if args.labels and len(args.labels) != len(args.reports):
        raise ValueError("--labels needs exactly one label per report")
    renames = dict(r.split("=", 1) for r in args.rename_folder)
    systems = []
    for k, path in enumerate(args.reports):
        with open(path) as f:
            report = json.load(f)
        label = args.labels[k] if args.labels else report["meta"]["label"]
        models = list(dict.fromkeys(r["model"] for r in report["results"]))
        for model in models:
            name = label if len(models) == 1 else f"{label}:{model}"
            results = {}
            for r in report["results"]:
                if r["model"] != model:
                    continue
                folder = "all" if args.pool_folders else renames.get(r["folder"], r["folder"])
                results.setdefault(folder, []).extend(r["rows"])
            git = report["meta"]["git"]
            systems.append({
                "name": name, "report": path, "model": model,
                "representation": next(r["representation"] for r in report["results"] if r["model"] == model),
                "git": f"{git['branch']} @ {git['commit']}{' (dirty)' if git['dirty'] else ''}",
                "created": report["meta"]["created"],
                "rows": results,
                "generalization": [g for g in report.get("generalization", []) if g["model"] == model],
            })
    names = [s["name"] for s in systems]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate system names {names}; pass distinct --labels")
    return systems


def folder_summary(rows):
    """Recompute the headline summary from rows (so pooled / renamed folders work too)."""
    return {"n_instances": len(rows),
            "feasibility_rate": sum(r["feasible"] for r in rows) / len(rows) if rows else None,
            "metrics": {m: describe([r.get(m) for r in rows]) for m in SUMMARY_METRICS}}


def _values(rows, metric):
    return [None if r.get(metric) is None else float(r[metric]) for r in rows]


def pairwise(a, b, rows_a, rows_b, metric, alpha):
    by_a = {r["name"]: r for r in rows_a}
    by_b = {r["name"]: r for r in rows_b}
    common = [n for n in by_a if n in by_b]
    if len(common) >= 2:
        res = compare(_values([by_a[n] for n in common], metric), _values([by_b[n] for n in common], metric), alpha)
        res.update(test="wilcoxon (paired)", p=res.get("wilcoxon_p"), effect=res.get("rank_biserial"),
                   effect_name="rank-biserial")
    else:
        res = compare_unpaired(_values(rows_a, metric), _values(rows_b, metric), alpha)
        res.update(test="mann-whitney (unpaired)", p=res.get("mannwhitney_p"), effect=res.get("cliffs_delta"),
                   effect_name="cliffs delta")
    better = BETTER.get(metric)
    winner = None
    if res.get("significant") and better and res.get("mean_diff"):
        a_lower = res["mean_diff"] < 0
        winner = a if (a_lower == (better == "lower")) else b
    res.update(system_a=a, system_b=b, metric=metric, better=better, winner=winner)
    return res


def compare_systems(systems, metrics, alpha):
    folders = {}
    for s in systems:
        for folder in s["rows"]:
            folders.setdefault(folder, []).append(s["name"])
    by_name = {s["name"]: s for s in systems}
    out = []
    for folder, names in folders.items():
        if len(names) < 2:
            continue
        summaries = {n: folder_summary(by_name[n]["rows"][folder]) for n in names}
        best = {}
        for m in metrics:
            better = BETTER.get(m)
            means = {n: summaries[n]["metrics"][m].get("mean") if m in summaries[n]["metrics"]
                     else summaries[n]["feasibility_rate"] for n in names}
            means = {n: v for n, v in means.items() if v is not None}
            if better and means:
                target = (min if better == "lower" else max)(means.values())
                winners = [n for n, v in means.items() if v == target]
                best[m] = winners[0] if len(winners) == 1 else f"tie ({len(winners)} systems)"
        tests = [pairwise(a, b, by_name[a]["rows"][folder], by_name[b]["rows"][folder], m, alpha)
                 for a, b in combinations(names, 2) for m in metrics]
        out.append({"folder": folder, "systems": names, "summaries": summaries, "best": best, "tests": tests})
    return out


def to_markdown(systems, folders, alpha):
    out = ["# Metrics comparison", "", f"- created: {datetime.now().isoformat(timespec='seconds')}", "",
           "## Systems", "",
           markdown_table(["System", "Model", "Repr.", "Git", "Report created", "Report file"],
                          [[s["name"], s["model"], s["representation"], s["git"], s["created"], s["report"]]
                           for s in systems]), ""]
    for f in folders:
        out += [f"## {f['folder']}", "", "Mean ± 95% CI half-width.", "",
                markdown_table(["System", "n", *HEADLINE_HEADER],
                               [[n, f["summaries"][n]["n_instances"], *headline_row(f["summaries"][n])]
                                for n in f["systems"]]), ""]
        if f["best"]:
            out += ["Best mean per metric: " + ", ".join(f"{m} → **{n}**" for m, n in f["best"].items()), ""]
        out += [f"Pairwise tests (alpha = {alpha}); winner only when significant.", "",
                markdown_table(["Metric", "A", "B", "Test", "n", "Mean diff (A−B)", "p", "Effect", "Winner"],
                               [[t["metric"], t["system_a"], t["system_b"], t["test"],
                                 t.get("n", f"{t.get('n_a')}/{t.get('n_b')}"), fmt(t.get("mean_diff")),
                                 fmt(t.get("p")), f"{fmt(t.get('effect'))} ({t['effect_name']})",
                                 t["winner"] or "-"] for t in f["tests"]]), ""]
    gen = [(s["name"], g) for s in systems for g in s["generalization"]]
    if gen:
        out += ["## Generalization to unseen sizes", "",
                "Gap = RE (LB) on the folder minus RE (LB) on each report's in-distribution folder.", "",
                markdown_table(["System", "Folder", "In training range", "RE (CP-SAT)", "RE (LB)", "Score",
                                "Feasible", "Gap"],
                               [[n, g["folder"], fmt(g["in_training_range_fraction"], 2), fmt(g["relative_error"]),
                                 fmt(g["relative_error_lb"]), fmt(g["scheduling_score"]),
                                 fmt(g["feasibility_rate"]), fmt(g["generalization_gap"])] for n, g in gen]), ""]
    return "\n".join(out)


def main(argv=None):
    args = parse_args(argv)
    systems = load_systems(args)
    if len(systems) < 2:
        raise ValueError("Need at least two systems (models) across the given reports to compare")
    folders = compare_systems(systems, args.metrics, args.alpha)
    if not folders:
        print("[COMPARE] No folder was evaluated by two or more systems - use --rename-folder or --pool-folders.")

    output = args.output or os.path.join("results", "metrics",
                                         f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "systems": [{k: v for k, v in s.items() if k != "rows"} for s in systems],
        "folders": folders,
    }
    with open(output + ".json", "w") as f:
        json.dump(result, f, indent=1)
    with open(output + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(systems, folders, args.alpha))
    for f_ in folders:
        print(f"[COMPARE] {f_['folder']}: best " + ", ".join(f"{m}={n}" for m, n in f_["best"].items()))
    print(f"[COMPARE] Comparison: {output}.json")
    print(f"[COMPARE] Summary: {output}.md")
    return output


if __name__ == "__main__":
    main()
