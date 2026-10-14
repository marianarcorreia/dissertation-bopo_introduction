"""Compile finished runs into one Excel workbook, from this folder or from every git branch.

A run counts as done when its results/<run>/ folder has a run_summary.json.
Settings and convergence metrics come from compare_runs.py, so the numbers
match the comparison plots.

Sheets
------
    Runs        one row per run: branch, settings, training info, val/test
                gaps, convergence metrics, plus any other config field the
                run saved (so branch-specific options show up too)
    Test gaps   one row per test instance, one column per tested run
                (from test_metrics.json)

With --all-branches (or --branches) the runs are read straight from the git
commits of each branch, so nothing needs to be checked out. Only committed
runs are seen there. A run that is identical on several branches (e.g. merged
from main) appears once, with every branch listed.

Usage
-----
    # finished runs in this folder -> results/runs_summary.xlsx
    python compile_runs.py

    # committed runs on every local branch
    python compile_runs.py --all-branches

    # some branches, some runs, custom output file
    python compile_runs.py --branches main batching_wait --runs "gat_*" --out results/gat.xlsx

Needs openpyxl (pip install openpyxl) and git on the PATH for branch mode.
"""
import argparse
import hashlib
import os
import subprocess
import tempfile

import pandas as pd

import compare_runs
from compare_runs import (SETTINGS, analyse_run, load_json, select_runs,
                          tidy_settings)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Files analyse_run() and the test sheet read from each run folder
RUN_FILES = ["run_summary.json", "validation_history.json",
             "episode_metrics.json", "test_metrics.json"]

COLUMN_ORDER = (
    ["branch", "run_name"] + SETTINGS
    + ["exclude_greedy_from_loss", "jm_design", "max_episodes",
       "episodes_completed", "updates_completed", "actor_params",
       "total_runtime_min", "best_val_gap", "best_validation_q80_gap",
       "final_val_gap", "test_gap", "test_std", "test_q80_gap",
       "generalization_gap", "lcs_first200_per_ep", "lcs_full_per_ep",
       "late_slope_per100ep", "late_tail_relrange", "actor_loss_first10",
       "actor_loss_last10", "entropy_norm_first10", "entropy_norm_last10",
       "converged", "best_difference", "best_model_path"]
)
# run_summary.json keys already in the sheet under another name, or not useful
SKIP_SUMMARY_KEYS = {"run_dir", "actor_param_count", "best_validation_avg_gap",
                     "test_avg_gap", "test_std_gap", "total_runtime_sec"}


# ---------------------------------------------------------------------------
# Reading runs from git branches
# ---------------------------------------------------------------------------
def git(args, stdin=None):
    return subprocess.run(["git"] + args, cwd=ROOT, input=stdin,
                          capture_output=True, check=True).stdout


def local_branches():
    out = git(["for-each-ref", "--format=%(refname:short)", "refs/heads"])
    return out.decode().split()


def export_branch(branch, dest):
    """Write the RUN_FILES of every run committed on `branch` to dest/<run>/."""
    listing = git(["ls-tree", "-r", "-z", "--name-only", branch, "--", "results"])
    paths = []
    for path in listing.split(b"\0"):
        parts = path.split(b"/")
        if len(parts) == 3 and parts[2].decode() in RUN_FILES:
            paths.append(path)
    if not paths:
        return
    # One `git cat-file --batch` call for all files instead of one `git show` each
    specs = b"".join(branch.encode() + b":" + p + b"\n" for p in paths)
    out = git(["cat-file", "--batch"], stdin=specs)
    pos = 0
    for path in paths:
        end = out.index(b"\n", pos)
        header, pos = out[pos:end], end + 1
        if header.endswith(b" missing"):
            continue
        size = int(header.split()[2])
        content, pos = out[pos:pos + size], pos + size + 1
        _, run, fname = path.decode("utf-8").split("/")
        os.makedirs(os.path.join(dest, run), exist_ok=True)
        with open(os.path.join(dest, run, fname), "wb") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------
def collect_runs(results_dir, branch, patterns, runs, test_gaps):
    """Add the runs in results_dir to `runs` (keyed by summary content) and `test_gaps`."""
    compare_runs.RESULTS = results_dir  # compare_runs reads run folders from here
    if not compare_runs.all_run_names():
        return
    for name, summary, settings in select_runs(patterns, {}):
        summary_path = os.path.join(results_dir, name, "run_summary.json")
        with open(summary_path, "rb") as f:
            key = (name, hashlib.sha1(f.read()).hexdigest())
        if key in runs:
            runs[key]["branch"] += f", {branch}"
            continue

        record, _val_df, _ep_df = analyse_run(name, summary, settings)
        for k, v in summary.items():
            if (k not in record and k not in SKIP_SUMMARY_KEYS
                    and not k.endswith("_file") and not isinstance(v, (dict, list))):
                record[k] = v
        runtime = summary.get("total_runtime_sec")
        record["total_runtime_min"] = runtime / 60 if runtime is not None else None
        record["branch"] = branch
        runs[key] = record

        test_path = os.path.join(results_dir, name, "test_metrics.json")
        if os.path.exists(test_path):
            test = load_json(test_path)
            label = name if branch == "(this folder)" else f"{branch}/{name}"
            test_gaps[label] = pd.Series(test["all_gaps"], index=test["instances"])


def build_tables(runs, test_gaps):
    runs_df = tidy_settings(pd.DataFrame(list(runs.values())))
    runs_df = runs_df[[c for c in COLUMN_ORDER if c in runs_df]
                      + [c for c in runs_df if c not in COLUMN_ORDER]]
    runs_df = runs_df.sort_values(["branch", "run_name"])
    gaps_df = pd.DataFrame(test_gaps).sort_index()
    gaps_df.index.name = "instance"
    return runs_df, gaps_df


def format_sheet(ws):
    """Freeze the header row, add filters and size columns to their content."""
    ws.freeze_panes = "C2" if ws.title == "Runs" else "B2"
    ws.auto_filter.ref = ws.dimensions
    for col in ws.columns:
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 50)
        for c in col[1:]:
            if isinstance(c.value, float):
                c.number_format = "0.0000"


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", default=["*"],
                    help="run folder names or glob patterns (default: all)")
    ap.add_argument("--all-branches", action="store_true",
                    help="read committed runs from every local git branch")
    ap.add_argument("--branches", nargs="+",
                    help="read committed runs from these git branches")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs_summary.xlsx"),
                    help="output .xlsx path")
    args = ap.parse_args()

    runs, test_gaps = {}, {}
    branches = local_branches() if args.all_branches else args.branches
    if branches:
        with tempfile.TemporaryDirectory() as tmp:
            for branch in branches:
                dest = os.path.join(tmp, branch)
                export_branch(branch, dest)
                before = len(runs)
                collect_runs(dest, branch, args.runs, runs, test_gaps)
                print(f"{branch}: {len(runs) - before} new runs")
    else:
        collect_runs(os.path.join(ROOT, "results"), "(this folder)", args.runs, runs, test_gaps)

    if not runs:
        raise SystemExit(f"No finished runs match {args.runs}")
    runs_df, gaps_df = build_tables(runs, test_gaps)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        runs_df.to_excel(writer, sheet_name="Runs", index=False)
        format_sheet(writer.sheets["Runs"])
        if not gaps_df.empty:
            gaps_df.to_excel(writer, sheet_name="Test gaps")
            format_sheet(writer.sheets["Test gaps"])

    print(f"Compiled {len(runs_df)} runs ({gaps_df.shape[1]} with test results) into {args.out}")


if __name__ == "__main__":
    main()
