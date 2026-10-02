"""Build one CSV per branch with every run's summary and test metrics, read from git."""
import csv
import json
import os
import subprocess
import sys

REPO = r"D:\Dissertação\dissertation-bopo_introduction"
OUT = os.path.join(REPO, "run_tables")
# local main is behind origin/main only in results, so main's table uses origin/main
BRANCH_REFS = {"main": "origin/main"}
CLASS_DIRS = ["Runs_usar", "Runs_a questionar", "Runs_nao usar", "Runs_incompletas", "Runs_teste"]
FIRST_COLS = [
    "branch", "location", "classification", "run_name", "representation", "gnn_type", "num_layers",
    "sel_k", "mask_option", "seed", "model_version", "hidden_channels", "heads", "jm_design", "logp_norm", "exclude_greedy_from_loss",
    "max_episodes", "episodes_completed", "updates_completed", "completed",
    "best_validation_avg_gap", "best_validation_q80_gap", "test_avg_gap", "test_std_gap", "test_q80_gap",
    "test_n_instances", "teacher_val_avg_gap", "teacher_val_q80_gap", "teacher_test_avg_gap",
    "teacher_test_q80_gap", "best_episode", "best_val_std_gap", "best_val_smoothed_avg_gap", "best_difference", "actor_param_count", "total_runtime_sec",
    "total_runtime_h", "best_model_path",
]


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, check=True).stdout


def read_blobs(ref, paths):
    """Read many files from a ref in one git cat-file --batch call."""
    data = "".join(f"{ref}:{p}\n" for p in paths).encode("utf-8")
    out = subprocess.run(["git", "-C", REPO, "cat-file", "--batch"], input=data,
                         capture_output=True, check=True).stdout
    res, pos = {}, 0
    for p in paths:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].split()
        if header[-1] == b"missing":
            pos = nl + 1
            continue
        size = int(header[2])
        res[p] = out[nl + 1:nl + 1 + size].decode("utf-8", "replace")
        pos = nl + 1 + size + 1
    return res


def classification(run_name):
    found = [d for d in CLASS_DIRS if os.path.isdir(os.path.join(REPO, d, run_name))]
    return ";".join(found)


def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        if k.startswith("plot_") or k.endswith("_file") or k in ("run_dir", "all_gaps", "instances"):
            continue
        key = prefix + k
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        elif isinstance(v, list):
            if v and all(isinstance(x, (int, float)) for x in v) and len(v) <= 10:
                flat[key] = ";".join(str(x) for x in v)
        else:
            flat[key] = v
    return flat


def build(branch):
    ref = BRANCH_REFS.get(branch, branch)
    files = git("ls-tree", "-r", "--name-only", ref).decode("utf-8").splitlines()
    summaries = [f for f in files if f.endswith("/run_summary.json")]
    tests = [f.rsplit("/", 1)[0] + "/test_metrics.json" for f in summaries]
    blobs = read_blobs(ref, summaries + [t for t in tests if t in set(files)] + ["candidate_models/model_params.json"])
    registry = {e["name"]: e for e in json.loads(blobs.get("candidate_models/model_params.json", "[]"))}
    rows = []
    for s, t in zip(summaries, tests):
        try:
            summ = json.loads(blobs[s])
        except (KeyError, json.JSONDecodeError):
            continue
        run_dir = s.rsplit("/", 1)[0]
        row = flatten(summ)
        if t in blobs:
            test = json.loads(blobs[t])
            row["test_n_instances"] = len(test.get("all_gaps", []))
            for k, v in flatten(test).items():
                tk = k if k.startswith("test_") or k == "best_model_path" else "test_" + k
                row.setdefault(tk, v)
        ckpt = registry.get(str(row.get("best_model_path", "")).replace("\\", "/").rsplit("/", 1)[-1])
        if ckpt:
            for k in ("hidden_channels", "heads", "gnn_type", "num_layers", "sel_k", "mask_option", "representation"):
                if k in ckpt:
                    row.setdefault(k, ckpt[k])
            for src, dst in (("episode", "best_episode"), ("std_gap", "best_val_std_gap"),
                             ("smoothed_avg_gap", "best_val_smoothed_avg_gap")):
                if src in ckpt:
                    row[dst] = ckpt[src]
        row["branch"] = branch
        row["location"] = run_dir
        row.setdefault("run_name", run_dir.rsplit("/", 1)[-1])
        row["classification"] = classification(row["run_name"])
        if "max_episodes" in row and "episodes_completed" in row:
            row["completed"] = row["episodes_completed"] >= row["max_episodes"]
        if isinstance(row.get("total_runtime_sec"), (int, float)):
            row["total_runtime_h"] = round(row["total_runtime_sec"] / 3600, 3)
        rows.append(row)
    rows.sort(key=lambda r: r["location"])
    cols = [c for c in FIRST_COLS if any(c in r for r in rows)]
    cols += sorted({k for r in rows for k in r} - set(cols))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{branch}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{branch} (from {ref}): {len(rows)} runs, {len(cols)} columns -> {path}")


if __name__ == "__main__":
    branches = sys.argv[1:] or git("for-each-ref", "--format=%(refname:short)", "refs/heads").decode().split()
    for b in branches:
        build(b)
