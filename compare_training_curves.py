"""Training curves of several runs on the same axes: one PNG per metric, one line per run.

Reads what src/train.py saves in every run folder: episode_metrics.json (one entry per BOPO
step) and validation_history.json (every validation). Per-step metrics are measured on a
different training instance each step, so each run is drawn as a faint raw line plus a moving
average; validation metrics are drawn as lines with markers. Colours follow the representation
(src/metrics/report.py), so they match the metrics report charts.

Checkpoint metrics: the metrics of src/metrics (relative error, score, feasibility, the
isomorphism / expressiveness metrics, heterophily, batching statistics, efficiency) are only
defined for a trained policy, so every checkpoint a run saved (one per validation improvement,
listed in sweep_logs/<run>.log) is evaluated with the same pipeline as run_metrics.py on
--instances, and each metric is drawn against the training step it was saved at, with its 95%
CI as a band. These are best-so-far models, so the points sit where validation improved. The
evaluation is cached in results/<run>/checkpoint_metrics.json (delete it to recompute).

Output: <out>/<metric>.png for every training metric and <out>/overview.png with all of them;
<out>/checkpoints/<metric>.png for every checkpoint metric and <out>/overview_checkpoints.png.

Usage:
    python compare_training_curves.py                      # the three GAT batching runs, seed 42
    python compare_training_curves.py --runs results/batching_gin_L3_ojmb_node_s42 \
        results/batching_gin_L3_ojmb_edge_s42 results/batching_gin_L3_ojmb_base_s42 \
        --out results/training_curves/batching_gin_L3_s42
    python compare_training_curves.py --no-checkpoints    # training curves only
"""
import argparse
import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics.report import (CHART_GROUPS, GRID, INK, INK_2, MUTED, REPRESENTATION_ORDER, SERIES_COLORS,
                                SURFACE, _value)

# (source, key, label, better). source: "episode" = every BOPO step, "validation" = every
# validation. The learning rate is left out: it follows the same schedule in every run.
METRICS = (
    ("validation", "avg_gap", "Validation gap to CP-SAT (mean)", "lower"),
    ("validation", "smoothed_avg_gap", "Validation gap, smoothed (checkpoint criterion)", "lower"),
    ("validation", "q80_gap", "Validation gap, 80th percentile", "lower"),
    ("validation", "std_gap", "Validation gap, standard deviation", "lower"),
    ("episode", "makespan", "Training makespan (best of the group)", "lower"),
    ("episode", "actor_loss", "Actor loss (BOPO)", "lower"),
    ("episode", "action_entropy", "Action entropy", None),
    ("episode", "action_entropy_max_entropy_normalized", "Action entropy / maximum", None),
    ("episode", "action_entropy_effective_action_count", "Effective number of actions", None),
    ("episode", "sro_margin_mean", "Preference margin (mean)", None),
    ("episode", "sro_saturated_frac", "Saturated preference pairs", None),
    ("episode", "greedy_is_best", "Greedy rollout is the best", "higher"),
    ("episode", "actor_grad_norm", "Actor gradient norm", None),
    ("episode", "steps", "Decisions per episode", None),
    ("episode", "update_duration_sec", "Update time (s)", "lower"),
)
DEFAULT_RUNS = [f"results/batching_gat_L2_{rep}_s42" for rep in ("ojmb_node", "ojmb_edge", "ojmb_base")]
LABELS = {"ojmb_node": "Family node", "ojmb_edge": "Batch edge", "ojmb_base": "Baseline (no batching info)"}


def load_run(path):
    with open(os.path.join(path, "run_summary.json")) as f:
        rep = json.load(f).get("representation", os.path.basename(path))
    data = {}
    for source, file in (("episode", "episode_metrics.json"), ("validation", "validation_history.json")):
        with open(os.path.join(path, file)) as f:
            data[source] = json.load(f)
    return rep, data


def series(data, source, key):
    pts = [(e["episode"], e[key]) for e in data[source] if e.get(key) is not None]
    if not pts:
        return None, None
    x, y = zip(*pts)
    return np.array(x, dtype=float), np.array(y, dtype=float)


def moving_average(y, window):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    head = np.cumsum(y[:window - 1]) / np.arange(1, window)  # growing window at the start
    return np.concatenate([head, np.convolve(y, kernel, mode="valid")])


def style(ax, title, better):
    arrow = {"lower": " (lower better)", "higher": " (higher better)"}.get(better, "")
    ax.set_facecolor(SURFACE)
    ax.set_title(title + arrow, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Training step (episode)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def draw(ax, runs, source, key, window):
    drawn = False
    for label, color, data in runs:
        x, y = series(data, source, key)
        if x is None:
            continue
        drawn = True
        if source == "episode":
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.22)
            ax.plot(x, moving_average(y, window), color=color, linewidth=2, label=label)
        else:
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    return drawn


def saved_checkpoints(run_path, log_dir):
    """Names of the checkpoints a run saved, in order, from its training log."""
    log = os.path.join(log_dir, os.path.basename(os.path.normpath(run_path)) + ".log")
    with open(log, encoding="utf-8", errors="replace") as f:
        return re.findall(r"saving candidate: (\d+\.pth)", f.read())


def checkpoint_metrics(run_path, args):
    """[{episode, name, summary}] for every checkpoint of the run, evaluated with the
    run_metrics.py pipeline on args.instances; cached in <run>/checkpoint_metrics.json."""
    cache_path = os.path.join(run_path, "checkpoint_metrics.json")
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("instances") != args.instances:
            cache = {}
    entries = cache.get("checkpoints", {})
    names = saved_checkpoints(run_path, args.log_dir)
    todo = [n for n in names if n not in entries]
    if todo:
        import torch
        from run_metrics import load_agent
        from src.metrics.evaluate import evaluate_agent, summarize
        from src.metrics.references import load_instances
        with open(args.models_file) as f:
            params = {p["name"]: p for p in json.load(f)}
        instances = load_instances(args.instances)
        models_dir = os.path.dirname(args.models_file) or "."
        for n in todo:
            if n not in params or not os.path.isfile(os.path.join(models_dir, n)):
                print(f"  {n}: checkpoint or its parameters missing, skipped")
                continue
            with torch.no_grad():
                rep, env, agent = load_agent(params[n], models_dir, instances)
                rows, repr_summary = evaluate_agent(agent, env, instances, rep)
                summary = summarize(rows, repr_summary, agent.policy.actor)
            entries[n] = {"episode": params[n]["episode"], "summary": summary}
            print(f"  {os.path.basename(run_path)} step {params[n]['episode']}: {n} "
                  f"RE={summary['metrics']['relative_error'].get('mean')}", flush=True)
            with open(cache_path, "w") as f:  # save after every checkpoint: safe to interrupt
                json.dump({"instances": args.instances, "checkpoints": entries}, f)
    return sorted((dict(e, name=n) for n, e in entries.items() if n in names), key=lambda e: e["episode"])


def draw_checkpoints(ax, runs, key):
    drawn = False
    for label, color, _, ckpts in runs:
        pts = [(c["episode"], *_value(c["summary"], key)) for c in ckpts]
        pts = [(x, m, h) for x, m, h in pts if m is not None]
        if not pts:
            continue
        drawn = True
        x, m, h = (np.array(v, dtype=float) for v in zip(*[(x, m, 0.0 if h is None else h) for x, m, h in pts]))
        ax.fill_between(x, m - h, m + h, color=color, alpha=0.12, linewidth=0, step=None)
        ax.plot(x, m, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    return drawn


def plot_checkpoint_metrics(runs, out):
    """One PNG per src/metrics metric (checkpoint step on x) plus an overview grid."""
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    note = "one point per saved checkpoint (validation improved); band: 95% CI over the instances"
    written = []
    for group, metrics in CHART_GROUPS:
        for key, label, better in metrics:
            values = [_value(c["summary"], key)[0] for _, _, _, ck in runs for c in ck]
            values = [v for v in values if v is not None]
            if not values:
                continue
            if len(set(values)) == 1:
                print(f"skip checkpoint metric {key} (constant {values[0]:g} for every checkpoint)")
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
            draw_checkpoints(ax, runs, key)
            style(ax, f"{group}: {label}", better)
            ax.set_xlabel("Training step the checkpoint was saved at", fontsize=8, color=MUTED)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
            ax.text(1.0, -0.19, note, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_2)
            path = os.path.join(out, "checkpoints", f"{key}.png")
            fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
            plt.close(fig)
            written.append((group, key, label, better))
            print(path)
    if not written:
        return
    ncols = 3
    nrows = math.ceil(len(written) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows + 0.8), facecolor=SURFACE,
                             squeeze=False)
    for ax, (group, key, label, better) in zip(axes.flat, written):
        draw_checkpoints(ax, runs, key)
        style(ax, f"{group}: {label}", better)
        ax.set_xlabel("Checkpoint step", fontsize=8, color=MUTED)
    for ax in list(axes.flat)[len(written):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Metrics of the saved checkpoints - {os.path.basename(os.path.normpath(out))} ({note})",
                 fontsize=11, color=INK, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out, "overview_checkpoints.png")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Run folders (results/<run>).")
    p.add_argument("--out", default="results/training_curves/batching_gat_L2_s42")
    p.add_argument("--window", type=int, default=15, help="Moving-average window for per-step metrics.")
    p.add_argument("--no-checkpoints", action="store_true", help="Skip the checkpoint metrics.")
    p.add_argument("--instances", default="data/batching/batching_test_split.json",
                   help="Instances the checkpoints are evaluated on (as in run_metrics.py --folders).")
    p.add_argument("--models-file", default="candidate_models/model_params.json")
    p.add_argument("--log-dir", default="sweep_logs", help="Where the training logs <run>.log are.")
    args = p.parse_args()

    runs, seen = [], {}
    for path in args.runs:
        rep, data = load_run(path)
        idx = REPRESENTATION_ORDER.index(rep) if rep in REPRESENTATION_ORDER else len(seen)
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        label = LABELS.get(rep, rep)
        seen[rep] = seen.get(rep, 0) + 1
        if seen[rep] > 1:  # the same representation twice (e.g. two seeds): keep them apart
            label = f"{label} ({os.path.basename(os.path.normpath(path))})"
        runs.append((label, color, data))
    os.makedirs(args.out, exist_ok=True)
    note = f"faint: raw per step; bold: {args.window}-step moving average"

    written = []
    for source, key, title, better in METRICS:
        values = [y for _, _, data in runs for y in (series(data, source, key)[1],) if y is not None]
        if values and len(np.unique(np.concatenate(values))) == 1:  # e.g. always 0: nothing to compare
            print(f"skip {key} (constant {float(values[0][0]):g} in every run)")
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
        if not draw(ax, runs, source, key, args.window):
            plt.close(fig)
            continue
        style(ax, title, better)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        if source == "episode":
            ax.text(1.0, -0.19, note, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_2)
        path = os.path.join(args.out, f"{key}.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        written.append((source, key, title, better))
        print(path)

    ncols = 3
    nrows = math.ceil(len(written) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows + 0.8), facecolor=SURFACE,
                             squeeze=False)
    for ax, (source, key, title, better) in zip(axes.flat, written):
        draw(ax, runs, source, key, args.window)
        style(ax, title, better)
    for ax in list(axes.flat)[len(written):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Training curves - {os.path.basename(os.path.normpath(args.out))} ({note})",
                 fontsize=11, color=INK, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(args.out, "overview.png")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(path)

    if not args.no_checkpoints:
        print(f"Evaluating the saved checkpoints on {args.instances} ...")
        ck_runs = [(label, color, data, checkpoint_metrics(path, args))
                   for (label, color, data), path in zip(runs, args.runs)]
        plot_checkpoint_metrics(ck_runs, args.out)


if __name__ == "__main__":
    main()
