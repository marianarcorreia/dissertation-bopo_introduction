"""Training curves of several runs on the same axes: one PNG per metric, one line per run.

Reads what src/train.py saves in every run folder: episode_metrics.json (one entry per BOPO
step) and validation_history.json (every validation). Per-step metrics are measured on a
different training instance each step, so each run is drawn as a faint raw line plus a moving
average; validation metrics are drawn as lines with markers. Colours follow the representation
(src/metrics/report.py), so they match the metrics report charts.

Output: <out>/<metric>.png for every metric, plus <out>/overview.png with all of them.

Usage:
    python compare_training_curves.py                      # the three GAT batching runs, seed 42
    python compare_training_curves.py --runs results/batching_gin_L3_ojmb_node_s42 \
        results/batching_gin_L3_ojmb_edge_s42 results/batching_gin_L3_ojmb_base_s42 \
        --out results/training_curves/batching_gin_L3_s42
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics.report import GRID, INK, INK_2, MUTED, REPRESENTATION_ORDER, SERIES_COLORS, SURFACE

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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Run folders (results/<run>).")
    p.add_argument("--out", default="results/training_curves/batching_gat_L2_s42")
    p.add_argument("--window", type=int, default=15, help="Moving-average window for per-step metrics.")
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


if __name__ == "__main__":
    main()
