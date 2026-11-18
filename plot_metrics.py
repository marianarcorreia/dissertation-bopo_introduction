import argparse
import json
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics.report import BETTER, HEADLINE

LABELS = {k:label for k, label, _ in HEADLINE}
HIGHLIGHT_COLORS = ["#2a78d6","#eb6834", "#1baf7a"]
MUTED = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

ALL_METRICS = ["relative_error_lb", "scheduling_score", "makespan", "time_per_decision_ms"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot a compare_metrics.py comparison.")
    p.add_argument("comparison", help="JSON written by compare_metrics.py.")
    p.add_argument("--metric", default="all")
    p.add_argument("--highlight", nargs="*", default=[], help="Up to 3 systems drawn in color.")
    p.add_argument("--exclude", nargs="*", default=[], help="Systems to leave out.")
    p.add_argument("--folder-order", nargs="*", default=None, help="Folders in plotting order (e.g. by size).")
    p.add_argument("--box-folder", default="data/test/large", help="Folder for the box and trade-off plots.")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=MUTED, alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[PLOT] {path}")

def color_of(name, highlight):
    if not highlight:
        return HIGHLIGHT_COLORS[0]
    return HIGHLIGHT_COLORS[highlight.index(name)] if name in highlight else MUTED

def stat(folder, name, metric):
    return folder["summaries"][name]["metrics"].get(metric, {})

def short(folder):
    return folder.split("/")[-1]

def best_first(folder, names, metric):
    # matplotlib draws y=0 at the bottom, so the best model goes last (on top)
    key = lambda n: stat(folder, n, metric).get("mean") or 0
    return sorted(names, key=key, reverse=BETTER.get(metric) != "higher")


def plot_forest(folders, names, metric, highlight, path):
    """One panel per folder: mean with 95% CI for every model."""
    order = best_first(folders[0], names, metric)
    fig, axes = plt.subplots(1, len(folders), figsize=(2.6 * len(folders), 0.32 * len(order) + 1.4),
                             sharey=True, squeeze=False)
    for ax, folder in zip(axes[0], folders):
        style(ax)
        for y, n in enumerate(order):
            s = stat(folder, n, metric)
            if s.get("mean") is None:
                continue
            c = color_of(n, highlight)
            ax.plot([s["ci95_low"], s["ci95_high"]], [y, y], color=c, linewidth=2, solid_capstyle="round")
            ax.plot(s["mean"], y, "o", color=c, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(short(folder["folder"]), fontsize=9, color=INK)
        ax.set_xlabel(LABELS.get(metric, metric), fontsize=8, color=INK_2)
    axes[0][0].set_yticks(range(len(order)), order)
    fig.suptitle(f"{LABELS.get(metric, metric)} per model (mean ± 95% CI)", fontsize=11, color=INK)
    save(fig, path)


def plot_generalization(folders, names, metric, highlight, path):
    """Metric mean per folder, one line per model; highlighted models in color."""
    xs = [short(f["folder"]) for f in folders]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    style(ax)
    for n in sorted(names, key=lambda n: n in highlight):  # highlighted drawn last, on top
        ys = [stat(f, n, metric).get("mean") for f in folders]
        hl = n in highlight
        ax.plot(xs, ys, marker="o", markersize=6 if hl else 3, linewidth=2 if hl else 1,
                color=color_of(n, highlight), zorder=3 if hl else 2, label=n if hl else None)
    if highlight:
        ax.plot([], [], color=MUTED, linewidth=1, label="other models")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    ax.set_ylabel(LABELS.get(metric, metric), fontsize=9, color=INK_2)
    ax.set_title(f"Generalization: {LABELS.get(metric, metric)} by instance folder", fontsize=11, color=INK)
    save(fig, path)


def plot_tradeoff(folder, names, metric, highlight, path):
    """Inference time vs quality on one folder."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    style(ax)
    for n in names:
        x = stat(folder, n, "time_per_decision_ms").get("mean")
        y = stat(folder, n, metric).get("mean")
        if x is None or y is None:
            continue
        ax.plot(x, y, "o", color=color_of(n, highlight), markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(n, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("ms / decision", fontsize=9, color=INK_2)
    ax.set_ylabel(LABELS.get(metric, metric), fontsize=9, color=INK_2)
    ax.set_title(f"Speed vs quality on {short(folder['folder'])}", fontsize=11, color=INK)
    save(fig, path)


def instance_values(system, folder, metric):
    with open(system["report"]) as f:
        report = json.load(f)
    for r in report["results"]:
        if r["model"] == system["model"] and r["folder"] == folder:
            return [row[metric] for row in r["rows"] if row.get(metric) is not None]
    return []


def plot_box(systems, folder, metric, highlight, path):
    """Per-instance distribution of the metric on one folder."""
    values = {s["name"]: instance_values(s, folder["folder"], metric) for s in systems}
    order = best_first(folder, [n for n in values if values[n]], metric)[::-1]  # best on the left
    fig, ax = plt.subplots(figsize=(0.55 * len(order) + 2, 4.5))
    style(ax)
    bp = ax.boxplot([values[n] for n in order], patch_artist=True, widths=0.6,
                    medianprops={"color": INK, "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED, "markeredgecolor": "none"})
    for box, n in zip(bp["boxes"], order):
        box.set_facecolor(color_of(n, highlight))
        box.set_edgecolor(SURFACE)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=45, ha="right")
    ax.set_ylabel(LABELS.get(metric, metric), fontsize=9, color=INK_2)
    ax.set_title(f"Per-instance {LABELS.get(metric, metric)} on {short(folder['folder'])}", fontsize=11, color=INK)
    save(fig, path)


def main(argv=None):
    args = parse_args(argv)
    with open(args.comparison) as f:
        comp = json.load(f)
    systems = [s for s in comp["systems"] if s["name"] not in set(args.exclude)]
    names = [s["name"] for s in systems]
    folders = comp["folders"]
    if args.folder_order:
        by = {f["folder"]: f for f in folders}
        folders = [by[f] for f in args.folder_order]
    by = {f["folder"]: f for f in folders}
    out = args.output_dir or os.path.splitext(args.comparison)[0] + "_plots"
    os.makedirs(out, exist_ok=True)
    m = args.metric

    metrics = ALL_METRICS if args.metric == "all" else [args.metric]
    for m in metrics:
        plot_forest(folders, names, m, args.highlight, os.path.join(out, f"forest_{m}.png"))
        plot_generalization(folders, names, m, args.highlight, os.path.join(out, f"generalization_{m}.png"))
        if args.box_folder in by:
            plot_tradeoff(by[args.box_folder], names, m, args.highlight, os.path.join(out, f"tradeoff_{m}.png"))
            plot_box(systems, by[args.box_folder], m, args.highlight, os.path.join(out, f"box_{m}.png"))

    plot_forest(folders, names, m, args.highlight, os.path.join(out, f"forest_{m}.png"))
    plot_generalization(folders, names, m, args.highlight, os.path.join(out, f"generalization_{m}.png"))
    if args.box_folder in by:
        plot_tradeoff(by[args.box_folder], names, m, args.highlight, os.path.join(out, f"tradeoff_{m}.png"))
        plot_box(systems, by[args.box_folder], m, args.highlight, os.path.join(out, f"box_{m}.png"))


if __name__ == "__main__":
    main()