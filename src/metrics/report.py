"""Shared by run_metrics.py and compare_metrics.py: which metrics are headlined, which
direction is better, and Markdown table formatting."""
import math

# (key in summary["metrics"] / rows, column label, better: "lower" | "higher" | None)
HEADLINE = (
    ("makespan", "Makespan", "lower"),
    ("relative_error", "RE (CP-SAT)", "lower"),
    ("relative_error_lb", "RE (LB)", "lower"),
    ("scheduling_score", "Score", "higher"),
    ("violation_rate_per_step", "Viol./step", "lower"),
    ("time_per_decision_ms", "ms/decision", "lower"),
    ("gpu_peak_mb", "GPU peak MB", "lower"),
    ("rss_mb", "RSS MB", "lower"),
    ("action_distinct_ratio", "Action distinct", "higher"),
    ("structure_gain", "Structure gain", None),
    ("effective_rank_ratio", "Eff. rank", "higher"),
    ("embedding_heterophily", "Heterophily (emb)", None),
    ("feature_heterophily", "Heterophily (feat)", None),
    # batching
    ("avg_batch_size", "Batch size", None),
    ("batched_fraction", "Batched ops", None),
)
BETTER = {k: b for k, _, b in HEADLINE}
BETTER["feasible"] = "higher"


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.{digits}f}"
    return str(value)


def mean_ci(desc, digits=4):
    """'mean ± half-width of the 95% CI' from a statistics.describe() dict."""
    if not desc or not desc.get("n"):
        return "-"
    mean = desc["mean"]
    half = desc.get("ci95_high", float("nan")) - mean
    if desc["n"] < 2 or math.isnan(half):
        return fmt(mean, digits)
    d = 1 if abs(mean) >= 100 else digits  # same precision for the mean and its interval
    return f"{mean:.{d}f} ± {half:.{d}f}"


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def headline_row(summary):
    """One table row of headline metrics for a (model, folder) summary."""
    m = summary["metrics"]
    return [mean_ci(m.get(k)) for k, _, _ in HEADLINE] + [fmt(summary.get("feasibility_rate"))]


HEADLINE_HEADER = [label for _, label, _ in HEADLINE] + ["Feasible"]


# ---------------------------------------------------------------- bar charts
# Panels of the metrics chart, grouped by what they measure. The isomorphism group comes from
# src/metrics/representation.py: whether nodes / actions that differ get different embeddings
# (a representation that maps non-isomorphic neighbourhoods to the same embedding cannot tell
# them apart). (key, label, better) as in HEADLINE.
CHART_GROUPS = (
    ("Schedule quality", (
        ("makespan", "Makespan", "lower"),
        ("relative_error", "RE vs CP-SAT", "lower"),
        ("relative_error_lb", "RE vs lower bound", "lower"),
        ("scheduling_score", "Scheduling score", "higher"),
    )),
    ("Constraints", (
        ("feasibility_rate", "Feasible schedules", "higher"),
        ("violation_rate_per_step", "Violations / decision", "lower"),
    )),
    ("Isomorphism / expressiveness", (
        ("action_distinct_ratio", "Distinct action scores", "higher"),
        ("embedding_distinct_ratio", "Distinct node embeddings", "higher"),
        ("input_distinct_ratio", "Distinct input features", None),
        ("structure_gain", "Structure gain (emb / input)", None),
        ("effective_rank_ratio", "Effective rank", "higher"),
        ("embedding_heterophily", "Heterophily (embeddings)", None),
        ("feature_heterophily", "Heterophily (features)", None),
    )),
    ("Batching", (
        ("num_batches", "Batches", None),
        ("avg_batch_size", "Mean batch size", None),
        ("batched_fraction", "Operations batched", None),
    )),
    ("Efficiency", (
        ("time_per_decision_ms", "ms / decision", "lower"),
        ("gpu_peak_mb", "GPU peak MB", "lower"),
        ("rss_mb", "RSS MB", "lower"),
    )),
)

# Categorical slots of the reference palette (the first three validate for all pairs), fixed
# per representation so a colour always means the same representation across charts.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
REPRESENTATION_ORDER = ("ojmb_node", "ojmb_edge", "ojmb_base", "oo", "om", "ojm")
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _series(report, folder):
    """[(label, representation, summary)] for one folder, in the fixed representation order."""
    results = [r for r in report["results"] if r["folder"] == folder]
    reps = [r["representation"] for r in results]
    rank = {rep: i for i, rep in enumerate(REPRESENTATION_ORDER)}
    results.sort(key=lambda r: (rank.get(r["representation"], len(rank)), r["model"]))
    return [(r["representation"] if reps.count(r["representation"]) == 1
             else f"{r['representation']} ({r['model']})", r["representation"], r["summary"]) for r in results]


def _value(summary, key):
    """(mean, 95% CI half-width or None) of a metric in a run_metrics summary."""
    if key == "feasibility_rate":
        v = summary.get("feasibility_rate")
        return (None, None) if v is None else (float(v), None)
    desc = summary["metrics"].get(key) or {}
    if not desc.get("n"):
        return None, None
    mean = desc["mean"]
    half = desc.get("ci95_high", float("nan")) - mean
    return mean, (None if desc["n"] < 2 or math.isnan(half) else half)


def plot_metric_bars(report, path, folder=None):
    """One bar per model for every metric of CHART_GROUPS - small multiples, one panel per
    metric so each keeps its own scale - with 95% CI error bars. Metrics without data for any
    model are left out. Returns the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    folder = folder or report["results"][0]["folder"]
    series = _series(report, folder)
    colors = {}
    for _, rep, _ in series:  # colour follows the representation, never the rank
        idx = REPRESENTATION_ORDER.index(rep) if rep in REPRESENTATION_ORDER else len(colors)
        colors.setdefault(rep, SERIES_COLORS[idx % len(SERIES_COLORS)])

    groups = []
    for title, metrics in CHART_GROUPS:
        panels = [m for m in metrics if any(_value(s, m[0])[0] is not None for _, _, s in series)]
        if panels:
            groups.append((title, panels))
    ncols = 4
    nrows = sum(math.ceil(len(p) / ncols) for _, p in groups)
    height = 2.7 * nrows + 1.2
    fig = plt.figure(figsize=(3.3 * ncols, height), facecolor=SURFACE)
    # fixed ~1.2 in above the first panels for the title, legend and group heading
    grid = fig.add_gridspec(nrows, ncols, hspace=0.95, wspace=0.35, top=1 - 1.2 / height, bottom=0.03)

    row = 0
    for title, panels in groups:
        for i, (key, label, better) in enumerate(panels):
            ax = fig.add_subplot(grid[row + i // ncols, i % ncols])
            ax.set_facecolor(SURFACE)
            means, errs, cols = [], [], []
            for _, rep, summary in series:
                mean, half = _value(summary, key)
                means.append(float("nan") if mean is None else mean)
                errs.append(0.0 if half is None else half)
                cols.append(colors[rep])
            x = list(range(len(series)))
            ax.bar(x, means, width=0.6, color=cols, edgecolor=SURFACE, linewidth=2, zorder=2)
            ax.errorbar(x, means, yerr=errs, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
            for xi, m, e in zip(x, means, errs):
                if not math.isnan(m):
                    ax.annotate(fmt(m, 3), (xi, m + e if m >= 0 else m - e), xytext=(0, 3 if m >= 0 else -3),
                                textcoords="offset points", ha="center", va="bottom" if m >= 0 else "top",
                                fontsize=7, color=INK_2)
            arrow = {"lower": " (lower better)", "higher": " (higher better)"}.get(better, "")
            ax.set_title(label + arrow, fontsize=8.5, color=INK, loc="left")
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)
            ax.margins(y=0.2)
            if i == 0:
                ax.text(0, 1.3, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK)
        row += math.ceil(len(panels) / ncols)

    handles = [Patch(color=colors[rep], label=name) for name, rep, _ in series]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    label = report.get("meta", {}).get("label", "")
    fig.suptitle(f"{label} - {folder} (mean, 95% CI)", fontsize=11, color=INK, y=1.03)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    # python -m src.metrics.report results/metrics/<run>.json [--folder F] [--out chart.png]
    import argparse
    import json
    import os

    p = argparse.ArgumentParser(description="Bar chart of every metric in a run_metrics.py report.")
    p.add_argument("report")
    p.add_argument("--folder", default=None, help="Instance folder / split to plot (default: the first).")
    p.add_argument("--out", default=None, help="Output image (default: the report path with .png).")
    a = p.parse_args()
    with open(a.report) as f:
        loaded = json.load(f)
    print(plot_metric_bars(loaded, a.out or os.path.splitext(a.report)[0] + ".png", a.folder))
