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
