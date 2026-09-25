"""Generic run comparison: pick some runs, pick what to compare, get the plots.

Same analysis as compare_final9.py (convergence metrics CSV + learning curves,
test-gap bars, actor loss, entropy), but instead of a hard-coded 3x3 grid you
choose:

    RUNS     which result folders to include (names or glob patterns)
    COMPARE  the setting that becomes the lines / bar colours   (e.g. sel_k)
    FACET    the setting that becomes one subplot per value     (e.g. representation)
    WHERE    keep only runs matching these settings              (e.g. gnn_type=transformer)

Settings available for COMPARE / FACET / WHERE:
    representation, gnn_type, num_layers, sel_k, mask_option, seed, logp_norm, run_name
They are read from run_summary.json; older runs that did not save them get them
parsed from the folder name (e.g. "_gat", "layers3", "selk2", "_s43").
Anything still unknown can be filled in by hand in OVERRIDES.

Usage
-----
Edit the "EDIT HERE" block below and run `python compare_runs.py`, or use the CLI:

    # see every run and the settings the script detected for it
    python compare_runs.py --list

    # sel_k 1 vs 2 for the 3-layer O-J-M transformer
    python compare_runs.py --runs "ablation_ojm_transformer_layers3*" --compare sel_k

    # backbones side by side, one subplot per representation
    python compare_runs.py --runs "convergence_check_*_gat" "convergence_check_*_gin" \
        "convergence_check_*_transformer" --compare gnn_type --facet representation

    # only keep some runs out of a broad pattern
    python compare_runs.py --runs "ablation_*" --compare gnn_type --where num_layers=3

Outputs go to results/compare_<name>/ (convergence_summary.csv + plots/*.png).
"""
import argparse
import fnmatch
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================================================
# EDIT HERE (used when the matching CLI flag is not given)
# ===========================================================================
RUNS = [
    "ablation_ojm_*_layers3", "train_run_*"
]
COMPARE = "gnn_type"        # lines / bar colours
FACET = None                # one subplot per value, or None for a single panel
WHERE = {}                  # e.g. {"representation": "ojm", "num_layers": 3}
OUT_NAME = None             # output folder suffix; auto-generated if None

# Fill in settings the script cannot detect for a run (older runs did not
# save their config). Values here win over everything else.
OVERRIDES = {
    "ablation_ojm_transformer_layers3": {"sel_k": 1},
    "ablation_ojm_transformer_layers3_selk2": {"sel_k": 2},
    "train_run_*": {"sel_k": 100}
}
# ===========================================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

SETTINGS = ["representation", "gnn_type", "num_layers", "sel_k",
            "mask_option", "seed", "logp_norm"]
REP_LABELS = {"oo": "Disjunctive (O-O)", "om": "Hetero O-M", "ojm": "Hetero O-J-M"}
SETTING_LABELS = {
    "representation": "Representation", "gnn_type": "Backbone",
    "num_layers": "Layers", "sel_k": "sel_k", "mask_option": "Mask option",
    "seed": "Seed", "logp_norm": "log-p norm", "run_name": "Run",
}

sns.set_theme(style="whitegrid", palette="pastel")


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_name(name):
    """Best-effort settings from a folder name, for runs that did not save them."""
    parsed = {}
    tokens = name.lower().split("_")
    for t in tokens:
        if t in REP_LABELS:
            parsed["representation"] = t
        if t in ("gat", "gin", "transformer"):
            parsed["gnn_type"] = t
    m = re.search(r"(?:layers?|_l)(\d+)", name.lower())
    if m:
        parsed["num_layers"] = int(m.group(1))
    m = re.search(r"sel_?k(\d+)", name.lower())
    if m:
        parsed["sel_k"] = int(m.group(1))
    m = re.search(r"_s(\d+)(?:_|$)", name.lower())
    if m:
        parsed["seed"] = int(m.group(1))
    return parsed


def all_run_names():
    if not os.path.isdir(RESULTS):
        return []
    return sorted(
        d for d in os.listdir(RESULTS)
        if os.path.exists(os.path.join(RESULTS, d, "run_summary.json"))
    )


def run_settings(name, summary):
    settings = parse_name(name)
    settings.update({k: summary[k] for k in SETTINGS if summary.get(k) is not None})
    settings.update(OVERRIDES.get(name, {}))
    settings["run_name"] = name
    return {k: settings.get(k) for k in SETTINGS + ["run_name"]}


def matches_where(settings, where):
    for key, wanted in where.items():
        if str(settings.get(key)) != str(wanted):
            return False
    return True


def select_runs(patterns, where):
    names = all_run_names()
    chosen = []
    for pattern in patterns:
        hits = [n for n in names if fnmatch.fnmatch(n, pattern)]
        if not hits:
            print(f"WARNING: no run matches '{pattern}'")
        chosen.extend(h for h in hits if h not in chosen)
    runs = []
    for name in chosen:
        summary = load_json(os.path.join(RESULTS, name, "run_summary.json"))
        settings = run_settings(name, summary)
        if matches_where(settings, where):
            runs.append((name, summary, settings))
    return runs


# ---------------------------------------------------------------------------
# Metrics (same definitions as compare_final9.py)
# ---------------------------------------------------------------------------
def linreg_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    slope, _intercept = np.polyfit(x, y, 1)
    return slope


def first_last_mean(ep_df, col, n=10):
    if ep_df.empty or col not in ep_df:
        return np.nan, np.nan
    return ep_df[col].iloc[:n].mean(), ep_df[col].iloc[-n:].mean()


def analyse_run(name, summary, settings):
    run_dir = os.path.join(RESULTS, name)
    val_path = os.path.join(run_dir, "validation_history.json")
    ep_path = os.path.join(run_dir, "episode_metrics.json")
    val_hist = load_json(val_path) if os.path.exists(val_path) else []
    ep_hist = load_json(ep_path) if os.path.exists(ep_path) else []

    episodes = [v["episode"] for v in val_hist]
    raw_gap = [v["avg_gap"] for v in val_hist]
    smooth_gap = [v.get("smoothed_avg_gap", v["avg_gap"]) for v in val_hist]
    val_df = pd.DataFrame({"episode": episodes, "avg_gap": raw_gap,
                           "smoothed_avg_gap": smooth_gap})
    ep_df = pd.DataFrame(ep_hist)

    # LCS over first 200 episodes (README definition) and over the full run
    early = [(e, g) for e, g in zip(episodes, raw_gap) if e <= 200]
    lcs_early = linreg_slope([e for e, _ in early], [g for _, g in early])
    lcs_full = linreg_slope(episodes, raw_gap)

    # Late-phase stability on the smoothed curve (last 5 checkpoints)
    n_tail = min(5, len(episodes))
    if n_tail:
        tail_ep, tail_smooth = episodes[-n_tail:], smooth_gap[-n_tail:]
        late_slope_per100 = linreg_slope(tail_ep, tail_smooth) * 100
        tail_mean = np.mean(tail_smooth)
        tail_relrange = (max(tail_smooth) - min(tail_smooth)) / tail_mean if tail_mean else np.nan
    else:
        late_slope_per100 = tail_relrange = np.nan

    best_val_gap = summary.get("best_validation_avg_gap")
    test_gap = summary.get("test_avg_gap")
    gen_gap = (test_gap - best_val_gap) if (test_gap is not None and best_val_gap is not None) else np.nan

    converged = bool(
        abs(late_slope_per100) < 0.02
        and tail_relrange < 0.30
        and abs(gen_gap) < 0.06
    )

    loss_first, loss_last = first_last_mean(ep_df, "actor_loss")
    ent_first, ent_last = first_last_mean(ep_df, "action_entropy_max_entropy_normalized")

    record = dict(settings)
    record.update({
        "episodes_completed": summary.get("episodes_completed"),
        "actor_params": summary.get("actor_param_count"),
        "lcs_first200_per_ep": lcs_early,
        "lcs_full_per_ep": lcs_full,
        "late_slope_per100ep": late_slope_per100,
        "late_tail_relrange": tail_relrange,
        "best_val_gap": best_val_gap,
        "final_val_gap": raw_gap[-1] if raw_gap else np.nan,
        "test_gap": test_gap,
        "test_std": summary.get("test_std_gap"),
        "generalization_gap": gen_gap,
        "actor_loss_first10": loss_first,
        "actor_loss_last10": loss_last,
        "entropy_norm_first10": ent_first,
        "entropy_norm_last10": ent_last,
        "converged": converged,
    })
    return record, val_df, ep_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def value_label(setting, value):
    if setting == "representation":
        return REP_LABELS.get(value, str(value))
    if setting == "gnn_type" and value is not None:
        return str(value).upper()
    if setting == "run_name":
        return str(value)
    return f"{SETTING_LABELS.get(setting, setting)}={value}"


def sort_key(v):
    return (v is None, isinstance(v, str), v if v is not None else 0)


def facet_axes(facet_values, sharey=True):
    n = len(facet_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 0.5, 4.5), sharey=sharey, squeeze=False)
    return fig, axes[0]


def plot_curves(df, curves, value_fn, ylabel, title, path, compare, facet,
                colors, logy=False, ylim=None, band=None):
    facet_values = sorted(df[facet].unique(), key=sort_key) if facet else [None]
    fig, axes = facet_axes(facet_values)
    for ax, fval in zip(axes, facet_values):
        sub = df if facet is None else df[df[facet] == fval]
        seen = set()
        for _, row in sub.iterrows():
            data = curves.get(row["run_name"])
            if data is None or data.empty:
                continue
            y = value_fn(data)
            if y is None:
                continue
            cval = row[compare]
            label = value_label(compare, cval)
            ax.plot(data["episode"], y, color=colors[cval], linewidth=1.8,
                    label=None if label in seen else label)
            seen.add(label)
            if band is not None:
                ax.fill_between(data["episode"], band(data), y, color=colors[cval], alpha=0.15)
        if facet:
            ax.set_title(value_label(facet, fval))
        ax.set_xlabel("Episode")
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    if any(ax.get_legend_handles_labels()[0] for ax in axes):
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        axes[0].legend(handles, labels, title=SETTING_LABELS.get(compare, compare), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test_bars(df, path, compare, facet, colors):
    bar_df = df.dropna(subset=["test_gap"]).copy()
    if bar_df.empty:
        print("No test_gap values — skipping bar chart")
        return
    bar_df["compare_label"] = bar_df[compare].map(lambda v: value_label(compare, v))
    palette = {value_label(compare, k): c for k, c in colors.items()}
    x = "facet_label" if facet else "compare_label"
    if facet:
        bar_df["facet_label"] = bar_df[facet].map(lambda v: value_label(facet, v))
    order = [value_label(facet, v) for v in sorted(bar_df[facet].unique(), key=sort_key)] if facet else None
    hue_order = [value_label(compare, v) for v in sorted(bar_df[compare].unique(), key=sort_key)]

    fig, ax = plt.subplots(figsize=(min(12, max(6, 1.1 * len(bar_df))), 5))
    # errorbar="sd" only shows when several runs (e.g. seeds) share a bar
    sns.barplot(data=bar_df, x=x, y="test_gap", hue="compare_label", order=order,
                hue_order=hue_order, palette=palette, ax=ax, errorbar="sd")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylabel("Test makespan gap (mean)")
    ax.set_xlabel("")
    ax.set_title(f"Held-out test gap by {SETTING_LABELS.get(compare, compare)}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title=SETTING_LABELS.get(compare, compare), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rolling(col):
    def fn(data):
        if col not in data:
            return None
        return data[col].rolling(10, min_periods=1).mean()
    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def tidy_settings(df):
    """Keep settings as ints/strings (pandas turns int columns with gaps into floats)."""
    for c in SETTINGS:
        if c in df:
            df[c] = pd.Series(
                [None if pd.isna(v) else int(v) if isinstance(v, float) and v.is_integer() else v
                 for v in df[c]], index=df.index, dtype=object)
    return df


def parse_where(items):
    where = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--where expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        where[key.strip()] = value.strip()
    return where


def print_run_list(patterns):
    rows = []
    for name in all_run_names():
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        summary = load_json(os.path.join(RESULTS, name, "run_summary.json"))
        s = run_settings(name, summary)
        s["episodes"] = summary.get("episodes_completed")
        s["test_gap"] = summary.get("test_avg_gap")
        rows.append(s)
    pd.set_option("display.width", 220)
    rows = tidy_settings(pd.DataFrame(rows)).fillna({c: "-" for c in SETTINGS})
    pd.set_option("display.max_rows", 500)
    print(rows.to_string(index=False))


def main():
    choices = SETTINGS + ["run_name"]
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", help="run folder names or glob patterns")
    ap.add_argument("--compare", choices=choices, help="setting shown as lines / bar colours")
    ap.add_argument("--facet", choices=choices + ["none"], help="setting shown as one subplot per value")
    ap.add_argument("--where", nargs="+", metavar="KEY=VALUE", help="keep only matching runs")
    ap.add_argument("--out", help="output folder name suffix (results/compare_<out>)")
    ap.add_argument("--list", action="store_true", help="list runs and detected settings, then exit")
    args = ap.parse_args()

    if args.list:
        print_run_list(args.runs)
        return

    patterns = args.runs or RUNS
    compare = args.compare or COMPARE
    facet = FACET if args.facet is None else (None if args.facet == "none" else args.facet)
    where = parse_where(args.where) if args.where else WHERE

    runs = select_runs(patterns, where)
    if not runs:
        raise SystemExit(f"No runs found for {patterns} with {where}")

    records, val_curves, ep_curves = [], {}, {}
    for name, summary, settings in runs:
        record, val_df, ep_df = analyse_run(name, summary, settings)
        records.append(record)
        val_curves[name] = val_df
        ep_curves[name] = ep_df
    df = tidy_settings(pd.DataFrame(records))

    # Make missing values explicit instead of silently dropping them
    for col in [compare] + ([facet] if facet else []):
        missing = df[df[col].isna()]["run_name"].tolist()
        if missing:
            print(f"WARNING: '{col}' unknown for {missing} — set it in OVERRIDES. Shown as '?'.")
        df[col] = df[col].where(df[col].notna(), "?")

    # Point out settings that differ but are not being compared: those are
    # confounders the plots would otherwise hide.
    varied = [c for c in SETTINGS if c not in (compare, facet) and df[c].nunique(dropna=False) > 1]
    for c in varied:
        values = sorted({"unknown" if v is None else str(v) for v in df[c]})
        print(f"NOTE: runs also differ in {c}: {values}")

    out_name = args.out or OUT_NAME or ("by_" + compare + (f"_per_{facet}" if facet else ""))
    out_dir = os.path.join(RESULTS, f"compare_{out_name}")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = df.sort_values([c for c in (facet, compare) if c] + ["run_name"],
                        key=lambda s: s.map(lambda v: str(sort_key(v))))
    df.to_csv(os.path.join(out_dir, "convergence_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show_cols = ["run_name"] + [c for c in SETTINGS if c in (compare, facet) or c in varied] + [
        "episodes_completed", "best_val_gap", "test_gap", "test_std",
        "generalization_gap", "late_slope_per100ep", "converged"]
    print(df[show_cols].to_string(index=False))

    compare_values = sorted(df[compare].unique(), key=sort_key)
    colors = dict(zip(compare_values, sns.color_palette("deep", len(compare_values))))
    what = f"by {SETTING_LABELS.get(compare, compare)}" + (
        f" per {SETTING_LABELS.get(facet, facet)}" if facet else "")
    common = dict(compare=compare, facet=facet, colors=colors)

    plot_curves(df, val_curves, lambda d: d["smoothed_avg_gap"],
                "Validation makespan gap", f"Validation gap learning curves {what}",
                os.path.join(plots_dir, "fig1_learning_curves.png"),
                band=lambda d: d["avg_gap"], **common)
    plot_test_bars(df, os.path.join(plots_dir, "fig2_test_gap_bars.png"), compare, facet, colors)
    plot_curves(df, ep_curves, rolling("actor_loss"),
                "Actor loss (10-ep rolling mean, log scale)", f"Actor loss trajectories {what}",
                os.path.join(plots_dir, "fig3_actor_loss.png"), logy=True, **common)
    plot_curves(df, ep_curves, rolling("action_entropy_max_entropy_normalized"),
                "Normalized action entropy (10-ep rolling mean)", f"Policy entropy trajectories {what}",
                os.path.join(plots_dir, "fig4_action_entropy.png"), ylim=(0, 1.05), **common)

    print("\nSaved CSV and figures to", out_dir)


if __name__ == "__main__":
    main()
