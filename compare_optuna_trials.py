"""Compare the trials of one Optuna study, one PNG per metric.

Every finished trial of the study becomes one line in each graph, so the trials can
be compared metric by metric along the episodes:

    per-episode metrics (episode_metrics.json)    ->  line chart, 5-episode rolling mean
    validation metrics  (validation_history.json) ->  line chart with markers at checkpoints
    objective (last-checkpoint val q80 gap)       ->  horizontal bar chart, best first
    src/metrics evaluation of the best checkpoint ->  horizontal bar chart per metric,
        (makespan, relative error vs CP-SAT / lower     mean with 95% CI, best first
         bound, scheduling score, violation rate,
         feasibility, time, memory, model size,
         expressiveness, heterophily)

Trial hyperparameters and objective values are read from the Optuna storage, the
curves from each trial's results folder (results/<study folder>/trial_N/). A study that
was resumed has its trials split over several study folders; all of them are merged.
Trials without a run_summary.json (failed/interrupted) are skipped.

The src/metrics evaluation rolls each trial's best checkpoint out greedily on the
held-out test split (val/test_dataset.json by default - never used to pick a checkpoint)
and caches the result as trial_N/eval_<split>.json, so re-running is cheap. --no-eval
skips it; --no-representation skips expressiveness/heterophily (about half the time).

Usage
-----
    # default: the old-loss GAT ojm study (2 layers)
    python compare_optuna_trials.py

    # any other study
    python compare_optuna_trials.py --study fjsp_tuning_ojm_L2 \
        --folders "optuna_ojm_gat_L2_*" --out optuna_gat_L2_newloss

Outputs go to results/compare_<out>/ (one PNG per metric + trials_summary.csv).
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

DEFAULT_STUDY = "fjsp_tuning_ojm_exgreedyFalse_logpsum_L2"
DEFAULT_FOLDERS = ["optuna_ojm_gat_exgreedyFalse_logpsum_L2_*"]
DEFAULT_OUT = "optuna_gat_L2_oldloss"

# Validated reference categorical palette (fixed order, never cycled). Trials beyond the
# 8th reuse the hues with a dashed line, so identity never rests on color alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

ROLLING = 5  # episodes; per-episode metrics are noisy (a new instance every episode)

EPISODE_METRICS = {
    "makespan": ("Makespan of the best rollout", "Makespan"),
    "actor_loss": ("Actor loss (BOPO)", "Loss"),
    "actor_grad_norm": ("Actor gradient norm", "Gradient norm"),
    "action_entropy": ("Action entropy", "Entropy (nats)"),
    "action_entropy_max_entropy_normalized": ("Normalized action entropy", "Entropy / max entropy"),
    "action_entropy_effective_action_count": ("Effective number of actions", "exp(entropy)"),
    "sro_margin_mean": ("SRO margin (mean)", "Margin"),
    "sro_saturated_frac": ("SRO saturated fraction", "Fraction of pairs"),
    "greedy_is_best": ("Greedy rollout is the best", "Fraction of episodes"),
    "steps": ("Decisions per episode", "Decisions"),
    "episode_duration_sec": ("Episode duration", "Seconds"),
    "update_duration_sec": ("Update duration", "Seconds"),
}
VALIDATION_METRICS = {
    "avg_gap": ("Validation makespan gap (mean)", "Gap to reference"),
    "q80_gap": ("Validation makespan gap (80th percentile)", "Gap to reference"),
    "std_gap": ("Validation makespan gap (std)", "Std of gap"),
    "smoothed_avg_gap": ("Validation makespan gap (smoothed mean)", "Gap to reference"),
}
# src/metrics (see src/metrics/evaluate.py SUMMARY_METRICS): key -> (title, axis label, lower is better)
EVAL_METRICS = {
    "makespan": ("Makespan", "Makespan", True),
    "relative_error": ("Relative error vs CP-SAT reference", "(makespan - CP-SAT) / CP-SAT", True),
    "relative_error_lb": ("Relative error vs lower bound", "(makespan - LB) / LB", True),
    "scheduling_score": ("Scheduling score", "Score", False),
    "violation_rate_per_step": ("Constraint violation rate per step", "Violating steps / steps", True),
    "time_sec": ("Inference time per instance", "Seconds", True),
    "time_per_decision_ms": ("Inference time per decision", "Milliseconds", True),
    "rss_mb": ("Process memory (RSS)", "MB", True),
    "rss_delta_mb": ("Process memory increase during the rollout", "MB", True),
    "gpu_peak_mb": ("GPU peak memory", "MB", True),
    "gpu_peak_delta_mb": ("GPU peak memory increase during the rollout", "MB", True),
    "action_distinct_ratio": ("Action expressiveness (distinct action embeddings)", "Ratio", False),
    "embedding_distinct_ratio": ("Embedding expressiveness (distinct node embeddings)", "Ratio", False),
    "input_distinct_ratio": ("Input distinctness (distinct node features)", "Ratio", False),
    "structure_gain": ("Structure gain (distinct embeddings / distinct inputs)", "Ratio", False),
    "effective_rank_ratio": ("Effective rank of the embeddings", "Effective rank / dimension", False),
    "embedding_heterophily": ("Embedding heterophily", "Heterophily", None),
    "feature_heterophily": ("Feature heterophily", "Heterophily", None),
}
# single values per trial (no per-instance distribution, so no CI)
EVAL_SCALARS = {
    "feasibility_rate": ("Feasibility rate", "Feasible schedules / instances", False),
    "param_count": ("Model size (parameters)", "Parameters", True),
    "model_size_mb": ("Model size (MB)", "MB", True),
}
HEADS = 3  # fixed in param.py's search space and not saved in run_summary.json


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def study_params(study_name, storage):
    """{trial number: (params, value)} from the Optuna storage, or {} if unavailable."""
    try:
        import optuna
        import param  # sibling param.py: reuses its storage handling (journal / sqlite)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(study_name=study_name,
                                  storage=param._make_storage(param._normalize_storage_url(storage)))
    except Exception as exc:  # noqa: BLE001 - plots still work without the params
        print(f"WARNING: could not load study '{study_name}' from {storage}: {exc}")
        return {}
    return {t.number: (t.params, t.value) for t in study.trials}


def last_val_q80(val_hist):
    """Same objective as param.py: 80th percentile of the last checkpoint's gaps."""
    if not val_hist or not val_hist[-1].get("all_gaps"):
        return np.nan
    return float(np.percentile(val_hist[-1]["all_gaps"], 80))


def collect_trials(patterns, params_by_trial):
    trials = []
    for pattern in patterns:
        for study_dir in sorted(glob.glob(os.path.join(RESULTS, pattern))):
            for trial_dir in glob.glob(os.path.join(study_dir, "trial_*")):
                m = re.fullmatch(r"trial_(\d+)", os.path.basename(trial_dir))
                if not m or not os.path.exists(os.path.join(trial_dir, "run_summary.json")):
                    continue
                number = int(m.group(1))
                summary = load_json(os.path.join(trial_dir, "run_summary.json"))
                episodes = pd.DataFrame(load_json(os.path.join(trial_dir, "episode_metrics.json")))
                val_hist = load_json(os.path.join(trial_dir, "validation_history.json"))
                params, value = params_by_trial.get(number, ({}, None))
                trials.append({
                    "number": number,
                    "dir": trial_dir,
                    "summary": summary,
                    "params": params,
                    "objective": value if value is not None else last_val_q80(val_hist),
                    "episodes": episodes,
                    "validation": pd.DataFrame(val_hist),
                })
    trials.sort(key=lambda t: t["number"])
    return trials


def evaluate_trial(t, split_path, representation_metrics):
    """src/metrics summary of the trial's best checkpoint on split_path (cached per trial)."""
    split = os.path.splitext(os.path.basename(split_path))[0]
    cache = os.path.join(t["dir"], f"eval_{split}{'' if representation_metrics else '_norepr'}.json")
    if os.path.exists(cache):
        return load_json(cache)

    import torch
    from run_metrics import load_agent  # sibling run_metrics.py: same agent construction
    from src.metrics.evaluate import evaluate_agent, summarize

    s, p = t["summary"], t["params"]
    checkpoint = s.get("best_model_path")
    if not checkpoint or not os.path.exists(os.path.join(ROOT, checkpoint)):
        print(f"WARNING: T{t['number']}: checkpoint {checkpoint!r} not found - skipped")
        return None
    param = {"name": os.path.basename(checkpoint), "representation": s["representation"],
             "gnn_type": s.get("gnn_type", "gat"), "jm_design": s.get("jm_design", "baseline"),
             "mask_option": s["mask_option"], "sel_k": s["sel_k"], "num_layers": s["num_layers"],
             "hidden_channels": p.get("hidden_channels", 128), "heads": HEADS}
    instances = load_json(split_path)
    print(f"[EVAL] T{t['number']}: {param['name']} on {split} ({len(instances)} instances) ...")
    with torch.no_grad():
        rep, env, agent = load_agent(param, os.path.join(ROOT, os.path.dirname(checkpoint)), instances)
        rows, repr_summary = evaluate_agent(agent, env, instances, rep,
                                            representation_metrics=representation_metrics)
        summary = summarize(rows, repr_summary, agent.policy.actor)
    result = {"checkpoint": checkpoint, "split": split_path, "summary": summary, "rows": rows}
    with open(cache, "w") as f:
        json.dump(result, f, indent=1, default=float)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def style(i):
    return {"color": SERIES[i % len(SERIES)],
            "linestyle": "-" if i < len(SERIES) else (0, (5, 2))}


def trial_label(t):
    sel_k = t["params"].get("sel_k", t["summary"].get("sel_k"))
    lr = t["params"].get("lr")
    parts = [f"T{t['number']}", f"sel_k {sel_k}"]
    if lr is not None:
        parts.append(f"lr {lr:.1e}")
    parts.append(f"q80 {t['objective']:.3f}")
    return " · ".join(parts)


def new_axes(title, ylabel, subtitle):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, length=0, pad=6)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_xlabel("Episode", color=TEXT_SECONDARY)
    fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=14, fontweight="bold")
    ax.set_title(subtitle, loc="left", color=TEXT_SECONDARY, fontsize=9.5, pad=10)
    return fig, ax


def finish(fig, ax, path):
    leg = ax.legend(title="Trial", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    frameon=False, fontsize=8.5, title_fontsize=9, handlelength=2.6)
    plt.setp(leg.get_texts(), color=TEXT_PRIMARY)
    plt.setp(leg.get_title(), color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_episode_metric(trials, key, title, ylabel, study, path):
    fig, ax = new_axes(title, ylabel, f"{study} · {ROLLING}-episode rolling mean · one line per trial")
    drawn = False
    for i, t in enumerate(trials):
        df = t["episodes"]
        if key not in df or df[key].dropna().empty:
            continue
        y = df[key].astype(float).rolling(ROLLING, min_periods=1).mean()
        ax.plot(df["episode"], y, linewidth=2, label=trial_label(t), **style(i))
        drawn = True
    if drawn:
        finish(fig, ax, path)
    else:
        plt.close(fig)
    return drawn


def plot_validation_metric(trials, key, title, ylabel, study, path):
    fig, ax = new_axes(title, ylabel, f"{study} · validation checkpoints · one line per trial")
    drawn = False
    for i, t in enumerate(trials):
        df = t["validation"]
        if key not in df or df[key].dropna().empty:
            continue
        ax.plot(df["episode"], df[key], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=trial_label(t), **style(i))
        drawn = True
    if drawn:
        ax.set_xticks(sorted({e for t in trials for e in t["validation"].get("episode", [])}))
        finish(fig, ax, path)
    else:
        plt.close(fig)
    return drawn


def fmt_value(v):
    if abs(v) >= 1e5:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def plot_bars(trials, values, title, xlabel, subtitle, path, lower_is_better=True):
    """values: {trial number: (value, ci_low, ci_high)}; ci may be None. One bar per trial,
    in the trial's line color, best first (by trial number when there is no better side)."""
    index = {t["number"]: i for i, t in enumerate(trials)}  # keep each trial's color
    ranked = [t for t in trials if t["number"] in values and np.isfinite(values[t["number"]][0])]
    if not ranked:
        return False
    if lower_is_better is not None:
        ranked.sort(key=lambda t: values[t["number"]][0], reverse=not lower_is_better)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(ranked) + 1.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(ranked))
    means = np.array([values[t["number"]][0] for t in ranked])
    ax.barh(y, means, color=[style(index[t["number"]])["color"] for t in ranked],
            height=0.62, edgecolor=SURFACE, linewidth=2)
    lows = [values[t["number"]][1] for t in ranked]
    highs = [values[t["number"]][2] for t in ranked]
    has_ci = all(v is not None and np.isfinite(v) for v in lows + highs)
    if has_ci:
        ax.errorbar(means, y, xerr=[means - np.array(lows), np.array(highs) - means],
                    fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.2, capsize=3)
    right = np.array(highs) if has_ci else means
    for yi, v, r in zip(y, means, right):
        ax.text(max(r, v, 0), yi, f"  {fmt_value(v)}", va="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_yticks(y, [trial_label(t).rsplit(" · q80", 1)[0] for t in ranked], color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, length=0)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    lo = min(0.0, float(np.min(lows)) if has_ci else float(means.min()))
    ax.set_xlim(lo * 1.15, max(float(np.max(right)), 0) * 1.18 or 1)
    fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=14, fontweight="bold")
    order = {True: "best (lowest) first", False: "best (highest) first", None: "by trial"}[lower_is_better]
    ax.set_title(f"{subtitle} · {order}" + (" · error bars: 95% CI" if has_ci else ""),
                 loc="left", color=TEXT_SECONDARY, fontsize=9.5, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_objective(trials, study, path):
    plot_bars(trials, {t["number"]: (t["objective"], None, None) for t in trials},
              "Optuna objective by trial",
              "Last-checkpoint validation q80 gap (Optuna objective, lower is better)",
              study, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--study", default=DEFAULT_STUDY, help="Optuna study name")
    ap.add_argument("--storage", default="optuna_journal.log", help="Optuna storage (as in main.py)")
    ap.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS,
                    help="study result folder(s) inside results/ (glob patterns)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder suffix (results/compare_<out>)")
    ap.add_argument("--eval-split", default="val/test_dataset.json",
                    help="instances (json list with CP-SAT 'score') for the src/metrics evaluation")
    ap.add_argument("--no-eval", action="store_true", help="skip the src/metrics evaluation")
    ap.add_argument("--no-representation", action="store_true",
                    help="skip expressiveness/heterophily in the evaluation (about half the time)")
    args = ap.parse_args()

    trials = collect_trials(args.folders, study_params(args.study, args.storage))
    if not trials:
        raise SystemExit(f"No finished trials found in results/{args.folders}")
    out_dir = os.path.join(RESULTS, f"compare_{args.out}")
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for n, (key, (title, ylabel)) in enumerate(VALIDATION_METRICS.items(), start=1):
        name = f"val{n}_{key}.png"
        if plot_validation_metric(trials, key, title, ylabel, args.study, os.path.join(out_dir, name)):
            written.append(name)
    for n, (key, (title, ylabel)) in enumerate(EPISODE_METRICS.items(), start=1):
        name = f"ep{n:02d}_{key}.png"
        if plot_episode_metric(trials, key, title, ylabel, args.study, os.path.join(out_dir, name)):
            written.append(name)
    plot_objective(trials, args.study, os.path.join(out_dir, "objective_q80_by_trial.png"))
    written.append("objective_q80_by_trial.png")

    evals = {}
    if not args.no_eval:
        split_path = os.path.join(ROOT, args.eval_split)
        for t in trials:
            result = evaluate_trial(t, split_path, not args.no_representation)
            if result is not None:
                evals[t["number"]] = result["summary"]
        split_name = os.path.splitext(os.path.basename(args.eval_split))[0]
        n_inst = next(iter(evals.values()))["n_instances"] if evals else 0
        subtitle = f"{args.study} · best checkpoint on {split_name} ({n_inst} instances)"
        for n, (key, (title, xlabel, lower)) in enumerate(EVAL_METRICS.items(), start=1):
            values = {}
            for number, s in evals.items():
                d = s["metrics"].get(key, {})
                if d.get("n"):
                    values[number] = (d["mean"], d.get("ci95_low"), d.get("ci95_high"))
            name = f"eval{n:02d}_{key}.png"
            if plot_bars(trials, values, title, f"{xlabel} (mean per instance)", subtitle,
                         os.path.join(out_dir, name), lower):
                written.append(name)
        for n, (key, (title, xlabel, lower)) in enumerate(EVAL_SCALARS.items(), start=len(EVAL_METRICS) + 1):
            values = {number: ((s.get("model") or {}).get(key, s.get(key)), None, None)
                      for number, s in evals.items()}
            values = {k: v for k, v in values.items() if v[0] is not None}
            name = f"eval{n:02d}_{key}.png"
            if plot_bars(trials, values, title, xlabel, subtitle, os.path.join(out_dir, name), lower):
                written.append(name)

    rows = []
    for t in trials:
        s = t["summary"]
        row = {"trial": t["number"], "objective_last_val_q80": t["objective"],
               **t["params"],
               "best_val_avg_gap": s.get("best_validation_avg_gap"),
               "test_avg_gap": s.get("test_avg_gap"), "test_q80_gap": s.get("test_q80_gap"),
               "runtime_min": (s.get("total_runtime_sec") or np.nan) / 60}
        e = evals.get(t["number"])
        if e:
            row.update({f"eval_{k}": e["metrics"].get(k, {}).get("mean") for k in EVAL_METRICS})
            row.update({f"eval_{k}": (e.get("model") or {}).get(k, e.get(k)) for k in EVAL_SCALARS})
        row["run_dir"] = os.path.relpath(t["dir"], ROOT)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("objective_last_val_q80")
    table.to_csv(os.path.join(out_dir, "trials_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    show = [c for c in table.columns if c != "run_dir" and not c.startswith("eval_")]
    show += [c for c in ("eval_relative_error", "eval_scheduling_score", "eval_feasibility_rate")
             if c in table]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n{len(trials)} trials, {len(written)} figures -> {out_dir}")


if __name__ == "__main__":
    main()
