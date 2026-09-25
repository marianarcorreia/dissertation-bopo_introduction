"""Convergence analysis + comparison figures for the 9 representation x backbone runs.

Reads results/convergence_check_<rep>_<backbone>/ for rep in {oo, om, ojm} and
backbone in {gat, gin, transformer} (parsed from the folder name — run_summary.json
does not carry the backbone). Produces:
  - results/comparison_final_3layers_omj/convergence_summary.csv
  - results/comparison_final_3layers_omj/plots/*.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
SELK2 = "--selk100" in sys.argv
OUT_DIR = os.path.join(RESULTS, "comparison_final_3layers_selk100" + ("_selk2" if SELK2 else ""))
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

REPS = ["oo", "om", "ojm"]
BACKBONES = ["gin", "gat", "transformer"]
REP_LABELS = {"oo": "Disjunctive (O-O)", "om": "Hetero O-M", "ojm": "Hetero O-J-M"}

sns.set_theme(style="whitegrid", palette="pastel")

# Prefer the 3-layer ablation run when it exists (currently only ojm), else
# fall back to the convergence-check run.
RUN_TEMPLATES = ["train_run_{rep}", "ablation_{rep}_{backbone}_layers3"]


def run_name(rep, backbone):
    templates = RUN_TEMPLATES
    if SELK2 and backbone == "transformer":
        templates = ["ablation_{rep}_{backbone}_layers3_selk2"] + RUN_TEMPLATES
    for tmpl in templates:
        name = tmpl.format(rep=rep, backbone=backbone)
        if os.path.exists(os.path.join(RESULTS, name, "run_summary.json")):
            return name
    return RUN_TEMPLATES[-1].format(rep=rep, backbone=backbone)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def linreg_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    slope, _intercept = np.polyfit(x, y, 1)
    return slope


records = []
val_curves = {}
episode_curves = {}

for rep in REPS:
    for backbone in BACKBONES:
        name = run_name(rep, backbone)
        run_dir = os.path.join(RESULTS, name)
        summary_path = os.path.join(run_dir, "run_summary.json")
        val_path = os.path.join(run_dir, "validation_history.json")
        ep_path = os.path.join(run_dir, "episode_metrics.json")
        test_path = os.path.join(run_dir, "test_metrics.json")
        if not os.path.exists(summary_path):
            print(f"MISSING: {name}")
            continue

        summary = load_json(summary_path)
        val_hist = load_json(val_path)
        ep_hist = load_json(ep_path)
        test_metrics = load_json(test_path) if os.path.exists(test_path) else {}

        episodes = [v["episode"] for v in val_hist]
        raw_gap = [v["avg_gap"] for v in val_hist]
        smooth_gap = [v["smoothed_avg_gap"] for v in val_hist]
        std_gap = [v["std_gap"] for v in val_hist]

        val_curves[name] = pd.DataFrame({
            "episode": episodes, "avg_gap": raw_gap,
            "smoothed_avg_gap": smooth_gap, "std_gap": std_gap,
        })
        episode_curves[name] = pd.DataFrame(ep_hist)

        # LCS over first 200 episodes (README definition)
        early_mask = [e <= 200 for e in episodes]
        early_ep = [e for e, m in zip(episodes, early_mask) if m]
        early_gap = [g for g, m in zip(raw_gap, early_mask) if m]
        lcs_early = linreg_slope(early_ep, early_gap)

        # Full-run slope
        lcs_full = linreg_slope(episodes, raw_gap)

        # Late-phase (last 5 validation checkpoints) stability, on the smoothed
        # curve so single-checkpoint noise (n=30-40 val instances) doesn't
        # dominate the convergence call.
        n_tail = min(5, len(episodes))
        tail_ep = episodes[-n_tail:]
        tail_smooth = smooth_gap[-n_tail:]
        late_slope = linreg_slope(tail_ep, tail_smooth)
        late_slope_per100 = late_slope * 100 if not np.isnan(late_slope) else np.nan
        tail_mean = np.mean(tail_smooth)
        tail_range = (max(tail_smooth) - min(tail_smooth))
        tail_relrange = tail_range / tail_mean if tail_mean else np.nan

        best_val_gap = summary.get("best_validation_avg_gap")
        test_gap = summary.get("test_avg_gap")
        test_std = summary.get("test_std_gap")
        gen_gap = (test_gap - best_val_gap) if (test_gap is not None and best_val_gap is not None) else np.nan

        # Convergence verdict: flat late-phase slope + bounded relative
        # fluctuation among the last checkpoints + test gap not blown up
        # relative to best validation gap (rules out overfitting to val set).
        converged = (
            abs(late_slope_per100) < 0.02
            and tail_relrange < 0.30
            and abs(gen_gap) < 0.06
        )

        ep_df = pd.DataFrame(ep_hist)
        actor_loss_first10 = ep_df["actor_loss"].iloc[:10].mean() if not ep_df.empty else np.nan
        actor_loss_last10 = ep_df["actor_loss"].iloc[-10:].mean() if not ep_df.empty else np.nan
        entropy_first10 = ep_df["action_entropy_max_entropy_normalized"].iloc[:10].mean() if not ep_df.empty else np.nan
        entropy_last10 = ep_df["action_entropy_max_entropy_normalized"].iloc[-10:].mean() if not ep_df.empty else np.nan

        records.append({
            "representation": rep,
            "backbone": backbone,
            "run_name": name,
            "episodes_completed": summary.get("episodes_completed"),
            "actor_params": summary.get("actor_param_count"),
            "lcs_first200_per_ep": lcs_early,
            "lcs_full_per_ep": lcs_full,
            "late_slope_per100ep": late_slope_per100,
            "late_tail_relrange": tail_relrange,
            "best_val_gap": best_val_gap,
            "final_val_gap": raw_gap[-1] if raw_gap else np.nan,
            "test_gap": test_gap,
            "test_std": test_std,
            "generalization_gap": gen_gap,
            "actor_loss_first10": actor_loss_first10,
            "actor_loss_last10": actor_loss_last10,
            "entropy_norm_first10": entropy_first10,
            "entropy_norm_last10": entropy_last10,
            "converged": converged,
        })

if not records:
    raise SystemExit(f"No runs found under {RESULTS} for templates {RUN_TEMPLATES}")
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "convergence_summary.csv"), index=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure 1: validation gap learning curves, one subplot per representation,
# one line per backbone.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
colors = dict(zip(BACKBONES, sns.color_palette("pastel", 3)))
for ax, rep in zip(axes, REPS):
    for backbone in BACKBONES:
        name = run_name(rep, backbone)
        if name not in val_curves:
            continue
        vdf = val_curves[name]
        ax.plot(vdf["episode"], vdf["smoothed_avg_gap"], label=backbone.upper(),
                color=colors[backbone], linewidth=2)
        ax.fill_between(vdf["episode"], vdf["avg_gap"], vdf["smoothed_avg_gap"],
                         color=colors[backbone], alpha=0.15)
    ax.set_title(REP_LABELS[rep])
    ax.set_xlabel("Episode")
    ax.axhline(0, color="gray", linewidth=0.5)
axes[0].set_ylabel("Validation makespan gap")
axes[0].legend(title="Backbone", frameon=False)
fig.suptitle("Validation gap learning curves — 9 representation x backbone runs")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig1_learning_curves.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: final test gap bar chart, grouped by representation, colored by backbone
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
bar_df = df.copy()
bar_df["rep_label"] = bar_df["representation"].map(REP_LABELS)
sns.barplot(data=bar_df, x="rep_label", y="test_gap", hue="backbone",
            palette="pastel", ax=ax, errorbar=None)
ax.set_ylabel("Test makespan gap (mean)")
ax.set_xlabel("")
ax.set_title("Held-out test gap by representation and GNN backbone")
ax.legend(title="Backbone", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig2_test_gap_bars.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: actor loss curves (log scale), same layout as fig 1
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, rep in zip(axes, REPS):
    for backbone in BACKBONES:
        name = run_name(rep, backbone)
        if name not in episode_curves:
            continue
        edf = episode_curves[name]
        smoothed = edf["actor_loss"].rolling(10, min_periods=1).mean()
        ax.plot(edf["episode"], smoothed, label=backbone.upper(),
                color=colors[backbone], linewidth=1.8)
    ax.set_title(REP_LABELS[rep])
    ax.set_xlabel("Episode")
    ax.set_yscale("log")
axes[0].set_ylabel("Actor loss (10-ep rolling mean, log scale)")
axes[0].legend(title="Backbone", frameon=False)
fig.suptitle("Actor loss trajectories — 9 representation x backbone runs")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig3_actor_loss.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: normalized action entropy curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, rep in zip(axes, REPS):
    for backbone in BACKBONES:
        name = run_name(rep, backbone)
        if name not in episode_curves:
            continue
        edf = episode_curves[name]
        smoothed = edf["action_entropy_max_entropy_normalized"].rolling(10, min_periods=1).mean()
        ax.plot(edf["episode"], smoothed, label=backbone.upper(),
                color=colors[backbone], linewidth=1.8)
    ax.set_title(REP_LABELS[rep])
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Normalized action entropy (10-ep rolling mean)")
axes[0].legend(title="Backbone", frameon=False)
fig.suptitle("Policy entropy trajectories — 9 representation x backbone runs")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig4_action_entropy.png"), dpi=150)
plt.close(fig)

print("\nSaved figures to", PLOTS_DIR)
