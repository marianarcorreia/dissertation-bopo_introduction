"""Q3 diagnostic: edge count per step and per instance size, for all 3 representations.

Read-only structural probe - no training, no model weights involved. Drives each
environment with its own expert_action() (the same earliest-completion-time dispatch
rule used for BOPO's warm-start teacher, see src/*.py:expert_action), and records the
total edge count across all edge types at every step of the episode. This measures
the representations' structural density/dynamism claims from README section 1
(G^disj = O(|O|^2) worst case vs G^OM/G^OJM ~ linear) directly, instead of asserting it.

Usage: python edge_scalability_probe.py
Output: results/q3_edge_scalability/edge_scalability.csv and edge_scalability.png
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.train import _resolve_representation_modules, generate_train_instances

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "q3_edge_scalability")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = ["oo", "om", "ojm"]
REP_LABELS = {"oo": "Disjunctive (O-O)", "om": "Hetero O-M", "ojm": "Hetero O-J-M"}

# Matches README section 6.1's R1-R3 (small) / R4-R6 (medium) instance sizes.
SIZES = {
    "small (6J x 5M)": dict(j_min=6, j_max=6, m_min=5, m_max=5, op_max=6, max_processing=100),
    "medium (10J x 8M)": dict(j_min=10, j_max=10, m_min=8, m_max=8, op_max=6, max_processing=100),
}

N_INSTANCES_PER_SIZE = 5  # averaged per (representation, size) to smooth over instance-to-instance variance

sns.set_theme(style="whitegrid", palette="pastel")


def total_edges(state):
    return sum(int(state[et].edge_index.shape[1]) for et in state.edge_types)


records = []
trajectories = {}  # (rep, size_label) -> list of edge-count trajectories (one per instance)

for rep in REPS:
    _, EnvClass, _ = _resolve_representation_modules(rep)
    for size_label, size_cfg in SIZES.items():
        train_config = {
            "n_cases": N_INSTANCES_PER_SIZE,
            "range_jobs": (size_cfg["j_min"], size_cfg["j_max"]),
            "range_machines": (size_cfg["m_min"], size_cfg["m_max"]),
            "range_op_per_job": (5, size_cfg["op_max"]),
            "max_processing": size_cfg["max_processing"],
        }
        instances = generate_train_instances(train_config)
        env = EnvClass(instances, 1, 1)  # mask_option=1, sel_k=1 (train.py defaults)

        key = (rep, size_label)
        trajectories[key] = []
        step_times = []

        for inst_idx in range(len(instances)):
            state = env.reset(sel_index=inst_idx)
            traj = [total_edges(state)]
            done = False
            while not done:
                action = env.expert_action()
                t0 = time.perf_counter()
                state, _reward, done, _info = env.step(action)
                step_times.append(time.perf_counter() - t0)
                traj.append(total_edges(state) if not done else traj[-1])
            trajectories[key].append(traj)

        edge_at_start = [t[0] for t in trajectories[key]]
        edge_at_half = [t[len(t) // 2] for t in trajectories[key]]
        edge_at_end = [t[-2] if len(t) > 1 else t[-1] for t in trajectories[key]]  # last pre-terminal state

        records.append({
            "representation": rep,
            "size": size_label,
            "n_instances": len(instances),
            "n_ops_total": env.num_operations if hasattr(env, "num_operations") else None,
            "n_machines": env.num_machines if hasattr(env, "num_machines") else None,
            "edges_at_start_mean": np.mean(edge_at_start),
            "edges_at_half_mean": np.mean(edge_at_half),
            "edges_at_end_mean": np.mean(edge_at_end),
            "mean_step_time_ms": np.mean(step_times) * 1000,
            "episode_length_mean": np.mean([len(t) - 1 for t in trajectories[key]]),
        })

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "edge_scalability.csv"), index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure: edge count trajectory (normalized to % of episode progress) per
# representation, one panel per instance size.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(SIZES), figsize=(6 * len(SIZES), 4.5), sharey=False)
colors = dict(zip(REPS, sns.color_palette("pastel", 3)))
for ax, size_label in zip(axes, SIZES):
    for rep in REPS:
        trajs = trajectories[(rep, size_label)]
        # resample each trajectory onto a common 0-100% progress axis, then average
        common_x = np.linspace(0, 1, 50)
        resampled = []
        for t in trajs:
            x = np.linspace(0, 1, len(t))
            resampled.append(np.interp(common_x, x, t))
        mean_traj = np.mean(resampled, axis=0)
        ax.plot(common_x * 100, mean_traj, label=REP_LABELS[rep], color=colors[rep], linewidth=2)
    ax.set_title(size_label)
    ax.set_xlabel("Episode progress (%)")
axes[0].set_ylabel("Total edge count (all edge types)")
axes[0].legend(title="Representation", frameon=False)
fig.suptitle("Edge count over an episode, by representation and instance size")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "edge_scalability.png"), dpi=150)
plt.close(fig)

print("\nSaved to", OUT_DIR)
