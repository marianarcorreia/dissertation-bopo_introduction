# import optuna

# # load optuna study
# study = optuna.load_study(
#     storage="sqlite:///optuna.db",
#     study_name="fjsp_tuning_oo",
# )

# # print best trial
# print("Best trial:")
# trial = study.best_trial
# print(f"  Value: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

import json


def normalize_metrics(metrics):
    if isinstance(metrics, dict):
        return metrics

    if isinstance(metrics, list):
        if not metrics:
            return {}

        if all(isinstance(item, dict) for item in metrics):
            keys = set()
            for item in metrics:
                keys.update(item.keys())

            normalized = {key: [] for key in keys}
            for item in metrics:
                for key in keys:
                    normalized[key].append(item.get(key))
            return normalized

    raise ValueError("Unsupported metrics JSON format. Expected dict or list of dicts.")

all_data = {}
for trial in range(6):

    val_data = normalize_metrics(json.load(open(f"results/optuna_oo_1775570534/trial_{trial}/validation_history.json")))
    episode_data = normalize_metrics(json.load(open(f"results/optuna_oo_1775570534/trial_{trial}/episode_metrics.json")))
    train_data = normalize_metrics(json.load(open(f"results/optuna_oo_1775570534/trial_{trial}/update_metrics.json")))
    all_data[trial] = {
        "validation": val_data,
        "episode": episode_data,
        "train": train_data
    }

import matplotlib

# Use a non-interactive backend so plotting works without a Tk installation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2)
for trial in range(6):
    val_data = all_data[trial]["validation"]
    episode_data = all_data[trial]["episode"]
    train_data = all_data[trial]["train"]

    # Loss evolution in (1, 1) — BOPO has no critic, so train.py logs this as "actor_loss"
    # (there is no separate "policy_loss" field in update_metrics.json).
    axes[0, 0].plot(train_data["actor_loss"], label=f"Trial {trial}")
    axes[0, 0].set_title("Actor Loss Evolution")
    axes[0, 0].set_xlabel("Update Step")
    axes[0, 0].set_ylabel("Actor Loss")
    axes[0, 0].legend()

    # Reward evolution in (1, 2)
    axes[0, 1].plot(episode_data["episode_reward"], label=f"Trial {trial}")
    axes[0, 1].set_title("Episode Reward Evolution")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Episode Reward")
    axes[0, 1].legend()

    # Validation performance in (2, 1)
    axes[1, 0].plot(val_data["avg_gap"], label=f"Trial {trial}")
    axes[1, 0].set_title("Average Gap Evolution")
    axes[1, 0].set_xlabel("Validation Step")
    axes[1, 0].set_ylabel("Average Gap")
    axes[1, 0].legend()

    # Update duration in (2, 2) — no action-entropy metric is tracked by train.py,
    # so this plots something that actually exists in update_metrics.json instead.
    axes[1, 1].plot(train_data["update_duration_sec"], label=f"Trial {trial}")
    axes[1, 1].set_title("BOPO Update Duration Evolution")
    axes[1, 1].set_xlabel("Update Step")
    axes[1, 1].set_ylabel("Duration (s)")
    axes[1, 1].legend()
plt.tight_layout()
# plt.show()
plt.savefig("results/optuna_oo_1775570534/trial_comparison.png", dpi=150)
print("Saved plot to results/optuna_oo_1775570534/trial_comparison.png")
