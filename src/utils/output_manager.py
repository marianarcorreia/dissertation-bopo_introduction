import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class OutputManager:
    def __init__(self, output_dir="results", run_name="run", history_file="validation_history.json", plot_dir="plots"):
        self.output_dir = output_dir
        self.run_name = run_name
        self.history_file = history_file
        self.episode_metrics_file = "episode_metrics.json"
        self.update_metrics_file = "update_metrics.json"
        self.run_summary_file = "run_summary.json"
        self.test_metrics_file = "test_metrics.json"
        self.run_dir = self._create_run_directory(self.run_name)
        self.plot_dir = os.path.join(self.run_dir, plot_dir)

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _sanitize_run_name(self, run_name):
        """Sanitize each path segment while preserving '/' as an intentional
        directory separator (e.g. param.py's "<study_folder>/trial_<N>" runs,
        which rely on nesting to produce results/<study_folder>/trial_<N>/)."""
        segments = str(run_name).replace("\\", "/").split("/")
        safe_segments = []
        for segment in segments:
            segment = segment.strip().replace(" ", "_")
            if segment in ("", ".", ".."):
                segment = "run"
            safe_segments.append(segment)
        return safe_segments

    def _create_run_directory(self, run_name):
        os.makedirs(self.output_dir, exist_ok=True)

        safe_segments = self._sanitize_run_name(run_name)
        safe_name = os.path.join(*safe_segments)
        base_path = os.path.join(self.output_dir, safe_name)

        if not os.path.exists(base_path):
            return base_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = os.path.join(self.output_dir, f"{safe_name}_{timestamp}")
        index = 1
        while os.path.exists(candidate):
            candidate = os.path.join(self.output_dir, f"{safe_name}_{timestamp}_{index}")
            index += 1
        return candidate

    def log(self, message, tag="TRAIN"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{tag}][{timestamp}] {message}")

    def history_path(self):
        return os.path.join(self.run_dir, self.history_file)

    def episode_metrics_path(self):
        return os.path.join(self.run_dir, self.episode_metrics_file)

    def update_metrics_path(self):
        return os.path.join(self.run_dir, self.update_metrics_file)

    def run_summary_path(self):
        return os.path.join(self.run_dir, self.run_summary_file)

    def test_metrics_path(self):
        return os.path.join(self.run_dir, self.test_metrics_file)

    def save_json(self, data, path):
        with open(path, "w") as outfile:
            json.dump(data, outfile)

    def append_json_entry(self, data_list, entry, path):
        data_list.append(entry)
        self.save_json(data_list, path)
        return data_list

    def save_validation_history(self, history):
        self.save_json(history, self.history_path())

    def append_validation_history(self, history, entry):
        return self.append_json_entry(history, entry, self.history_path())

    def append_episode_metrics(self, metrics, entry):
        return self.append_json_entry(metrics, entry, self.episode_metrics_path())

    def append_update_metrics(self, metrics, entry):
        return self.append_json_entry(metrics, entry, self.update_metrics_path())

    def save_run_summary(self, summary):
        self.save_json(summary, self.run_summary_path())

    def save_test_metrics(self, metrics):
        """One-time, post-training report on the held-out test split (see
        src/train.py's evaluate_on_test_set call and
        src/utils/validation_utils.py:get_test_dataset). Kept in its own file, separate
        from validation_history.json, so it's never mistaken for one of the periodic
        during-training checkpoints."""
        self.save_json(metrics, self.test_metrics_path())

    def _plot_series(self, x_values, y_values, title, x_label, y_label, save_name, color="#8da0cb"):
        if len(x_values) == 0 or len(y_values) == 0:
            return None

        plt.figure(figsize=(9, 5))
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.plot(x_values, y_values, marker="o", linewidth=2, color=color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()

        output_path = os.path.join(self.plot_dir, save_name)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def plot_validation_gap(self, history, save_name="validation_gap_curve.png"):
        if len(history) == 0:
            return None

        episodes = [row["episode"] for row in history]
        avg_gaps = [row["avg_gap"] for row in history]
        std_gaps = [row.get("std_gap", 0.0) for row in history]

        plt.figure(figsize=(9, 5))
        plt.style.use("seaborn-v0_8-whitegrid")

        line_color = "#8da0cb"
        fill_color = "#c6dbef"

        plt.plot(episodes, avg_gaps, marker="o", linewidth=2, color=line_color, label="Average gap")
        lower = [m - s for m, s in zip(avg_gaps, std_gaps)]
        upper = [m + s for m, s in zip(avg_gaps, std_gaps)]
        plt.fill_between(episodes, lower, upper, alpha=0.35, color=fill_color, label="±1 std")

        plt.title("Validation Gap Over Training")
        plt.xlabel("Episode")
        plt.ylabel("Gap (makespan_model / makespan_ref - 1)")
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(self.plot_dir, save_name)
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_episode_metrics(self, episode_metrics):
        if len(episode_metrics) == 0:
            return {}

        episodes = [row["episode"] for row in episode_metrics]
        rewards = [row.get("episode_reward", 0.0) for row in episode_metrics]
        makespans = [row.get("makespan", 0.0) for row in episode_metrics]
        durations = [row.get("episode_duration_sec", 0.0) for row in episode_metrics]
        val_avg_gaps = [row.get("validation_avg_gap") for row in episode_metrics]

        outputs = {}
        outputs["reward"] = self._plot_series(
            episodes,
            rewards,
            "Episode Reward Over Training",
            "Episode",
            "Reward",
            "episode_reward_curve.png",
            color="#66c2a5",
        )
        outputs["makespan"] = self._plot_series(
            episodes,
            makespans,
            "Episode Makespan Over Training",
            "Episode",
            "Makespan",
            "episode_makespan_curve.png",
            color="#fc8d62",
        )
        outputs["duration"] = self._plot_series(
            episodes,
            durations,
            "Episode Wall Time Over Training",
            "Episode",
            "Seconds",
            "episode_duration_curve.png",
            color="#8da0cb",
        )

        val_points_x = [episodes[i] for i, v in enumerate(val_avg_gaps) if v is not None]
        val_points_y = [v for v in val_avg_gaps if v is not None]
        outputs["validation_avg_gap"] = self._plot_series(
            val_points_x,
            val_points_y,
            "Validation Average Gap Over Training",
            "Episode",
            "Avg Gap",
            "validation_avg_gap_points.png",
            color="#e78ac3",
        )

        return outputs

    def plot_update_metrics(self, update_metrics):
        if len(update_metrics) == 0:
            return {}

        episodes = [row["episode"] for row in update_metrics]
        actor_losses = [row.get("actor_loss", 0.0) for row in update_metrics]
        critic_losses = [row.get("critic_loss", 0.0) for row in update_metrics]
        durations = [row.get("update_duration_sec", 0.0) for row in update_metrics]

        outputs = {}
        outputs["actor_loss"] = self._plot_series(
            episodes,
            actor_losses,
            "Actor Loss Over Updates",
            "Episode",
            "Actor Loss",
            "actor_loss_curve.png",
            color="#a6d854",
        )
        outputs["critic_loss"] = self._plot_series(
            episodes,
            critic_losses,
            "Critic Loss Over Updates",
            "Episode",
            "Critic Loss",
            "critic_loss_curve.png",
            color="#ffd92f",
        )
        outputs["update_duration"] = self._plot_series(
            episodes,
            durations,
            "BOPO Update Time Over Training",
            "Episode",
            "Seconds",
            "update_duration_curve.png",
            color="#e5c494",
        )

        return outputs
