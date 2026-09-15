"""
Optuna Trials Comparison Dashboard
=================================
Run with:
    streamlit run optuna_dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.chart_theme import color_by, register_theme
from src.utils.dashboard_data import to_long_metrics

RESULTS_DIR = Path("results")


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_metrics(raw: Any) -> pd.DataFrame:
    """Convert metrics JSON into a DataFrame.

    Supports:
    - list[dict]: row-wise metric records
    - dict[str, list]: column-wise metric series
    """
    if isinstance(raw, list):
        if not raw:
            return pd.DataFrame()
        if not all(isinstance(item, dict) for item in raw):
            raise ValueError("Expected list of dictionaries in metrics JSON.")
        return pd.DataFrame(raw)

    if isinstance(raw, dict):
        # Keep only keys that can become DataFrame columns safely.
        serializable = {}
        for key, value in raw.items():
            if isinstance(value, list):
                serializable[key] = value
            elif _is_scalar_number(value) or isinstance(value, str):
                serializable[key] = [value]
        if not serializable:
            return pd.DataFrame()
        return pd.DataFrame(serializable)

    raise ValueError("Unsupported metrics JSON format. Expected dict or list.")


def find_optuna_studies(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []

    studies: list[Path] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        has_trial_dirs = any(p.is_dir() and p.name.startswith("trial_") for p in child.iterdir())
        if has_trial_dirs:
            studies.append(child)
    return studies


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = load_json(str(path))
        return normalize_metrics(raw)
    except Exception:
        return pd.DataFrame()


def _best_trial_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "best_validation_avg_gap" in out.columns:
        out["rank_best_validation_avg_gap"] = (
            out["best_validation_avg_gap"].rank(method="min", ascending=True)
        )
    else:
        out["rank_best_validation_avg_gap"] = pd.NA
    return out


def _to_long_metrics(df: pd.DataFrame, trial_name: str, source: str) -> pd.DataFrame:
    return to_long_metrics(df, trial_name, source, series_col="trial")


@st.cache_data(show_spinner=True)
def load_study_data(study_dir_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_dir = Path(study_dir_str)
    trial_dirs = sorted([p for p in study_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")])

    summary_rows: list[dict[str, Any]] = []
    long_frames: list[pd.DataFrame] = []

    for trial_dir in trial_dirs:
        trial_name = trial_dir.name

        run_summary_path = trial_dir / "run_summary.json"
        val_path = trial_dir / "validation_history.json"
        epi_path = trial_dir / "episode_metrics.json"
        upd_path = trial_dir / "update_metrics.json"

        run_summary = {}
        if run_summary_path.exists():
            try:
                run_summary = load_json(str(run_summary_path))
            except Exception:
                run_summary = {}

        val_df = safe_load_df(val_path)
        epi_df = safe_load_df(epi_path)
        upd_df = safe_load_df(upd_path)

        long_frames.append(_to_long_metrics(val_df, trial_name, "validation"))
        long_frames.append(_to_long_metrics(epi_df, trial_name, "episode"))
        long_frames.append(_to_long_metrics(upd_df, trial_name, "update"))

        row: dict[str, Any] = {
            "trial": trial_name,
            "episodes_completed": run_summary.get("episodes_completed"),
            "updates_completed": run_summary.get("updates_completed"),
            "best_validation_avg_gap": run_summary.get("best_validation_avg_gap"),
            "best_validation_q80_gap": run_summary.get("best_validation_q80_gap"),
            "best_difference": run_summary.get("best_difference"),
            "total_runtime_sec": run_summary.get("total_runtime_sec"),
            "actor_param_count": run_summary.get("actor_param_count"),
        }

        if not val_df.empty:
            if "avg_gap" in val_df.columns:
                avg_gap = pd.to_numeric(val_df["avg_gap"], errors="coerce")
                row["min_avg_gap_from_history"] = avg_gap.min()
                row["final_avg_gap_from_history"] = avg_gap.iloc[-1]
            if "q80_gap" in val_df.columns:
                q80_gap = pd.to_numeric(val_df["q80_gap"], errors="coerce")
                row["min_q80_gap_from_history"] = q80_gap.min()

        if not epi_df.empty and "episode_reward" in epi_df.columns:
            reward = pd.to_numeric(epi_df["episode_reward"], errors="coerce")
            row["mean_reward_last_50"] = reward.tail(50).mean()
            row["final_reward"] = reward.iloc[-1]

        if not epi_df.empty and "action_entropy_uncertainty_gap" in epi_df.columns:
            gap = pd.to_numeric(epi_df["action_entropy_uncertainty_gap"], errors="coerce")
            if gap.notna().any():
                row["mean_uncertainty_gap_last_50"] = gap.tail(50).mean()
                row["final_uncertainty_gap"] = gap.iloc[-1]

        if not upd_df.empty and "policy_loss" in upd_df.columns:
            pl = pd.to_numeric(upd_df["policy_loss"], errors="coerce")
            row["mean_policy_loss_last_10_updates"] = pl.tail(10).mean()

        if not upd_df.empty and "action_entropy_uncertainty_gap" in upd_df.columns:
            upd_gap = pd.to_numeric(upd_df["action_entropy_uncertainty_gap"], errors="coerce")
            if upd_gap.notna().any():
                row["mean_uncertainty_gap_last_10_updates"] = upd_gap.tail(10).mean()

        if not upd_df.empty and "action_entropy_max_entropy_normalized" in upd_df.columns:
            upd_norm = pd.to_numeric(upd_df["action_entropy_max_entropy_normalized"], errors="coerce")
            if upd_norm.notna().any():
                row["mean_max_entropy_norm_last_10_updates"] = upd_norm.tail(10).mean()

        if not upd_df.empty and "action_entropy_effective_action_count" in upd_df.columns:
            upd_eff_count = pd.to_numeric(upd_df["action_entropy_effective_action_count"], errors="coerce")
            if upd_eff_count.notna().any():
                row["mean_effective_action_count_last_10_updates"] = upd_eff_count.tail(10).mean()

        if not upd_df.empty and "action_entropy_effective_action_ratio" in upd_df.columns:
            upd_eff_ratio = pd.to_numeric(upd_df["action_entropy_effective_action_ratio"], errors="coerce")
            if upd_eff_ratio.notna().any():
                row["mean_effective_action_ratio_last_10_updates"] = upd_eff_ratio.tail(10).mean()

        if not upd_df.empty and "action_entropy_symmetry_corrected" in upd_df.columns:
            upd_sym = pd.to_numeric(upd_df["action_entropy_symmetry_corrected"], errors="coerce")
            if upd_sym.notna().any():
                row["mean_symmetry_corrected_entropy_last_10_updates"] = upd_sym.tail(10).mean()

        if not upd_df.empty and "action_entropy_random_ref" in upd_df.columns:
            upd_ref = pd.to_numeric(upd_df["action_entropy_random_ref"], errors="coerce")
            if upd_ref.notna().any():
                row["mean_random_entropy_ref_last_10_updates"] = upd_ref.tail(10).mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = _best_trial_rank(summary_df)

    long_df = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame(
        columns=["trial", "source", "step", "metric", "value"]
    )
    return summary_df, long_df


def main() -> None:
    try:
        st.set_page_config(page_title="Optuna Trial Comparison", layout="wide", page_icon="🔬")
    except st.errors.StreamlitAPIException:
        pass  # already set by the caller (e.g. when embedded as a dashboard page)
    register_theme()
    st.title("Optuna Study Trial Comparison")
    st.caption("Compare all trial histories and summary metrics for one Optuna study folder.")

    studies = find_optuna_studies(RESULTS_DIR)
    if not studies:
        st.error("No Optuna-style study folders found under results/.")
        st.stop()

    study_options = [p.name for p in studies]
    selected_study_name = st.sidebar.selectbox("Study folder", study_options)
    selected_study = next(p for p in studies if p.name == selected_study_name)

    with st.spinner("Loading study data..."):
        summary_df, long_df = load_study_data(str(selected_study))

    if summary_df.empty:
        st.warning("No trial data found in the selected study folder.")
        st.stop()

    all_trials = summary_df["trial"].tolist()
    selected_trials = st.sidebar.multiselect("Trials", all_trials, default=all_trials)
    if not selected_trials:
        st.warning("Select at least one trial in the sidebar.")
        st.stop()

    filtered_summary = summary_df[summary_df["trial"].isin(selected_trials)].copy()
    filtered_long = long_df[long_df["trial"].isin(selected_trials)].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trials in study", len(all_trials))
    c2.metric("Selected trials", len(selected_trials))
    c3.metric(
        "Best validation avg gap",
        f"{filtered_summary['best_validation_avg_gap'].min():.4f}"
        if "best_validation_avg_gap" in filtered_summary and filtered_summary["best_validation_avg_gap"].notna().any()
        else "n/a",
    )
    c4.metric(
        "Fastest runtime (s)",
        f"{filtered_summary['total_runtime_sec'].min():.1f}"
        if "total_runtime_sec" in filtered_summary and filtered_summary["total_runtime_sec"].notna().any()
        else "n/a",
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Trial Overview", "Metric Curves", "Distribution", "Raw Data"])

    with tab1:
        st.subheader("Trial summary")
        display_cols = [
            "trial",
            "rank_best_validation_avg_gap",
            "best_validation_avg_gap",
            "min_avg_gap_from_history",
            "final_avg_gap_from_history",
            "mean_max_entropy_norm_last_10_updates",
            "mean_effective_action_count_last_10_updates",
            "mean_effective_action_ratio_last_10_updates",
            "mean_symmetry_corrected_entropy_last_10_updates",
            "mean_uncertainty_gap_last_10_updates",
            "mean_random_entropy_ref_last_10_updates",
            "mean_reward_last_50",
            "total_runtime_sec",
            "episodes_completed",
            "updates_completed",
        ]
        display_cols = [c for c in display_cols if c in filtered_summary.columns]
        st.dataframe(
            filtered_summary[display_cols].sort_values(by=display_cols[1] if len(display_cols) > 1 else "trial"),
            width='stretch',
            hide_index=True,
        )

        if "best_validation_avg_gap" in filtered_summary.columns:
            chart_df = filtered_summary[["trial", "best_validation_avg_gap"]].dropna()
            if not chart_df.empty:
                bar = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("trial:N", sort="-y", title="Trial"),
                    y=alt.Y("best_validation_avg_gap:Q", title="Best validation avg gap (lower is better)"),
                    color=color_by("trial", sorted(chart_df["trial"].unique().tolist()), title="Trial"),
                    tooltip=["trial", alt.Tooltip("best_validation_avg_gap:Q", format=".4f")],
                )
                st.altair_chart(bar, width='stretch')

    with tab2:
        st.subheader("Metric curves across trials")
        if filtered_long.empty:
            st.info("No numeric metric history found for selected trials.")
        else:
            source_options = sorted(filtered_long["source"].dropna().unique().tolist())
            default_source = "update" if "update" in source_options else source_options[0]
            selected_source = st.selectbox(
                "Metric source",
                source_options,
                index=source_options.index(default_source),
            )

            source_df = filtered_long[filtered_long["source"] == selected_source]
            metric_options = sorted(source_df["metric"].dropna().unique().tolist())
            preferred_metrics = [
                "action_entropy_symmetry_corrected",
                "action_entropy_max_entropy_normalized",
                "action_entropy_effective_action_ratio",
                "action_entropy_effective_action_count",
                "action_entropy_uncertainty_gap",
                "action_entropy_random_ref",
                "action_entropy",
            ]
            default_metric = next((m for m in preferred_metrics if m in metric_options), metric_options[0])
            selected_metric = st.selectbox(
                "Metric",
                metric_options,
                index=metric_options.index(default_metric),
            )

            plot_df = source_df[source_df["metric"] == selected_metric].copy()
            plot_df = plot_df.sort_values(["trial", "step"])

            smooth_window = st.slider("Smoothing window", min_value=1, max_value=50, value=1)
            if smooth_window > 1:
                plot_df["value"] = (
                    plot_df.groupby("trial", as_index=False)["value"]
                    .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
                )

            line = alt.Chart(plot_df).mark_line().encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("value:Q", title=selected_metric),
                color=color_by("trial", selected_trials, title="Trial"),
                tooltip=["trial", "step", alt.Tooltip("value:Q", format=".6f")],
            ).interactive()
            st.altair_chart(line, width='stretch')

    with tab3:
        st.subheader("Metric distribution by trial")
        if filtered_long.empty:
            st.info("No numeric metric history found for selected trials.")
        else:
            source_options = sorted(filtered_long["source"].dropna().unique().tolist())
            selected_source = st.selectbox("Distribution source", source_options, key="dist_source")
            source_df = filtered_long[filtered_long["source"] == selected_source]

            metric_options = sorted(source_df["metric"].dropna().unique().tolist())
            selected_metric = st.selectbox("Distribution metric", metric_options, key="dist_metric")

            dist_df = source_df[source_df["metric"] == selected_metric]
            box = alt.Chart(dist_df).mark_boxplot().encode(
                x=alt.X("trial:N", title="Trial"),
                y=alt.Y("value:Q", title=selected_metric),
                color=color_by("trial", selected_trials, title="Trial", show_legend=False),
                tooltip=["trial", alt.Tooltip("value:Q", format=".6f")],
            )
            st.altair_chart(box, width='stretch')

    with tab4:
        st.subheader("Raw tables")
        st.markdown("Summary table")
        st.dataframe(filtered_summary, width='stretch', hide_index=True)

        st.markdown("Long metrics table")
        st.dataframe(filtered_long, width='stretch', hide_index=True)

        st.download_button(
            "Download summary CSV",
            data=filtered_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_study_name}_trial_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download long metrics CSV",
            data=filtered_long.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_study_name}_long_metrics.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
