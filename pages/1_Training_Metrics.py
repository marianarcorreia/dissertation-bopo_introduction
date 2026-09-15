"""Training metrics deep dive: episode/update/validation curves across runs."""

from __future__ import annotations

import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.chart_theme import color_by, register_theme
from src.utils.dashboard_data import episode_df, to_long_metrics, update_df, validation_df
from src.utils.dashboard_state import shared_training_run_selector

st.set_page_config(page_title="Training Metrics", layout="wide", page_icon="📈")
register_theme()

st.title("Training Metrics")
st.caption("Episode, update and validation curves for one or more training runs.")

selection = shared_training_run_selector(label="Runs to compare", max_default_runs=6)
selected_runs = selection["selected_runs"]

if not selected_runs:
    st.warning("Select at least one training run in the sidebar.")
    st.stop()

SOURCE_LOADERS = {
    "episode": episode_df,
    "update": update_df,
    "validation": validation_df,
}

long_frames = []
for run in selected_runs:
    for source, loader in SOURCE_LOADERS.items():
        df = loader(run["path"])
        long_frames.append(to_long_metrics(df, run["name"], source, series_col="run"))

long_df = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

if long_df.empty:
    st.info("No numeric metric history found for the selected run(s).")
    st.stop()

run_names = [r["name"] for r in selected_runs]

tab_curves, tab_dist, tab_raw = st.tabs(["Metric curves", "Distribution", "Raw data"])

with tab_curves:
    source_options = sorted(long_df["source"].dropna().unique().tolist())
    default_source = "validation" if "validation" in source_options else source_options[0]
    selected_source = st.selectbox(
        "Metric source", source_options, index=source_options.index(default_source)
    )

    source_df = long_df[long_df["source"] == selected_source]
    metric_options = sorted(source_df["metric"].dropna().unique().tolist())
    preferred_metrics = ["avg_gap", "episode_reward", "makespan", "actor_loss"]
    default_metric = next((m for m in preferred_metrics if m in metric_options), metric_options[0])
    selected_metric = st.selectbox(
        "Metric", metric_options, index=metric_options.index(default_metric)
    )

    plot_df = source_df[source_df["metric"] == selected_metric].copy()
    plot_df = plot_df.sort_values(["run", "step"])

    smooth_window = st.slider("Smoothing window", min_value=1, max_value=50, value=1)
    if smooth_window > 1:
        plot_df["value"] = (
            plot_df.groupby("run", as_index=False)["value"]
            .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
        )

    line = alt.Chart(plot_df).mark_line(point=len(plot_df) < 200).encode(
        x=alt.X("step:Q", title="Step"),
        y=alt.Y("value:Q", title=selected_metric),
        color=color_by("run", run_names, title="Run"),
        tooltip=["run", "step", alt.Tooltip("value:Q", format=".6f")],
    ).interactive()
    st.altair_chart(line, width='stretch')

with tab_dist:
    source_options = sorted(long_df["source"].dropna().unique().tolist())
    selected_source_d = st.selectbox("Distribution source", source_options, key="dist_source")
    source_df_d = long_df[long_df["source"] == selected_source_d]

    metric_options_d = sorted(source_df_d["metric"].dropna().unique().tolist())
    selected_metric_d = st.selectbox("Distribution metric", metric_options_d, key="dist_metric")

    dist_df = source_df_d[source_df_d["metric"] == selected_metric_d]
    box = alt.Chart(dist_df).mark_boxplot().encode(
        x=alt.X("run:N", title="Run"),
        y=alt.Y("value:Q", title=selected_metric_d),
        color=color_by("run", run_names, title="Run", show_legend=False),
        tooltip=["run", alt.Tooltip("value:Q", format=".6f")],
    )
    st.altair_chart(box, width='stretch')

with tab_raw:
    st.dataframe(long_df, width='stretch', hide_index=True)
    st.download_button(
        "Download long metrics CSV",
        data=long_df.to_csv(index=False).encode("utf-8"),
        file_name="training_metrics_long.csv",
        mime="text/csv",
    )
