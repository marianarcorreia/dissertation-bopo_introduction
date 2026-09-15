"""Benchmark (instance_gaps.csv) and test.py score-file comparison across runs.

Two data sources are shown because they're genuinely different things
produced by this codebase today:

- "Benchmark gap %" reads results/<run>/benchmark/instance_gaps.csv — a
  per-checkpoint optimality-gap format some part of the pipeline is designed
  to produce, if a run ever writes one.
- "Test scores" reads the results_<folder>.json files test.py actually
  writes today (see test.py --output-prefix): raw model makespan per
  instance, with no reference/optimal value joined in, so no gap % is
  available for these — only the makespan and wall time.
"""

from __future__ import annotations

import os
import sys

import altair as alt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.chart_theme import color_by, register_theme
from src.utils.dashboard_data import (
    benchmark_runs,
    combined_benchmark_df,
    combined_test_score_df,
    test_runs,
)

st.set_page_config(page_title="Benchmark & Test Results", layout="wide", page_icon="🧪")
register_theme()

st.title("Benchmark & Test Results")

bench_runs = benchmark_runs()
t_runs = test_runs()


def _rep_of(run: dict) -> str:
    return run.get("representation") or run["summary"].get("representation") or "unknown"


all_reps = sorted({_rep_of(r) for r in bench_runs + t_runs})
if all_reps:
    selected_reps = st.sidebar.multiselect("Representation", all_reps, default=all_reps)
    bench_runs = [r for r in bench_runs if _rep_of(r) in selected_reps]
    t_runs = [r for r in t_runs if _rep_of(r) in selected_reps]

tab_bench, tab_test = st.tabs(["Benchmark gap % (instance_gaps.csv)", "Test scores (results_*.json)"])

with tab_bench:
    if not bench_runs:
        st.info(
            "No run currently has results/<run>/benchmark/instance_gaps.csv. "
            "This view will populate once a run writes that file."
        )
    else:
        names = [r["name"] for r in bench_runs]
        selected = st.multiselect("Runs", names, default=names, key="bench_runs")
        chosen = [r for r in bench_runs if r["name"] in selected]
        df = combined_benchmark_df(chosen)
        if df.empty:
            st.info("Selected run(s) have no readable benchmark rows.")
        else:
            datasets = sorted(df["dataset"].dropna().unique().tolist())
            selected_datasets = st.multiselect(
                "Dataset(s)", datasets, default=datasets, key="bench_datasets"
            )
            plot_df = df[df["dataset"].isin(selected_datasets)]

            if plot_df.empty:
                st.info("Select at least one dataset.")
            else:
                run_names = sorted(plot_df["run"].dropna().unique().tolist())
                bar = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X("name:N", title="Instance", sort=None),
                    y=alt.Y("gap_percent:Q", title="Gap to BKS (%)"),
                    color=color_by("run", run_names, title="Run"),
                    xOffset="run:N",
                    tooltip=["run", "dataset", "name", alt.Tooltip("gap_percent:Q", format=".2f")],
                )
                if len(selected_datasets) > 1:
                    bar = bar.facet(row=alt.Row("dataset:N", title="Dataset"))
                st.altair_chart(bar, width='stretch')
                st.dataframe(plot_df, width='stretch', hide_index=True)

with tab_test:
    if not t_runs:
        st.info("No test-score files (results_*.json) found yet. Run `python test.py` first.")
    else:
        names = [r["name"] for r in t_runs]
        selected = st.multiselect("Runs", names, default=names, key="test_runs")
        chosen = [r for r in t_runs if r["name"] in selected]
        df = combined_test_score_df(chosen)
        if df.empty:
            st.info("Selected run(s) have no readable test-score rows.")
        else:
            datasets = sorted(df["dataset"].dropna().unique().tolist())
            selected_datasets = st.multiselect(
                "Dataset(s)", datasets, default=datasets, key="test_datasets"
            )
            plot_df = df[df["dataset"].isin(selected_datasets)].copy()

            if plot_df.empty:
                st.info("Select at least one dataset.")
            else:
                run_names = sorted(plot_df["run"].dropna().unique().tolist())
                metric = st.radio("Metric", ["score", "time"], horizontal=True)
                y_title = "Makespan (score)" if metric == "score" else "Wall time (s)"

                bar = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X("name:N", title="Instance", sort=None),
                    y=alt.Y(f"{metric}:Q", title=y_title),
                    color=color_by("run", run_names, title="Run"),
                    xOffset="run:N",
                    tooltip=["run", "dataset", "name", alt.Tooltip(f"{metric}:Q", format=".2f")],
                )
                if len(selected_datasets) > 1:
                    bar = bar.facet(row=alt.Row("dataset:N", title="Dataset"))
                st.altair_chart(bar, width='stretch')

                st.caption(
                    "No optimal/reference makespan is joined into these files today, "
                    "so only raw makespan and wall time are available here — not a gap %."
                )
                st.dataframe(plot_df, width='stretch', hide_index=True)
