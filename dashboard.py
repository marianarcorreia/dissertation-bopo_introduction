"""
BOPO FJSP Results Dashboard — Home
==================================
Shows every run under results/ (training, testing, benchmarking, Optuna
studies) and lets you compare validation-gap learning curves across runs.

Run with:
    streamlit run dashboard.py

This is auto-launched at the end of main.py / test.py / param.py via
src/utils/dashboard_launcher.py, so it normally opens by itself; running the
command above is only needed if you closed the tab and the server is still
up, or want it fresh.

Additional views (training-curve deep dive, benchmark/test comparisons,
Optuna trial comparison) live in pages/ and appear in the sidebar nav that
Streamlit generates automatically for this file's `pages/` folder.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.chart_theme import color_by, register_theme
from src.utils.dashboard_data import list_all_runs, training_runs, validation_df
from src.utils.dashboard_state import shared_training_run_selector

register_theme()

st.set_page_config(page_title="BOPO FJSP Dashboard", layout="wide", page_icon="📊")

st.title("BOPO FJSP — Results Dashboard")
st.caption(
    "Every run recorded under results/ so far. Data refreshes automatically "
    "(cache TTL: 5s) — reload the page after a new run finishes."
)

if st.sidebar.button("🔄 Force refresh now"):
    st.cache_data.clear()
    st.rerun()

all_runs = list_all_runs()

if not all_runs:
    st.warning(
        "No runs found under results/. Run `python main.py` (training), "
        "`python test.py` (testing) or `python param.py` (Optuna tuning) first."
    )
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────

train_runs_list = training_runs()
optuna_studies = [r for r in all_runs if r["type"] == "optuna_study"]
test_runs_list = [r for r in all_runs if r.get("has_test_scores")]

best_gap = None
for r in train_runs_list:
    g = r["summary"].get("best_validation_avg_gap")
    if g is not None and (best_gap is None or g < best_gap):
        best_gap = g

most_recent = all_runs[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total runs", len(all_runs))
c2.metric("Training runs", len(train_runs_list))
c3.metric("Optuna studies", len(optuna_studies))
c4.metric("Test / benchmark runs", len(test_runs_list))
c5.metric(
    "Best validation avg gap so far",
    f"{best_gap:.4f}" if best_gap is not None else "n/a",
)

st.caption(
    f"Most recent activity: **{most_recent['name']}** "
    f"({most_recent['type']}) — "
    f"{datetime.fromtimestamp(most_recent['last_modified']).strftime('%Y-%m-%d %H:%M:%S')}"
)

# ── All runs table ────────────────────────────────────────────────────────────

st.subheader("All runs")

table_rows = []
for r in all_runs:
    summary = r["summary"]
    table_rows.append({
        "name": r["name"],
        "type": r["type"],
        "representation": r["representation"],
        "best_validation_avg_gap": summary.get("best_validation_avg_gap"),
        "episodes_completed": summary.get("episodes_completed"),
        "total_runtime_sec": summary.get("total_runtime_sec"),
        "last_modified": datetime.fromtimestamp(r["last_modified"]),
    })
runs_df = pd.DataFrame(table_rows)

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    type_filter = st.multiselect(
        "Filter by run type",
        sorted(runs_df["type"].unique().tolist()),
        default=sorted(runs_df["type"].unique().tolist()),
    )
with filter_col2:
    rep_options = sorted(runs_df["representation"].fillna("unknown").unique().tolist())
    rep_filter = st.multiselect(
        "Filter by representation",
        rep_options,
        default=rep_options,
    )
filtered_runs_df = runs_df[
    runs_df["type"].isin(type_filter)
    & runs_df["representation"].fillna("unknown").isin(rep_filter)
]

st.dataframe(
    filtered_runs_df.sort_values("last_modified", ascending=False),
    width='stretch',
    hide_index=True,
    column_config={
        "last_modified": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
        "best_validation_avg_gap": st.column_config.NumberColumn(format="%.4f"),
        "total_runtime_sec": st.column_config.NumberColumn(format="%.1f s"),
    },
)

# ── Flagship chart: validation gap learning curves across runs ────────────────

st.subheader("Validation gap across training runs")
st.caption(
    "Gap = model makespan / OR-Tools reference makespan − 1. Lower is better. "
    "Shaded band is ±1 std across the validation set at each checkpoint."
)

if not train_runs_list:
    st.info("No training runs yet — nothing to plot.")
else:
    selection = shared_training_run_selector(label="Runs to compare", max_default_runs=6)
    selected_runs = selection["selected_runs"]

    group_by_rep = st.checkbox(
        "Color by representation instead of individual run",
        value=False,
        help="Useful for the classic oo/om/ojm learning-curve comparison from the study design.",
    )

    frames = []
    for run in selected_runs:
        df = validation_df(run["path"])
        if df.empty or "avg_gap" not in df.columns:
            continue
        df = df.copy()
        df["run"] = run["name"]
        df["representation"] = run.get("representation") or "unknown"
        frames.append(df)

    if not frames:
        st.info("Selected run(s) have no validation_history.json data yet.")
    else:
        long_df = pd.concat(frames, ignore_index=True)
        long_df["std_gap"] = long_df.get("std_gap", 0.0).fillna(0.0)
        long_df["lower"] = long_df["avg_gap"] - long_df["std_gap"]
        long_df["upper"] = long_df["avg_gap"] + long_df["std_gap"]

        color_field = "representation" if group_by_rep else "run"
        domain = sorted(long_df[color_field].unique().tolist())

        if group_by_rep:
            plot_df = (
                long_df.groupby(["representation", "episode"], as_index=False)
                .agg(avg_gap=("avg_gap", "mean"), lower=("lower", "mean"), upper=("upper", "mean"))
            )
        else:
            plot_df = long_df

        band = alt.Chart(plot_df).mark_area(opacity=0.15).encode(
            x=alt.X("episode:Q", title="Training step"),
            y=alt.Y("lower:Q", title="Validation avg gap"),
            y2="upper:Q",
            color=color_by(color_field, domain, title=color_field.replace("_", " ").title()),
        )
        line = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("episode:Q", title="Training step"),
            y=alt.Y("avg_gap:Q", title="Validation avg gap"),
            color=color_by(color_field, domain, title=color_field.replace("_", " ").title()),
            tooltip=[
                alt.Tooltip(f"{color_field}:N"),
                alt.Tooltip("episode:Q", title="step"),
                alt.Tooltip("avg_gap:Q", format=".4f"),
            ],
        )
        st.altair_chart((band + line).interactive(), width='stretch')

        with st.expander("Show underlying data"):
            st.dataframe(plot_df, width='stretch', hide_index=True)

st.divider()
st.page_link("pages/1_Training_Metrics.py", label="→ Training metrics deep dive", icon="📈")
st.page_link("pages/2_Benchmark_and_Test_Results.py", label="→ Benchmark / test results", icon="🧪")
st.page_link("pages/3_Optuna_Trials.py", label="→ Optuna trial comparison", icon="🔬")
