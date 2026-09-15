"""Shared Streamlit state helpers for dashboard pages."""

from __future__ import annotations

import streamlit as st

from .dashboard_data import training_runs


SHARED_TRAINING_SELECTION_KEY = "shared_training_run_selection"
SHARED_TRAINING_SELECTED_NAMES_KEY = "shared_training_selected_names"


def shared_training_run_selector(
    *,
    label: str = "Runs to compare",
    max_default_runs: int = 4,
) -> dict:
    """Render/use a shared training-run multiselect across pages.

    Returns a dict with keys: all_runs, run_options, selected_names, selected_runs.
    Also stores the same object in st.session_state[SHARED_TRAINING_SELECTION_KEY].
    """
    runs = training_runs()
    run_options = {r["name"]: r for r in runs}
    available_names = list(run_options.keys())

    prev_selected = st.session_state.get(SHARED_TRAINING_SELECTED_NAMES_KEY, [])
    valid_prev_selected = [name for name in prev_selected if name in run_options]

    default_selection = valid_prev_selected or available_names[:min(max_default_runs, len(available_names))]
    selected_names = st.sidebar.multiselect(
        label,
        available_names,
        default=default_selection,
        key=SHARED_TRAINING_SELECTED_NAMES_KEY,
    )

    selected_names = [name for name in selected_names if name in run_options]
    selection = {
        "all_runs": runs,
        "run_options": run_options,
        "selected_names": selected_names,
        "selected_runs": [run_options[name] for name in selected_names],
    }
    st.session_state[SHARED_TRAINING_SELECTION_KEY] = selection
    return selection
