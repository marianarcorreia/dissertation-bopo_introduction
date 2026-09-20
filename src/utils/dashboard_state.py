"""Shared Streamlit state helpers for dashboard pages."""

from __future__ import annotations

import streamlit as st

from .dashboard_data import training_runs


SHARED_TRAINING_SELECTION_KEY = "shared_training_run_selection"
SHARED_TRAINING_SELECTED_NAMES_KEY = "shared_training_selected_names"
SHARED_TRAINING_REPRESENTATION_KEY = "shared_training_representation_filter"


def shared_training_run_selector(
    *,
    label: str = "Runs to compare",
    max_default_runs: int = 4,
) -> dict:
    """Render/use a shared training-run multiselect across pages.

    Also renders a "Representation" multiselect above it that narrows which
    runs are offered/defaulted, so you can e.g. only compare oo runs.

    Returns a dict with keys: all_runs, run_options, selected_names, selected_runs,
    representations, selected_representations. Also stores the same object in
    st.session_state[SHARED_TRAINING_SELECTION_KEY].
    """
    runs = training_runs()
    run_options = {r["name"]: r for r in runs}

    representations = sorted({r.get("representation") or "unknown" for r in runs})
    # A widget bound to `key` restores its value from session_state rather than
    # `default` on rerun, so stale entries left over from a shrunk options list
    # must be pruned first or Streamlit raises. Passing `default` only on the
    # very first render (when the key doesn't exist yet) avoids Streamlit's
    # "default value but also set via Session State" warning on every rerun.
    if SHARED_TRAINING_REPRESENTATION_KEY in st.session_state:
        st.session_state[SHARED_TRAINING_REPRESENTATION_KEY] = [
            r
            for r in st.session_state[SHARED_TRAINING_REPRESENTATION_KEY]
            if r in representations
        ]
    else:
        st.session_state[SHARED_TRAINING_REPRESENTATION_KEY] = representations.copy()
    selected_reps = st.sidebar.multiselect(
        "Representation",
        representations,
        key=SHARED_TRAINING_REPRESENTATION_KEY,
    )

    available_names = [
        name for name, r in run_options.items()
        if (r.get("representation") or "unknown") in selected_reps
    ]

    if SHARED_TRAINING_SELECTED_NAMES_KEY in st.session_state:
        st.session_state[SHARED_TRAINING_SELECTED_NAMES_KEY] = [
            n for n in st.session_state[SHARED_TRAINING_SELECTED_NAMES_KEY] if n in available_names
        ]
        names_kwargs = {}
    else:
        names_kwargs = {
            "default": available_names[:min(max_default_runs, len(available_names))]
        }
    selected_names = st.sidebar.multiselect(
        label,
        available_names,
        key=SHARED_TRAINING_SELECTED_NAMES_KEY,
        **names_kwargs,
    )

    selected_names = [name for name in selected_names if name in available_names]
    selection = {
        "all_runs": runs,
        "run_options": run_options,
        "representations": representations,
        "selected_representations": selected_reps,
        "selected_names": selected_names,
        "selected_runs": [run_options[name] for name in selected_names],
    }
    st.session_state[SHARED_TRAINING_SELECTION_KEY] = selection
    return selection
