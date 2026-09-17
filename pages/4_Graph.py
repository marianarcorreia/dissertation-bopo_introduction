"""Disjunctive graph visualization for FJSP instances.

Renders the classic disjunctive-graph representation (nodes = operations,
solid arrows = precedence/conjunctive arcs within a job, dashed curved arrows
= disjunctive arcs between operations from different jobs that could conflict
over a shared machine) for an instance in the same {"jobs": ...,
"operations": ...} format consumed by the "oo" environment
(src/envo_o.py::FJSPEnvOO).

Originally this used job_shop_lib's `plot_disjunctive_graph`, but that
package hard-pins `pyarrow<21`, which has no prebuilt wheel for this
project's Python 3.14 venv (pyarrow<21 would have to be compiled from
source, needing Visual Studio + cmake + Arrow C++). So the graph is instead
built directly with networkx (already installed via torch-geometric) and
matplotlib, mirroring the same arrow styling ("<|-|>" curved disjunctive
edges) the job_shop_lib snippet asked for. This also avoids a limitation of
job_shop_lib's own Operation model, which only stores one duration per
operation regardless of machine — real FJSP durations vary by machine.
"""

from __future__ import annotations

import os
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import generate_instance_list
from src.parsedata import get_data, parse

st.set_page_config(page_title="Graph", layout="wide", page_icon="🕸️")

st.title("Disjunctive Graph")
st.caption(
    "Visualizes the disjunctive graph of an FJSP instance shaped like the ones "
    "the 'oo' representation trains on. Not tied to a specific past run — "
    "results/ doesn't persist the instance a run used, so pick or generate one below."
)


def build_disjunctive_graph(jobs: list[list[int]], operations: list[list[int]]):
    """Build a directed graph with conjunctive (precedence) edges, plus the
    set of disjunctive (potential machine-conflict) operation pairs."""
    graph = nx.DiGraph()
    for job_id, job_ops in enumerate(jobs):
        for pos, op_id in enumerate(job_ops):
            eligible = [m for m, t in enumerate(operations[op_id]) if t > 0]
            mean_duration = sum(operations[op_id][m] for m in eligible) / len(eligible) if eligible else 0.0
            graph.add_node(op_id, job=job_id, pos_in_job=pos, machines=eligible, mean_duration=mean_duration)
        for a, b in zip(job_ops, job_ops[1:]):
            graph.add_edge(a, b, kind="conjunctive")

    disjunctive_pairs = set()
    op_ids = list(graph.nodes)
    for i, op_a in enumerate(op_ids):
        for op_b in op_ids[i + 1:]:
            if graph.nodes[op_a]["job"] == graph.nodes[op_b]["job"]:
                continue
            if set(graph.nodes[op_a]["machines"]) & set(graph.nodes[op_b]["machines"]):
                disjunctive_pairs.add((op_a, op_b))

    return graph, disjunctive_pairs


def plot_disjunctive_graph(graph, disjunctive_pairs, figsize=(6, 4), show_labels=True):
    """Draw `graph`'s conjunctive edges as solid arrows and `disjunctive_pairs`
    as single curved bidirectional arrows, matching job_shop_lib's
    draw_disjunctive_edges="single_edge" style with arrowstyle "<|-|>" and
    connectionstyle "arc3,rad=0.15"."""
    fig, ax = plt.subplots(figsize=figsize)
    pos = {op_id: (data["pos_in_job"], -data["job"]) for op_id, data in graph.nodes(data=True)}

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#8da0cb", node_size=700, edgecolors="black")
    if show_labels:
        labels = {op_id: f"O{op_id}" for op_id in graph.nodes}
        nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=8)

    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edge_color="black",
        arrows=True,
        arrowstyle="-|>",
        node_size=700,
    )

    for op_a, op_b in disjunctive_pairs:
        ax.annotate(
            "",
            xy=pos[op_b], xytext=pos[op_a],
            arrowprops=dict(
                arrowstyle="<|-|>",
                connectionstyle="arc3,rad=0.15",
                color="#fc8d62",
                linestyle="dashed",
                shrinkA=14, shrinkB=14,
            ),
        )

    ax.plot([], [], color="black", linewidth=1.5, label="Precedence (conjunctive)")
    ax.plot([], [], color="#fc8d62", linewidth=1.5, linestyle="dashed", label="Machine conflict (disjunctive)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


source = st.radio(
    "Instance source",
    ["Generate random instance", "Upload a .fjs / benchmark file"],
    horizontal=True,
)

instance_name = None

if source == "Generate random instance":
    c1, c2, c3 = st.columns(3)
    with c1:
        n_jobs = st.slider("Jobs", 2, 10, 3)
    with c2:
        n_machines = st.slider("Machines", 2, 10, 3)
    with c3:
        ops_per_job = st.slider("Operations per job", 1, 6, (2, 3))

    seed = st.number_input("Random seed (for reproducibility)", min_value=0, value=0, step=1)

    if st.button("Generate", type="primary") or "graph_instance" not in st.session_state:
        random.seed(int(seed))
        raw_instance = generate_instance_list(
            n_cases=1,
            range_jobs=(n_jobs, n_jobs),
            range_machines=(n_machines, n_machines),
            range_op_per_job=ops_per_job,
        )[0]
        jobs, operations, info, _ = get_data(parse(raw_instance))
        st.session_state["graph_instance"] = (jobs, operations)
        st.session_state["graph_instance_name"] = f"random_{n_jobs}j_{n_machines}m_seed{seed}"

    jobs, operations = st.session_state["graph_instance"]
    instance_name = st.session_state["graph_instance_name"]

else:
    uploaded = st.file_uploader("Upload a Brandimarte/Taillard-style FJSP text file (.fjs/.txt)")
    if uploaded is None:
        st.info("Upload an instance file to render its disjunctive graph.")
        st.stop()
    text = uploaded.read().decode("utf-8")
    jobs, operations, info, _ = get_data(parse(text))
    instance_name = uploaded.name

st.caption(f"Instance: **{instance_name}** — {len(jobs)} jobs, {len(operations)} operations")

graph, disjunctive_pairs = build_disjunctive_graph(jobs, operations)
fig = plot_disjunctive_graph(graph, disjunctive_pairs)
st.pyplot(fig)
