"""Checks for the HeGMN similarity test (src/hegmn_data.py, src/hegmn.py, graph_similarity_test.py).

Plain asserts, no pytest needed:
    python tests/test_hegmn.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from src.hegmn import DEFAULT_CONFIG, HeGMN
from src.hegmn_data import (build_vocab, collect_states, edge_key, hetero_to_nx, hged, identity_mapping_cost,
                            make_instances, norm_similarity, nx_to_pyg)

REPS = ["oo", "om", "ojm"]


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def small_graph(nodes, edges):
    """nodes: {label: node_type}, edges: [(u, v, edge_type)]"""
    g = nx.Graph()
    for n, t in nodes.items():
        g.add_node(n, node_type=t)
    for u, v, t in edges:
        g.add_edge(u, v, edge_type=t)
    return g


_CACHE = {}


def real_states():
    """Same 4 tiny instances in every representation, reset state and mid-episode."""
    if not _CACHE:
        seed(1)
        inst = make_instances(4, (2, 3), (2, 3), (2, 3), 20)
        states = {rep: collect_states(rep, inst, [0.0, 0.5]) for rep in REPS}
        nt, et = build_vocab([s for rep in REPS for s in states[rep].values()])
        _CACHE.update(instances=inst, states=states, nt=nt, et=et)
    return _CACHE


# ---------------------------------------------------------------------------
# HGED against hand-computed values (HeGMN costs: ins/del 1, type substitution 2)
# ---------------------------------------------------------------------------
def test_hged_known_values():
    a = small_graph({0: 0, 1: 0}, [(0, 1, 0)])
    assert hged(a, a)[0] == 0
    # one extra node + one extra edge
    b = small_graph({0: 0, 1: 0, 2: 0}, [(0, 1, 0), (1, 2, 0)])
    assert hged(a, b)[0] == 2
    # a node of a different type: substitute (2) == delete + insert (1 + 1)
    c = small_graph({0: 0}, [])
    d = small_graph({0: 1}, [])
    assert hged(c, d)[0] == 2
    # same nodes, edge of a different type
    e = small_graph({0: 0, 1: 0}, [(0, 1, 1)])
    assert hged(a, e)[0] == 2
    # isomorphic but differently labelled -> 0 (the search, not the identity bound)
    f = small_graph({"x": 0, "y": 1, "z": 0}, [("x", "y", 0), ("y", "z", 0)])
    g = small_graph({"p": 0, "q": 0, "r": 1}, [("p", "r", 0), ("r", "q", 0)])
    assert identity_mapping_cost(f, g) > 0
    assert hged(f, g)[0] == 0


def test_norm_similarity():
    assert norm_similarity(0, 5, 7) == 1.0
    assert abs(norm_similarity(6, 5, 7) - np.exp(-1)) < 1e-12


# ---------------------------------------------------------------------------
# Graph export from the real environments
# ---------------------------------------------------------------------------
def test_edge_key_merges_reverse_relations_only():
    assert edge_key(("machine", "exec", "operation")) == edge_key(("operation", "exec", "machine"))
    assert edge_key(("operation", "prec", "operation")) != edge_key(("operation", "disj", "operation"))


def test_vocab_covers_every_representation():
    d = real_states()
    assert set(d["nt"]) == {"operation", "machine", "job"}
    for rep in REPS:
        for s in d["states"][rep].values():
            assert all(edge_key(et) in d["et"] for et in s.edge_types)


def test_hetero_to_nx_matches_state():
    d = real_states()
    for rep in REPS:
        for s in d["states"][rep].values():
            g = hetero_to_nx(s, d["nt"], d["et"])
            assert g.number_of_nodes() == sum(s[nt].num_nodes for nt in s.node_types)
            assert nx.number_of_selfloops(g) == 0
            # every non-self-loop hetero edge is in the graph, and nothing else is
            expected = set()
            for et in s.edge_types:
                src, _, dst = et
                for u, v in s[et].edge_index.t().tolist():
                    if (src, u) != (dst, v):
                        expected.add(frozenset([(src, u), (dst, v)]))
            assert {frozenset(e) for e in g.edges} == expected
            for n, t in g.nodes(data="node_type"):
                assert t == d["nt"][n[0]]


def test_same_instance_same_operations_in_every_rep():
    d = real_states()
    for i in range(len(d["instances"])):
        n_ops = {rep: d["states"][rep][(i, 0.0)]["operation"].num_nodes for rep in REPS}
        assert len(set(n_ops.values())) == 1, n_ops


def test_nx_to_pyg():
    d = real_states()
    g = hetero_to_nx(d["states"]["ojm"][(0, 0.0)], d["nt"], d["et"])
    data = nx_to_pyg(g, len(d["nt"]))
    assert data.num_nodes == g.number_of_nodes()
    assert data.edge_index.size(1) == 2 * g.number_of_edges()
    assert torch.equal(data.x.argmax(1), data.node_type)
    assert data.edge_type.numel() == data.edge_index.size(1)
    # undirected: every edge also present reversed with the same type
    fwd = {(u, v, t) for (u, v), t in zip(data.edge_index.t().tolist(), data.edge_type.tolist())}
    assert all((v, u, t) in fwd for u, v, t in fwd)


def test_hged_on_real_graphs_symmetric_and_bounded():
    d = real_states()
    graphs = [hetero_to_nx(d["states"][rep][(0, 0.0)], d["nt"], d["et"]) for rep in REPS]
    for a in graphs:
        for b in graphs:
            ab, ba = hged(a, b, timeout=20)[0], hged(b, a, timeout=20)[0]
            assert ab == ba
            assert ab <= identity_mapping_cost(a, b)
            # lower bound: node-count difference + edge-count difference
            assert ab >= abs(a.number_of_nodes() - b.number_of_nodes()) + abs(a.number_of_edges() - b.number_of_edges())
            if a is b:
                assert ab == 0


# ---------------------------------------------------------------------------
# HeGMN
# ---------------------------------------------------------------------------
def make_model(d, graphs, node_match=True):
    cfg = dict(DEFAULT_CONFIG, node_match=node_match, num_features=len(d["nt"]),
               num_etypes=len(d["et"]), max_nums=max(g.num_nodes for g in graphs), dropout=0.0)
    return HeGMN(cfg)


def pyg_graphs():
    d = real_states()
    return d, [nx_to_pyg(hetero_to_nx(s, d["nt"], d["et"]), len(d["nt"]))
               for rep in REPS for s in d["states"][rep].values()]


def test_model_output():
    seed()
    d, gs = pyg_graphs()
    for nm in (True, False):
        m = make_model(d, gs, nm).eval()
        out = m(gs[0], gs[-1])
        assert out.shape == (1,) and 0 < float(out) < 1


def test_cross_attention_only_matches_same_type():
    seed()
    d, gs = pyg_graphs()
    m = make_model(d, gs).eval()
    g1, g2 = gs[0], gs[-1]  # an oo graph (operations only) vs an ojm graph (all 3 types)
    with torch.no_grad():
        u1 = m._pad(m._nconv(g1.x, g1.edge_index, g1.edge_type))
        u2 = m._pad(m._nconv(g2.x, g2.edge_index, g2.edge_type))
        a = m._nca(u1, u2, g1.node_type, g2.node_type)
    heads = m.config["num_heads"]
    assert a.shape == (2 * heads, m.config["max_nums"], m.config["max_nums"])
    n1, n2 = g1.num_nodes, g2.num_nodes
    diff = g1.node_type.view(-1, 1) != g2.node_type.view(1, -1)
    for mat in a:  # both directions, every head
        assert torch.all(mat[:n1, :n2][diff] == 0)
        assert torch.all(mat[n1:] == 0) and torch.all(mat[:, n2:] == 0)
        assert torch.any(mat[:n1, :n2][~diff] != 0)


def test_graph_level_model_is_node_order_invariant():
    """Without node matching HeGMN is a function of the graphs, not of node order.
    (The node-matching CNN reads the similarity matrix as an image, so it is order-
    dependent by design in the original too.)"""
    seed()
    d, gs = pyg_graphs()
    m = make_model(d, gs, node_match=False).eval()
    g = gs[-1]
    perm = torch.randperm(g.num_nodes)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(g.num_nodes)
    gp = g.clone()
    gp.x, gp.node_type = g.x[perm], g.node_type[perm]
    gp.edge_index = inv[g.edge_index]
    with torch.no_grad():
        assert torch.allclose(m(gs[0], g), m(gs[0], gp), atol=1e-5)


def test_predict_is_symmetric():
    import graph_similarity_test as gst
    seed()
    d, gs = pyg_graphs()
    m = make_model(d, gs)
    pairs = [(0, len(gs) - 1), (3, 10)]
    ab = gst.predict(m, gs, pairs, "cpu")
    ba = gst.predict(m, gs, [(j, i) for i, j in pairs], "cpu")
    assert np.allclose(ab, ba)


def test_model_learns_hged_targets():
    """Gradients reach every parameter group and training fits real HGED targets."""
    seed()
    d, gs = pyg_graphs()
    m = make_model(d, gs)
    nxg = [hetero_to_nx(s, d["nt"], d["et"]) for rep in REPS for s in d["states"][rep].values()]
    pairs = [(i, j) for i in range(len(gs)) for j in range(i + 1, len(gs))][:40]
    target = torch.tensor([norm_similarity(hged(nxg[i], nxg[j], 5)[0], gs[i].num_nodes, gs[j].num_nodes)
                           for i, j in pairs], dtype=torch.float)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def loss_fn():
        return F.mse_loss(torch.cat([m(gs[i], gs[j]) for i, j in pairs]), target)

    first = loss_fn()
    first.backward()
    for name in ("_nconv", "_nca", "_nmatch", "_fcl"):
        grads = [p.grad for n, p in m.named_parameters() if n.startswith(name)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), name
    opt.step()
    for _ in range(80):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    final = float(loss_fn())
    assert final < 0.2 * float(first), (float(first), final)


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report every test, not just the first failure
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
