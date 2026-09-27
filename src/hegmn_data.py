"""Graph export + heterogeneous graph edit distance (HGED) for the HeGMN similarity test.

Turns the HeteroData state of any representation env (oo / om / ojm, or a future one
with extra node types such as workers) into the graph format HeGMN works on
(Sang et al. 2025, arXiv:2503.08739, https://github.com/alvinsang1906/HeGMN):
a single graph whose nodes carry a global `node_type` id and whose undirected edges
carry a global `edge_type` id. Node features are the one-hot node type, so the test
compares STRUCTURE (topology + node/edge types) only - the per-representation
numeric features have different dimensions and meanings and are not comparable.

HGED follows HeGMN's utils/computeHGED.py exactly: node/edge insertion and deletion
cost 1, substituting a node/edge by one of a different type costs 2 (never cheaper
than delete+insert, so types are never "renamed"), and the similarity target is
exp(-nGED) with nGED = GED / ((|V1| + |V2|) / 2).
"""
import math
import time
from multiprocessing import Pool

import networkx as nx
import torch
from torch_geometric.data import Data

from src.train import _resolve_representation_modules, generate_train_instances


# ---------------------------------------------------------------------------
# Type vocabularies
# ---------------------------------------------------------------------------
def edge_key(edge_type):
    """Undirected key of a hetero relation: ('machine','exec','operation') and
    ('operation','exec','machine') are the same connection seen from both ends, so they
    share one edge type; ('operation','prec','operation') and ('operation','disj',
    'operation') stay distinct because the relation name differs."""
    src, rel, dst = edge_type
    a, b = sorted((src, dst))
    return f"{a}-{rel}-{b}"


def build_vocab(states):
    """Global node/edge type ids over every state being compared, so the same type gets
    the same id in every representation. Sorted so ids don't depend on visiting order."""
    ntypes, etypes = set(), set()
    for s in states:
        ntypes.update(s.node_types)
        etypes.update(edge_key(et) for et in s.edge_types)
    return ({t: i for i, t in enumerate(sorted(ntypes))},
            {t: i for i, t in enumerate(sorted(etypes))})


# ---------------------------------------------------------------------------
# HeteroData state -> networkx (for HGED) -> PyG Data (for HeGMN)
# ---------------------------------------------------------------------------
def hetero_to_nx(state, ntype_vocab, etype_vocab, keep_self_loops=False):
    """Nodes are labelled (node_type_name, local_index), so the same operation/machine/
    job has the same label in every representation of the same instance - used by
    identity_mapping_cost() as the natural correspondence between two representations.
    Self-loops (om/ojm add one per operation on the prec relation, oo has none) are a
    GNN implementation detail rather than part of the scheduling structure, so they are
    dropped by default."""
    g = nx.Graph()
    for nt in state.node_types:
        for i in range(state[nt].num_nodes):
            g.add_node((nt, i), node_type=ntype_vocab[nt])
    for et in state.edge_types:
        src, _, dst = et
        etype = etype_vocab[edge_key(et)]
        for u, v in state[et].edge_index.t().tolist():
            a, b = (src, u), (dst, v)
            if a == b and not keep_self_loops:
                continue
            if not g.has_edge(a, b):
                g.add_edge(a, b, edge_type=etype)
    return g


def nx_to_pyg(g, num_ntypes):
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    node_type = torch.tensor([g.nodes[n]["node_type"] for n in nodes], dtype=torch.long)
    src, dst, etypes = [], [], []
    for u, v, d in g.edges(data=True):
        # both directions, like HeGMN's data loaders
        src += [idx[u], idx[v]]
        dst += [idx[v], idx[u]]
        etypes += [d["edge_type"], d["edge_type"]]
    return Data(
        x=torch.nn.functional.one_hot(node_type, num_ntypes).float(),
        edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
        node_type=node_type,
        edge_type=torch.tensor(etypes, dtype=torch.long),
        num_nodes=len(nodes),
    )


# ---------------------------------------------------------------------------
# States: the same instance, rolled out in every representation
# ---------------------------------------------------------------------------
def make_instances(n, jobs, machines, ops_per_job, max_processing):
    return generate_train_instances({
        "n_cases": n,
        "range_jobs": jobs,
        "range_machines": machines,
        "range_op_per_job": ops_per_job,
        "max_processing": max_processing,
    })


def collect_states(rep, instances, snapshots, mask_option=1, sel_k=1):
    """Returns {(instance_idx, snapshot): state}. A snapshot is a fraction of the episode
    (0.0 = reset state); the env is driven by its own expert_action() - the same
    earliest-completion-time dispatch rule every representation implements (see
    edge_scalability_probe.py), so all representations walk through the same schedule."""
    _, EnvClass, _ = _resolve_representation_modules(rep)
    env = EnvClass(instances, mask_option, sel_k)
    out = {}
    for i in range(len(instances)):
        # the terminal state is not kept: it has nothing left to schedule
        trajectory = [env.reset(sel_index=i).clone()]
        done = False
        while not done:
            state, _r, done, _info = env.step(env.expert_action())
            if not done:
                trajectory.append(state.clone())
        for f in snapshots:
            out[(i, f)] = trajectory[min(int(round(f * len(trajectory))), len(trajectory) - 1)]
    return out


# ---------------------------------------------------------------------------
# HGED
# ---------------------------------------------------------------------------
def _node_subst(a, b):
    return 0 if a["node_type"] == b["node_type"] else 2


def _edge_subst(a, b):
    return 0 if a["edge_type"] == b["edge_type"] else 2


def identity_mapping_cost(g1, g2):
    """Edit cost of the natural correspondence: node (type, i) in g1 <-> (type, i) in g2,
    everything else inserted/deleted. Any node mapping gives a valid GED upper bound, and
    for two representations of the same instance this one is the obvious alignment (same
    operations, same machines), so it seeds the exact search's pruning bound."""
    n1, n2 = set(g1.nodes), set(g2.nodes)
    cost = len(n1 ^ n2)
    cost += sum(_node_subst(g1.nodes[n], g2.nodes[n]) for n in n1 & n2)
    e1 = {frozenset(e): d["edge_type"] for *e, d in g1.edges(data=True)}
    e2 = {frozenset(e): d["edge_type"] for *e, d in g2.edges(data=True)}
    for e in e1.keys() | e2.keys():
        if e in e1 and e in e2:
            cost += 0 if e1[e] == e2[e] else 2
        else:
            cost += 1
    return float(cost)


def hged(g1, g2, timeout=10.0):
    """Returns (ged, exact). Exact branch-and-bound search (networkx, as in HeGMN) bounded
    by the identity-mapping cost and a wall-clock timeout; on timeout the best edit path
    found so far is kept, so `ged` is then an upper bound (exact=False)."""
    ub = identity_mapping_cost(g1, g2)
    t0 = time.perf_counter()
    found = nx.graph_edit_distance(g1, g2, node_subst_cost=_node_subst, edge_subst_cost=_edge_subst,
                                   upper_bound=ub, timeout=timeout)
    elapsed = time.perf_counter() - t0
    ged = ub if found is None else min(ub, float(found))
    return ged, elapsed < timeout


def _hged_job(args):
    key, g1, g2, timeout = args
    ged, exact = hged(g1, g2, timeout)
    return key, ged, exact


def hged_many(pairs, graphs, timeout=10.0, processes=None, cache=None, progress=None):
    """pairs: iterable of (i, j) graph ids. Returns {(i, j): (ged, exact)}, reusing and
    filling `cache` (a dict keyed the same way) so re-runs skip already computed pairs."""
    cache = {} if cache is None else cache
    todo = [(p, graphs[p[0]], graphs[p[1]], timeout) for p in pairs if p not in cache]
    if todo:
        with Pool(processes) as pool:
            for n, (key, ged, exact) in enumerate(pool.imap_unordered(_hged_job, todo, chunksize=4), 1):
                cache[key] = (ged, exact)
                if progress and (n % 200 == 0 or n == len(todo)):
                    progress(n, len(todo))
    return cache


def norm_similarity(ged, n1, n2):
    """HeGMN target: exp(-GED / mean node count)."""
    return math.exp(-ged / ((n1 + n2) / 2))
