"""Representation-quality metrics: expressiveness and heterophily, measured on the states
the trained policy actually sees during a greedy rollout.

The probe attaches forward hooks to the actor (no change to the model code): every time
the policy scores a state it sees the normalised input graph, the GNN's node embeddings
and the action logits. Works for every representation because it only uses the graph's
own node / edge types and the action store (edge type for om/ojm, node type for oo).

Expressiveness (can the representation + GNN tell things apart?)
  action_distinct_ratio   distinct logits among the valid actions / valid actions. < 1 means
                          the policy scores different actions identically - it cannot
                          separate them and has to break ties arbitrarily.
  embedding_distinct_ratio  distinct node embeddings / nodes (per node type, pooled).
  input_distinct_ratio    distinct input feature rows / nodes - what the raw features alone
                          separate.
  structure_gain          distinct embeddings / distinct inputs. > 1: message passing
                          separates nodes whose own features are identical (the graph
                          structure adds information); < 1: the GNN merges nodes the input
                          could still tell apart (information lost).
  effective_rank_ratio    effective rank (Roy & Vetterli, 2007) of each node type's embedding
                          matrix / its maximum possible rank. Near 0 = embeddings collapse
                          onto a few directions (over-smoothing).

Heterophily (how different are the nodes each edge connects?)
  For every relation, h = mean over edges of (1 - cos(z_u, z_v)) / 2, in [0, 1]:
  0 = connected nodes look the same (homophilous), 0.5 = unrelated, 1 = opposite.
  embedding_heterophily   on the GNN's output embeddings, every relation (all node types
                          share the output dimension).
  feature_heterophily     on the input features, only for relations between nodes of the
                          same type (different types have different feature dimensions).
  Both are also reported per relation and as edge-weighted means over relations. This is
  a label-free (feature) heterophily: the scheduling graphs have no node classes to use for
  the usual edge-homophily ratio.
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROUND = 1e4  # values equal to 4 decimals count as identical


def _n_unique_rows(x):
    if x.numel() == 0:
        return 0
    return int(torch.unique(torch.round(x.float() * ROUND), dim=0).shape[0])


def _effective_rank_ratio(z):
    n, d = z.shape
    if n < 2:
        return None
    s = torch.linalg.svdvals(z.double())
    if float(s.sum()) <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    erank = float(torch.exp(-(p * torch.log(p)).sum()))
    return erank / min(n, d)


def _heterophily(a, b):
    if a.shape[0] == 0:
        return None
    return float(((1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1)) / 2.0).mean())


def _rel_name(edge_type):
    return "__".join(edge_type)


class RepresentationProbe:
    """probe = RepresentationProbe(bopo_agent.policy.actor); ...rollout...; probe.summary()"""

    def __init__(self, actor, action_store):
        self.action_store = action_store  # edge type tuple (om/ojm) or node type (oo)
        self._actor = actor
        self._emb = None
        self._records = []
        self._handles = []
        self.resume()

    def resume(self):
        """(Re)attach the hooks; a no-op if they are already attached."""
        if not self._handles:
            self._handles = [
                self._actor.gnn.register_forward_hook(self._on_gnn),
                self._actor.register_forward_hook(self._on_actor),
            ]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _on_gnn(self, module, args, output):
        self._emb = {k: v.detach() for k, v in output.items()}

    def _on_actor(self, module, args, output):
        data = args[0]
        emb, self._emb = self._emb, None
        if emb is None:
            return
        rec = {"node": {}, "rel": {}}

        valid = ~data[self.action_store].mask.cpu()
        logits = output.detach().cpu().reshape(-1)[valid]
        if logits.numel() >= 2:
            rec["action_distinct_ratio"] = _n_unique_rows(logits[:, None]) / logits.numel()

        for ntype, z in emb.items():
            z = z.cpu()
            x = data.x_dict[ntype].detach().cpu()
            rec["node"][ntype] = {
                "n": z.shape[0],
                "distinct_emb": _n_unique_rows(z),
                "distinct_in": _n_unique_rows(x),
                "effective_rank_ratio": _effective_rank_ratio(z),
            }

        for etype, ei in data.edge_index_dict.items():
            src, _, dst = etype
            if ei.shape[1] == 0 or src not in emb or dst not in emb:
                continue
            ei = ei.cpu()
            keep = ei[0] != ei[1] if src == dst else torch.ones(ei.shape[1], dtype=torch.bool)
            u, v = ei[0][keep], ei[1][keep]
            if u.numel() == 0:
                continue
            r = {"n_edges": int(u.numel()),
                 "embedding": _heterophily(emb[src].cpu()[u], emb[dst].cpu()[v])}
            if src == dst:
                x = data.x_dict[src].detach().cpu()
                r["feature"] = _heterophily(x[u], x[v])
            rec["rel"][_rel_name(etype)] = r
        self._records.append(rec)

    def mark(self):
        """Position to pass to summary(start=...) to summarise only the states seen after now."""
        return len(self._records)

    def summary(self, start=0):
        """Aggregate over the states seen since `start` (pooled over nodes / edges)."""
        recs = self._records[start:]
        if not recs:
            return {"n_states": 0}
        n = emb_d = in_d = 0
        eranks = []
        per_type = defaultdict(lambda: {"n": 0, "distinct_emb": 0, "distinct_in": 0, "eranks": []})
        rel = defaultdict(lambda: {"n_edges": 0, "emb_sum": 0.0, "feat_sum": 0.0, "feat_edges": 0})
        for r in recs:
            for t, s in r["node"].items():
                n += s["n"]; emb_d += s["distinct_emb"]; in_d += s["distinct_in"]
                pt = per_type[t]
                pt["n"] += s["n"]; pt["distinct_emb"] += s["distinct_emb"]; pt["distinct_in"] += s["distinct_in"]
                if s["effective_rank_ratio"] is not None:
                    pt["eranks"].append(s["effective_rank_ratio"]); eranks.append(s["effective_rank_ratio"])
            for name, s in r["rel"].items():
                a = rel[name]
                a["n_edges"] += s["n_edges"]
                a["emb_sum"] += s["embedding"] * s["n_edges"]
                if s.get("feature") is not None:
                    a["feat_sum"] += s["feature"] * s["n_edges"]; a["feat_edges"] += s["n_edges"]

        action = [r["action_distinct_ratio"] for r in recs if "action_distinct_ratio" in r]
        total_edges = sum(a["n_edges"] for a in rel.values())
        feat_edges = sum(a["feat_edges"] for a in rel.values())
        return {
            "n_states": len(recs),
            "expressiveness": {
                "action_distinct_ratio": float(np.mean(action)) if action else None,
                "embedding_distinct_ratio": emb_d / n if n else None,
                "input_distinct_ratio": in_d / n if n else None,
                "structure_gain": emb_d / in_d if in_d else None,
                "effective_rank_ratio": float(np.mean(eranks)) if eranks else None,
                "per_node_type": {
                    t: {"embedding_distinct_ratio": s["distinct_emb"] / s["n"],
                        "input_distinct_ratio": s["distinct_in"] / s["n"],
                        "effective_rank_ratio": float(np.mean(s["eranks"])) if s["eranks"] else None}
                    for t, s in per_type.items() if s["n"]
                },
            },
            "heterophily": {
                "embedding_heterophily": sum(a["emb_sum"] for a in rel.values()) / total_edges if total_edges else None,
                "feature_heterophily": sum(a["feat_sum"] for a in rel.values()) / feat_edges if feat_edges else None,
                "per_relation": {
                    name: {"embedding": a["emb_sum"] / a["n_edges"],
                           "feature": a["feat_sum"] / a["feat_edges"] if a["feat_edges"] else None,
                           "n_edges": a["n_edges"]}
                    for name, a in rel.items() if a["n_edges"]
                },
            },
        }


def flatten_summary(summary):
    """Headline scalars of a probe summary, for per-instance tables."""
    if not summary or not summary.get("n_states"):
        return {}
    e, h = summary["expressiveness"], summary["heterophily"]
    return {
        "action_distinct_ratio": e["action_distinct_ratio"],
        "embedding_distinct_ratio": e["embedding_distinct_ratio"],
        "input_distinct_ratio": e["input_distinct_ratio"],
        "structure_gain": e["structure_gain"],
        "effective_rank_ratio": e["effective_rank_ratio"],
        "embedding_heterophily": h["embedding_heterophily"],
        "feature_heterophily": h["feature_heterophily"],
    }
