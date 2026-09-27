"""Graph similarity test between representations with HeGMN (arXiv:2503.08739).

Question: for the SAME FJSP instance and scheduling state, how structurally similar are
the graphs each representation builds (oo / om / ojm, or any representation added later,
e.g. with worker constraints)? And is a representation change a bigger change than
switching to a different instance?

Pipeline (mirrors HeGMN's own):
  1. Generate small instances, roll each out with every representation's expert_action()
     and snapshot the state at the given episode fractions (0.0 = reset state).
  2. Convert every state to a heterogeneous graph (src/hegmn_data.py) with global node /
     edge type ids and one-hot node-type features -> the comparison is purely structural.
  3. Ground truth: heterogeneous GED (HGED), exp(-HGED / mean node count) as similarity.
  4. Train HeGMN (src/hegmn.py: relational-GIN encoder + same-type cross-graph matching)
     to predict that similarity, split BY INSTANCE into train / val / test, MSE loss,
     early stopping on validation MSE.
  5. Test: HeGMN's metrics (MSE, Spearman rho, Kendall tau, p@10) ranking the training
     database for each test graph, plus the representation similarity matrix (mean HGED
     and HeGMN similarity per representation pair, same instance vs different instance).

Exact GED is exponential, so instances are kept tiny (HeGMN's own graphs have 8-28
nodes); pairs whose search hits --ged-timeout keep the best edit path found (an upper
bound on GED) and are counted in the summary as inexact.

Usage:
    python graph_similarity_test.py                       # defaults below
    python graph_similarity_test.py --reps oo om ojm --n-instances 40 --snapshots 0.0 0.5
    python graph_similarity_test.py --smoke               # few graphs, few epochs
Output: results/graph_similarity/<run-name>/
"""
import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

from src.hegmn import DEFAULT_CONFIG, HeGMN
from src.hegmn_data import (build_vocab, collect_states, hetero_to_nx, hged_many, make_instances,
                            norm_similarity, nx_to_pyg)

REP_LABELS = {"oo": "Disjunctive (O-O)", "om": "Hetero O-M", "ojm": "Hetero O-J-M"}


def parse_args():
    p = argparse.ArgumentParser(description="HeGMN graph-similarity test between FJSP representations.")
    p.add_argument("--reps", nargs="+", default=["oo", "om", "ojm"])
    # HeGMN generalizes to unseen graphs only with enough distinct training graphs (its own
    # datasets have 700-1200); with ~100 it memorizes them. Graphs are cheap - only the
    # sampled pairs below need a GED - so the defaults favour many graphs, few pairs each.
    p.add_argument("--n-instances", type=int, default=120)
    p.add_argument("--jobs", type=int, nargs=2, default=(2, 3))
    p.add_argument("--machines", type=int, nargs=2, default=(2, 3))
    p.add_argument("--ops", type=int, nargs=2, default=(2, 3), help="operations per job (min max)")
    p.add_argument("--max-processing", type=int, default=20)
    p.add_argument("--snapshots", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75],
                   help="episode fractions at which the state is taken (0.0 = reset)")
    p.add_argument("--split", type=float, nargs=2, default=(0.6, 0.2),
                   help="train and val fraction of the INSTANCES; the rest is test")
    p.add_argument("--ged-timeout", type=float, default=5.0, help="seconds per GED search")
    p.add_argument("--processes", type=int, default=None, help="GED worker processes (default: all cores)")
    p.add_argument("--max-train-pairs", type=int, default=6000,
                   help="random subsample of training pairs (all same-instance pairs always kept)")
    p.add_argument("--max-val-pairs", type=int, default=1500, help="random subsample of validation pairs")
    p.add_argument("--db-size", type=int, default=30,
                   help="training graphs used as the database val/test graphs are ranked against")
    p.add_argument("--max-test-queries", type=int, default=100,
                   help="test graphs ranked against the database (HeGMN ranking metrics)")
    p.add_argument("--max-matrix-pairs", type=int, default=1500,
                   help="different-instance test pairs sampled for the representation matrix "
                        "(all same-instance pairs are always kept)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--no-node-match", action="store_true",
                   help="graph-level matching only (HeGMN ablation)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run to check the pipeline")
    args = p.parse_args()
    if args.smoke:
        args.n_instances, args.epochs, args.patience = 10, 3, 3
        args.max_train_pairs, args.max_val_pairs, args.db_size, args.ged_timeout = 300, 150, 12, 1.0
        args.max_test_queries, args.max_matrix_pairs = 12, 100
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_graphs(args):
    instances = make_instances(args.n_instances, tuple(args.jobs), tuple(args.machines),
                               tuple(args.ops), args.max_processing)
    states = {rep: collect_states(rep, instances, args.snapshots) for rep in args.reps}
    ntype_vocab, etype_vocab = build_vocab([s for rep in args.reps for s in states[rep].values()])

    n_train = int(round(args.split[0] * len(instances)))
    n_val = int(round(args.split[1] * len(instances)))
    order = list(range(len(instances)))
    random.shuffle(order)
    split_of = {i: ("train" if k < n_train else "val" if k < n_train + n_val else "test")
                for k, i in enumerate(order)}

    meta, nx_graphs, pyg_graphs = [], [], []
    for rep in args.reps:
        for (inst, snap), state in sorted(states[rep].items()):
            g = hetero_to_nx(state, ntype_vocab, etype_vocab)
            meta.append({"gid": len(meta), "rep": rep, "instance": inst, "snapshot": snap,
                         "split": split_of[inst], "nodes": g.number_of_nodes(), "edges": g.number_of_edges()})
            nx_graphs.append(g)
            pyg_graphs.append(nx_to_pyg(g, len(ntype_vocab)))
    return pd.DataFrame(meta), nx_graphs, pyg_graphs, ntype_vocab, etype_vocab


def make_pairs(meta, args):
    ids = {s: meta.index[meta.split == s].tolist() for s in ("train", "val", "test")}
    same_inst = lambda i, j: meta.at[i, "instance"] == meta.at[j, "instance"]

    train_all = list(combinations(ids["train"], 2))
    keep = [p for p in train_all if same_inst(*p)]
    rest = [p for p in train_all if not same_inst(*p)]
    random.shuffle(rest)
    train = keep + rest[:max(0, args.max_train_pairs - len(keep))]

    db = random.sample(ids["train"], min(args.db_size, len(ids["train"])))
    val = [(d, q) for q in ids["val"] for d in db]
    val = random.sample(val, min(args.max_val_pairs, len(val)))
    queries = random.sample(ids["test"], min(args.max_test_queries, len(ids["test"])))
    test_rank = [(d, q) for q in queries for d in db]
    # test graph pairs at the same snapshot -> representation similarity matrix: every
    # same-instance pair, plus a sample of different-instance pairs as the reference
    same_snap = [p for p in combinations(ids["test"], 2) if meta.at[p[0], "snapshot"] == meta.at[p[1], "snapshot"]]
    diff = [p for p in same_snap if not same_inst(*p)]
    test_matrix = [p for p in same_snap if same_inst(*p)] + random.sample(diff, min(args.max_matrix_pairs, len(diff)))
    return {"train": train, "val": val, "test_rank": test_rank, "test_matrix": test_matrix, "db": db}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def predict(model, graphs, pairs, device):
    """HeGMN is not symmetric in (G1, G2) (FCL over [h1, h2], one-directional cross-
    attention), while similarity is - so a pair's score is the mean of both orders."""
    model.eval()
    out = []
    with torch.no_grad():
        for i, j in pairs:
            gi, gj = graphs[i].to(device), graphs[j].to(device)
            out.append(0.5 * (float(model(gi, gj)) + float(model(gj, gi))))
    return np.array(out)


def fit(model, graphs, pairs, target, args, out_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    val_t = np.array([target[p] for p in pairs["val"]])
    best, patience, history = float("inf"), 0, []
    train_pairs = list(pairs["train"])
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        losses = []
        for b in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[b:b + args.batch_size]
            # random order per pair: the pair lists come out of combinations() with i < j,
            # and graph ids are grouped by representation, so without this G1 is always the
            # "earlier" representation (e.g. never ojm-vs-oo) and HeGMN, which is not
            # symmetric, fails on the other orientation of unseen pairs
            oriented = [(i, j) if random.random() < 0.5 else (j, i) for i, j in batch]
            optimizer.zero_grad()
            pred = torch.cat([model(graphs[i].to(args.device), graphs[j].to(args.device)) for i, j in oriented])
            targ = torch.tensor([target[p] for p in batch], dtype=torch.float, device=args.device)
            loss = F.mse_loss(pred, targ)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        vloss = float(np.mean((predict(model, graphs, pairs["val"], args.device) - val_t) ** 2))
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": vloss})
        if vloss < best:
            best, patience = vloss, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            patience += 1
        print(f"[HeGMN] epoch {epoch:4d} | train_mse={history[-1]['train_mse']:.5f} | val_mse={vloss:.5f}"
              f" | best={best:.5f} | patience={patience}/{args.patience}")
        if patience >= args.patience:
            print("[HeGMN] early stop")
            break
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=args.device))
    return history


# ---------------------------------------------------------------------------
# Metrics (as reported in HeGMN: MSE, Spearman rho, Kendall tau, p@10 per query graph)
# ---------------------------------------------------------------------------
def prec_at_k(true, pred, k):
    """HeGMN's p@k: tie-inclusive top-k on both sides, intersection capped at k."""
    def top_k(v):
        order = np.argsort(-v, kind="stable")
        kk = k
        while kk < len(v) and v[order[kk - 1]] == v[order[kk]]:
            kk += 1
        return set(order[:kk].tolist())
    return min(len(top_k(true) & top_k(pred)), k) / k


def ranking_metrics(pairs, pred, target, k=10):
    per_query = defaultdict(lambda: ([], []))
    for (d, q), p in zip(pairs, pred):
        per_query[q][0].append(target[(d, q)])
        per_query[q][1].append(p)
    rho, tau, pk = [], [], []
    for t, p in per_query.values():
        t, p = np.array(t), np.array(p)
        if np.ptp(t) > 0 and np.ptp(p) > 0:  # correlation undefined for a constant ranking
            rho.append(spearmanr(p, t).correlation)
            tau.append(kendalltau(p, t).correlation)
        if len(t) > k:
            pk.append(prec_at_k(t, p, k))
    true_all = np.array([target[pr] for pr in pairs])
    return {"mse_x1e-3": float(np.mean((np.asarray(pred) - true_all) ** 2) * 1e3),
            "spearman_rho": float(np.mean(rho)) if rho else float("nan"),
            "kendall_tau": float(np.mean(tau)) if tau else float("nan"),
            f"p@{k}": float(np.mean(pk)) if pk else float("nan"),
            "n_queries": len(per_query)}


def representation_matrix(meta, pairs, pred, target, ged):
    rows = []
    for (i, j), p in zip(pairs, pred):
        a, b = meta.loc[i], meta.loc[j]
        ra, rb = sorted((a.rep, b.rep))
        rows.append({"rep_a": ra, "rep_b": rb,
                     "same_instance": a.instance == b.instance,
                     "same_snapshot": a.snapshot == b.snapshot,
                     "snapshot": a.snapshot if a.snapshot == b.snapshot else None,
                     "hged": ged[(i, j)][0], "hged_exact": ged[(i, j)][1],
                     "true_sim": target[(i, j)], "hegmn_sim": p})
    df = pd.DataFrame(rows)
    df = df[df.same_snapshot]
    summary = (df.groupby(["snapshot", "same_instance", "rep_a", "rep_b"])
                 .agg(n_pairs=("true_sim", "size"), hged_mean=("hged", "mean"),
                      true_sim_mean=("true_sim", "mean"), true_sim_std=("true_sim", "std"),
                      hegmn_sim_mean=("hegmn_sim", "mean"), hegmn_sim_std=("hegmn_sim", "std"),
                      exact_frac=("hged_exact", "mean"))
                 .reset_index())
    return df, summary


def plot_matrices(summary, reps, out_dir):
    snaps = sorted(summary.snapshot.unique())
    fig, axes = plt.subplots(len(snaps), 2, figsize=(10, 4.2 * len(snaps)), squeeze=False)
    for r, snap in enumerate(snaps):
        sub = summary[(summary.snapshot == snap) & summary.same_instance]
        for c, (col, title) in enumerate([("true_sim_mean", "HGED similarity (ground truth)"),
                                          ("hegmn_sim_mean", "HeGMN predicted similarity")]):
            # diagonal: a representation vs itself on the same state is the same graph
            mat = pd.DataFrame(np.eye(len(reps)), index=reps, columns=reps).where(np.eye(len(reps)) == 1)
            for _, row in sub.iterrows():
                mat.loc[row.rep_a, row.rep_b] = mat.loc[row.rep_b, row.rep_a] = row[col]
            labels = [REP_LABELS.get(x, x) for x in reps]
            sns.heatmap(mat.astype(float), annot=True, fmt=".3f", vmin=0, vmax=1, cmap="Blues",
                        xticklabels=labels, yticklabels=labels, ax=axes[r][c], cbar=c == 1)
            axes[r][c].set_title(f"{title}\nsame instance, episode fraction {snap:g}")
    fig.suptitle("Structural similarity between representations of the same FJSP state")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "representation_similarity.png"), dpi=150)
    plt.close(fig)


def plot_history(history, out_dir):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h.epoch, h.train_mse, label="train")
    ax.plot(h.epoch, h.val_mse, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE vs exp(-nHGED)")
    ax.legend(frameon=False)
    ax.set_title("HeGMN training")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    sns.set_theme(style="whitegrid", palette="pastel")
    run_name = args.run_name or f"hegmn_{'_'.join(args.reps)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "graph_similarity", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[SIM] output: {out_dir}")

    # 1-2. graphs
    meta, nx_graphs, graphs, ntype_vocab, etype_vocab = build_graphs(args)
    meta.to_csv(os.path.join(out_dir, "graphs.csv"), index=False)
    print(f"[SIM] {len(meta)} graphs | node types {ntype_vocab} | edge types {etype_vocab}")
    print(meta.groupby(["rep", "split"])[["nodes", "edges"]].mean().round(1).to_string())
    pairs = make_pairs(meta, args)
    print("[SIM] pairs | " + " | ".join(f"{k}={len(v)}" for k, v in pairs.items() if k != "db"))

    # 3. HGED ground truth (cached per run folder, so an interrupted run resumes)
    cache_path = os.path.join(out_dir, "hged_cache.pkl")
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    all_pairs = sorted(set(p for k in ("train", "val", "test_rank", "test_matrix") for p in pairs[k]))
    t0 = time.time()
    hged_many(all_pairs, nx_graphs, timeout=args.ged_timeout, processes=args.processes, cache=cache,
              progress=lambda n, tot: print(f"[SIM] HGED {n}/{tot} | {time.time() - t0:.0f}s"))
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    target = {p: norm_similarity(cache[p][0], meta.at[p[0], "nodes"], meta.at[p[1], "nodes"]) for p in all_pairs}
    exact_frac = float(np.mean([cache[p][1] for p in all_pairs]))
    print(f"[SIM] HGED done in {time.time() - t0:.0f}s | exact for {exact_frac:.1%} of pairs")

    # 4. HeGMN
    config = dict(DEFAULT_CONFIG, hidden_dim=args.hidden_dim, pool_forth_dim=2 * args.hidden_dim,
                  node_match=not args.no_node_match, num_features=len(ntype_vocab),
                  num_etypes=len(etype_vocab), max_nums=max(3, int(meta.nodes.max())))
    model = HeGMN(config).to(args.device)
    print(f"[SIM] HeGMN | params={sum(p.numel() for p in model.parameters()):,} | config={config}")
    history = fit(model, graphs, pairs, target, args, out_dir)
    plot_history(history, out_dir)

    # 5. evaluation
    rank_pred = predict(model, graphs, pairs["test_rank"], args.device)
    metrics = ranking_metrics(pairs["test_rank"], rank_pred, target)
    print("[SIM] test ranking metrics:", json.dumps(metrics))

    mat_pred = predict(model, graphs, pairs["test_matrix"], args.device)
    pair_df, summary = representation_matrix(meta, pairs["test_matrix"], mat_pred, target, cache)
    pair_df.to_csv(os.path.join(out_dir, "test_pairs.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "representation_similarity.csv"), index=False)
    plot_matrices(summary, args.reps, out_dir)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))

    torch.save({"state_dict": model.state_dict(), "config": config,
                "ntype_vocab": ntype_vocab, "etype_vocab": etype_vocab},
               os.path.join(out_dir, "hegmn_model.pt"))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "config": config, "ntype_vocab": ntype_vocab,
                   "etype_vocab": etype_vocab, "n_graphs": len(meta),
                   "n_pairs": {k: len(v) for k, v in pairs.items()}, "hged_exact_frac": exact_frac,
                   "epochs_trained": len(history), "best_val_mse": min(h["val_mse"] for h in history),
                   "test_ranking": metrics}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[SIM] saved to {out_dir}")


if __name__ == "__main__":
    main()
