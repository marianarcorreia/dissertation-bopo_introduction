"""Checks for the v2 batching envs (src/env_batching.py: WAIT_ACTION and BATCH_AWARE_MASK).

Plain asserts, no pytest needed:
    python tests/test_batching_v2.py
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.batch_solver import check_schedule as batch_check
from src.env_batching import SENTINEL
from src.metrics.batching import BATCHING_CONSTRAINTS, metric_schedule
from src.metrics.schedule import check_schedule
from src.train import BATCHING_V1, BATCHING_V2, _resolve_representation_modules

DATA = json.load(open("data/batching/batching_dataset.json"))


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def env_of(rep, instances, sel_k=1):
    return _resolve_representation_modules(rep)[1](instances, 1, sel_k, jm_design="edges")


def test_v2_schedules_are_feasible():
    instances = DATA[:10]
    for rep in BATCHING_V2:
        env = env_of(rep, instances)
        for i, inst in enumerate(instances):
            for policy in ("random", "expert"):
                seed(i)
                env.reset(sel_index=i)
                done = False
                while not done:
                    _, _, done, _ = env.step(env.sample() if policy == "random" else env.expert_action())
                assert batch_check(inst, env.schedule) == [], (rep, i, policy)
                c = check_schedule(inst, metric_schedule(env.schedule), BATCHING_CONSTRAINTS, step_key="batch")
                assert c["feasible"], (rep, i, policy, c["violations"])
                assert math.isclose(round(max(e["end"] for e in env.schedule), 2), env.mk)


def test_wait_actions_do_what_they_show():
    """Every wait action is the twin of an allowed immediate action, starts later, shows its
    own start, and - when taken - schedules a larger batch starting exactly then."""
    instances = DATA[:10]
    taken = 0
    for rep in BATCHING_V2:
        env = env_of(rep, instances)
        for i in range(len(instances)):
            seed(i)
            env.reset(sel_index=i)
            done = False
            while not done:
                store = env.state['machine', 'exec', 'job']
                assert store.wait.shape[0] == store.edge_index.shape[1] == store.mask.shape[0]
                immediate = {(int(m), int(j)): e for e, (m, j) in enumerate(store.edge_index.T.tolist())
                             if not store.wait[e]}
                cands = env._family_candidates()
                wait_edges = store.wait.nonzero().flatten().tolist()
                for e in wait_edges:
                    m, j = store.edge_index[:, e].tolist()
                    twin = immediate[(m, j)]
                    assert not store.mask[twin] and not store.mask[e]
                    s_now = float(env.job_start_machines[j, m])
                    s_wait = float(store.wait_start[e])
                    assert s_wait > s_now
                    assert abs(float(store.edge_attr[e, 2]) - s_wait) < 1e-4  # the edge shows it
                    assert len(env._batch_members(j, m, s_wait, cands)) > len(env._batch_members(j, m, s_now, cands))
                # take a wait action whenever there is one, to exercise step()
                if wait_edges and random.random() < 0.5:
                    e = random.choice(wait_edges)
                    m, j = store.edge_index[:, e].tolist()
                    expected = len(env._batch_members(j, m, float(store.wait_start[e]), cands))
                    s_wait = float(store.wait_start[e])
                    before = len(env.schedule)
                    _, _, done, _ = env.step(e)
                    first = env.schedule[before]
                    batch = [x for x in env.schedule[before:] if x["batch"] == first["batch"]]
                    assert first["job"] == j and first["machine"] == m and first["start"] == s_wait
                    assert len(batch) == expected >= 2
                    taken += 1
                else:
                    _, _, done, _ = env.step(env.sample())
    assert taken > 0


def test_batch_aware_mask_keeps_the_best_batch_machine():
    instances = DATA[:5]
    for rep in BATCHING_V2:
        env = env_of(rep, instances, sel_k=1)
        for i in range(len(instances)):
            seed(i)
            env.reset(sel_index=i)
            done = False
            while not done:
                store = env.state['machine', 'exec', 'job']
                start, proc = env._batch_start_proc()
                completion = start + proc
                imm = ~store.wait
                for j in store.edge_index[1][imm].unique().tolist():
                    idx = ((store.edge_index[1] == j) & imm).nonzero().flatten()
                    valid = idx[start[idx] < SENTINEL]
                    allowed = valid[~store.mask[valid]]
                    assert allowed.numel() == 1  # sel_k = 1
                    assert completion[allowed[0]] == completion[valid].min()
                _, _, done, _ = env.step(env.expert_action())


def test_v2_batches_more_than_v1():
    """The point of the wait actions: a random policy batches more operations with them."""
    instances = DATA[:10]
    share = {}
    for rep in ("ojmb_node", "ojmb_node_v2"):
        env = env_of(rep, instances)
        batched = total = 0
        for i in range(len(instances)):
            seed(i)
            env.reset(sel_index=i)
            done = False
            while not done:
                _, _, done, _ = env.step(env.sample())
            sizes = {}
            for e in env.schedule:
                sizes[e["batch"]] = sizes.get(e["batch"], 0) + 1
            batched += sum(s for s in sizes.values() if s > 1)
            total += len(env.schedule)
        share[rep] = batched / total
    assert share["ojmb_node_v2"] > share["ojmb_node"], share


def test_batch_size_column_shows_what_the_action_dispatches():
    """Column 5 of the action edge = operations the action dispatches (0 for the baseline, which
    stays blind); after normalisation every attributed relation has the v2 width."""
    instances = DATA[:5]
    for rep in BATCHING_V2:
        env = env_of(rep, instances)
        assert env.EDGE_DIM == 6
        for i in range(len(instances)):
            seed(i)
            env.reset(sel_index=i)
            done = False
            while not done:
                store = env.state['machine', 'exec', 'job']
                assert store.edge_attr.shape[1] == 6
                norm = env.normalize_state(env.state)
                for et in norm.edge_types:
                    if "edge_attr" in norm[et]:
                        assert norm[et].edge_attr.shape[1] == 6, (rep, et)
                a = env.sample()
                shown = float(store.edge_attr[a, 5])
                before = len(env.schedule)
                _, _, done, _ = env.step(a)
                first = env.schedule[before]
                size = sum(1 for x in env.schedule[before:] if x["batch"] == first["batch"])
                assert shown == (0.0 if rep == "ojmb_base_v2" else size), (rep, shown, size)


def test_representation_c_uses_only_the_base_graph():
    """C has exactly the node and edge types of the baseline; the batching lives in features."""
    instances = DATA[:3]
    base = env_of("ojmb_base_v2", instances).reset(sel_index=0)
    env = env_of("ojmb_feat_v2", instances)
    feat = env.reset(sel_index=0)
    assert feat.metadata() == base.metadata()
    assert feat["operation"].x.shape[1] == base["operation"].x.shape[1] + 6  # flag, kappa, B, p, c_o, remaining
    assert feat["job"].x.shape[1] == base["job"].x.shape[1] + 3  # flag, B, pi_j
    assert feat["machine"].x.shape[1] == base["machine"].x.shape[1] + 1  # batching opportunity
    done = False
    while not done:
        opp = env.state["machine"].x[:, -1]
        assert float(opp.min()) >= 0 and float(opp.max()) <= 1
        # exec edge column 3: partners of the operation eligible on the machine
        oid = env.state["operation"].oid.tolist()
        store = env.state["operation", "exec", "machine"]
        for (n, m), value in zip(store.edge_index.T.tolist(), store.edge_attr[:, 3].tolist()):
            o = oid[n]
            expected = sum(1 for o2 in env.feat_partners[o] if o2 in oid and env.operations[o2][m] > 0)
            assert value == expected
        _, _, done, _ = env.step(env.sample())


def test_v2_trains():
    instances = DATA[:2]
    for rep in BATCHING_V2:
        seed(0)
        rep, EnvClass, BOPOClass = _resolve_representation_modules(rep)
        env = EnvClass(instances, 1, 1, jm_design="edges")
        agent = BOPOClass(0.001, env, env.reset().metadata(), 16, 2, 1, B=4, K=2, gnn_type="gat",
                          jm_design="edges")
        agent.update(0)
        with torch.no_grad():
            state = env.reset(sel_index=0)
            done = False
            q = 1
            while not done:
                state, _, done, _ = env.step(agent.select_action(state, 2, q))
                q += 1
        assert batch_check(instances[0], env.schedule) == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
