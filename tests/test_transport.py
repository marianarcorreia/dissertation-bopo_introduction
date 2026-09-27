"""Checks for the FJSP with batching and transport (docs/transport_formulation.tex):
src/transport.py, the CP-SAT model and checkers, and the nine batching x transport envs.

Plain asserts, no pytest needed:
    python tests/test_transport.py
"""
import copy
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.batch_generator import generate_batching_instance_list
from src.batch_solver import check_schedule as batch_check, solve_batching
from src.metrics.batching import BATCHING_CONSTRAINTS, metric_schedule
from src.metrics.evaluate import evaluate_agent
from src.metrics.schedule import check_schedule, lower_bound
from src.train import TRANSPORT_REPRESENTATIONS, _resolve_representation_modules
from src.transport import job_lower_bound, transfer_times, transport_matrix

BATCH_REPS = ("node", "edge", "base")
TRANSPORT_CODES = ("t0", "tf", "te")

# Section 11 of the formulation: M1 (0,0), M2 (4,3), M3 (10,0), depot (0,5).
# job 0: op 0 on M1 (p=5), then op 1 (family 0) on M2
# job 1: op 2 on M3 (p=1), then op 3 (family 0) on M2
# job 2: op 4 on M2 (p=9), alone - keeps M2 busy until 15
EXAMPLE = {
    "jobs": [[0, 1], [2, 3], [4]],
    "operations": [[5, 0, 0], [0, 4, 0], [0, 0, 1], [0, 3, 0], [0, 9, 0]],
    "num_machines": 3, "maximum": 9,
    "family": [-1, 0, -1, 0, -1], "capacities": [2], "delta": 2,
    "coords": [[0, 0], [4, 3], [10, 0]], "depot": [0, 5],
}


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def env_class(rep):
    return _resolve_representation_modules(rep)[1]


def transport_instances(n=3, rho=0.3):
    seed(3)
    return generate_batching_instance_list(n_cases=n, range_jobs=(4, 5), range_machines=(3, 4),
                                           range_op_per_job=(3, 4), max_processing=20, transport_rho=rho)


def batching_instances(n=3):
    from src.metrics.references import load_dataset
    return load_dataset("data/batching/batching_dataset.json")[:n]


def rollout(env, index, policy="random"):
    env.reset(sel_index=index)
    done = False
    while not done:
        action = env.sample() if policy == "random" else env.expert_action()
        _, _, done, _ = env.step(action)
    return env.schedule


def dispatch(env, mach, job):
    """Step the (machine, job) action if it is still pending (a forced last action is taken
    by the env itself)."""
    ei = env.state['machine', 'exec', 'job'].edge_index
    hit = ((ei[0] == mach) & (ei[1] == job)).nonzero().flatten()
    if hit.numel():
        env.step(int(hit[0]))


# ------------------------------------------------------------------ layout / bounds
def test_transport_matrix_and_bounds():
    tau = transport_matrix(EXAMPLE)
    # locations 0..2 = M1..M3, 3 = depot (the table of the formulation, reordered)
    assert tau.tolist() == [[0, 7, 10, 5], [7, 0, 9, 6], [10, 9, 0, 15], [5, 6, 15, 0]]
    assert transfer_times(EXAMPLE)[1] == (7.0, 7.0)  # M1 -> M2
    assert transfer_times(EXAMPLE)[0] == (5.0, 5.0)  # depot -> M1
    # job 0: 5 + 5 + 7 + 4 = 21; job 1: 15 + 1 + 9 + 3 = 28; job 2: 6 + 9 = 15
    assert job_lower_bound(EXAMPLE) == 28.0
    assert lower_bound(EXAMPLE) >= 28.0
    no_layout = {k: v for k, v in EXAMPLE.items() if k not in ("coords", "depot")}
    assert not transport_matrix(no_layout).any()
    assert job_lower_bound(no_layout) == 9.0


def test_generator_layout():
    for inst in transport_instances(5, rho=0.5):
        assert len(inst["coords"]) == inst["num_machines"] and len(inst["depot"]) == 2
        assert inst["transport_rho"] == 0.5
    seed(3)
    plain = generate_batching_instance_list(n_cases=1, range_jobs=(4, 5), range_machines=(3, 4),
                                            range_op_per_job=(3, 4), max_processing=20)[0]
    seed(3)
    with_layout = generate_batching_instance_list(n_cases=1, range_jobs=(4, 5), range_machines=(3, 4),
                                                  range_op_per_job=(3, 4), max_processing=20,
                                                  transport_rho=0.3)[0]
    # the layout is drawn last: the rest of the instance is unchanged
    assert {k: v for k, v in with_layout.items() if k not in ("coords", "depot", "transport_rho")} == plain


# ------------------------------------------------------------------ solver / checkers
def test_solver_respects_transport():
    sol = solve_batching(EXAMPLE, time_limit=10, workers=1)
    assert sol["status"] == "OPTIMAL", sol["status"]
    assert batch_check(EXAMPLE, sol["final_schedule"]) == []
    sched = metric_schedule(sol["final_schedule"])
    assert check_schedule(EXAMPLE, sched, BATCHING_CONSTRAINTS, step_key="batch")["feasible"]
    no_layout = {k: v for k, v in EXAMPLE.items() if k not in ("coords", "depot")}
    free = solve_batching(no_layout, time_limit=10, workers=1)
    assert sol["makespan"] >= free["makespan"]
    # waiting for job 1 (arrives at M2 at 25) to batch it with job 0 would end at 29; running
    # job 0 alone (17-21) and job 1 at 25-28 meets the transport lower bound of job 1 (28)
    assert sol["makespan"] == lower_bound(EXAMPLE) == 28
    ends = {e["op_id"]: e for e in sol["final_schedule"]}
    assert ends[1]["batch"] != ends[3]["batch"]


def test_checkers_detect_transport_violation():
    sol = solve_batching(EXAMPLE, time_limit=10, workers=1)
    bad = copy.deepcopy(sol["final_schedule"])
    for e in bad:  # shift the M2 batch to start right when op 0 ends: no time to travel
        if e["op_id"] in (1, 3):
            p = e["end"] - e["start"]
            e["start"], e["end"] = 10, 10 + p
    assert any("transport" in err or "precedence" in err for err in batch_check(EXAMPLE, bad))
    c = check_schedule(EXAMPLE, metric_schedule(bad), BATCHING_CONSTRAINTS, step_key="batch")
    assert c["violations"]["transport"] > 0
    first = copy.deepcopy(sol["final_schedule"])
    for e in first:
        if e["op_id"] == 0:
            e["start"], e["end"] = 0, 5  # before arriving from the depot (tau = 5)
    assert any("depot" in err for err in batch_check(EXAMPLE, first))


# ------------------------------------------------------------------ envs
def test_zero_transport_reproduces_batching_env():
    """Regression: without a layout (or with every location at the same point), every
    batching x transport env takes exactly the same schedules as the batching-only env, and
    T-0 has exactly the same node features."""
    instances = batching_instances(3)
    same_point = [dict(i, coords=[[4, 4]] * i["num_machines"], depot=[4, 4]) for i in instances]
    for b in BATCH_REPS:
        legacy = env_class(f"ojmb_{b}")(instances, 0, 100, jm_design="edges")
        for t in TRANSPORT_CODES:
            for data in (instances, same_point):
                env = env_class(f"ojmb_{b}_{t}")(data, 0, 100, jm_design="edges")
                for i in range(len(instances)):
                    seed(i)
                    ref = rollout(legacy, i)
                    seed(i)
                    got = rollout(env, i)
                    assert got == ref, (b, t, i)
                    seed(i)
                    s_ref, s_got = legacy.reset(sel_index=i), env.reset(sel_index=i)
                    if t == "t0":
                        for nt in s_ref.node_types:
                            assert torch.equal(s_ref[nt].x, s_got[nt].x), (b, nt)
                        assert torch.equal(s_ref['machine', 'exec', 'job'].edge_attr,
                                           s_got['machine', 'exec', 'job'].edge_attr)


def test_example_batch_depends_on_transport():
    """Section 11: dispatching job 0 on M2 starts at 17 (its arrival) and job 1, arriving at
    25, cannot join; dispatching job 1 starts at 25 and job 0 (arrived at 17) joins."""
    for rep in TRANSPORT_REPRESENTATIONS:
        for opener, start, members in ((0, 17, {0}), (1, 25, {0, 1})):
            env = env_class(rep)([EXAMPLE], 0, 100, jm_design="edges")
            env.reset(sel_index=0)
            for mach, job in ((0, 0), (2, 1), (1, 2)):  # the three first operations
                dispatch(env, mach, job)
            assert env.operations_ends == [10, 16, 15], (rep, env.operations_ends)
            ei = env.state['machine', 'exec', 'job'].edge_index
            s = {int(j): float(env.job_start_machines[int(j), 1]) for j in ei[1]}
            assert s == {0: 17.0, 1: 25.0}, (rep, s)
            dispatch(env, 1, opener)
            batch = [e for e in env.schedule if e["op_id"] in (1, 3) and e["start"] == start]
            assert {e["job"] for e in batch} == members, (rep, opener, env.schedule)


def test_transport_features_and_edges():
    env = env_class("ojmb_node_tf")([EXAMPLE], 0, 100, jm_design="edges")
    state = env.reset(sel_index=0)
    for mach, job in ((0, 0), (2, 1), (1, 2)):
        dispatch(env, mach, job)
    state = env.state
    # job 0 is at M1 = (0, 0), job 1 at M3 = (10, 0); scale W = 10 -> raw (0, 0) and (1, 0)
    assert state["job"].x[0, -2:].tolist() == [0.0, 0.0] and state["job"].x[1, -2:].tolist() == [1.0, 0.0]
    mej = state['machine', 'exec', 'job']
    assert mej.edge_attr.shape[1] == 6
    assert sorted(mej.edge_attr[:, 5].tolist()) == [7.0, 9.0]  # M1 -> M2 and M3 -> M2
    norm = env.normalize_state(state)
    assert norm["job"].x[1, -2:].tolist() == [1.0, -1.0]  # shared scale, mapped to [-1, 1]
    assert all(norm[et].edge_attr.shape[1] == 6 for et in norm.edge_types if "edge_attr" in norm[et])

    env = env_class("ojmb_edge_te")([EXAMPLE], 0, 100, jm_design="edges")
    env.reset(sel_index=0)
    assert env.state["job", "at", "machine"].edge_index.shape[1] == 0  # all at the depot
    for mach, job in ((0, 0), (2, 1), (1, 2)):
        dispatch(env, mach, job)
    at = env.state["job", "at", "machine"].edge_index.T.tolist()
    assert sorted(at) == [[0, 0], [1, 2]], at  # job 2 is done
    reach = env.state["job", "reach", "machine"]
    assert sorted(zip(reach.edge_index[0].tolist(), reach.edge_attr[:, 0].tolist())) == [(0, 7.0), (1, 9.0)]
    assert env.state["operation", "transfer", "operation"].edge_index.shape[1] == 0  # no pending pairs


def test_transport_env_schedules_are_feasible():
    instances = transport_instances(3)
    for rep in TRANSPORT_REPRESENTATIONS + ("ojmb_node",):
        env = env_class(rep)(instances, 0, 100, jm_design="edges")
        for i, inst in enumerate(instances):
            for policy in ("random", "expert"):
                seed(i)
                sched = rollout(env, i, policy)
                assert batch_check(inst, sched) == [], (rep, i, batch_check(inst, sched))
                c = check_schedule(inst, metric_schedule(sched), BATCHING_CONSTRAINTS, step_key="batch")
                assert c["feasible"], (rep, i, c["violations"])
                assert math.isclose(round(max(e["end"] for e in sched), 2), env.mk)
                assert env.mk >= lower_bound(inst) - 1e-9


def test_solver_bound_and_env_agree():
    """On small instances CP-SAT (optimal) is never worse than the env's teacher, and the
    lower bound never exceeds the optimum."""
    instances = transport_instances(3)
    env = env_class("ojmb_base_t0")(instances, 0, 100, jm_design="edges")
    for i, inst in enumerate(instances):
        sol = solve_batching(inst, time_limit=20, workers=4)
        assert batch_check(inst, sol["final_schedule"]) == []
        rollout(env, i, "expert")
        assert lower_bound(inst) <= sol["best_bound"] + 1e-9
        if sol["status"] == "OPTIMAL":
            assert sol["makespan"] <= env.mk + 1e-9, (i, sol["makespan"], env.mk)


def test_every_representation_runs_a_model():
    instances = transport_instances(2)
    for rep in TRANSPORT_REPRESENTATIONS:
        seed(0)
        rep, EnvClass, BOPOClass = _resolve_representation_modules(rep)
        env = EnvClass(instances, 0, 100, jm_design="edges")
        agent = BOPOClass(0.001, env, env.reset().metadata(), 16, 1, 1, gnn_type="gat", jm_design="edges")
        rows, _ = evaluate_agent(agent, env, instances, rep, references=[None] * len(instances))
        for r in rows:
            assert r["feasible"], (rep, r["violations"])
            assert r["violations"]["transport"] == 0
        # a BOPO update (parallel rollouts normalised through agent.env) also runs
        agent.B, agent.K = 4, 2
        agent.update(0)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
