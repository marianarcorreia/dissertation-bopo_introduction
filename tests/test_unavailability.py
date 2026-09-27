"""Checks for machine unavailability on the blocking FJSP: window arithmetic, the Weibull
scenario sampler, the env (feasibility, common random numbers, no information leak,
information parity of the two representations, normalisation), the CP-SAT reference and the
end-to-end evaluation.

Plain asserts, no pytest needed:
    python tests/test_unavailability.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.env_unavailability import FJSPEnvUnavailability
from src.metrics.evaluate import constraints_for, evaluate_agent
from src.metrics.references import load_dataset
from src.metrics.schedule import check_schedule, lower_bound
from src.metrics.unavailability import stored_windows, unavailability_constraints
from src.train import _resolve_representation_modules
from src.unavailability import (BREAKDOWN, MODES, SCHEDULED, earliest_start, information_view, p_fail,
                                resume_end, sample_scenario, scenario_to_json)
from src.unavailability_config import UNAVAIL_CONFIG

REPRS = ("feat", "dummy", "none")


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def blocking_instances(n=3):
    return load_dataset("val/test_dataset_blocking.json")[:n]


def with_scenario(inst, mode, s=0, windows=None):
    """The instance with a stored realisation (or the given windows)."""
    scenario = sample_scenario(inst["operations"], lower_bound(inst), mode, np.random.default_rng(s), UNAVAIL_CONFIG)
    if windows is not None:
        scenario["windows"] = windows
    return {**inst, "unavailability": {mode: scenario_to_json(scenario)}}


def rollout(env, index, rule="sample"):
    env.reset(sel_index=index)
    done, actions = False, []
    while not done:
        a = {"sample": env.sample, "expert": env.expert_action, "blind": env.blind_expert_action}[rule]()
        actions.append(a)
        _, _, done, _ = env.step(a)
    return actions


# ── window arithmetic ──────────────────────────────────────────────────────────

def test_window_arithmetic():
    sched, bd = [(2, 4, SCHEDULED)], [(2, 4, BREAKDOWN)]
    assert earliest_start(0, 2, sched) == 0          # fits before the maintenance
    assert earliest_start(0, 3, sched) == 4          # would cross it: waits
    assert earliest_start(3, 1, bd) == 4             # never starts inside a window
    assert earliest_start(0, 3, bd) == 0             # a breakdown may be crossed (resume)
    assert resume_end(0, 5, bd) == 7                 # paused for 2
    assert resume_end(0, 2, bd) == 2                 # ends exactly when the window starts
    assert resume_end(0, 5, [(1, 2, BREAKDOWN), (4, 6, BREAKDOWN)]) == 8
    view = information_view([[(1, 3, BREAKDOWN), (5, 9, BREAKDOWN), (10, 12, SCHEDULED)]], [4.0], 6)
    assert view == [[(1, 3, BREAKDOWN), (5, 10.0, BREAKDOWN), (10, 12, SCHEDULED)]]  # future unseen, current estimated
    view = information_view([[(7, 9, BREAKDOWN)]], [4.0], 6)
    assert view == [[]]


def test_scenarios_are_reproducible_and_consistent():
    inst = blocking_instances(1)[0]
    lb = lower_bound(inst)
    a = sample_scenario(inst["operations"], lb, "mixed", np.random.default_rng(5), UNAVAIL_CONFIG)
    b = sample_scenario(inst["operations"], lb, "mixed", np.random.default_rng(5), UNAVAIL_CONFIG)
    c = sample_scenario(inst["operations"], lb, "scheduled", np.random.default_rng(5), UNAVAIL_CONFIG)
    assert a == b
    assert [[w for w in ws if w[2] == SCHEDULED] for ws in a["windows"]] == c["windows"]
    assert c["eta"] is None
    assert len(set(a["eta"])) == len(a["eta"])  # heterogeneous reliability
    for ws in a["windows"]:
        assert all(s < e for s, e, _ in ws)
        assert all(ws[k][1] <= ws[k + 1][0] for k in range(len(ws) - 1))  # disjoint, sorted
    # increasing failure rate: an older machine is more likely to fail within the horizon
    assert p_fail(0, 100, 2.0, 50) < p_fail(100, 100, 2.0, 50) < p_fail(200, 100, 2.0, 50)
    assert abs(p_fail(0, 100, 1.0, 50) - p_fail(100, 100, 1.0, 50)) < 1e-12  # beta = 1: memoryless


# ── environment ────────────────────────────────────────────────────────────────

def test_env_schedules_are_feasible():
    instances = blocking_instances(3)
    for mode in MODES:
        for rep in REPRS:
            seed(0)
            env = FJSPEnvUnavailability(instances, 0, 100, unavail_repr=rep, unavail_mode=mode)
            for i in range(len(instances)):
                rollout(env, i, "expert" if i % 2 else "sample")
                c = check_schedule(instances[i], env.schedule, constraints_for(env))
                assert c["feasible"], (mode, rep, i, c["violations"])
                assert c["n_steps"] == len(instances[i]["operations"])


def test_same_dynamics_and_mask_for_every_representation():
    """Only the graph differs: the same actions give the same schedule in every representation."""
    inst = with_scenario(blocking_instances(1)[0], "mixed", 3)
    envs = [FJSPEnvUnavailability([inst], 0, 100, unavail_repr=r, unavail_mode="mixed") for r in REPRS]
    states = [e.reset(sel_index=0) for e in envs]
    done = False
    while not done:
        masks = [e.state['machine', 'exec', 'job'].mask for e in envs]
        assert all(torch.equal(masks[0], m) for m in masks[1:])
        a = envs[0].expert_action()
        assert all(e.expert_action() == a for e in envs)
        done = [e.step(a)[2] for e in envs][0]
    assert len({e.mk for e in envs}) == 1


def test_common_random_numbers():
    instances = blocking_instances(1)
    a = FJSPEnvUnavailability(instances, 0, 100, unavail_mode="breakdown")
    b = FJSPEnvUnavailability(instances, 0, 100, **a.env_kwargs())
    s = a.draw_scenario()
    a.fix_scenario(s)
    b.fix_scenario(s)
    seed(1)
    actions = rollout(a, 0)
    b.reset(sel_index=0)
    for act in actions:
        b.step(act)
    assert a.true_windows == b.true_windows and a.mk == b.mk
    b.fix_scenario(s + 1)
    b.reset(sel_index=0)
    assert a.true_windows != b.true_windows


def _state_tensors(env):
    st = env.state
    out = {nt: st[nt].x.clone() for nt in st.node_types}
    for et in st.edge_types:
        out[et] = st[et].edge_index.clone()
        if "edge_attr" in st[et]:
            out[(et, "attr")] = st[et].edge_attr.clone()
    out["mask"] = st['machine', 'exec', 'job'].mask.clone()
    return out


def test_no_information_leak():
    """Changing breakdowns that have not happened yet (and the true end of one in progress)
    changes nothing the policy sees: state, mask, action-edge features, teacher."""
    instances = blocking_instances(3)
    checked = 0
    for rep in REPRS:
        for i in range(len(instances)):
            seed(i)
            env = FJSPEnvUnavailability(instances, 0, 100, unavail_repr=rep, unavail_mode="breakdown")
            env.reset(sel_index=i)
            for _ in range(15):
                _, _, done, _ = env.step(env.sample())
                assert not done
            before = _state_tensors(env)
            expert = env.expert_action()
            t = env.t
            changed = []
            for ws in env.true_windows:
                # drop every breakdown that has not started, make a repair in progress longer,
                # and add a new breakdown after the last known window
                past = [(s, e + 50 if s <= t < e else e, k) for s, e, k in ws if s <= t]
                start = int(max([t] + [e for _, e, _ in past])) + 7
                changed.append(past + [(start, start + 23, BREAKDOWN)])
            env.true_windows = changed
            env._build_state()
            after = _state_tensors(env)
            assert before.keys() == after.keys()
            for k in before:
                assert torch.equal(before[k], after[k]), (rep, i, k)
            assert env.expert_action() == expert
            checked += 1
    assert checked == 3 * len(REPRS)


def test_representations_carry_the_same_information():
    """The feature columns of "feat" can be rebuilt from the dummy operations of "dummy"."""
    inst = with_scenario(blocking_instances(1)[0], "mixed", 4)
    fa = FJSPEnvUnavailability([inst], 0, 100, unavail_repr="feat", unavail_mode="mixed")
    fb = FJSPEnvUnavailability([inst], 0, 100, unavail_repr="dummy", unavail_mode="mixed")
    fa.reset(sel_index=0)
    fb.reset(sel_index=0)
    M = fa.num_machines
    done, steps = False, 0
    while not done:
        cols = fa.state['machine'].x[:M, -7:]
        x = fb.state['operation'].x
        rebuilt = torch.zeros((M, 7))
        rebuilt[:, 2] = 1.0  # no next window: availability period = H
        rebuilt[:, 5:] = fb.state['machine'].x[:M, -2:]
        ei = fb.state['operation', 'exec', 'machine'].edge_index
        for o, m in ei.T.tolist():
            if x[o, 2] < 0.5:
                continue
            _, _, _, offset, length, planned, breakdown = x[o].tolist()
            if offset == 0:  # current window
                rebuilt[m, 0], rebuilt[m, 1], rebuilt[m, 4] = planned, breakdown, length
            else:
                rebuilt[m, 2], rebuilt[m, 3] = offset, length
        assert torch.allclose(cols, rebuilt, atol=1e-6), (steps, cols, rebuilt)
        a = fa.expert_action()
        done = fa.step(a)[2]
        fb.step(a)
        steps += 1


def test_normalisation_keeps_flags_apart():
    """Fixed scaling: 'every machine down' and 'no machine down' normalise differently
    (a per-state min-max maps both to -1)."""
    env = FJSPEnvUnavailability(blocking_instances(1), 0, 100, unavail_repr="feat", unavail_mode="mixed")
    state = env.reset(sel_index=0)
    for value in (0.0, 1.0):
        s = state.clone()
        s['machine'].x[:, -7] = value
        n = env.normalize_state(s)
        assert torch.all(n['machine'].x[:, -7] == 2 * value - 1)
    env = FJSPEnvUnavailability(blocking_instances(1), 0, 100, unavail_repr="dummy", unavail_mode="mixed")
    n = env.normalize_state(env.reset(sel_index=0))
    x = n['operation'].x
    dummy = x[:, 2] > 0
    assert dummy.any() and torch.all(x[dummy, :2] == -1)
    real = x[~dummy, 1]
    assert abs(float(real.min()) + 1) < 1e-5 and abs(float(real.max()) - 1) < 1e-5  # min-max over real rows only


def test_aware_equals_blind_without_windows():
    inst = with_scenario(blocking_instances(1)[0], "mixed", 0, windows=[[] for _ in range(len(blocking_instances(1)[0]["operations"][0]))])
    env = FJSPEnvUnavailability([inst], 0, 100, unavail_mode="mixed")
    rollout(env, 0, "expert")
    aware = env.mk
    rollout(env, 0, "blind")
    assert env.mk == aware


# ── CP-SAT reference ───────────────────────────────────────────────────────────

def test_solver_respects_windows():
    from src.generate_unavailability_references import expert_makespan, solver_records
    from src.solver_blocking import solve_blocking_fjsp
    inst = {"jobs": [[0, 1], [2, 3], [4, 5]],
            "operations": [[3, 2], [2, 0], [0, 4], [3, 3], [2, 2], [0, 3]]}
    windows = [[(2, 5, SCHEDULED)], [(3, 6, BREAKDOWN)]]
    inst = with_scenario(inst, "mixed", 0, windows=windows)
    status, ms, schedule = solve_blocking_fjsp(inst["jobs"], inst["operations"], 2, 1, 10.0, windows=windows)
    assert status == "OPTIMAL"
    c = check_schedule(inst, solver_records(inst, schedule), unavailability_constraints(2, 1, stored_windows("mixed")))
    assert c["feasible"], c["violations"]
    ect, _ = expert_makespan(inst, "mixed")
    assert ms <= ect
    # a schedule that ignores the windows is caught
    bad = solver_records(inst, schedule)
    for r in bad:
        r["end"] = r["start"] + inst["operations"][r["operation"]][r["machine"]]
    c = check_schedule(inst, bad, unavailability_constraints(2, 1, stored_windows("mixed")))
    assert not c["feasible"] or all(r["end"] <= 2 or r["start"] >= 6 for r in bad)


# ── end to end ─────────────────────────────────────────────────────────────────

def test_evaluate_unavailability_representations():
    instances = [with_scenario(inst, "mixed", k) for k, inst in enumerate(blocking_instances(2))]
    for rep in ("ojmb_uf", "ojmb_uo", "ojmb_u0"):
        seed(0)
        rep, EnvClass, BOPOClass = _resolve_representation_modules(rep)
        env = EnvClass(instances, 0, 100, unavail_mode="mixed")
        agent = BOPOClass(0.001, env, env.reset().metadata(), 16, 1, 1, gnn_type="gat", jm_design=env.jm_design)
        rows, repr_summary = evaluate_agent(agent, env, instances, rep)
        for r in rows:
            assert r["feasible"], (rep, r["violations"])
            assert "machine_unavailability" in r["violations"]
            assert r["trapped_time"] >= 0 and r["num_waits"] >= 0
        assert repr_summary["n_states"] > 0


def test_bopo_group_shares_the_scenario():
    """sample_group gives every rollout of a group the same scenario (common random numbers)."""
    from src.bopo import BOPO
    instances = blocking_instances(1)
    env = FJSPEnvUnavailability(instances, 0, 100, unavail_mode="breakdown")
    agent = BOPO(0.001, env, env.reset().metadata(), 16, 1, 1, 4, 2, True, gnn_type="gat", jm_design=env.jm_design)
    seen = []
    original = FJSPEnvUnavailability._load_instance

    def spy(self, instance):
        original(self, instance)
        seen.append(self.true_windows)

    FJSPEnvUnavailability._load_instance = spy
    try:
        agent.sample_group(0, record=True)
    finally:
        FJSPEnvUnavailability._load_instance = original
    assert len(seen) == 4 and all(w == seen[0] for w in seen)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
