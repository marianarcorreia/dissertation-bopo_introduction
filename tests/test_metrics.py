"""Checks for src/metrics (schedule / constraint checks, statistics, representation probe,
end-to-end evaluation on every representation).

Plain asserts, no pytest needed:
    python tests/test_metrics.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.metrics.evaluate import evaluate_agent, summarize
from src.metrics.representation import RepresentationProbe
from src.metrics.schedule import check_schedule, lower_bound, relative_error, schedule_metrics, scheduling_score
from src.metrics.statistics import compare, compare_unpaired, describe, required_sample_size
from src.train import _resolve_representation_modules, generate_train_instances

REPS = ["oo", "om", "ojm"]

# 2 jobs x 2 machines: job 0 = ops 0 -> 1, job 1 = op 2. op 0 only runs on machine 0.
TOY = {"jobs": [[0, 1], [2]], "operations": [[3, 0], [2, 4], [5, 1]]}
TOY_OK = [
    {"job": 0, "operation": 0, "machine": 0, "start": 0, "end": 3},
    {"job": 1, "operation": 2, "machine": 1, "start": 0, "end": 1},
    {"job": 0, "operation": 1, "machine": 0, "start": 3, "end": 5},
]


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def with_entry(i, **changes):
    sched = [dict(r) for r in TOY_OK]
    sched[i].update(changes)
    return sched


def test_feasible_schedule():
    c = check_schedule(TOY, TOY_OK)
    assert c["feasible"] and c["n_steps"] == 3 and c["n_violating_steps"] == 0
    assert c["violation_rate_per_step"] == 0.0
    m = schedule_metrics(TOY, TOY_OK, reference=5)
    assert m["makespan"] == 5 and m["relative_error"] == 0.0 and m["scheduling_score"] == 1.0
    assert m["reference_source"] == "cp_sat"


def test_each_violation_is_detected():
    cases = {
        "machine_capacity": with_entry(2, start=2, end=4),          # overlaps op 0 on machine 0 ... and precedence
        "precedence": with_entry(2, machine=1, start=1, end=5),     # starts before op 0 ends (at 3)
        "eligibility": with_entry(0, machine=1, start=0, end=3),    # op 0 cannot run on machine 1
        "processing_time": with_entry(1, end=2),                    # op 2 takes 1 on machine 1
        "completeness": TOY_OK[:2],                                  # op 1 never scheduled
    }
    for name, sched in cases.items():
        c = check_schedule(TOY, sched)
        assert not c["feasible"], name
        assert c["violations"][name] > 0, (name, c["violations"])
    dup = TOY_OK + [dict(TOY_OK[1], start=1, end=2)]
    c = check_schedule(TOY, dup)
    assert c["violations"]["completeness"] == 1 and c["n_violating_steps"] == 1
    assert math.isclose(c["violation_rate_per_step"], 1 / 4)


def test_bounds_and_scores():
    # longest job: 3 + 2 = 5; work: (3 + 2 + 1) / 2 = 3
    assert lower_bound(TOY) == 5.0
    # machine 0 is a bottleneck: 3 operations only it can run -> 9, above the longest job (3)
    # and the average load (10 / 2)
    bottleneck = {"jobs": [[0], [1], [2], [3]], "operations": [[3, 0], [3, 0], [3, 0], [0, 1]]}
    assert lower_bound(bottleneck) == 9.0
    assert lower_bound(bottleneck, exhaustive_up_to=0) == 9.0  # eligibility sets only
    assert math.isclose(relative_error(6, 5), 0.2) and relative_error(6, None) is None
    assert scheduling_score(10, 5, True) == 0.5 and scheduling_score(10, 5, False) == 0.0
    m = schedule_metrics(TOY, TOY_OK)
    assert m["reference_source"] == "lower_bound" and m["relative_error"] is None
    assert m["relative_error_lb"] == 0.0 and m["scheduling_score"] == 1.0


def test_statistics():
    d = describe([1, 2, 3, 4, None, float("nan")])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["ci95_low"] < 2.5 < d["ci95_high"]
    assert describe([])["n"] == 0
    a = [10, 11, 12, 13, 14, 15, 16, 17]
    b = [x + 1 for x in a]
    c = compare(a, b)
    assert c["a_better_count"] == 8 and c["rank_biserial"] == -1.0 and c["mean_diff"] == -1.0
    assert compare(a, a)["wilcoxon_p"] == 1.0
    assert required_sample_size(0.5) == 33 and required_sample_size(0.2) == 198
    u = compare_unpaired([1, 2, 3, 4, 5], [10, 11, 12, 13])
    assert u["cliffs_delta"] == -1.0 and u["significant"] and u["mean_diff"] < 0
    assert compare_unpaired([1, 1], [1, 1])["mannwhitney_p"] == 1.0


def small_instances(n=2):
    seed(1)
    return generate_train_instances({"n_cases": n, "range_jobs": (3, 4), "range_machines": (2, 3),
                                     "range_op_per_job": (2, 3), "max_processing": 20})


def build(rep, instances):
    rep, EnvClass, BOPOClass = _resolve_representation_modules(rep)
    env = EnvClass(instances, 0, 100)
    metadata = env.reset().metadata()
    agent = BOPOClass(0.001, env, metadata, 16, 1, 1, gnn_type="gat")
    return rep, env, agent


def test_env_schedules_are_feasible():
    """Every env records a schedule the independent checker accepts, and whose makespan is env.mk."""
    instances = small_instances(3)
    for rep in REPS:
        _, env, _ = build(rep, instances)
        for i in range(len(instances)):
            env.reset(sel_index=i)
            done = False
            while not done:
                _, _, done, _ = env.step(env.sample())
            c = check_schedule(instances[i], env.schedule)
            assert c["feasible"], (rep, i, c)
            assert c["n_steps"] == len(instances[i]["operations"])
            assert math.isclose(round(max(r["end"] for r in env.schedule), 2), env.mk), rep


def test_evaluate_every_representation():
    instances = small_instances(2)
    for rep in REPS:
        seed(0)
        rep, env, agent = build(rep, instances)
        rows, repr_summary = evaluate_agent(agent, env, instances, rep)
        assert len(rows) == 2
        for r in rows:
            assert r["feasible"] and r["violation_rate_per_step"] == 0.0
            assert math.isclose(round(r["makespan"], 2), r["env_makespan"])
            assert r["relative_error"] is None and r["relative_error_lb"] >= -1e-9
            assert 0.0 < r["scheduling_score"] <= 1.0
            assert r["time_sec"] > 0 and r["rss_mb"] > 0
            for k in ("action_distinct_ratio", "embedding_distinct_ratio", "effective_rank_ratio",
                      "embedding_heterophily"):
                assert r[k] is None or 0.0 <= r[k] <= 1.0 + 1e-9, (rep, k, r[k])
        assert repr_summary["n_states"] > 0
        assert repr_summary["heterophily"]["per_relation"], rep
        s = summarize(rows, repr_summary, agent.policy.actor)
        assert s["feasibility_rate"] == 1.0 and s["model"]["param_count"] > 0
        assert s["metrics"]["makespan"]["n"] == 2
        # the probe's hooks must be gone after evaluation
        assert not agent.policy.actor._forward_hooks and not agent.policy.actor.gnn._forward_hooks, rep


def test_probe_detects_ties():
    """A constant actor scores every action the same -> action_distinct_ratio = 1 / n_valid."""
    instances = small_instances(1)
    rep, env, agent = build("oo", instances)
    with torch.no_grad():
        state = env.reset(sel_index=0)
        agent.select_action(state, 2, 1)  # materialises the lazy Linear(-1, 1) before zeroing it
        agent.policy.actor.lin3.weight.zero_()
        probe = RepresentationProbe(agent.policy.actor, "operation")
        agent.select_action(state, 2, 1)
    probe.remove()
    n_valid = int((~state["operation"].mask).sum())
    assert math.isclose(probe.summary()["expressiveness"]["action_distinct_ratio"], 1 / n_valid)


def test_blocking_env_schedules_are_feasible():
    """Random rollouts of the blocking env satisfy every blocking check (buffers, held parts,
    flow), for the default buffers and tighter ones where swaps happen."""
    from src.env_blocking import FJSPEnvBlocking
    from src.metrics.blocking import blocking_constraints, blocking_stats
    from src.metrics.references import load_dataset
    instances = load_dataset("val/test_dataset_blocking.json")[:3]
    for in_cap, out_cap in ((2, 1), (1, 1), (1, 0), (0, 0)):
        seed(0)
        env = FJSPEnvBlocking(instances, 0, 100, in_cap=in_cap, out_cap=out_cap)
        for i in range(len(instances)):
            env.reset(sel_index=i)
            done = False
            while not done:
                _, _, done, _ = env.step(env.sample())
            c = check_schedule(instances[i], env.schedule, blocking_constraints(in_cap, out_cap))
            assert c["feasible"], (in_cap, out_cap, i, c["violations"])
            assert c["n_steps"] == len(instances[i]["operations"])
            assert math.isclose(round(max(r["end"] for r in env.schedule), 2), env.mk)
            assert blocking_stats(env.schedule)["blocked_time"] >= 0


def test_blocking_violations_are_detected():
    from src.metrics.blocking import blocking_constraints
    # 3 one-operation jobs + a 2-operation job, 2 machines
    inst = {"jobs": [[0], [1], [2], [3, 4]], "operations": [[2, 0], [2, 0], [2, 0], [1, 0], [0, 1]]}

    def rec(job, op, m, routed, start, depart=None, leave_out=None, p=None):
        p = p if p is not None else inst["operations"][op][m]
        depart = start + p if depart is None else depart
        return {"job": job, "operation": op, "machine": m, "routed": routed, "start": start,
                "end": start + p, "depart": depart, "leave_out": depart if leave_out is None else leave_out}

    ok = [rec(3, 3, 0, 0, 0, depart=1, leave_out=1), rec(3, 4, 1, 1, 1),
          rec(0, 0, 0, 0, 1), rec(1, 1, 0, 0, 3), rec(2, 2, 0, 1, 5)]
    assert check_schedule(inst, ok, blocking_constraints(2, 1))["feasible"]
    # op 1 and op 2 wait in machine 0's input buffer at the same time: fine with 2 slots, not with 1
    c = check_schedule(inst, ok, blocking_constraints(1, 1))
    assert c["violations"]["input_buffer_capacity"] > 0, c
    # op 3 held on machine 0 until 2 (blocked) while op 0 starts at 1 -> machine overlap
    held = [rec(3, 3, 0, 0, 0, depart=2, leave_out=2), rec(3, 4, 1, 2, 2),
            rec(0, 0, 0, 0, 1), rec(1, 1, 0, 0, 3), rec(2, 2, 0, 1, 5)]
    c = check_schedule(inst, held, blocking_constraints(2, 1))
    assert c["violations"]["machine_capacity"] > 0, c
    # op 3 waits in the output buffer [1, 3) with 0 output slots
    out = [rec(3, 3, 0, 0, 0, depart=1, leave_out=3), rec(3, 4, 1, 3, 3),
           rec(0, 0, 0, 0, 1), rec(1, 1, 0, 0, 3), rec(2, 2, 0, 1, 5)]
    assert check_schedule(inst, out, blocking_constraints(2, 1))["feasible"]
    assert check_schedule(inst, out, blocking_constraints(2, 0))["violations"]["output_buffer_capacity"] > 0
    # the job's next operation routed before the part left its machine
    flow = [rec(3, 3, 0, 0, 0, depart=2, leave_out=1), rec(3, 4, 1, 1, 2),
            rec(0, 0, 0, 0, 2), rec(1, 1, 0, 0, 4), rec(2, 2, 0, 1, 6)]
    assert check_schedule(inst, flow, blocking_constraints(2, 1))["violations"]["blocking_flow"] > 0


def test_evaluate_blocking_representations():
    from src.metrics.references import load_dataset
    instances = load_dataset("val/test_dataset_blocking.json")[:2]
    for rep in ("ojmb", "ojmd", "ojm_blk"):
        seed(0)
        rep, EnvClass, BOPOClass = _resolve_representation_modules(rep)
        env = EnvClass(instances, 0, 100)
        agent = BOPOClass(0.001, env, env.reset().metadata(), 16, 1, 1, gnn_type="gat", jm_design=env.jm_design)
        rows, repr_summary = evaluate_agent(agent, env, instances, rep)
        for r in rows:
            assert r["feasible"], (rep, r["violations"])
            assert "input_buffer_capacity" in r["violations"]
            assert r["relative_error"] is not None and r["relative_error"] >= -1e-9  # vs blocking CP-SAT
            assert r["num_swaps"] >= 0 and r["blocked_time"] >= 0
        assert repr_summary["n_states"] > 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
