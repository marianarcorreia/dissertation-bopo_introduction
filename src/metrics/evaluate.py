"""Runs a trained policy greedily over a list of instances and computes every metric
(schedule, constraint, efficiency, representation) per instance, plus their summaries.

Kept independent of src/train.py (the caller builds the env and agent, see run_metrics.py),
so it can also be reused from training code.
"""
from src.metrics.efficiency import MemoryTracker, model_size
from src.metrics.representation import RepresentationProbe, flatten_summary
from src.metrics.schedule import FJSP_CONSTRAINTS, schedule_metrics
from src.metrics.statistics import describe

# where each representation keeps its action mask (see Policy.action_distribution)
ACTION_STORE = {
    "oo": "operation",
    "om": ("machine", "exec", "operation"),
    "ojm": ("machine", "exec", "job"),
}

SUMMARY_METRICS = (
    "makespan", "relative_error", "relative_error_lb", "scheduling_score",
    "violation_rate_per_step", "time_sec", "time_per_decision_ms",
    "rss_mb", "rss_delta_mb", "gpu_peak_mb", "gpu_peak_delta_mb",
    "action_distinct_ratio", "embedding_distinct_ratio", "input_distinct_ratio",
    "structure_gain", "effective_rank_ratio", "embedding_heterophily", "feature_heterophily",
)


def instance_size(instance):
    return {
        "num_jobs": len(instance["jobs"]),
        "num_machines": len(instance["operations"][0]),
        "num_operations": len(instance["operations"]),
    }


def _rollout(agent, env, index):
    state = env.reset(sel_index=index)
    decisions = 0
    for q in range(1, 10 ** 10):
        action = agent.select_action(state, 2, q)  # 2 = greedy (argmax)
        decisions += 1
        state, _, done, _ = env.step(action)
        if done:
            return decisions


def evaluate_agent(agent, env, instances, representation, representation_metrics=True,
                   constraints=FJSP_CONSTRAINTS, references=None):
    """env must have been built on `instances` (in the same order). references[i] is the
    CP-SAT makespan of instance i, or None; defaults to each instance's own "score"
    (how the validation / test splits store it).

    Returns (rows, representation_summary): one dict of metrics per instance, and the probe
    summary pooled over all instances (None when representation_metrics is False)."""
    if references is None:
        references = [inst.get("score") for inst in instances]
    rows = []
    probe = None
    for i, instance in enumerate(instances):
        # 1st rollout: timing / memory, with no probe hooks running
        with MemoryTracker() as mem:
            decisions = _rollout(agent, env, i)
        ref = references[i]
        row = {
            "index": i,
            "name": instance.get("name", str(i)),
            **instance_size(instance),
            **schedule_metrics(instance, env.schedule, None if ref is None else float(ref), constraints),
            "env_makespan": float(env.mk),
            "decisions": decisions,
            **mem.result,
            "time_per_decision_ms": 1000.0 * mem.result["time_sec"] / max(decisions, 1),
        }
        # 2nd rollout: representation probe (greedy, so the same decisions are replayed)
        if representation_metrics:
            if probe is None:
                probe = RepresentationProbe(agent.policy.actor, ACTION_STORE[representation])
            start = probe.mark()
            probe.resume()
            try:
                _rollout(agent, env, i)
            finally:
                probe.remove()
            row.update(flatten_summary(probe.summary(start)))
        rows.append(row)
    return rows, (probe.summary() if probe is not None else None)


def summarize(rows, repr_summary=None, model=None):
    """Per-metric descriptive statistics plus feasibility / violation totals."""
    summary = {
        "n_instances": len(rows),
        "feasibility_rate": sum(r["feasible"] for r in rows) / len(rows) if rows else None,
        "total_violations": {},
        "metrics": {m: describe([r.get(m) for r in rows]) for m in SUMMARY_METRICS},
        "size_range": {k: [min(r[k] for r in rows), max(r[k] for r in rows)]
                       for k in ("num_jobs", "num_machines", "num_operations")} if rows else {},
    }
    for r in rows:
        for name, n in r["violations"].items():
            summary["total_violations"][name] = summary["total_violations"].get(name, 0) + n
    if repr_summary is not None:
        summary["representation"] = repr_summary
    if model is not None:
        summary["model"] = model_size(model)
    return summary
