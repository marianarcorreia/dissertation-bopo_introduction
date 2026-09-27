"""Constraint checks and statistics for the FJSP with parallel batching (src/env_batching.py),
mirroring src/batch_solver.py:check_schedule and the CP-SAT model's (B1)-(B7).

The env records one entry per operation with "operation" = its position in the job (kappa)
and "op_id" = its global id (the format batch_solver.check_schedule expects).
metric_schedule() maps it to the shared metrics format ("operation" = global id, "position" =
kappa); a decision schedules a whole batch, so violation rates are counted per "batch".
"""
from src.metrics.schedule import TOL, Completeness, Constraint, Eligibility, Precedence


def metric_schedule(schedule):
    return [dict(e, operation=int(e["op_id"]), position=int(e["operation"])) for e in schedule]


def _members(schedule, batch):
    return [r for r in schedule if r["batch"] == batch]


def _first_of_batch(schedule, i):
    return all(r["batch"] != schedule[i]["batch"] for r in schedule[:i])


class BatchProcessingTime(Constraint):
    """The batch lasts exactly the longest processing time of its members on its machine (B4a),
    and nothing starts before 0."""
    name = "processing_time"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        m = int(rec["machine"])
        p = max(instance["operations"][int(r["operation"])][m] for r in _members(schedule, rec["batch"]))
        return int(abs((float(rec["end"]) - float(rec["start"])) - p) > TOL) + int(float(rec["start"]) < -TOL)


class BatchComposition(Constraint):
    """Who can share a batch (B2, B2a, B3, B5): one family (operations without a family run
    alone), at most B_f members, order difference <= delta, one operation per job, and all
    members on the same machine at the same time. Counted once per batch."""
    name = "batch_composition"

    def check_step(self, instance, schedule, i):
        if not _first_of_batch(schedule, i):
            return 0
        members = _members(schedule, schedule[i]["batch"])
        family = instance["family"]
        fams = {family[int(r["operation"])] for r in members}
        errors = 0
        if len(members) > 1:
            errors += len(fams) != 1 or -1 in fams
            f = next(iter(fams))
            errors += f >= 0 and len(members) > instance["capacities"][f]
            positions = [int(r["position"]) for r in members]
            errors += max(positions) - min(positions) > instance["delta"]
            errors += len({r["job"] for r in members}) != len(members)
        errors += len({(r["machine"], float(r["start"]), float(r["end"])) for r in members}) != 1
        return int(errors)


class BatchMachineCapacity(Constraint):
    """No overlap on a machine between different batches (B7); members of one batch share it."""
    name = "machine_capacity"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        m, s, e = int(rec["machine"]), float(rec["start"]), float(rec["end"])
        return sum(1 for r in schedule[:i]
                   if r["batch"] != rec["batch"] and int(r["machine"]) == m
                   and float(r["start"]) < e - TOL and s < float(r["end"]) - TOL)


BATCHING_CONSTRAINTS = (Completeness(), Eligibility(), BatchProcessingTime(), Precedence(),
                        BatchMachineCapacity(), BatchComposition())


def batch_stats(schedule):
    """How much the policy batched: number of batches, their mean size, and the share of
    operations processed together with at least one other."""
    sizes = {}
    for r in schedule:
        sizes[r["batch"]] = sizes.get(r["batch"], 0) + 1
    n_ops = len(schedule)
    return {
        "num_batches": len(sizes),
        "avg_batch_size": n_ops / len(sizes) if sizes else None,
        "batched_fraction": sum(s for s in sizes.values() if s > 1) / n_ops if n_ops else None,
    }
