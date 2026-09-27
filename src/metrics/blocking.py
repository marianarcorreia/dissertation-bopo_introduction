"""Constraint checks and statistics for the blocking FJSP with finite buffers
(src/env_blocking.py), mirroring the CP-SAT model in src/solver_blocking.py.

A blocking schedule entry (see FJSPEnvBlocking._record_schedule) has, besides job /
operation / machine / start / end:
    routed     the part enters the machine's input buffer   -> input-buffer interval [routed, start)
    depart     the part leaves the machine (>= end: a finished part held on a blocked machine
               keeps occupying it)                            -> machine interval      [start, depart)
    leave_out  the part leaves the output buffer (= the job's next operation is routed)
                                                              -> output-buffer interval [depart, leave_out)
Intervals are half-open, so a part that passes through a buffer at a single instant (e.g. is
routed and starts at the same time) never occupies it - as with the solver's zero-length
intervals.
"""
from src.metrics.schedule import TOL, Completeness, Constraint, Eligibility, Precedence, ProcessingTime


def _exceeds(new, previous, cap):
    """True if adding interval `new` to `previous` (same resource) puts more than `cap`
    intervals at the same time somewhere inside `new`."""
    a, b = new
    if b - a <= TOL:
        return False
    points = [a] + [s for s, _ in previous if a < s < b]
    for p in points:
        if 1 + sum(1 for s, e in previous if s <= p + TOL and p < e - TOL and e - s > TOL) > cap:
            return True
    return False


class _Resource(Constraint):
    """Capacity of a per-machine resource whose occupation interval is given by interval()."""
    cap = 1

    def interval(self, rec):
        raise NotImplementedError

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        m = int(rec["machine"])
        previous = [self.interval(r) for r in schedule[:i] if int(r["machine"]) == m]
        return int(_exceeds(self.interval(rec), previous, self.cap))


class BlockingMachineCapacity(_Resource):
    """One part on a machine at a time, from its start until it physically leaves (depart):
    holding a finished part while blocked occupies the machine."""
    name = "machine_capacity"

    def interval(self, rec):
        return float(rec["start"]), float(rec["depart"])


class InputBufferCapacity(_Resource):
    name = "input_buffer_capacity"

    def __init__(self, cap):
        self.cap = cap

    def interval(self, rec):
        return float(rec["routed"]), float(rec["start"])


class OutputBufferCapacity(_Resource):
    name = "output_buffer_capacity"

    def __init__(self, cap):
        self.cap = cap

    def interval(self, rec):
        return float(rec["depart"]), float(rec["leave_out"])


class BlockingFlow(Constraint):
    """The part's path is consistent: routed before it starts, leaves the machine no earlier
    than it ends (and exactly then if it is the job's last operation, which leaves the system),
    and enters the next machine only after leaving the previous one."""
    name = "blocking_flow"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        errors = 0
        if any(rec.get(k) is None for k in ("routed", "start", "end", "depart", "leave_out")):
            return 1
        errors += float(rec["routed"]) > float(rec["start"]) + TOL
        errors += float(rec["depart"]) < float(rec["end"]) - TOL
        errors += float(rec["leave_out"]) < float(rec["depart"]) - TOL
        job = instance["jobs"][int(rec["job"])]
        op = int(rec["operation"])
        if op in job:
            k = job.index(op)
            if k == len(job) - 1:
                errors += abs(float(rec["depart"]) - float(rec["end"])) > TOL
            if k > 0:
                prev = next((r for r in schedule[:i] if int(r["operation"]) == job[k - 1]), None)
                if prev is not None and float(prev["depart"]) > float(rec["routed"]) + TOL:
                    errors += 1
        return int(errors)


def blocking_constraints(in_cap, out_cap):
    return (Completeness(), Eligibility(), ProcessingTime(), Precedence(), BlockingMachineCapacity(),
            InputBufferCapacity(in_cap), OutputBufferCapacity(out_cap), BlockingFlow())


def blocking_stats(schedule):
    """How much blocking the schedule suffered: operations whose machine was blocked after they
    finished, and the total machine time lost holding finished parts."""
    held = [float(r["depart"]) - float(r["end"]) for r in schedule
            if r.get("depart") is not None and r.get("end") is not None]
    return {
        "blocked_ops": sum(1 for h in held if h > TOL),
        "blocked_time": float(sum(h for h in held if h > TOL)),
    }
