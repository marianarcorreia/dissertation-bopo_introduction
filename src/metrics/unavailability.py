"""Constraint checks and statistics for machine unavailability on the blocking FJSP
(src/env_unavailability.py), with the semantics of src/unavailability.py:
    - no operation starts inside a window of its machine;
    - an operation that starts before a scheduled window fits before it (nominal time);
    - processing pauses during windows (preempt-resume): end = resume_end(start, p, windows).
Holding a finished part on a blocked machine during a window is allowed (unloading continues),
so the blocking checks of src/metrics/blocking.py apply unchanged, except ProcessingTime, which
resume replaces.

The windows of the episode are not in the schedule entries: each check takes a function
instance -> windows (see src/metrics/evaluate.py:constraints_for, which reads them from the env
that produced the schedule).
"""
from src.metrics.blocking import (BlockingFlow, BlockingMachineCapacity, InputBufferCapacity,
                                  OutputBufferCapacity)
from src.metrics.schedule import TOL, Completeness, Constraint, Eligibility, Precedence
from src.unavailability import SCHEDULED, resume_end


def stored_windows(mode):
    """windows_fn reading the realisation stored in the instance (validation / test splits)."""
    def windows_fn(instance):
        return [[tuple(w) for w in ws] for ws in instance["unavailability"][mode]["windows"]]
    return windows_fn


class ProcessingTimeResume(Constraint):
    """end - start equals the processing time plus the downtime of the windows it crossed,
    and nothing starts before time 0."""
    name = "processing_time"

    def __init__(self, windows_fn):
        self.windows_fn = windows_fn

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        m = int(rec["machine"])
        p = instance["operations"][int(rec["operation"])][m]
        start, end = float(rec["start"]), float(rec["end"])
        expected = resume_end(start, p, self.windows_fn(instance)[m])
        return int(abs(end - expected) > TOL) + int(start < -TOL)


class Unavailability(Constraint):
    """No start inside a window, and no scheduled window crossed by an operation that started
    before it (with its nominal processing time)."""
    name = "machine_unavailability"

    def __init__(self, windows_fn):
        self.windows_fn = windows_fn

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        m = int(rec["machine"])
        p = instance["operations"][int(rec["operation"])][m]
        x = float(rec["start"])
        errors = 0
        for s, e, kind in self.windows_fn(instance)[m]:
            errors += s - TOL <= x < e - TOL
            errors += kind == SCHEDULED and x < s - TOL and x + p > s + TOL
        return int(errors)


def unavailability_constraints(in_cap, out_cap, windows_fn):
    return (Completeness(), Eligibility(), ProcessingTimeResume(windows_fn), Precedence(),
            BlockingMachineCapacity(), InputBufferCapacity(in_cap), OutputBufferCapacity(out_cap),
            BlockingFlow(), Unavailability(windows_fn))


def _overlap(a, b, windows_m):
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e, _ in windows_m)


def unavailability_stats(schedule, windows):
    """How much the windows hurt the schedule:
        interrupted_ops / interrupted_time  operations paused by a window, and the time lost
        trapped_parts / trapped_time        parts waiting in the input buffer of a machine while
                                            it was down (routed there before or during a window)
    The trapped parts are the downtime-induced blocking cascade: they hold input-buffer slots."""
    interrupted = [float(r.get("interrupted", 0.0)) for r in schedule]
    trapped = [_overlap(float(r["routed"]), float(r["start"]), windows[int(r["machine"])])
               for r in schedule if r.get("routed") is not None]
    return {
        "interrupted_ops": sum(1 for x in interrupted if x > TOL),
        "interrupted_time": float(sum(x for x in interrupted if x > TOL)),
        "trapped_parts": sum(1 for x in trapped if x > TOL),
        "trapped_time": float(sum(trapped)),
    }
