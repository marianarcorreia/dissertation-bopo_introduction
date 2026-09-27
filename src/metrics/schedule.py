"""Schedule-quality metrics: makespan, lower bound, relative error, scheduling score and
constraint feasibility / violation rate per step.

All of these work on the finished schedule each env records in `env.schedule` (one dict
per scheduled operation, in decision order: job, operation, machine, start, end) plus the
instance ({"jobs", "operations"}), never on env internals. So the same code applies to
every representation (oo / om / ojm), and a constraint branch only has to add its own
checks: subclass `Constraint` and pass the list to `check_schedule(...)` (or extend
`FJSP_CONSTRAINTS`), e.g. a blocking or machine-unavailability check.
"""
from collections import defaultdict
from itertools import combinations

from src.transport import job_lower_bound

TOL = 1e-6


# ---------------------------------------------------------------------- objective
def makespan(schedule):
    return max((float(r["end"]) for r in schedule), default=0.0)


def lower_bound(instance, exhaustive_up_to=10):
    """Cheap, solver-free makespan lower bound for an FJSP instance: the larger of
    (a) the longest job, taking every operation on its fastest eligible machine, and
    (b) for a set S of machines, the fastest-machine work of the operations that can ONLY run
        on machines of S, spread evenly over S. With S = all machines this is the average
        load; smaller sets catch bottlenecks (operations with few eligible machines).
    (b) is taken over every S when there are at most `exhaustive_up_to` machines, otherwise over
    the operations' own eligibility sets and the full set (as tight on every set checked).
    With parallel batching (instance "family" / "capacities"), up to B_f operations of family f
    share one batch lasting at least as long as each of them, so each one's work in (b) counts
    1 / B_f.
    With transport (instance "coords" / "depot", src/transport.py), (a) is the shortest
    depot -> ... -> last machine path of the job including transport times; transport does
    not occupy machines, so (b) is unchanged. Without transport (a) is the same as before.
    Used as the reference for relative error when no CP-SAT solution exists (e.g. the
    unseen-size test sets), so relative error is always defined."""
    ops = instance["operations"]
    num_machines = len(ops[0])
    min_proc = [min(p for p in row if p > 0) for row in ops]
    family, capacities = instance.get("family"), instance.get("capacities")
    load = [p / capacities[f] if family and (f := family[o]) >= 0 else p for o, p in enumerate(min_proc)]
    eligible = [frozenset(m for m, p in enumerate(row) if p > 0) for row in ops]
    best = job_lower_bound(instance)
    if num_machines <= exhaustive_up_to:
        sets = [frozenset(s) for k in range(1, num_machines + 1) for s in combinations(range(num_machines), k)]
    else:
        sets = set(eligible) | {frozenset(range(num_machines))}
    for s in sets:
        work = sum(w for w, e in zip(load, eligible) if e <= s)
        best = max(best, work / len(s))
    return float(best)


def relative_error(value, reference):
    """(C - C_ref) / C_ref. Same definition as the validation `gap` in
    src/utils/validation_utils.py:run_validation, so the numbers are comparable."""
    if reference is None or reference <= 0:
        return None
    return float(value) / float(reference) - 1.0


def scheduling_score(value, reference, feasible):
    """C_ref / C in (0, 1] when the policy is no better than the reference (1 = matches it,
    > 1 = beats it), and 0 for an infeasible schedule - so a policy can't score well by
    breaking a constraint. Unlike the relative error it is bounded and averages without
    being dominated by a few very bad instances."""
    if not feasible or reference is None or value <= 0:
        return 0.0
    return float(reference) / float(value)


# -------------------------------------------------------------------- constraints
class Constraint:
    """One family of constraints. check_step() sees the schedule up to and including
    decision i and returns how many violations decision i introduced; check_final() runs
    once on the whole schedule (for things only decidable at the end, e.g. completeness)."""
    name = "constraint"

    def check_step(self, instance, schedule, i):
        return 0

    def check_final(self, instance, schedule):
        return 0


class Completeness(Constraint):
    """Every operation scheduled exactly once, and on the job it belongs to."""
    name = "completeness"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        op = int(rec["operation"])
        errors = 0
        if not 0 <= op < len(instance["operations"]):
            return 1
        if any(int(r["operation"]) == op for r in schedule[:i]):
            errors += 1  # duplicate
        if op not in instance["jobs"][int(rec["job"])]:
            errors += 1  # wrong job
        return errors

    def check_final(self, instance, schedule):
        done = {int(r["operation"]) for r in schedule}
        return sum(1 for o in range(len(instance["operations"])) if o not in done)


class Eligibility(Constraint):
    """The machine can process the operation (processing time > 0)."""
    name = "eligibility"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        return int(instance["operations"][int(rec["operation"])][int(rec["machine"])] <= 0)


class ProcessingTime(Constraint):
    """end - start equals the operation's processing time on the chosen machine, and
    nothing starts before time 0."""
    name = "processing_time"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        p = instance["operations"][int(rec["operation"])][int(rec["machine"])]
        start, end = float(rec["start"]), float(rec["end"])
        return int(abs((end - start) - p) > TOL) + int(start < -TOL)


class Precedence(Constraint):
    """The job's previous operation was scheduled earlier and ends before this one starts."""
    name = "precedence"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        job = instance["jobs"][int(rec["job"])]
        op = int(rec["operation"])
        if op not in job or job.index(op) == 0:
            return 0
        prev_op = job[job.index(op) - 1]
        prev = next((r for r in schedule[:i] if int(r["operation"]) == prev_op), None)
        return int(prev is None or float(prev["end"]) > float(rec["start"]) + TOL)


class MachineCapacity(Constraint):
    """A machine processes one operation at a time: no overlap with anything already
    placed on the same machine (order-independent interval check)."""
    name = "machine_capacity"

    def check_step(self, instance, schedule, i):
        rec = schedule[i]
        m, s, e = int(rec["machine"]), float(rec["start"]), float(rec["end"])
        return sum(1 for r in schedule[:i]
                   if int(r["machine"]) == m and float(r["start"]) < e - TOL and s < float(r["end"]) - TOL)


FJSP_CONSTRAINTS = (Completeness(), Eligibility(), ProcessingTime(), Precedence(), MachineCapacity())


def check_schedule(instance, schedule, constraints=FJSP_CONSTRAINTS, step_key=None):
    """Independent feasibility check of a recorded schedule.

    step_key: when one decision schedules several entries (e.g. a whole batch), the entry
    field that identifies the decision; by default every entry is its own decision.

    Returns:
        feasible                   - no violation of any constraint, and every operation scheduled
        n_steps                    - scheduling decisions
        n_violating_steps          - decisions that broke at least one constraint
        violation_rate_per_step    - n_violating_steps / n_steps
        violations                 - {constraint name: number of violations}
    """
    violations = defaultdict(int)
    step_violated = {}
    for i in range(len(schedule)):
        step_errors = 0
        for c in constraints:
            n = c.check_step(instance, schedule, i)
            violations[c.name] += n
            step_errors += n
        step = i if step_key is None else schedule[i][step_key]
        step_violated[step] = step_violated.get(step, False) or step_errors > 0
    for c in constraints:
        violations[c.name] += c.check_final(instance, schedule)

    n_steps = len(step_violated)
    violating_steps = sum(step_violated.values())
    return {
        "feasible": all(v == 0 for v in violations.values()),
        "n_steps": n_steps,
        "n_violating_steps": violating_steps,
        "violation_rate_per_step": violating_steps / n_steps if n_steps else 0.0,
        "violations": {c.name: int(violations[c.name]) for c in constraints},
    }


def schedule_metrics(instance, schedule, reference=None, constraints=FJSP_CONSTRAINTS, step_key=None):
    """All schedule-level metrics for one solved instance. `reference` is the CP-SAT
    makespan when one exists; the lower bound is always computed as well."""
    mk = makespan(schedule)
    lb = lower_bound(instance)
    check = check_schedule(instance, schedule, constraints, step_key)
    ref = reference if reference is not None else lb
    return {
        "makespan": mk,
        "reference": reference,
        "lower_bound": lb,
        "reference_source": "cp_sat" if reference is not None else "lower_bound",
        "relative_error": relative_error(mk, reference),
        "relative_error_lb": relative_error(mk, lb),
        "scheduling_score": scheduling_score(mk, ref, check["feasible"]),
        **check,
    }
