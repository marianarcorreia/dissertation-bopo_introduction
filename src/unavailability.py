"""Machine-unavailability windows: sampling (scheduled maintenance + Weibull breakdowns),
the information view a policy may use, and the time arithmetic shared by the environment
(src/env_unavailability.py), the CP-SAT reference (src/solver_blocking.py) and the metrics
(src/metrics/unavailability.py).

A window is a tuple (s, e, kind) on one machine, half-open [s, e), with integer times and
kind SCHEDULED (0) or BREAKDOWN (1). `windows` is always a list with one sorted, pairwise
disjoint list of windows per machine.

Semantics (identical in the env, the solver and the checks):
    - no operation starts inside a window;
    - scheduled windows are non-preemptive: an operation starting before a scheduled window
      must fit before it with its nominal processing time (x + p <= s);
    - breakdowns are preempt-resume: processing pauses during the window and continues after
      it, so an operation's end is resume_end(start, p, windows). A breakdown can push an
      operation into a later scheduled window, which then pauses it as well.
"""
import math

import numpy as np

SCHEDULED, BREAKDOWN = 0, 1
MODES = ("scheduled", "breakdown", "mixed")


# ── time arithmetic ─────────────────────────────────────────────────────────────

def earliest_start(r, p, windows_m):
    """Earliest x >= r at which an operation of nominal length p may start on a machine
    with these windows: not inside a window, and not crossing a scheduled window."""
    x = r
    moved = True
    while moved:
        moved = False
        for s, e, kind in windows_m:
            if s <= x < e or (kind == SCHEDULED and x < s < x + p):
                x = e
                moved = True
    return x


def resume_end(x, p, windows_m):
    """End of an operation started at x with p units of work, pausing during every window
    it meets (preempt-resume)."""
    cur, rem = x, p
    for s, e, _ in windows_m:
        if e <= cur:
            continue
        if s >= cur + rem:
            break
        rem -= max(s - cur, 0)
        cur = e
    return cur + rem


def window_at(windows_m, t):
    """The window containing t, or None."""
    for w in windows_m:
        if w[0] <= t < w[1]:
            return w
    return None


def next_window(windows_m, t):
    """The first window starting after t, or None."""
    for w in windows_m:
        if w[0] > t:
            return w
    return None


def last_renewal(windows_m, t):
    """End of the last window finished by t (a repair or a maintenance renews the machine)."""
    ends = [e for _, e, _ in windows_m if e <= t]
    return max(ends) if ends else 0.0


def p_fail(age, eta, beta, horizon):
    """Weibull probability that a machine of this age fails within `horizon`:
    1 - R(age + H) / R(age), with R(a) = exp(-(a / eta)^beta)."""
    return 1.0 - math.exp((age / eta) ** beta - ((age + horizon) / eta) ** beta)


def information_view(windows, mttr, t0):
    """What may be known at time t0: every scheduled window, the breakdowns that have already
    started, and, for a breakdown still in progress, only an estimated end t0 + MTTR_m (the
    repair time is exponential, so the expected remaining repair is MTTR_m whatever the
    elapsed time). Breakdowns that start after t0 are not in the view."""
    view = []
    for m, ws in enumerate(windows):
        seen = []
        for s, e, kind in ws:
            if kind == SCHEDULED or e <= t0:
                seen.append((s, e, kind))
            elif s <= t0:
                seen.append((s, t0 + mttr[m], kind))
        view.append(seen)
    return view


# ── sampling ────────────────────────────────────────────────────────────────────

def machine_loads(operations):
    """Expected work per machine if every operation is spread evenly over its eligible
    machines - used to find the bottleneck without relying on machine indices."""
    ops = np.asarray(operations, dtype=float)
    eligible = (ops > 0).sum(axis=1, keepdims=True)
    return (ops / np.maximum(eligible, 1)).sum(axis=0)


def sample_scheduled(operations, lb, rng, cfg):
    """Planned maintenance windows: cfg["per_machine"] per machine, plus
    cfg["bottleneck_extra"] on the most loaded machine, placed without overlap."""
    num_machines = len(operations[0])
    lo, hi = cfg["per_machine"]
    counts = [int(rng.integers(lo, hi + 1)) for _ in range(num_machines)]
    counts[int(np.argmax(machine_loads(operations)))] += cfg["bottleneck_extra"]
    windows = []
    for m in range(num_machines):
        placed = []
        for _ in range(counts[m]):
            for _attempt in range(50):
                length = max(1, int(round(rng.uniform(*cfg["length"]) * lb)))
                s = int(round(rng.uniform(*cfg["start"]) * lb))
                if all(s + length <= a or s >= b for a, b, _ in placed):
                    placed.append((s, s + length, SCHEDULED))
                    break
        windows.append(sorted(placed))
    return windows


def sample_reliability(num_machines, lb, rng, cfg):
    """Per machine: MTTR_m and the Weibull scale eta_m giving the drawn availability
    A_m = MTBF / (MTBF + MTTR) with MTBF = eta * Gamma(1 + 1/beta)."""
    beta = cfg["beta"]
    mttr, eta = [], []
    for _ in range(num_machines):
        r = rng.uniform(*cfg["mttr"]) * lb
        a = rng.uniform(*cfg["availability"])
        mtbf = r * a / (1.0 - a)
        mttr.append(float(r))
        eta.append(float(mtbf / math.gamma(1.0 + 1.0 / beta)))
    return {"beta": beta, "eta": eta, "mttr": mttr}


def sample_breakdowns(scheduled, reliability, t_end, rng):
    """Breakdowns on [0, t_end], machine by machine: from the last renewal, draw a Weibull
    time to failure; if a scheduled maintenance starts first, the machine is renewed at its
    end and a new failure time is drawn from there. A repair is exponential with mean MTTR_m
    and is cut short by a maintenance that starts during it (the maintenance takes over)."""
    beta = reliability["beta"]
    out = []
    for m, planned in enumerate(scheduled):
        eta, mttr = reliability["eta"][m], reliability["mttr"][m]
        t, fails = 0.0, []
        while t < t_end:
            failure = t + eta * rng.weibull(beta)
            maint = next((w for w in planned if w[1] > t), None)
            if maint is not None and maint[0] <= failure:
                t = maint[1]
                continue
            if failure >= t_end:
                break
            s = int(math.ceil(failure))
            e = s + max(1, int(round(rng.exponential(mttr))))
            if maint is not None:
                e = min(e, maint[0])
            if e > s:
                fails.append((s, e, BREAKDOWN))
            t = max(float(e), t + 1.0)
        out.append(fails)
    return out


def sample_scenario(operations, lb, mode, rng, cfg):
    """One realisation for an instance: {"windows", "beta", "eta", "mttr"}.
    The draws are always made in the same order, so a seed gives the same scheduled windows
    in "scheduled" and "mixed" mode. In "scheduled" mode the reliability entries are None."""
    if mode not in MODES:
        raise ValueError(f"unavailability mode must be one of {MODES}, got {mode!r}")
    num_machines = len(operations[0])
    planned = sample_scheduled(operations, lb, rng, cfg["scheduled"])
    reliability = sample_reliability(num_machines, lb, rng, cfg["breakdown"])
    renewals = planned if mode == "mixed" else [[] for _ in range(num_machines)]
    breakdowns = sample_breakdowns(renewals, reliability, cfg["timeline_factor"] * lb, rng)
    windows = []
    for m in range(num_machines):
        ws = (planned[m] if mode != "breakdown" else []) + (breakdowns[m] if mode != "scheduled" else [])
        windows.append(sorted(ws))
    if mode == "scheduled":
        reliability = {"beta": None, "eta": None, "mttr": None}
    return {"windows": windows, **reliability}


def scenario_to_json(scenario):
    return {**scenario, "windows": [[list(w) for w in ws] for ws in scenario["windows"]]}


def scenario_from_json(data):
    return {**data, "windows": [sorted(tuple(int(v) for v in w) for w in ws) for ws in data["windows"]]}
