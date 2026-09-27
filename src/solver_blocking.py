"""CP-SAT model of the blocking FJSP with finite buffers (reference for src/env_blocking.py).

For every operation k of a job, with the chosen machine m_k:
    r_k  time the part enters m_k's input buffer (for k = 0: when it enters the system)
    s_k  start on m_k                                  (in-buffer interval [r_k, s_k])
    d_k  time the part leaves m_k, d_k >= s_k + p      (machine interval [s_k, d_k]: holding
                                                        a finished part while blocked occupies
                                                        the machine)
    the part then waits in m_k's output buffer until it enters the next machine's input
    buffer: out-buffer interval [d_k, r_{k+1}]. The last operation leaves at d = s + p.
Capacities: cumulative constraints with capacity in_cap / out_cap per machine buffer;
machines: no-overlap on the machine intervals. Deadlocks are excluded by feasibility.

Not modelled (they are dispatching rules of the env, not physical constraints, so the env's
makespan is >= this model's optimum): FIFO order inside the input buffer, non-delay routing,
and "a machine with a full output buffer accepts no routing".

Machine unavailability (windows, src/unavailability.py; the reference of
src/env_unavailability.py): per chosen machine, with c_k the end of processing,
    - s_k is not inside a window: for every window w, a_w -> s_k >= e_w, not a_w -> s_k < s_w;
    - scheduled windows are non-preemptive: not a_w -> s_k + p <= s_w;
    - breakdowns are preempt-resume: y_w (processing crosses w) <-> not a_w and
      s_k + p + sum(len_w' y_w' for earlier w') > s_w, and c_k = s_k + p + sum(len_w y_w).
    The machine interval [s_k, d_k] is not blocked by windows: a finished part held on a
    blocked machine can stay there, or leave, during a window (unloading continues).
With breakdowns the windows are the realised ones, known in advance: the result is a
clairvoyant (perfect-information) bound for an online policy, not a reachable optimum.
"""
from ortools.sat.python import cp_model


def solve_blocking_fjsp(jobs, operations, in_cap=5, out_cap=2, time_limit=60.0, workers=8,
                        windows=None, horizon=None, hint=None):
    """jobs: list of operation-id lists; operations[o][m] = processing time (0 = ineligible).
    windows: optional unavailability windows, one sorted list of (start, end, kind) per machine.
    horizon: optional upper bound on the makespan (e.g. a heuristic's makespan); windows
    starting after it are dropped.
    hint: optional feasible schedule {o: dict(machine, r, s, c, d)} (e.g. the env's, see
    src/generate_unavailability_references.py) given to CP-SAT as a starting solution.
    Returns (status_name, makespan, schedule) where schedule[o] = dict(machine, r, s, c, d)
    (c = end of processing, later than s + p when a breakdown interrupted it)."""
    from src.unavailability import SCHEDULED
    num_machines = len(operations[0])
    if horizon is None:
        horizon = int(sum(max(row) for row in operations))  # a serial schedule never uses buffers
        if windows:
            # a serial schedule after every window
            horizon += max((e for ws in windows for _, e, _ in ws), default=0)
    horizon = int(horizon)
    if windows:
        windows = [[(int(a), int(b), k) for a, b, k in ws if a < horizon] for ws in windows]
    model = cp_model.CpModel()

    r, s, c, d, presence = {}, {}, {}, {}, {}
    window_vars, cross_vars = [], []
    machine_iv = [[] for _ in range(num_machines)]
    in_iv = [[] for _ in range(num_machines)]
    out_iv = [[] for _ in range(num_machines)]
    ends = []

    for job in jobs:
        for k, o in enumerate(job):
            r[o] = model.new_int_var(0, horizon, f"r{o}")
            s[o] = model.new_int_var(0, horizon, f"s{o}")
            d[o] = model.new_int_var(0, horizon, f"d{o}")
            c[o] = model.new_int_var(0, horizon, f"c{o}")
        for k, o in enumerate(job):
            last = k == len(job) - 1
            occ = model.new_int_var(0, horizon, f"occ{o}")
            wait_in = model.new_int_var(0, horizon, f"win{o}")
            wait_out = None if last else model.new_int_var(0, horizon, f"wout{o}")
            if not last:
                model.add(r[job[k + 1]] >= d[o])
            lits = []
            for m, p in enumerate(operations[o]):
                if p == 0:
                    continue
                lit = model.new_bool_var(f"x{o}_{m}")
                lits.append(lit)
                presence[o, m] = lit
                p = int(p)
                stretch = []
                for w, (ws, we, kind) in enumerate(windows[m] if windows else []):
                    after = model.new_bool_var(f"a{o}_{m}_{w}")
                    window_vars.append((o, m, ws, we, after))
                    model.add(s[o] >= we).only_enforce_if([lit, after])
                    model.add(s[o] <= ws - 1).only_enforce_if([lit, after.Not()])
                    if kind == SCHEDULED:
                        model.add(s[o] + p <= ws).only_enforce_if([lit, after.Not()])
                    cross = model.new_bool_var(f"y{o}_{m}_{w}")
                    cross_vars.append((o, m, ws, cross))
                    reach = s[o] + p + sum(stretch)  # where processing reaches without w
                    model.add_implication(cross, lit)
                    model.add_implication(cross, after.Not())
                    model.add(reach >= ws + 1).only_enforce_if(cross)
                    model.add(reach <= ws).only_enforce_if([lit, after.Not(), cross.Not()])
                    stretch.append((we - ws) * cross)
                model.add(c[o] == s[o] + p + sum(stretch)).only_enforce_if(lit)
                if last:
                    model.add(d[o] == c[o]).only_enforce_if(lit)
                else:
                    model.add(d[o] >= c[o]).only_enforce_if(lit)
                machine_iv[m].append(model.new_optional_interval_var(s[o], occ, d[o], lit, f"mach{o}_{m}"))
                in_iv[m].append(model.new_optional_interval_var(r[o], wait_in, s[o], lit, f"in{o}_{m}"))
                if not last:
                    out_iv[m].append(model.new_optional_interval_var(d[o], wait_out, r[job[k + 1]], lit, f"out{o}_{m}"))
            model.add_exactly_one(lits)
        ends.append(d[job[-1]])

    for m in range(num_machines):
        if machine_iv[m]:
            model.add_no_overlap(machine_iv[m])
        if in_iv[m]:
            model.add_cumulative(in_iv[m], [1] * len(in_iv[m]), in_cap)
        if out_iv[m]:
            model.add_cumulative(out_iv[m], [1] * len(out_iv[m]), out_cap)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)

    if hint:
        for o, h in hint.items():
            for var, key in ((r, "r"), (s, "s"), (c, "c"), (d, "d")):
                model.add_hint(var[o], int(round(h[key])))
        for (o, m), lit in presence.items():
            model.add_hint(lit, int(hint[o]["machine"] == m))
        for o, m, ws, we, after in window_vars:
            model.add_hint(after, int(hint[o]["machine"] == m and hint[o]["s"] >= we))
        for o, m, ws, cross in cross_vars:
            h = hint[o]
            model.add_hint(cross, int(h["machine"] == m and h["s"] < ws < h["c"]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.status_name(status), None, None

    schedule = {}
    for (o, m), lit in presence.items():
        if solver.value(lit):
            schedule[o] = {"machine": m, "r": solver.value(r[o]), "s": solver.value(s[o]),
                           "c": solver.value(c[o]), "d": solver.value(d[o])}
    return solver.status_name(status), float(solver.value(makespan)), schedule
