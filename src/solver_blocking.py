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
"""
from ortools.sat.python import cp_model


def solve_blocking_fjsp(jobs, operations, in_cap=5, out_cap=2, time_limit=60.0, workers=8):
    """jobs: list of operation-id lists; operations[o][m] = processing time (0 = ineligible).
    Returns (status_name, makespan, schedule) where schedule[o] = dict(machine, r, s, d)."""
    num_machines = len(operations[0])
    horizon = int(sum(max(row) for row in operations))  # a serial schedule never uses buffers
    model = cp_model.CpModel()

    r, s, d, presence = {}, {}, {}, {}
    machine_iv = [[] for _ in range(num_machines)]
    in_iv = [[] for _ in range(num_machines)]
    out_iv = [[] for _ in range(num_machines)]
    ends = []

    for job in jobs:
        for k, o in enumerate(job):
            r[o] = model.new_int_var(0, horizon, f"r{o}")
            s[o] = model.new_int_var(0, horizon, f"s{o}")
            d[o] = model.new_int_var(0, horizon, f"d{o}")
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
                if last:
                    model.add(d[o] == s[o] + p).only_enforce_if(lit)
                else:
                    model.add(d[o] >= s[o] + p).only_enforce_if(lit)
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

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.status_name(status), None, None

    schedule = {}
    for (o, m), lit in presence.items():
        if solver.value(lit):
            schedule[o] = {"machine": m, "r": solver.value(r[o]), "s": solver.value(s[o]), "d": solver.value(d[o])}
    return solver.status_name(status), float(solver.value(makespan)), schedule
