"""
CP-SAT solver for the FJSP with parallel batching (docs/batching_formulation.tex, (B1)-(B7)).

Model (per family f with members O^f, potential batches B_f = one per member):
  x[o,b]   operation o is in batch b                               (B1), (B2)
  representative symmetry breaking: batch b is "owned" by the b-th member of the family,
  so member i can only join batches b <= i and batch b is used iff its owner is in it.
  order difference: x[o,b] + x[o',b] <= 1 if |kappa(o)-kappa(o')| > delta   (B2a)
  capacity: sum_o x[o,b] <= B_f                                     (B3)
  one eligible machine per used batch, optional interval per (b, machine)   (B4)
  parallel batching: duration on machine mu = max_{o in b} p[o,mu]  (B4a)
  all members start/end with the batch                              (B5)
  job precedence                                                    (B6)
  no overlap of batch intervals on each machine                     (B7)
Operations with family -1 are singleton families with capacity 1.

Transport (docs/transport_formulation.tex, (T-a)-(T-d)), when the instance has a layout
(src/transport.py): z[o,mu] <=> operation o runs on machine mu, and consecutive operations
of a job on machines mu != nu are separated by the transport time tau[mu][nu]; the first
operation of a job cannot start before the transport from the depot.
"""
from ortools.sat.python import cp_model

from src.transport import depot_index, has_transport, transport_matrix


def _groups(instance):
    """Return a list of (members, capacity) - every operation appears in exactly one group."""
    family = instance["family"]
    caps = instance["capacities"]
    groups = {}
    singles = []
    for o, f in enumerate(family):
        if f < 0:
            singles.append(([o], 1))
        else:
            groups.setdefault(f, []).append(o)
    return [(sorted(groups[f]), caps[f]) for f in sorted(groups)] + singles


def solve_batching(instance, time_limit=30.0, workers=8, seed=0):
    jobs = instance["jobs"]
    ops = instance["operations"]
    delta = instance["delta"]
    n_mach = instance["num_machines"]
    n_ops = len(ops)

    op_job, op_kappa = [0] * n_ops, [0] * n_ops
    for j, job in enumerate(jobs):
        for k, o in enumerate(job):
            op_job[o], op_kappa[o] = j, k + 1

    tau = transport_matrix(instance)
    depot = depot_index(instance)
    transport = has_transport(instance) and bool(tau.any())
    horizon = sum(max(row) for row in ops) + (n_ops * int(tau.max()) if transport else 0)
    model = cp_model.CpModel()
    # z[o][mu]: operation o runs on machine mu (transport only)
    z = [{mu: model.new_bool_var(f"z_{o}_{mu}") for mu in range(n_mach) if ops[o][mu] > 0}
         for o in range(n_ops)] if transport else None

    op_start = [model.new_int_var(0, horizon, f"S_o{o}") for o in range(n_ops)]
    op_end = [model.new_int_var(0, horizon, f"C_o{o}") for o in range(n_ops)]
    machine_intervals = [[] for _ in range(n_mach)]
    batch_records = []  # (members, x[i], used, start, end, {mu: presence})

    for members, cap in _groups(instance):
        eligible = [mu for mu in range(n_mach) if all(ops[o][mu] > 0 for o in members)]
        if not eligible:
            raise ValueError(f"Family {members} has no common eligible machine")
        n = len(members)
        # x[i][b]: member i in batch b (only b <= i)
        x = [[model.new_bool_var(f"x_{members[i]}_{members[b]}") if b <= i else None
              for b in range(n)] for i in range(n)]
        for i in range(n):
            model.add_exactly_one(x[i][b] for b in range(i + 1))                      # (B1)
        for b in range(n):
            used = x[b][b]
            for i in range(b + 1, n):
                model.add_implication(x[i][b], used)
                oi, ob = members[i], members[b]
                if op_job[oi] == op_job[ob]:
                    model.add(x[i][b] == 0)  # (F1) safety net
            # (B2a) order difference and (F1) between any two non-owner members
            for i in range(b, n):
                for i2 in range(i + 1, n):
                    o1, o2 = members[i], members[i2]
                    if abs(op_kappa[o1] - op_kappa[o2]) > delta or op_job[o1] == op_job[o2]:
                        model.add_bool_or([x[i][b].Not(), x[i2][b].Not()])
            model.add(sum(x[i][b] for i in range(b, n)) <= cap)                         # (B3)

            start = model.new_int_var(0, horizon, f"S_b{members[b]}")
            end = model.new_int_var(0, horizon, f"C_b{members[b]}")
            presences = {}
            for mu in eligible:
                pres = model.new_bool_var(f"y_{members[b]}_{mu}")
                pmax = max(ops[members[i]][mu] for i in range(b, n))
                dur = model.new_int_var(0, pmax, f"d_{members[b]}_{mu}")
                # (B4a) parallel batching: duration = longest member on this machine
                model.add_max_equality(dur, [ops[members[i]][mu] * x[i][b] for i in range(b, n)])
                itv = model.new_optional_interval_var(start, dur, end, pres, f"I_{members[b]}_{mu}")
                machine_intervals[mu].append(itv)
                presences[mu] = pres
            model.add(sum(presences.values()) == used)                                 # (B4)
            if transport:                                                               # (T-a)
                for i in range(b, n):
                    for mu, pres in presences.items():
                        model.add_bool_or([x[i][b].Not(), pres.Not(), z[members[i]][mu]])
            model.add(start == 0).only_enforce_if(used.Not())
            model.add(end == 0).only_enforce_if(used.Not())
            for i in range(b, n):                                                       # (B5)
                o = members[i]
                model.add(op_start[o] == start).only_enforce_if(x[i][b])
                model.add(op_end[o] == end).only_enforce_if(x[i][b])
            batch_records.append((members, b, x, used, start, end, presences))

    for job in jobs:                                                                    # (B6)
        for a, c in zip(job[:-1], job[1:]):
            model.add(op_start[c] >= op_end[a])
    if transport:
        for o in range(n_ops):                                                          # (T-b)
            model.add_exactly_one(z[o].values())
        for job in jobs:
            for nu, lit in z[job[0]].items():                                           # (T-d)
                if tau[depot][nu] > 0:
                    model.add(op_start[job[0]] >= int(tau[depot][nu])).only_enforce_if(lit)
            for a, c in zip(job[:-1], job[1:]):                                         # (T-c)
                for mu, za in z[a].items():
                    for nu, zc in z[c].items():
                        if mu != nu and tau[mu][nu] > 0:
                            model.add(op_start[c] >= op_end[a] + int(tau[mu][nu])).only_enforce_if([za, zc])
    for mu in range(n_mach):                                                            # (B7)
        if len(machine_intervals[mu]) > 1:
            model.add_no_overlap(machine_intervals[mu])

    makespan = model.new_int_var(0, horizon, "Cmax")
    model.add_max_equality(makespan, [op_end[job[-1]] for job in jobs])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status_name, "makespan": None, "final_schedule": []}

    schedule = []
    batch_id = 0
    for members, b, x, used, start, end, presences in batch_records:
        if not solver.value(used):
            continue
        mu = next(m for m, p in presences.items() if solver.value(p))
        for i in range(b, len(members)):
            if solver.value(x[i][b]):
                o = members[i]
                schedule.append({
                    "job": op_job[o],
                    "operation": op_kappa[o] - 1,
                    "op_id": o,
                    "family": instance["family"][o],
                    "batch": batch_id,
                    "machine": mu,
                    "start": solver.value(start),
                    "end": solver.value(end),
                })
        batch_id += 1
    schedule.sort(key=lambda e: (e["job"], e["operation"]))

    return {
        "status": status_name,
        "makespan": int(solver.objective_value),
        "best_bound": int(solver.best_objective_bound),
        "wall_time": solver.wall_time,
        "num_batches": batch_id,
        "num_multi_op_batches": sum(
            1 for bid in range(batch_id) if sum(e["batch"] == bid for e in schedule) > 1),
        "final_schedule": schedule,
    }


def check_schedule(instance, schedule):
    """Independent feasibility check of a batched schedule. Returns a list of violations."""
    jobs, ops = instance["jobs"], instance["operations"]
    family, caps, delta = instance["family"], instance["capacities"], instance["delta"]
    errors = []
    by_op = {e["op_id"]: e for e in schedule}
    if sorted(by_op) != list(range(len(ops))):
        errors.append("not every operation is scheduled exactly once")
        return errors

    batches = {}
    for e in schedule:
        batches.setdefault(e["batch"], []).append(e)
    for bid, members in batches.items():
        ids = [e["op_id"] for e in members]
        fams = {family[o] for o in ids}
        mu = {e["machine"] for e in members}
        starts = {e["start"] for e in members}
        ends = {e["end"] for e in members}
        if len(ids) > 1 and (len(fams) != 1 or -1 in fams):
            errors.append(f"batch {bid}: mixed or missing family {fams}")
        if len(ids) > 1 and len(ids) > caps[family[ids[0]]]:
            errors.append(f"batch {bid}: capacity exceeded")
        kappas = [e["operation"] for e in members]
        if max(kappas) - min(kappas) > delta:
            errors.append(f"batch {bid}: order difference {max(kappas) - min(kappas)} > {delta}")
        if len({e["job"] for e in members}) != len(members):
            errors.append(f"batch {bid}: two operations of the same job")
        if len(mu) != 1 or len(starts) != 1 or len(ends) != 1:
            errors.append(f"batch {bid}: members not on one machine / not simultaneous")
            continue
        m = mu.pop()
        if any(ops[o][m] == 0 for o in ids):
            errors.append(f"batch {bid}: machine {m} not eligible for all members")
        if ends.pop() - starts.pop() != max(ops[o][m] for o in ids):
            errors.append(f"batch {bid}: duration is not the longest member time")

    tau = transport_matrix(instance)
    depot = depot_index(instance)
    for job in jobs:
        first = by_op[job[0]]
        if first["start"] < tau[depot][first["machine"]]:
            errors.append(f"op {job[0]} starts before arriving from the depot")
        for a, c in zip(job[:-1], job[1:]):
            if by_op[c]["start"] < by_op[a]["end"]:
                errors.append(f"precedence violated between op {a} and op {c}")
            elif by_op[c]["start"] < by_op[a]["end"] + tau[by_op[a]["machine"]][by_op[c]["machine"]]:
                errors.append(f"transport violated between op {a} and op {c}")

    per_machine = {}
    for bid, members in batches.items():
        e = members[0]
        per_machine.setdefault(e["machine"], []).append((e["start"], e["end"], bid))
    for m, items in per_machine.items():
        items.sort()
        for (s1, e1, b1), (s2, e2, b2) in zip(items, items[1:]):
            if s2 < e1:
                errors.append(f"machine {m}: batches {b1} and {b2} overlap")
    return errors
