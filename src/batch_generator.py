"""
Instance generator for the FJSP with parallel batching (see docs/batching_formulation.tex).

An instance extends the usual FJSP dict produced by src.parsedata.get_data (jobs,
operations, maximum, num_machines) with:
  family      : list[int], one entry per operation id. -1 = the operation belongs to no
                family and is always processed alone.
  capacities  : list[int], B_f for every family f (max operations per batch).
  delta       : int, maximum order difference allowed inside a batch (Delta = 2).

Families are built so that they respect the formulation:
  (F1) at most one operation per job in each family;
  "similar characteristics": all members share the same eligible machines M_f and have
      processing times drawn around a common family mean (same +-20% deviation as
      src.generator.CaseGenerator);
  order window: members are drawn from a window of order indices of width family_span.
      With family_span > delta, some members of a family are too far apart to be batched
      together, so the order difference constraint (F2)/(B2a) actually binds.
"""
import random

from src.transport import add_layout


def _proc_times_around(mean, machines, num_machines, proc_min, proc_max, dev):
    low = max(proc_min, round(mean * (1 - dev)))
    high = min(proc_max, round(mean * (1 + dev)))
    row = [0] * num_machines
    for m in machines:
        row[m] = random.randint(low, high)
    return row


def generate_batching_instance(
    n_jobs,
    n_machines,
    ops_per_job,
    max_processing=100,
    delta=2,
    family_span=3,
    family_ratio=0.6,
    family_size_range=(2, 4),
    capacity_range=(2, 3),
    proc_dev=0.2,
    transport_rho=None,
):
    """
    :param ops_per_job: list with the number of operations of every job
    :param family_ratio: target fraction of operations that belong to a family
    :param family_size_range: (min, max) number of operations in a family (bounded by n_jobs, F1)
    :param capacity_range: (min, max) batch capacity B_f
    :param transport_rho: None = no transport. Otherwise machines and depot get random
        positions with expected transport time ~ rho x mean processing time
        (src/transport.py, docs/transport_formulation.tex). Drawn last, so the rest of the
        instance is the same as without transport for the same random state.
    """
    proc_min = 1
    jobs = []
    o_id = 0
    for n_ops in ops_per_job:
        jobs.append(list(range(o_id, o_id + n_ops)))
        o_id += n_ops
    num_ops = o_id

    op_job = [0] * num_ops
    op_kappa = [0] * num_ops  # 1-based order index, as kappa(o) in the formulation
    for j, job in enumerate(jobs):
        for k, o in enumerate(job):
            op_job[o] = j
            op_kappa[o] = k + 1

    # ---- families -------------------------------------------------------------------
    family = [-1] * num_ops
    capacities = []
    target = int(round(family_ratio * num_ops))
    assigned = 0
    max_k = max(ops_per_job)
    attempts = 0
    while assigned < target and attempts < 50 * num_ops:
        attempts += 1
        k0 = random.randint(1, max(1, max_k - family_span))
        window = range(k0, k0 + family_span + 1)
        # one free operation per job inside the order window (F1)
        candidates_per_job = []
        for job in jobs:
            free = [o for o in job if family[o] == -1 and op_kappa[o] in window]
            if free:
                candidates_per_job.append(random.choice(free))
        lo, hi = family_size_range
        hi = min(hi, len(candidates_per_job))
        if hi < lo:
            continue
        size = random.randint(lo, hi)
        members = random.sample(candidates_per_job, size)
        f = len(capacities)
        for o in members:
            family[o] = f
        capacities.append(random.randint(*capacity_range))
        assigned += size

    # ---- eligible machines and processing times --------------------------------------
    operations = [None] * num_ops
    for f in range(len(capacities)):
        members = [o for o in range(num_ops) if family[o] == f]
        n_elig = random.randint(1, n_machines)
        machines = sorted(random.sample(range(n_machines), n_elig))
        mean = random.randint(proc_min, max_processing)
        for o in members:
            operations[o] = _proc_times_around(mean, machines, n_machines, proc_min, max_processing, proc_dev)
    for o in range(num_ops):
        if operations[o] is None:
            n_elig = random.randint(1, n_machines)
            machines = sorted(random.sample(range(n_machines), n_elig))
            mean = random.randint(proc_min, max_processing)
            operations[o] = _proc_times_around(mean, machines, n_machines, proc_min, max_processing, proc_dev)

    maximum = max(max(row) for row in operations)
    instance = {
        "jobs": jobs,
        "operations": operations,
        "maximum": maximum,
        "num_machines": n_machines,
        "family": family,
        "capacities": capacities,
        "delta": delta,
    }
    if transport_rho is not None:
        add_layout(instance, transport_rho)
    return instance


def generate_batching_instance_list(
    n_cases=1,
    range_jobs=(8, 10),
    range_machines=(5, 10),
    range_op_per_job=(5, 6),
    max_processing=100,
    **family_kwargs,
):
    """Same size ranges and defaults as src.generator.generate_instance_list."""
    instances = []
    for _ in range(n_cases):
        n_jobs = random.randint(*range_jobs)
        n_machines = random.randint(*range_machines)
        ops_per_job = [random.randint(*range_op_per_job) for _ in range(n_jobs)]
        instances.append(
            generate_batching_instance(n_jobs, n_machines, ops_per_job, max_processing, **family_kwargs)
        )
    return instances
