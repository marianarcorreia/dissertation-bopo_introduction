"""
Shop-floor layout and transport times for the FJSP with batching and transport
(docs/transport_formulation.tex).

An instance may carry
  coords : [[x, y], ...]  integer position of every machine, in time units
  depot  : [x, y]         load/unload station where every job starts
Transport between two locations takes their Manhattan distance (T1-T6 in the document:
unlimited transport capacity, starts when the operation ends, no return to the depot).
Instances without these keys have every location at (0, 0), i.e. no transport, so every
function here reduces to the transport-free problem.

Locations are indexed 0..m-1 for the machines and m (DEPOT) for the depot.
"""
import math
import random

import numpy as np


def depot_index(instance):
    return len(instance["operations"][0])


def has_transport(instance):
    return "coords" in instance


def locations(instance):
    """(m + 1) x 2 array: machine positions, then the depot."""
    m = depot_index(instance)
    if not has_transport(instance):
        return np.zeros((m + 1, 2), dtype=np.int64)
    return np.array(list(instance["coords"]) + [instance["depot"]], dtype=np.int64)


def transport_matrix(instance):
    """tau[a][b] = Manhattan distance between locations a and b, (m + 1) x (m + 1) ints."""
    pos = locations(instance)
    return np.abs(pos[:, None, :] - pos[None, :, :]).sum(axis=2)


def transfer_times(instance, tau=None):
    """Per operation o: (min, mean) transport time into o, over every pair (machine of the
    previous operation of the job, machine of o) - tau^min / tau^avg of the formulation. For
    the first operation the origin is the depot."""
    if tau is None:
        tau = transport_matrix(instance)
    ops = instance["operations"]
    depot = depot_index(instance)
    out = [(0.0, 0.0)] * len(ops)
    for job in instance["jobs"]:
        prev = [depot]
        for o in job:
            elig = [m for m, p in enumerate(ops[o]) if p > 0]
            block = tau[np.ix_(prev, elig)]
            out[o] = (float(block.min()), float(block.mean()))
            prev = elig
    return out


def job_lower_bound(instance, tau=None):
    """Longest job, with transport: shortest path through the layered graph of eligible
    machines (depot -> machine of op 1 -> ... -> machine of the last op), weighting each node
    by its processing time and each arc by its transport time. Valid with batching too, since
    a batch lasts at least as long as each member. Without transport it equals the sum of the
    fastest processing times."""
    if tau is None:
        tau = transport_matrix(instance)
    ops = instance["operations"]
    depot = depot_index(instance)
    best = 0.0
    for job in instance["jobs"]:
        prev, dist = [depot], np.zeros(1)
        for o in job:
            elig = [m for m, p in enumerate(ops[o]) if p > 0]
            proc = np.array([ops[o][m] for m in elig], dtype=float)
            dist = (dist[:, None] + tau[np.ix_(prev, elig)]).min(axis=0) + proc
            prev = elig
        best = max(best, float(dist.min()))
    return best


def floor_side(rho, mean_proc):
    """Side L of the square floor {0..L}^2 that makes the expected transport time between two
    uniform positions (2L/3) a fraction rho of the mean processing time."""
    return max(1, math.ceil(1.5 * rho * mean_proc))


def add_layout(instance, rho, rng=random):
    """Adds uniform random machine and depot positions to `instance` (in place) so that the
    expected transport time is about rho x the instance's mean processing time."""
    procs = [p for row in instance["operations"] for p in row if p > 0]
    side = floor_side(rho, sum(procs) / len(procs))
    m = depot_index(instance)
    instance["coords"] = [[rng.randint(0, side), rng.randint(0, side)] for _ in range(m)]
    instance["depot"] = [rng.randint(0, side), rng.randint(0, side)]
    instance["transport_rho"] = rho
    return instance
