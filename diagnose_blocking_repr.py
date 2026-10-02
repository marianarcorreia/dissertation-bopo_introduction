"""Diagnostic of the blocking representations (src/env_blocking.py): does each graph describe
the real blocking problem, and do the buffer-aware ones carry the same information?

    python diagnose_blocking_repr.py
    python diagnose_blocking_repr.py --n-instances 5 --policies expert random --no-model

For ojmb (buffer nodes), ojmd (dummy machines), ojmf (machine/operation features) and ojm_blk
(buffer-blind control), on instances of val/test_dataset_blocking.json:
  1. same trajectory - every env is driven with the same action sequence (expert ECT rule or a
     seeded random policy); the action edges and masks must be identical across representations
  2. decode - at every decision the buffer state (FIFO content of every input/output buffer,
     capacities, blocked machines, waiting times) is read back from the RAW graph and compared
     with the env's true state
  3. equivalence - the decoded states of ojmb/ojmd/ojmf are identical, raw and normalised
  4. realism - invariants of the real problem at every step (capacities, blocking only with a
     full output buffer, parts in the right place) and, at the end, the schedule checked by
     src/metrics (blocking_constraints) and compared with the CP-SAT reference
  5. normalised state - finite and within [-1, 1]
  6. model - a GAT policy per representation plays one greedy episode (src.metrics.evaluate)
Writes results/blocking/diagnostics/blocking_repr_diagnostic.json.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import torch

from src.env_blocking import HELD, INBUF, OUTBUF, RUN
from src.metrics.blocking import blocking_constraints
from src.metrics.references import load_dataset
from src.metrics.schedule import check_schedule
from src.train import _resolve_representation_modules

BUFFER_REPRS = ("ojmb", "ojmd", "ojmf")
ALL_REPRS = BUFFER_REPRS + ("ojm_blk",)
TOL = 1e-4


# ── decoders: RAW graph -> buffer state ─────────────────────────────────────────

def _empty_state(M):
    return {"in_cap": [None] * M, "out_cap": [None] * M, "in": [[] for _ in range(M)],
            "out": [[] for _ in range(M)], "blocked": [False] * M, "wait": {}}


def _fill_parts(st, M, op_ref, pairs):
    """pairs: (op index, machine, is_output, priority, waiting time) -> FIFO lists."""
    per_buf = defaultdict(list)
    for i, m, is_out, prio, wait in pairs:
        o = int(op_ref[i])
        per_buf[(m, is_out)].append((-prio, o))
        st["wait"][o] = round(float(wait), 4)
    for (m, is_out), items in per_buf.items():
        st["out" if is_out else "in"][m] = [o for _, o in sorted(items)]


def decode_ojmb(s):
    M = s["machine"].x.shape[0]
    st = _empty_state(M)
    bx = s["buffer"].x
    buf_machine = {int(b): int(m) for b, m in s["buffer", "of", "machine"].edge_index.T.tolist()}
    for b, m in buf_machine.items():
        is_out = bx[b, 3] > 0.5
        st["out_cap" if is_out else "in_cap"][m] = int(round(float(bx[b, 0])))
        if is_out:
            st["blocked"][m] = bool(bx[b, 4] > 0.5)
    ei, ea = s["operation", "in", "buffer"].edge_index, s["operation", "in", "buffer"].edge_attr
    _fill_parts(st, M, s["operation"].op_ref,
                [(int(i), buf_machine[int(b)], bool(a[4] > 0.5), float(a[1]), float(a[2]))
                 for (i, b), a in zip(ei.T.tolist(), ea)])
    return st


def decode_ojmd(s):
    x = s["machine"].x
    M = int((x[:, 3] == 0).sum())
    st = _empty_state(M)
    buf_machine = {int(d): int(m) for d, m in s["machine", "buffer_of", "machine"].edge_index.T.tolist()}
    for d, m in buf_machine.items():
        is_out = x[d, 7] > 0.5
        st["out_cap" if is_out else "in_cap"][m] = int(round(float(x[d, 4])))
        if is_out:
            st["blocked"][m] = bool(x[d, 8] > 0.5)
    ei, ea = s["operation", "in", "machine"].edge_index, s["operation", "in", "machine"].edge_attr
    _fill_parts(st, M, s["operation"].op_ref,
                [(int(i), buf_machine[int(d)], bool(a[4] > 0.5), float(a[1]), float(a[2]))
                 for (i, d), a in zip(ei.T.tolist(), ea)])
    return st


def decode_ojmf(s):
    x, ox = s["machine"].x, s["operation"].x
    M = x.shape[0]
    st = _empty_state(M)
    for m in range(M):
        st["in_cap"][m] = int(round(float(x[m, 3])))
        st["out_cap"][m] = int(round(float(x[m, 8])))
        st["blocked"][m] = bool(x[m, 13] > 0.5)
    # the machine of a buffered operation = its assigned exec edge (column 3 = 1)
    ei, ea = s["operation", "exec", "machine"].edge_index, s["operation", "exec", "machine"].edge_attr
    assigned = defaultdict(list)
    for (i, m), a in zip(ei.T.tolist(), ea):
        if a[3] > 0.5:
            assigned[int(i)].append(int(m))
    pairs = []
    for i in range(ox.shape[0]):
        if ox[i, 2] > 0.5 or ox[i, 3] > 0.5:
            ms = assigned.get(i, [])
            if len(ms) != 1:
                raise AssertionError(f"buffered operation row {i} has {len(ms)} assigned exec edges")
            pairs.append((i, ms[0], bool(ox[i, 3] > 0.5), float(ox[i, 4]), float(ox[i, 5])))
    _fill_parts(st, M, s["operation"].op_ref, pairs)
    return st


DECODERS = {"ojmb": decode_ojmb, "ojmd": decode_ojmd, "ojmf": decode_ojmf}


def true_state(env):
    M = env.num_machines
    st = _empty_state(M)
    st["in_cap"], st["out_cap"] = [env.in_cap] * M, [env.out_cap] * M
    st["in"] = [list(q) for q in env.in_queue]
    st["out"] = [list(q) for q in env.out_queue]
    st["blocked"] = [h is not None for h in env.held]
    st["wait"] = {o: round(env.t - env.entry[o], 4) for q in env.in_queue + env.out_queue for o in q}
    return st


def diff_states(a, b):
    return sorted(k for k in a if a[k] != b[k])


# ── realism invariants on the env's true state ──────────────────────────────────

def invariant_violations(env):
    v = []
    M = env.num_machines
    placed = Counter()
    for m in range(M):
        if len(env.in_queue[m]) > max(env.in_cap, 0) and env.in_cap > 0:
            v.append(f"input buffer of m{m} over capacity")
        if len(env.out_queue[m]) > env.out_cap:
            v.append(f"output buffer of m{m} over capacity")
        if env.held[m] is not None:
            if len(env.out_queue[m]) < env.out_cap:
                v.append(f"m{m} blocked with a free output slot")
            if env.running[m] is not None:
                v.append(f"m{m} blocked and processing at the same time")
        for o in env.in_queue[m]:
            placed[o] += 1
            if env.status[o] != INBUF or env.assigned[o] != m:
                v.append(f"op {o} in input buffer of m{m} with status {env.status[o]} / machine {env.assigned[o]}")
        for o in env.out_queue[m]:
            placed[o] += 1
            if env.status[o] != OUTBUF or env.assigned[o] != m:
                v.append(f"op {o} in output buffer of m{m} not processed there")
            if env.end[o] is None or env.end[o] > env.t + TOL:
                v.append(f"op {o} in output buffer of m{m} before finishing")
        for o in (env.running[m], env.held[m]):
            if o is not None:
                placed[o] += 1
                if env.assigned[o] != m:
                    v.append(f"op {o} on m{m} but assigned to m{env.assigned[o]}")
        o = env.running[m]
        if o is not None and not (env.start[o] <= env.t + TOL and env.t <= env.end[o] + TOL):
            v.append(f"op {o} running on m{m} outside [start, end]")
    for o, s in enumerate(env.status):
        if s in (INBUF, RUN, HELD, OUTBUF) and placed[o] != 1:
            v.append(f"op {o} (status {s}) found {placed[o]} times in buffers/machines")
    for q in env.in_queue + env.out_queue:
        for o in q:
            if env.t - env.entry[o] < -TOL:
                v.append(f"op {o} negative waiting time")
    return v


# ── normalised state ────────────────────────────────────────────────────────────

def normalized_problems(env):
    s = env.normalize_state(env.state)
    bad = []
    for store in s.stores:
        for key in ("x", "edge_attr"):
            t = store.get(key) if hasattr(store, "get") else getattr(store, key, None)
            if t is None or not torch.is_tensor(t) or t.numel() == 0:
                continue
            if not torch.isfinite(t).all():
                bad.append(f"{store._key} {key}: non-finite")
            elif t.min() < -1 - TOL or t.max() > 1 + TOL:
                bad.append(f"{store._key} {key}: outside [-1, 1] ({t.min():.3f}, {t.max():.3f})")
    return s, bad


def normalized_buffer_view(rep, s, M):
    """(machine -> in occupancy, out occupancy, blocked) and (op -> priority, waiting) after
    normalisation, to compare the numbers the network actually sees."""
    mach, ops = {}, {}
    op_ref = s["operation"].op_ref
    if rep == "ojmf":
        x, ox = s["machine"].x, s["operation"].x
        for m in range(M):
            mach[m] = (float(x[m, 4]), float(x[m, 9]), float(x[m, 13]))
        for i in range(ox.shape[0]):
            if ox[i, 2] > 0 or ox[i, 3] > 0:
                ops[int(op_ref[i])] = (float(ox[i, 4]), float(ox[i, 5]))
        return mach, ops
    if rep == "ojmb":
        bx, link = s["buffer"].x, s["buffer", "of", "machine"].edge_index
        occ, blocked_col, out_col, part = bx[:, 1], bx[:, 4], bx[:, 3], ("operation", "in", "buffer")
    else:
        bx, link = s["machine"].x, s["machine", "buffer_of", "machine"].edge_index
        occ, blocked_col, out_col, part = bx[:, 5], bx[:, 8], bx[:, 7], ("operation", "in", "machine")
    tmp = defaultdict(dict)
    for b, m in link.T.tolist():
        is_out = out_col[b] > 0
        tmp[m]["out" if is_out else "in"] = float(occ[b])
        if is_out:
            tmp[m]["blocked"] = float(blocked_col[b])
    for m, d in tmp.items():
        mach[m] = (d["in"], d["out"], d["blocked"])
    for (i, _), a in zip(s[part].edge_index.T.tolist(), s[part].edge_attr):
        ops[int(op_ref[i])] = (float(a[1]), float(a[2]))
    return mach, ops


def close(a, b):
    if a.keys() != b.keys():
        return False
    return all(abs(x - y) <= 1e-4 for k in a for x, y in zip(a[k], b[k]))


# ── one instance, one policy ────────────────────────────────────────────────────

def run_instance(instance, policy, seed):
    envs = {}
    for rep in ALL_REPRS:
        _, EnvClass, _ = _resolve_representation_modules(rep)
        envs[rep] = EnvClass([instance], 1, 100)
        envs[rep].reset(sel_index=0)
    rng = random.Random(seed)
    res = {rep: Counter() for rep in ALL_REPRS}
    examples = defaultdict(list)
    steps = 0
    while True:
        ref = envs["ojm_blk"]
        ei = ref.state["machine", "exec", "job"].edge_index
        mask = ref.state["machine", "exec", "job"].mask
        truth = true_state(ref)
        decoded, norm_views = {}, {}
        for rep, env in envs.items():
            r = res[rep]
            r["decisions"] += 1
            e = env.state["machine", "exec", "job"]
            if not torch.equal(e.edge_index, ei) or not torch.equal(e.mask, mask):
                r["action_mismatch"] += 1
            for msg in invariant_violations(env):
                r["invariant_violations"] += 1
                if len(examples[f"{rep}/invariant"]) < 5:
                    examples[f"{rep}/invariant"].append(f"t={env.t}: {msg}")
            s_norm, bad = normalized_problems(env)
            if bad:
                r["normalization_problems"] += 1
                if len(examples[f"{rep}/normalization"]) < 5:
                    examples[f"{rep}/normalization"].extend(bad[:2])
            if rep in DECODERS:
                d = DECODERS[rep](env.state)
                decoded[rep] = d
                keys = diff_states(d, true_state(env))
                if keys:
                    r["decode_mismatch"] += 1
                    if len(examples[f"{rep}/decode"]) < 5:
                        examples[f"{rep}/decode"].append(f"t={env.t}: differs in {keys}")
                norm_views[rep] = normalized_buffer_view(rep, s_norm, env.num_machines)
            if truth["in"] != true_state(env)["in"] or truth["out"] != true_state(env)["out"]:
                r["trajectory_divergence"] += 1
        for rep in BUFFER_REPRS[1:]:
            if diff_states(decoded[rep], decoded["ojmb"]):
                res[rep]["raw_not_equal_to_ojmb"] += 1
            m_a, o_a = norm_views[rep]
            m_b, o_b = norm_views["ojmb"]
            if not (close(m_a, m_b) and close(o_a, o_b)):
                res[rep]["normalized_not_equal_to_ojmb"] += 1
                if len(examples[f"{rep}/normalized_vs_ojmb"]) < 3:
                    examples[f"{rep}/normalized_vs_ojmb"].append(
                        f"t={ref.t}: machines {m_a} vs {m_b} | ops {o_a} vs {o_b}")
        if policy == "expert":
            action = ref.expert_action()
        else:
            action = rng.choice([i for i in range(len(mask)) if not mask[i]])
        dones = set()
        for rep, env in envs.items():
            _, _, done, _ = env.step(action)
            dones.add(done)
        steps += 1
        if len(dones) != 1:
            for r in res.values():
                r["trajectory_divergence"] += 1
            break
        if dones == {True}:
            break
    out = {}
    for rep, env in envs.items():
        chk = check_schedule(instance, env.schedule, blocking_constraints(env.in_cap, env.out_cap))
        out[rep] = dict(res[rep])
        out[rep].update({
            "makespan": env.mk, "reference": instance["score"],
            "gap_vs_cpsat": env.mk / instance["score"] - 1.0,
            "below_cpsat_reference": env.mk < instance["score"] - TOL,
            "feasible": chk["feasible"],
            "violations": {k: v for k, v in chk["violations"].items() if v},
            "swaps": env.num_swaps, "blocked_completions": env.num_blocked,
            "max_occupancy": dict(env.max_occupancy),
        })
    return out, dict(examples)


def graph_sizes(instance):
    rows = {}
    for rep in ALL_REPRS:
        _, EnvClass, _ = _resolve_representation_modules(rep)
        s = EnvClass([instance], 1, 100).reset(sel_index=0)
        rows[rep] = {
            "node_types": {t: [int(s[t].num_nodes), int(s[t].x.shape[1])] for t in s.node_types},
            "edge_types": {"/".join(t): int(s[t].edge_index.shape[1]) for t in s.edge_types},
        }
    return rows


def exec_edge_normalization_note(instance):
    """Pre-existing issue shared by every representation: FJSSPEnv.normalize_state min-maxes the
    op->machine exec edge with ONE scalar over all 5 columns, so the 0/1 'assigned' flag is
    scaled together with processing times."""
    _, EnvClass, _ = _resolve_representation_modules("ojmf")
    env = EnvClass([instance], 1, 100)
    env.reset(sel_index=0)
    for _ in range(6):
        _, _, done, _ = env.step(env.expert_action())
        if done:
            break
    raw = env.state["operation", "exec", "machine"].edge_attr
    norm = env.normalize_state(env.state)["operation", "exec", "machine"].edge_attr
    flag = raw[:, 3] > 0.5
    return {
        "assigned_flag_raw": [0.0, 1.0],
        "assigned_flag_normalized": [round(float(norm[~flag, 3].mean()), 4) if (~flag).any() else None,
                                     round(float(norm[flag, 3].mean()), 4) if flag.any() else None],
        "raw_edge_attr_max": round(float(raw.max()), 2),
    }


def model_smoke(instances, num_layers):
    from src.metrics.evaluate import evaluate_agent
    rows = {}
    for rep in ALL_REPRS:
        torch.manual_seed(0)
        random.seed(0)
        _, EnvClass, BOPOClass = _resolve_representation_modules(rep)
        env = EnvClass(instances, 1, 100)
        agent = BOPOClass(0.001, env, env.reset().metadata(), 128, num_layers, 3, gnn_type="gat",
                          jm_design=env.jm_design)
        results, _ = evaluate_agent(agent, env, instances, rep)
        # after the first forward pass: the Linear(-1, ...) layers are lazy until then
        n_params = sum(p.numel() for p in agent.policy.parameters())
        rows[rep] = {"params": int(n_params),
                     "feasible": all(r["feasible"] for r in results),
                     "mean_relative_error_untrained": sum(r["relative_error"] for r in results) / len(results)}
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="val/test_dataset_blocking.json")
    p.add_argument("--n-instances", type=int, default=20)
    p.add_argument("--policies", nargs="+", default=["expert", "random"], choices=["expert", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--no-model", action="store_true", help="skip the untrained-policy smoke test")
    p.add_argument("--out", default="results/blocking/diagnostics/blocking_repr_diagnostic.json")
    args = p.parse_args()

    instances = load_dataset(args.dataset)[:args.n_instances]
    print(f"[DIAG] {len(instances)} instance(s) from {args.dataset} | policies={args.policies}")
    per_run, totals, examples = [], {rep: Counter() for rep in ALL_REPRS}, defaultdict(list)
    for policy in args.policies:
        for k, inst in enumerate(instances):
            out, ex = run_instance(inst, policy, args.seed + k)
            per_run.append({"instance": inst["name"], "policy": policy, "results": out})
            for key, msgs in ex.items():
                examples[key].extend(msgs[:5 - len(examples[key])])
            for rep, r in out.items():
                t = totals[rep]
                for key in ("decisions", "action_mismatch", "trajectory_divergence", "invariant_violations",
                            "decode_mismatch", "raw_not_equal_to_ojmb", "normalized_not_equal_to_ojmb",
                            "normalization_problems", "swaps", "blocked_completions"):
                    t[key] += int(r.get(key, 0))
                t["episodes"] += 1
                t["infeasible"] += int(not r["feasible"])
                t["below_cpsat"] += int(r["below_cpsat_reference"])
                t[f"gap_sum_{policy}"] += r["gap_vs_cpsat"]
                t[f"episodes_{policy}"] += 1
            print(f"[DIAG] {policy:6s} {inst['name']}: " + " | ".join(
                f"{rep} mk={out[rep]['makespan']:.0f}{'' if out[rep]['feasible'] else ' INFEASIBLE'}"
                for rep in ALL_REPRS))

    summary = {}
    for rep, t in totals.items():
        s = {k: v for k, v in t.items() if not k.startswith(("gap_sum_", "episodes_"))}
        for policy in args.policies:
            if t[f"episodes_{policy}"]:
                s[f"mean_gap_vs_cpsat_{policy}"] = t[f"gap_sum_{policy}"] / t[f"episodes_{policy}"]
        s["decodable"] = rep in DECODERS
        summary[rep] = s

    report = {"dataset": args.dataset, "n_instances": len(instances), "policies": args.policies,
              "summary": summary, "graph_sizes_first_instance": graph_sizes(instances[0]),
              "exec_edge_normalization": exec_edge_normalization_note(instances[0]),
              "examples": dict(examples), "runs": per_run}
    if not args.no_model:
        print("[DIAG] model smoke test (untrained GAT policy, greedy)...")
        report["model_smoke"] = model_smoke(instances[:2], args.num_layers)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    cols = ["episodes", "decisions", "action_mismatch", "trajectory_divergence", "invariant_violations",
            "decode_mismatch", "raw_not_equal_to_ojmb", "normalized_not_equal_to_ojmb",
            "normalization_problems", "infeasible", "below_cpsat"]
    print("\n" + "=" * 100)
    print(f"{'check':32s}" + "".join(f"{rep:>14s}" for rep in ALL_REPRS))
    for c in cols:
        print(f"{c:32s}" + "".join(
            f"{('n/a' if c in ('decode_mismatch', 'raw_not_equal_to_ojmb', 'normalized_not_equal_to_ojmb') and (rep not in DECODERS or (rep == 'ojmb' and c != 'decode_mismatch')) else summary[rep].get(c, 0))!s:>14s}"
            for rep in ALL_REPRS))
    for policy in args.policies:
        c = f"mean_gap_vs_cpsat_{policy}"
        print(f"{c:32s}" + "".join(f"{summary[rep][c]:>14.4f}" for rep in ALL_REPRS))
    print(f"{'swaps / blocked completions':32s}" + "".join(
        f"{str(summary[rep]['swaps']) + ' / ' + str(summary[rep]['blocked_completions']):>14s}" for rep in ALL_REPRS))
    print("-" * 100)
    for rep, g in report["graph_sizes_first_instance"].items():
        print(f"{rep:8s} nodes {g['node_types']} | edge types {len(g['edge_types'])}")
    print(f"exec edge normalisation (all reprs): {report['exec_edge_normalization']}")
    if "model_smoke" in report:
        for rep, m in report["model_smoke"].items():
            print(f"model {rep:8s} params={m['params']} feasible={m['feasible']} "
                  f"untrained_gap={m['mean_relative_error_untrained']:.4f}")
    for key, msgs in examples.items():
        print(f"example {key}: {msgs[:2]}")
    print(f"[DIAG] report -> {args.out}")


if __name__ == "__main__":
    main()
