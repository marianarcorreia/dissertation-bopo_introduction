"""
FJSP with parallel batching on the operation-job-machine graph (src/env.py), in the two
representations of docs/batching_formulation.tex, plus a baseline to compare them with:

  FJSPBatchNodeEnv  ("ojmb_node", Representation A): a fourth node type, 'family', linked
                    to its operations and to its eligible machines.
  FJSPBatchEdgeEnv  ("ojmb_edge", Representation B): a new ('operation','batch','operation')
                    edge between operations that can be batched together.
  FJSPBatchBaseEnv  ("ojmb_base", baseline): the same batching problem and batch rule, but
                    the plain ojm graph - no family node, no batch edge, no family features.
                    Any gain of A or B over it comes from representing the batching.

All three keep the ojm action space - one ('machine','exec','job') edge per decision - so BOPO
(src/bopo.py) is reused unchanged and the two representations differ only in the graph.
When the selected job's current operation belongs to a family, the batch is completed
deterministically with other ready operations of the same family:
  - their job's current operation is in the same family and is eligible on the machine;
  - they are ready no later than the batch start (joining never delays the batch);
  - at most B_f operations, and max(kappa) - min(kappa) <= delta inside the batch;
  - candidates are added by earliest ready time.
Parallel batching: the batch runs from start to start + max_o p[o, machine].

Family ids are per-instance labels (family 3 of one instance has nothing in common with
family 3 of another), so they are not fed to the network as a number. Membership is
encoded structurally (family node / batch edges) and by numeric descriptors.

Instances must carry "family", "capacities" and "delta" (see src/batch_generator.py).
Every episode records the schedule in env.schedule, in the format checked by
src.batch_solver.check_schedule.
"""
import numpy as np
import torch
from src.env import FJSSPEnv, _dbg, normalize_columns, select_with_tiebreak, remove_operation_nodes

SENTINEL = 10000
EDGE_DIM = 5  # every attributed edge type must match GATv2Conv(edge_dim=5) in src/gat.py


def _empty_edges():
    return torch.zeros((2, 0), dtype=torch.long)


def _keep_edges(store, keep):
    store.edge_index = store.edge_index[:, keep]
    for key in ("edge_attr", "mask"):
        if key in store and store[key].shape[0] == keep.shape[0]:
            store[key] = store[key][keep]


class _FJSPBatchEnv(FJSSPEnv):
    BATCH_REP = None  # "node", "edge" or "base"

    # ------------------------------------------------------------------ instance / graph
    def generate_instance(self, instance):
        super().generate_instance(instance)
        n_ops = self.num_operations
        self.family = list(instance.get("family", [-1] * n_ops))
        self.capacities = list(instance.get("capacities", []))
        self.delta = int(instance.get("delta", 2))
        self.num_families = len(self.capacities)

        self.op_job = [0] * n_ops
        self.op_kappa = [0] * n_ops  # 0-based position in the job
        for j, job in enumerate(self.jobs):
            for k, o in enumerate(job):
                self.op_job[o], self.op_kappa[o] = j, k

        ops = self.operations
        self.op_elig = [frozenset(m for m in range(self.num_machines) if ops[o][m] > 0) for o in range(n_ops)]
        self.fam_members = [[o for o in range(n_ops) if self.family[o] == f] for f in range(self.num_families)]
        self.fam_machines = [
            [m for m in range(self.num_machines) if all(ops[o][m] > 0 for o in members)]
            for members in self.fam_members
        ]
        # p_bar[f][m]: longest member time on m = batch processing time upper bound
        self.fam_pbar_mach = [
            [max(ops[o][m] for o in members) if m in self.fam_machines[f] else 0 for m in range(self.num_machines)]
            for f, members in enumerate(self.fam_members)
        ]
        self.fam_pbar = [
            float(np.mean([self.fam_pbar_mach[f][m] for m in self.fam_machines[f]])) if self.fam_machines[f] else 0.0
            for f in range(self.num_families)
        ]

        self.data["operation"].oid = torch.arange(n_ops)  # original op id, kept through node removal
        if self.BATCH_REP == "base":  # plain ojm graph: the batching stays invisible to the network
            return

        # ---- family feature on operation nodes (static) ----
        extra = []
        for o in range(n_ops):
            f = self.family[o]
            in_fam = 1.0 if f >= 0 else 0.0
            kappa_norm = (self.op_kappa[o] + 1) / len(self.jobs[self.op_job[o]])
            row = [in_fam, kappa_norm]
            if self.BATCH_REP == "edge":  # no family node: batch-level parameters live on the op
                row += [float(self.capacities[f]) if f >= 0 else 1.0, self.fam_pbar[f] if f >= 0 else 0.0]
            extra.append(row)
        self.data["operation"].x = torch.cat([self.data["operation"].x, torch.tensor(extra)], dim=1)

        # ---- family feature on job nodes (dynamic, filled in calculate_next_state) ----
        # cols 4..6: current op in a family, its capacity B_f, compatible ready partners
        self.data["job"].x = torch.cat([self.data["job"].x, torch.zeros((self.num_jobs, 3))], dim=1)

        if self.BATCH_REP == "node":
            self._build_family_node()
        else:
            self._build_batch_edges()

    def _build_family_node(self):
        n_fam_nodes = max(self.num_families, 1)  # dummy node keeps the node type present
        self.data["family"].x = torch.zeros((n_fam_nodes, 6))
        loops = torch.arange(n_fam_nodes)
        self.data["family", "self", "family"].edge_index = torch.stack([loops, loops])

        of = [[o, f] for f, members in enumerate(self.fam_members) for o in members]
        of = torch.LongTensor(of).T if of else _empty_edges()
        self.data["operation", "in_family", "family"].edge_index = of
        self.data["family", "has", "operation"].edge_index = of.flip(0)

        fm, attrs = [], []
        for f in range(self.num_families):
            total = sum(self.fam_pbar_mach[f][m] for m in self.fam_machines[f]) or 1.0
            for m in self.fam_machines[f]:
                p = float(self.fam_pbar_mach[f][m])
                fm.append([f, m])
                attrs.append([p, p / total, 0.0, 0.0, 0.0])
        fm = torch.LongTensor(fm).T if fm else _empty_edges()
        attrs = torch.tensor(attrs, dtype=torch.float) if attrs else torch.zeros((0, EDGE_DIM))
        self.data["family", "batch_exec", "machine"].edge_index = fm
        self.data["family", "batch_exec", "machine"].edge_attr = attrs
        self.data["machine", "batch_exec", "family"].edge_index = fm.flip(0)
        self.data["machine", "batch_exec", "family"].edge_attr = attrs.clone()

    def _build_batch_edges(self):
        ops = self.operations
        elig = [{m for m in range(self.num_machines) if ops[o][m] > 0} for o in range(self.num_operations)]
        pairs, attrs = [], []
        for f, members in enumerate(self.fam_members):
            for a in members:
                for b in members:
                    if a == b or self.op_job[a] == self.op_job[b]:
                        continue
                    dk = abs(self.op_kappa[a] - self.op_kappa[b])
                    shared = elig[a] & elig[b]
                    if dk > self.delta or not shared:
                        continue
                    jaccard = len(shared) / len(elig[a] | elig[b])
                    p_pair = float(np.mean([max(ops[a][m], ops[b][m]) for m in shared]))
                    pairs.append([a, b])
                    attrs.append([dk / max(self.delta, 1), jaccard, p_pair, float(self.capacities[f]), 0.0])
        store = self.data["operation", "batch", "operation"]
        store.edge_index = torch.LongTensor(pairs).T if pairs else _empty_edges()
        store.edge_attr = torch.tensor(attrs, dtype=torch.float) if attrs else torch.zeros((0, EDGE_DIM))

    # ------------------------------------------------------------------ episode
    def reset(self, sel_index=None):
        self.schedule = []
        self.num_batches = 0
        self.job_done = None  # set once the instance is known (see calculate_next_state)
        self.fam_remaining = None  # unscheduled members per family, set in calculate_next_state
        return super().reset(sel_index)

    def _done(self, j):
        return self.job_done is not None and self.job_done[j]

    def _family_candidates(self):
        """Per family: the active jobs whose current operation belongs to it, as
        (ready time, job, operation, kappa) sorted by (ready time, job) - the order in which
        _batch_members adds partners. Built once per decision and shared by every action
        edge instead of rescanning all jobs for each edge."""
        cands = {}
        for j in range(self.num_jobs):
            if self._done(j):
                continue
            o = self.current_operations[j]
            f = self.family[o]
            if f >= 0:
                cands.setdefault(f, []).append((float(self.operations_ends[j]), j, o, self.op_kappa[o]))
        for lst in cands.values():
            lst.sort(key=lambda c: (c[0], c[1]))
        return cands

    def _batch_members(self, sel_job, mach, start, cands=None):
        """Jobs dispatched together when sel_job is dispatched on mach at start: sel_job plus
        ready (ready time <= start) same-family jobs eligible on mach, added by earliest ready
        time while the capacity B_f and the order difference delta allow."""
        o = self.current_operations[sel_job]
        f = self.family[o]
        members = [sel_job]
        if f < 0:
            return members
        if cands is None:
            cands = self._family_candidates()
        cap = self.capacities[f]
        k_min = k_max = self.op_kappa[o]
        for ready, j, o2, k in cands.get(f, ()):
            if len(members) >= cap:
                break
            if j == sel_job or ready > start or self.operations[o2][mach] <= 0:
                continue
            lo, hi = min(k_min, k), max(k_max, k)
            if hi - lo <= self.delta:
                members.append(j)
                k_min, k_max = lo, hi
        return members

    def _earliest_start(self, j):
        """Earliest time job j's current operation can start on any eligible machine."""
        return float(self.job_start_machines[j].min())

    def _ready_partners(self, j, cands=None, start=None):
        """Jobs that the batch rule (_batch_members) would add if job j were dispatched at
        its earliest start: same family, within delta positions, a shared eligible
        machine, and ready by then. Capped at B_f - 1."""
        o = self.current_operations[j]
        f = self.family[o]
        if f < 0:
            return 0
        if cands is None:
            cands = self._family_candidates()
        if start is None:
            start = self._earliest_start(j)
        count = 0
        for ready, j2, o2, k in cands.get(f, ()):
            if (j2 != j and ready <= start and abs(k - self.op_kappa[o]) <= self.delta
                    and self.op_elig[o] & self.op_elig[o2]):
                count += 1
        return min(count, self.capacities[f] - 1)

    def _refresh_jm_edges(self):
        """Action-edge features of FJSSPEnv._refresh_jm_edges, except that for the node and
        edge representations columns 0 and 3 hold the processing and completion time of the
        BATCH the action would dispatch (its longest member, see _batch_members), which is
        what the action actually costs. The baseline keeps the individual operation's time,
        so its graph stays free of batching information."""
        super()._refresh_jm_edges()
        if self.BATCH_REP == "base":
            return
        store = self.state['machine', 'exec', 'job']
        attr = store.edge_attr
        start, proc = self._batch_start_proc()
        attr[:, 0] = proc
        attr[:, 3] = start + proc
        self.state['job', 'exec', 'machine'].edge_attr = attr.clone()

    def _batch_start_proc(self):
        """Per (machine, job) action edge: earliest start and processing time of the batch
        the action would dispatch (the longest member, see _batch_members)."""
        edge_index = self.state['machine', 'exec', 'job'].edge_index
        machine_idx, job_idx = edge_index[0], edge_index[1]
        start = self.job_start_machines[job_idx, machine_idx].clone().float()
        proc = self.current_job_proc[job_idx, machine_idx].clone().float()
        cands = self._family_candidates()
        if not cands:
            return start, proc
        starts, machines, jobs = start.tolist(), machine_idx.tolist(), job_idx.tolist()
        for e, (m, j, s) in enumerate(zip(machines, jobs, starts)):
            # only edges whose job has a family with another candidate can form a batch
            f = self.family[self.current_operations[j]]
            if s >= SENTINEL or f < 0 or len(cands.get(f, ())) < 2:
                continue
            members = self._batch_members(j, m, s, cands)
            if len(members) > 1:
                proc[e] = max(self.operations[self.current_operations[j2]][m] for j2 in members)
        return start, proc

    def expert_action(self):
        """Warm-start teacher. For the node and edge representations the completion time
        used as criterion / tie-break is that of the BATCH the action would dispatch -
        exactly what their action edges show (see _refresh_jm_edges) - so the teacher can
        be imitated from the features they see. The baseline keeps the individual time,
        which is what its action edges show."""
        if self.BATCH_REP == "base":
            return super().expert_action()
        start, proc = self._batch_start_proc()
        completion = start + proc
        primary, secondary = (start, completion) if self.mask_option == 0 else (completion, start)
        return select_with_tiebreak(primary, secondary, self.state['machine', 'exec', 'job'].mask)

    def calculate_next_state(self):
        if self.job_done is None:
            self.job_done = [False] * self.num_jobs
            self.fam_remaining = [len(m) for m in self.fam_members]
        super().calculate_next_state()
        if self.BATCH_REP == "base":
            return

        cands = self._family_candidates()
        earliest = self.job_start_machines.min(dim=1).values.tolist()
        rows = []
        for j in range(self.num_jobs):
            if self.job_done[j]:
                rows.append((0.0, 0.0, 0.0))
                continue
            f = self.family[self.current_operations[j]]
            if f < 0:
                rows.append((0.0, 1.0, 0.0))
            else:
                rows.append((1.0, float(self.capacities[f]), float(self._ready_partners(j, cands, earliest[j]))))
        self.state["job"].x[:, 4:7] = torch.tensor(rows, dtype=self.state["job"].x.dtype)

        if self.BATCH_REP == "node":
            fx = self.state["family"].x
            for f in range(self.num_families):
                cap = self.capacities[f]
                # members that could share the family's earliest possible batch: current
                # operations whose job is ready by then (the rule of _batch_members)
                current = cands.get(f, [])
                t_f = min((earliest[j] for _, j, _, _ in current), default=0.0)
                ready = [k for r, _, _, k in current if r <= t_f]
                remaining = self.fam_remaining[f]
                spread = (max(ready) - min(ready)) if ready else 0
                fx[f] = torch.tensor([
                    float(cap),
                    self.fam_pbar[f],
                    len(self.fam_machines[f]) / self.num_machines,
                    len(ready) / cap,
                    spread / max(self.delta, 1),
                    remaining / len(self.fam_members[f]),
                ])

    def step(self, action):
        self.num_steps += 1
        edge = self.state['machine', 'exec', 'job'].edge_index[:, action]
        sel_job, sel_mach = int(edge[1]), int(edge[0])

        prev_ms = float(torch.max(self.state["machine"].x[:, 0]))
        start = max(float(self.state["machine"].x[sel_mach, 0]), float(self.operations_ends[sel_job]))
        members = self._batch_members(sel_job, sel_mach, start)
        proc = max(self.operations[self.current_operations[j]][sel_mach] for j in members)
        end = start + proc
        _dbg(2, f"  step #{self.num_steps} | sel_job={sel_job} | sel_mach={sel_mach} | batch_jobs={members} | [{start}, {end}]")

        # machine
        self.state["machine"].x[sel_mach, 0] = end
        self.job_start_machines[self.job_start_machines[:, sel_mach] < end, sel_mach] = end
        self.machines_occupations[sel_mach] += proc
        self.state["machine"].x[sel_mach, 1] = self.machines_occupations[sel_mach] / end

        batch_id = self.num_batches
        self.num_batches += 1
        scheduled_ops = []
        for j in members:
            o = self.current_operations[j]
            scheduled_ops.append(o)
            if self.family[o] >= 0:
                self.fam_remaining[self.family[o]] -= 1
            self.schedule.append({
                "job": j, "operation": self.op_kappa[o], "op_id": o, "family": self.family[o],
                "batch": batch_id, "machine": sel_mach, "start": start, "end": end,
            })
            self.operations_ends[j] = end
            self.job_start_machines[j, :] = SENTINEL
            self.current_job_proc[j, :] = 0

        # drop the action edges of every job in the batch
        mej = self.state['machine', 'exec', 'job']
        _keep_edges(mej, ~torch.isin(mej.edge_index[1], torch.tensor(members)))

        new_edges, new_feats = [], []
        for j in members:
            if self.current_operations[j] == self.jobs[j][-1]:
                self.job_done[j] = True
                self.state["job"].x[j, :] = 0
                self.state["job"].x[j, 0] = 1
                ll = self.state['job', 'listens', 'job']
                ll.edge_index = ll.edge_index[:, ll.edge_index[0, :] != j]
                continue
            self.current_operations[j] += 1
            oper = np.array(self.operations[self.current_operations[j]])
            feats, total_gap = [], 0
            for m in range(len(oper)):
                t = oper[m]
                if t != 0:
                    calcu = t + max(self.operations_ends[j] - self.state["machine"].x[m, 0], 0)
                    total_gap += calcu
                    new_edges.append([m, j])
                    feats.append([calcu, t / np.sum(oper), t + max(self.operations_ends[j], self.state["machine"].x[m, 0])])
                    self.job_start_machines[j, m] = max(self.operations_ends[j], self.state["machine"].x[m, 0])
                    self.current_job_proc[j, m] = t
            for row in feats:
                row.append(row[0] / total_gap)
                row.append(0)
            new_feats += feats
        if new_edges:
            mej.edge_index = torch.cat([mej.edge_index, torch.LongTensor(new_edges).T], dim=1)
            mej.edge_attr = torch.cat([mej.edge_attr, torch.tensor(np.array(new_feats, dtype=np.float32))], dim=0)

        reward = prev_ms - float(torch.max(self.state["machine"].x[:, 0]))

        if all(self.job_done):
            self.mk = round(float(torch.max(self.state["machine"].x[:, 0])), 2)
            _dbg(1, f"  episode DONE after {self.num_steps} steps | makespan={self.mk} | batches={self.num_batches}")
            return self.state, reward, True, {"current_machine": sel_mach}

        self._remove_operations(scheduled_ops)
        self.calculate_next_state()
        self.calculate_mask()

        done = False
        total_reward = reward
        if len(self.state['machine', 'exec', 'job'].mask) - sum(self.state['machine', 'exec', 'job'].mask) == 1:
            self.state, reward, done, _ = self.step(self.sample())
            total_reward += reward
        return self.state, total_reward, done, {"current_machine": sel_mach}

    def _remove_operations(self, op_ids):
        oid = self.state["operation"].oid
        nodes = torch.nonzero(torch.isin(oid, torch.tensor(op_ids))).flatten()

        def drop(edge_type, src=True, dst=True):
            store = self.state[edge_type]
            keep = torch.ones(store.edge_index.shape[1], dtype=torch.bool)
            if src:
                keep &= ~torch.isin(store.edge_index[0], nodes)
            if dst:
                keep &= ~torch.isin(store.edge_index[1], nodes)
            _keep_edges(store, keep)

        drop(('operation', 'belongs', 'job'), dst=False)
        drop(('operation', 'prec', 'operation'), src=False)
        drop(('operation', 'exec', 'machine'), dst=False)
        drop(('machine', 'exec', 'operation'), src=False)
        if self.BATCH_REP == "node":
            drop(('operation', 'in_family', 'family'), dst=False)
            drop(('family', 'has', 'operation'), src=False)
            finished = torch.tensor([f for f in range(self.num_families) if self.fam_remaining[f] == 0],
                                    dtype=torch.long)
            if finished.numel():
                for et, col in ((('family', 'batch_exec', 'machine'), 0), (('machine', 'batch_exec', 'family'), 1)):
                    store = self.state[et]
                    _keep_edges(store, ~torch.isin(store.edge_index[col], finished))
        elif self.BATCH_REP == "edge":
            drop(('operation', 'batch', 'operation'))
        self.state = remove_operation_nodes(self.state, nodes)

    def normalize_state(self, state):
        state = super().normalize_state(state)
        if self.BATCH_REP == "base":
            return state
        if self.BATCH_REP == "node":
            state["family"].x = normalize_columns(state["family"].x)
            edge_types = (('family', 'batch_exec', 'machine'), ('machine', 'batch_exec', 'family'))
        else:
            edge_types = (('operation', 'batch', 'operation'),)
        for et in edge_types:
            state[et].edge_attr = normalize_columns(state[et].edge_attr)
        return state


class FJSPBatchNodeEnv(_FJSPBatchEnv):
    """Representation A: family node type."""
    BATCH_REP = "node"


class FJSPBatchEdgeEnv(_FJSPBatchEnv):
    """Representation B: operation-operation batch edge type."""
    BATCH_REP = "edge"


class FJSPBatchBaseEnv(_FJSPBatchEnv):
    """Baseline: batching problem on the plain ojm graph (no batching information)."""
    BATCH_REP = "base"
