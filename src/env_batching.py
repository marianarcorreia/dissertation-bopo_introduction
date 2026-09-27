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

Transport (docs/transport_formulation.tex). When the instance has a shop-floor layout
("coords", "depot", see src/transport.py), a job must travel from the machine of its last
operation (the depot at the start) to the machine of the next one, which takes the Manhattan
distance tau. This is part of the problem, so every env applies it: a job's arrival at machine
mu is a_{j,mu} = end of its last operation + tau[location][mu], an operation starts no earlier
than its arrival, a partner joins a batch only if it has arrived by the batch start, and after
a batch its members are located at its machine. Without a layout tau = 0 and nothing changes.
How the transport is shown to the network is a second, independent choice, TRANSPORT_REP:
  None    the three batching-only envs above (unchanged; batching features keep their
          transport-free definition)
  "none"  T-0: no transport information in the graph (the start times on the action edges
          still include it)
  "feat"  T-F: machine and job-location coordinates, incoming transport on the operations,
          tau on the machine 'listens' edges and a 6th action-edge column tau[location][mu]
  "edge"  T-E: relations ('machine','transport','machine'), ('job','at','machine') +
          ('machine','hosts','job'), ('job','reach','machine') and
          ('operation','transfer','operation'), with transport times as attributes
The nine combinations are registered below as "ojmb_<node|edge|base>_<t0|tf|te>". They use
EDGE_DIM = 6 (every attributed edge type is zero-padded to it in normalize_state).
With transport, the batching features that assumed one ready time per job (pi_j on the job,
the fill level and spread of the family node) are evaluated at the machine where the job
(or the family) can start earliest.
"""
import numpy as np
import torch
from src.env import FJSSPEnv, _dbg, normalize_columns, select_with_tiebreak, remove_operation_nodes
from src.transport import locations, transfer_times, transport_matrix

SENTINEL = 10000
EDGE_DIM = 5  # attributed edge width of the batching-only envs (GATv2Conv(edge_dim=5) in src/gat.py)
TRANSPORT_EDGE_DIM = 6  # the transport envs add one action-edge column


def _empty_edges():
    return torch.zeros((2, 0), dtype=torch.long)


def _keep_edges(store, keep):
    store.edge_index = store.edge_index[:, keep]
    for key in ("edge_attr", "mask"):
        if key in store and store[key].shape[0] == keep.shape[0]:
            store[key] = store[key][keep]


def _pad(attr, width):
    if attr.shape[1] >= width:
        return attr
    return torch.cat([attr, attr.new_zeros((attr.shape[0], width - attr.shape[1]))], dim=1)


class _FJSPBatchEnv(FJSSPEnv):
    BATCH_REP = None  # "node", "edge" or "base"
    TRANSPORT_REP = None  # None (batching-only env), "none", "feat" or "edge"
    EDGE_DIM = EDGE_DIM  # read by src/bopo.py:BOPO to size the GNN's edge projections

    # ------------------------------------------------------------------ instance / graph
    def generate_instance(self, instance):
        self._generate_batching(instance)
        self.tau = transport_matrix(instance)
        self.depot = self.num_machines
        self.has_transport = bool(self.tau.any())
        if self.TRANSPORT_REP == "feat":
            self._build_transport_features(instance)
        elif self.TRANSPORT_REP == "edge":
            self._build_transport_edges(instance)

    def _build_transport_features(self, instance):
        """T-F: coordinates on machine and job nodes, incoming transport on operation nodes,
        tau on the machine 'listens' edges. Coordinates use one scale W for both axes and for
        jobs and machines (see normalize_state); the job columns are filled per decision."""
        pos = locations(instance).astype(np.float64)
        pos -= pos.min(axis=0)
        self.pos_scaled = torch.tensor(pos / max(pos.max(), 1.0), dtype=torch.float)
        d = self.data
        self.mach_coord_cols = [d["machine"].x.shape[1], d["machine"].x.shape[1] + 1]
        d["machine"].x = torch.cat([d["machine"].x, self.pos_scaled[:self.num_machines]], dim=1)
        self.job_coord_cols = [d["job"].x.shape[1], d["job"].x.shape[1] + 1]
        d["job"].x = torch.cat([d["job"].x, torch.zeros((self.num_jobs, 2))], dim=1)
        self.op_transfer_cols = [d["operation"].x.shape[1], d["operation"].x.shape[1] + 1]
        d["operation"].x = torch.cat([d["operation"].x, torch.tensor(transfer_times(instance, self.tau))], dim=1)
        listens = d["machine", "listens", "machine"]
        src, dst = listens.edge_index
        listens.edge_attr = torch.tensor(self.tau[src.numpy(), dst.numpy()], dtype=torch.float).unsqueeze(1)

    def _build_transport_edges(self, instance):
        """T-E: new relations whose attributes are transport times (no new node feature)."""
        d = self.data
        transfer = transfer_times(instance, self.tau)
        pairs, tf_edges, tf_attr = set(), [], []
        for job in self.jobs:
            for a, c in zip(job[:-1], job[1:]):
                for mu in self.op_elig[a]:
                    for nu in self.op_elig[c]:
                        if mu != nu:
                            pairs.update(((mu, nu), (nu, mu)))
                tf_edges.append([c, a])  # successor -> predecessor, like 'prec'
                tf_attr.append(list(transfer[c]))
        pairs = sorted(pairs)
        store = d["machine", "transport", "machine"]
        store.edge_index = torch.LongTensor(pairs).T if pairs else _empty_edges()
        store.edge_attr = torch.tensor([[float(self.tau[a][b])] for a, b in pairs], dtype=torch.float).reshape(-1, 1)
        store = d["operation", "transfer", "operation"]
        store.edge_index = torch.LongTensor(tf_edges).T if tf_edges else _empty_edges()
        store.edge_attr = torch.tensor(tf_attr, dtype=torch.float).reshape(-1, 2)
        d["job", "at", "machine"].edge_index = _empty_edges()  # every job starts at the depot
        d["machine", "hosts", "job"].edge_index = _empty_edges()
        d["job", "reach", "machine"].edge_index = _empty_edges()  # filled with the action edges
        d["job", "reach", "machine"].edge_attr = torch.zeros((0, 1))

    def _generate_batching(self, instance):
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

    def _arrival(self, j, mach):
        """Time job j arrives at machine mach: end of its last operation plus the transport
        from where it is (the depot before its first operation)."""
        return float(self.operations_ends[j]) + float(self.tau[self.job_loc[j]][mach])

    def _start_machine(self, j):
        """mu*_j: the machine where job j's current operation can start earliest (ties: earliest
        completion, then lowest index), with that start."""
        start = self.job_start_machines[j].tolist()
        proc = self.current_job_proc[j].tolist()
        mach = min(range(self.num_machines), key=lambda m: (start[m], start[m] + proc[m], m))
        return mach, start[mach]

    def _batch_members(self, sel_job, mach, start, cands=None):
        """Jobs dispatched together when sel_job is dispatched on mach at start: sel_job plus
        same-family jobs eligible on mach that are ready there by start (arrived, with
        transport), added by earliest ready/arrival time while the capacity B_f and the order
        difference delta allow."""
        o = self.current_operations[sel_job]
        f = self.family[o]
        members = [sel_job]
        if f < 0:
            return members
        if cands is None:
            cands = self._family_candidates()
        cap = self.capacities[f]
        k_min = k_max = self.op_kappa[o]
        cand = cands.get(f, ())
        if self.has_transport:  # readiness depends on the machine: order by arrival at mach
            cand = sorted(((self._arrival(j, mach), j, o2, k) for _, j, o2, k in cand),
                          key=lambda c: (c[0], c[1]))
        for ready, j, o2, k in cand:
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
        machine, and ready by then. Capped at B_f - 1.
        With transport, readiness depends on the machine, so this is evaluated exactly as the
        batch rule would apply it at mu*_j (see _start_machine): partners eligible on mu*_j
        that have arrived there by j's start on it."""
        o = self.current_operations[j]
        f = self.family[o]
        if f < 0:
            return 0
        if cands is None:
            cands = self._family_candidates()
        if self.has_transport:
            mach, start = self._start_machine(j)
            count = sum(1 for _, j2, o2, k in cands.get(f, ())
                        if j2 != j and abs(k - self.op_kappa[o]) <= self.delta
                        and self.operations[o2][mach] > 0 and self._arrival(j2, mach) <= start)
            return min(count, self.capacities[f] - 1)
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

    def _init_transport(self):
        """Every job starts at the depot, so its first operation can start on machine m only
        after the transport tau[depot][m]."""
        self.job_loc = [self.depot] * self.num_jobs
        if self.has_transport:
            from_depot = torch.tensor(self.tau[self.depot][:self.num_machines], dtype=self.job_start_machines.dtype)
            eligible = self.job_start_machines < SENTINEL
            self.job_start_machines = torch.where(
                eligible, torch.maximum(self.job_start_machines, from_depot.expand_as(self.job_start_machines)),
                self.job_start_machines)

    def _update_transport_state(self):
        """Dynamic part of the transport representations (after every decision)."""
        if self.TRANSPORT_REP == "feat":
            active = [j for j in range(self.num_jobs) if not self.job_done[j]]
            if not active:
                return
            jx = self.state["job"].x
            jx[active, self.job_coord_cols[0]:self.job_coord_cols[1] + 1] = self.pos_scaled[[self.job_loc[j] for j in active]]
            # the current operation's incoming transport is now exact: from where the job is
            oid = self.state["operation"].oid
            node_of = torch.full((self.num_operations,), -1, dtype=torch.long)
            node_of[oid] = torch.arange(oid.shape[0])
            ox = self.state["operation"].x
            for j in active:
                o = self.current_operations[j]
                taus = self.tau[self.job_loc[j]][sorted(self.op_elig[o])]
                ox[node_of[o], self.op_transfer_cols[0]] = float(taus.min())
                ox[node_of[o], self.op_transfer_cols[1]] = float(taus.mean())
        elif self.TRANSPORT_REP == "edge":
            at = [[j, self.job_loc[j]] for j in range(self.num_jobs)
                  if not self.job_done[j] and self.job_loc[j] != self.depot]
            at = torch.LongTensor(at).T if at else _empty_edges()
            self.state["job", "at", "machine"].edge_index = at
            self.state["machine", "hosts", "job"].edge_index = at.flip(0)

    def calculate_mask(self):
        super().calculate_mask()
        if self.TRANSPORT_REP is None:
            return
        # transport of every action: tau[location of the job][machine]
        mej = self.state['machine', 'exec', 'job']
        mach, jobs = mej.edge_index
        loc = torch.tensor(self.job_loc, dtype=torch.long)[jobs]
        tau = torch.tensor(self.tau, dtype=torch.float)[loc, mach].unsqueeze(1)
        if self.TRANSPORT_REP == "feat":  # 6th action-edge column
            mej.edge_attr = torch.cat([mej.edge_attr[:, :EDGE_DIM], tau], dim=1)
            if ('job', 'exec', 'machine') in self.state.edge_types:
                self.state['job', 'exec', 'machine'].edge_attr = mej.edge_attr.clone()
        elif self.TRANSPORT_REP == "edge":
            reach = self.state['job', 'reach', 'machine']
            reach.edge_index = mej.edge_index.flip(0)
            reach.edge_attr = tau

    def calculate_next_state(self):
        if self.job_done is None:
            self.job_done = [False] * self.num_jobs
            self.fam_remaining = [len(m) for m in self.fam_members]
            self._init_transport()
        super().calculate_next_state()
        self._update_transport_state()
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
                if self.has_transport and current:
                    # readiness depends on the machine: evaluated at mu_f, where the family's
                    # earliest batch can start
                    t_f, mu_f = min((s, m) for m, s in (self._start_machine(j) for _, j, _, _ in current))
                    ready = [k for _, j, o2, k in current
                             if self.operations[o2][mu_f] > 0 and self._arrival(j, mu_f) <= t_f]
                else:
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
        start = max(float(self.state["machine"].x[sel_mach, 0]), self._arrival(sel_job, sel_mach))
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
            self.job_loc[j] = sel_mach  # every member leaves from the batch's machine
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
                    arrival = self._arrival(j, m)
                    calcu = t + max(arrival - self.state["machine"].x[m, 0], 0)
                    total_gap += calcu
                    new_edges.append([m, j])
                    feats.append([calcu, t / np.sum(oper), t + max(arrival, self.state["machine"].x[m, 0])])
                    self.job_start_machines[j, m] = max(arrival, self.state["machine"].x[m, 0])
                    self.current_job_proc[j, m] = t
            for row in feats:
                row.append(row[0] / total_gap)
                row.append(0)
                row += [0] * (mej.edge_attr.shape[1] - len(row))  # T-F: transport column
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
        if self.TRANSPORT_REP == "edge":
            drop(('operation', 'transfer', 'operation'))
        self.state = remove_operation_nodes(self.state, nodes)

    def normalize_state(self, state):
        coords = []
        if self.TRANSPORT_REP == "feat":  # already on the shared scale W (see _build_transport_features)
            # the coordinates are the last two columns of machine and job nodes; read from the
            # state itself because BOPO normalises its rollout envs' states through self.env
            for nt in ("machine", "job"):
                width = state[nt].x.shape[1]
                cols = [width - 2, width - 1]
                coords.append((nt, cols, state[nt].x[:, cols].clone()))
        state = super().normalize_state(state)
        edge_types = []
        if self.BATCH_REP == "node":
            state["family"].x = normalize_columns(state["family"].x)
            edge_types += [('family', 'batch_exec', 'machine'), ('machine', 'batch_exec', 'family')]
        elif self.BATCH_REP == "edge":
            edge_types.append(('operation', 'batch', 'operation'))
        if self.TRANSPORT_REP == "feat":
            edge_types.append(('machine', 'listens', 'machine'))
        elif self.TRANSPORT_REP == "edge":
            edge_types += [('machine', 'transport', 'machine'), ('job', 'reach', 'machine'),
                           ('operation', 'transfer', 'operation')]
        for et in edge_types:
            state[et].edge_attr = normalize_columns(state[et].edge_attr)
        # per-type min-max would put job and machine coordinates in different frames, so the
        # network could not relate a job's location to a machine: keep the shared scale
        for nt, cols, raw in coords:
            state[nt].x[:, cols] = 2 * raw - 1
        if self.TRANSPORT_REP is not None:
            for et in state.edge_types:
                if "edge_attr" in state[et]:
                    state[et].edge_attr = _pad(state[et].edge_attr, self.EDGE_DIM)
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


# ---------------------------------------------------------------- batching x transport
# "ojmb_<batching>_<transport>" -> env class, e.g. "ojmb_node_tf" = family node + T-F.
_BATCH_NAMES = {"node": "Node", "edge": "Edge", "base": "Base"}
_TRANSPORT_NAMES = {"t0": ("none", "T0"), "tf": ("feat", "TF"), "te": ("edge", "TE")}
TRANSPORT_ENVS = {}
for _b, _bn in _BATCH_NAMES.items():
    for _t, (_rep, _tn) in _TRANSPORT_NAMES.items():
        _name = f"FJSPBatch{_bn}Transport{_tn}Env"
        _cls = type(_name, (_FJSPBatchEnv,), {
            "BATCH_REP": _b, "TRANSPORT_REP": _rep, "EDGE_DIM": TRANSPORT_EDGE_DIM,
            "__doc__": f"Batching representation '{_b}' with transport representation {_tn} "
                       "(docs/transport_formulation.tex).",
        })
        globals()[_name] = _cls
        TRANSPORT_ENVS[f"ojmb_{_b}_{_t}"] = _name
