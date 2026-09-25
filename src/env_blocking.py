"""Blocking FJSP environment with finite buffers (representations "ojmb", "ojmd", "ojm_blk").

Problem (blocking constraint with buffers):
    Every machine m has an INPUT buffer (capacity in_cap) and an OUTPUT buffer
    (capacity out_cap). A part flows

        input buffer of m  ->  machine m  ->  output buffer of m  ->  input buffer of next machine

    - The agent routes the job's next operation to a machine m: the part leaves its previous
      machine's output buffer (or leaves the machine directly if it was being held there) and
      enters m's input buffer. Routing to m is legal while m's input buffer has a free slot.
      (Standard blocking semantics: a machine whose output buffer is full can still receive
      parts - it is only blocked while it holds a finished part. The stricter rule "a full
      output buffer accepts nothing" was dropped because it created circular waits.)
    - A machine processes its input buffer in FIFO order, one part at a time.
    - When a part finishes and it is not the job's last operation, it moves to the machine's
      own output buffer. If that buffer is full the machine is BLOCKED: it holds the finished
      part and cannot start anything else until a slot is freed. A job's last operation leaves
      the system as soon as it finishes.

Why a new env instead of extending src/env.py:
    FJSSPEnv builds the schedule by appending operations (start = max(machine free, job ready))
    with no global clock, so "the buffer is full" is not defined there. This env keeps an
    event-driven clock t that only moves forward: decisions are taken at t, and t advances to
    the next processing completion whenever no legal routing exists.

Deadlock handling:
    - Avoidance: a routing whose one-step simulation ends in a deadlock is masked, unless every
      legal routing does (self.num_masked_deadlock_actions). One step of lookahead only.
    - Resolution: if nothing is processing and no routing is legal, the blocked machines form a
      cycle (each one's held part can only go to another blocked machine with a full input
      buffer). The parts of the shortest such cycle move at the same instant: each machine
      releases its held part, starts its next queued part (freeing a slot), and receives the
      part of its predecessor in the cycle - "blocking with swap" (Mascis & Pacciarelli, 2002).
      Swaps are feasible (capacities are never exceeded) and the CP-SAT reference allows them
      implicitly; they are counted in self.num_swaps.

Graph: the blocking information is added to the OJM graph of src/env.py in one of two ways,
chosen with blocking_repr (hypothesis of the thesis: which representation is better):

blocking_repr="node" ("OJMB") - new node type:
    node 'buffer' (2 per machine: input buffer of m = m, output buffer of m = M + m)
        x = [capacity, occupancy, occupancy/capacity, is_output, blocked]
    ('operation','in','buffer') / ('buffer','holds','operation'): parts currently in a buffer
        attr = [in_buffer, priority (1 = next to leave), waiting time, processing time, is_output]
    ('buffer','of','machine') / ('machine','has','buffer'): permanent, 2 per machine
        attr = [full, free slots / capacity, is_output, 0, 0]
    ('buffer','self','buffer'): self-loops (GATv2 runs with add_self_loops=False and has no root
        weight, so without them a buffer loses its own features after the first layer).
    Operations that are routed but have not left the system yet stay in the graph, linked to
    their buffer or (while processing/held) to their assigned machine through the exec edge,
    whose column 3 flags "assigned".
    The buffer <-> machine correspondence is encoded by the edges, not as a machine-id feature:
    an id number is an arbitrary label that would break permutation invariance.
    All edge attributes have 5 columns because the GNN layers use a shared edge_dim=5.

blocking_repr="dummy" ("OJMD") - dummy machines:
    each buffer is an extra node of the existing 'machine' type (rows M..3M-1), with the same
    5 buffer features appended to the machine features plus an is_dummy flag; the same
    information travels through ('operation','in','machine') / ('machine','holds','operation')
    and ('machine','buffer_of','machine') / ('machine','has_buffer','machine'), and a dummy's
    self-loop is added to ('machine','listens','machine'). The difference between the two
    representations is therefore only structural: dedicated buffer parameters (node) versus
    parameters shared with the real machines, distinguished by a flag (dummy).

blocking_repr="none" ("OJM_BLK") gives the plain OJM graph on the same blocking dynamics,
so every graph is compared on exactly the same problem.
"""
import copy
import random

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.blocking_config import BLOCKING_CONFIG
from src.env import FJSSPEnv, _dbg

IN_CAP, OUT_CAP = BLOCKING_CONFIG["in_cap"], BLOCKING_CONFIG["out_cap"]

FUTURE, INBUF, RUN, HELD, OUTBUF, GONE = range(6)
SENTINEL = 10000.0


def _edges(pairs):
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.LongTensor(pairs).T


def _attrs(rows, dim=5):
    if not rows:
        return torch.empty((0, dim), dtype=torch.float)
    return torch.tensor(rows, dtype=torch.float)


class FJSPEnvBlocking(FJSSPEnv):
    # "node": buffer node type | "dummy": buffers as dummy machines | "none": plain OJM graph
    BLOCKING_REPRS = ("node", "dummy", "none")

    def __init__(self, instances, mask_option=3, sel_k=5, jm_design="edges",
                 in_cap=IN_CAP, out_cap=OUT_CAP, blocking_repr="node", avoid_deadlocks=True):
        # job-machine edges are rebuilt from scratch every decision, so the "baseline" layout
        # (features frozen at creation time) does not exist here: always use the fixed layout
        if jm_design == "baseline":
            jm_design = "edges"
        if blocking_repr not in self.BLOCKING_REPRS:
            raise ValueError(f"blocking_repr must be one of {self.BLOCKING_REPRS}, got {blocking_repr!r}")
        super().__init__(instances, mask_option, sel_k, jm_design)
        self.in_cap = in_cap
        self.out_cap = out_cap
        self.blocking_repr = blocking_repr
        self.avoid_deadlocks = avoid_deadlocks

    def env_kwargs(self):
        return {"jm_design": self.jm_design, "in_cap": self.in_cap, "out_cap": self.out_cap,
                "blocking_repr": self.blocking_repr, "avoid_deadlocks": self.avoid_deadlocks}

    # ── instance ────────────────────────────────────────────────────────────────

    def _load_instance(self, instance):
        self.jobs = instance["jobs"]
        self.operations = instance["operations"]
        self.num_jobs = len(self.jobs)
        self.num_operations = len(self.operations)
        self.num_machines = len(self.operations[0])
        self.proc = np.array(self.operations, dtype=float)
        self.op_job = [0] * self.num_operations
        self.op_pos = [0] * self.num_operations
        for j, ops in enumerate(self.jobs):
            for k, o in enumerate(ops):
                self.op_job[o] = j
                self.op_pos[o] = k
        # remaining average work from each operation to the end of its job (as in src/env.py)
        self.all_pendings = [0.0] * self.num_operations
        for ops in self.jobs:
            acc = 0.0
            for o in reversed(ops):
                row = self.proc[o]
                acc += float(np.mean(row[row != 0]))
                self.all_pendings[o] = acc

    def reset(self, sel_index=None):
        if sel_index is None:
            self._load_instance(self.instances[self.current_instance])
            self.current_instance = (self.current_instance + 1) % len(self.instances)
        else:
            self._load_instance(self.instances[sel_index])
        M, N = self.num_machines, self.num_operations
        self.t = 0.0
        self.num_steps = 0
        self.num_swaps = 0
        self.num_masked_deadlock_actions = 0
        self.num_blocked = 0  # completions that found the output buffer full (machine blocked)
        self.status = [FUTURE] * N
        self.assigned = [-1] * N
        self.start = [None] * N
        self.end = [None] * N
        self.entry = [0.0] * N
        self.routed_at = [None] * N  # time the part entered its machine's input buffer
        self.depart = [None] * N     # time the part left its machine (later than end if held)
        self.in_queue = [[] for _ in range(M)]
        self.out_queue = [[] for _ in range(M)]
        self.running = [None] * M
        self.held = [None] * M
        self.busy_time = [0.0] * M
        self.job_next = [0] * self.num_jobs
        self.job_done = [False] * self.num_jobs
        self.max_occupancy = {"in": 0, "out": 0}
        self.mk = None
        self._advance()
        self._build_state()
        _dbg(1, f"  reset() blocking | {self.num_jobs} jobs | {N} ops | {M} machines | "
                f"in_cap={self.in_cap} out_cap={self.out_cap} | blocking_repr={self.blocking_repr}")
        return self.state

    # ── dynamics ────────────────────────────────────────────────────────────────

    def _next_op(self, j):
        k = self.job_next[j]
        return self.jobs[j][k] if k < len(self.jobs[j]) else None

    def _released(self, j):
        """The job's next operation can be routed: it is the first one, or its predecessor
        has finished processing (sitting in an output buffer or held on a blocked machine)."""
        k = self.job_next[j]
        if self.job_done[j] or k >= len(self.jobs[j]):
            return False
        return k == 0 or self.status[self.jobs[j][k - 1]] in (OUTBUF, HELD)

    def _legal(self, j, m):
        o = self._next_op(j)
        if o is None or self.proc[o, m] == 0 or not self._released(j):
            return False
        if self.in_cap == 0:
            return self.running[m] is None and self.held[m] is None and not self.in_queue[m]
        return len(self.in_queue[m]) < self.in_cap

    def _any_legal(self):
        return any(self._legal(j, m) for j in range(self.num_jobs) for m in range(self.num_machines))

    def _start_idle_machines(self):
        for m in range(self.num_machines):
            if self.running[m] is None and self.held[m] is None and self.in_queue[m]:
                o = self.in_queue[m].pop(0)
                self.status[o] = RUN
                self.start[o] = self.t
                self.end[o] = self.t + float(self.proc[o, m])
                self.busy_time[m] += float(self.proc[o, m])
                self.running[m] = o

    def _free_output_slot(self, m):
        """A slot of m's output buffer became free: a part held on m (if any) moves into it,
        which unblocks m."""
        if self.held[m] is not None and len(self.out_queue[m]) < self.out_cap:
            o = self.held[m]
            self.held[m] = None
            self.depart[o] = self.t
            self.status[o] = OUTBUF
            self.entry[o] = self.t
            self.out_queue[m].append(o)

    def _finish(self, m):
        o = self.running[m]
        self.running[m] = None
        j = self.op_job[o]
        self.depart[o] = self.t
        if self.op_pos[o] == len(self.jobs[j]) - 1:
            self.status[o] = GONE
            self.job_done[j] = True
        elif len(self.out_queue[m]) < self.out_cap:
            self.status[o] = OUTBUF
            self.entry[o] = self.t
            self.out_queue[m].append(o)
        else:
            self.depart[o] = None
            self.num_blocked += 1
            self.status[o] = HELD
            self.held[m] = o

    def _advance(self):
        """Run the clock until a decision is needed or every job is done."""
        while True:
            self._start_idle_machines()
            self._track_occupancy()
            if all(self.job_done):
                return
            if self._any_legal():
                return
            running = [m for m in range(self.num_machines) if self.running[m] is not None]
            if running:
                self.t = min(self.end[self.running[m]] for m in running)
                for m in running:
                    if self.end[self.running[m]] <= self.t:
                        self._finish(m)
                continue
            # nothing processing and no legal routing: circular blocking
            self._swap()

    def _swap_cycle(self):
        """Shortest cycle in the graph 'blocked machine a -> machine b that the next operation
        of a's held part is eligible on'. In a deadlock every such b is itself blocked with a full
        input buffer (an idle machine with a queue would have started, and a machine with a free
        slot would make the routing legal), so every node has an outgoing arc and a cycle exists.
        Ties between equally short cycles: smallest total processing time of the moved parts."""
        succ = {}
        for a in range(self.num_machines):
            if self.held[a] is not None:
                o = self.held[a]
                nxt = self.jobs[self.op_job[o]][self.op_pos[o] + 1]
                succ[a] = [int(b) for b in np.nonzero(self.proc[nxt])[0]]
        best, best_key = None, None
        for root in succ:
            # breadth-first search from root back to root
            parent, frontier, found = {root: None}, [root], None
            while frontier and found is None:
                next_frontier = []
                for a in frontier:
                    for b in succ.get(a, []):
                        if b == root:
                            found = a
                            break
                        if b not in parent:
                            parent[b] = a
                            next_frontier.append(b)
                    if found is not None:
                        break
                frontier = next_frontier
            if found is None:
                continue
            cycle, a = [], found
            while a is not None:
                cycle.append(a)
                a = parent[a]
            cycle.reverse()  # root -> ... -> found, and found -> root closes the cycle
            cost = 0.0
            for i, a in enumerate(cycle):
                o = self.held[a]
                nxt = self.jobs[self.op_job[o]][self.op_pos[o] + 1]
                cost += float(self.proc[nxt, cycle[(i + 1) % len(cycle)]])
            key = (len(cycle), cost)
            if best_key is None or key < best_key:
                best, best_key = cycle, key
        return best

    def _swap(self):
        cycle = self._swap_cycle()
        assert cycle is not None, f"deadlock without a swap cycle at t={self.t}"
        moved = []
        for a in cycle:  # every machine of the cycle releases its held part...
            o = self.held[a]
            self.held[a] = None
            self.depart[o] = self.t
            self.status[o] = GONE
            moved.append(self.op_job[o])
        self._start_idle_machines()  # ...starts its next queued part, freeing a slot...
        for i, j in enumerate(moved):  # ...and receives the part of its predecessor in the cycle
            self._enter_input_buffer(j, cycle[(i + 1) % len(cycle)])
        self.num_swaps += 1
        _dbg(1, f"  deadlock at t={self.t} -> swap #{self.num_swaps} over machines {cycle}")

    def _route(self, j, m):
        k = self.job_next[j]
        if k > 0:
            prev = self.jobs[j][k - 1]
            a = self.assigned[prev]
            if self.status[prev] == OUTBUF:
                self.out_queue[a].remove(prev)
                self._free_output_slot(a)
            else:  # HELD: the part leaves the blocked machine directly
                self.held[a] = None
                self.depart[prev] = self.t
            self.status[prev] = GONE
        self._enter_input_buffer(j, m)

    def _enter_input_buffer(self, j, m):
        o = self.jobs[j][self.job_next[j]]
        self.status[o] = INBUF
        self.assigned[o] = m
        self.entry[o] = self.t
        self.routed_at[o] = self.t
        self.in_queue[m].append(o)
        self.job_next[j] += 1

    def _track_occupancy(self):
        self.max_occupancy["in"] = max(self.max_occupancy["in"], max(len(q) for q in self.in_queue))
        self.max_occupancy["out"] = max(self.max_occupancy["out"], max(len(q) for q in self.out_queue))

    def _projected_times(self):
        """Estimated end time of every routed operation and free time of every machine,
        assuming each machine keeps processing its input buffer in FIFO order from now on.
        A blocked machine is assumed to be released now (its release time is still unknown)."""
        proj_end = {}
        proj_free = [self.t] * self.num_machines
        for m in range(self.num_machines):
            free = self.t
            o = self.running[m]
            if o is not None:
                free = self.end[o]
                proj_end[o] = free
            for o in self.in_queue[m]:
                free += float(self.proc[o, m])
                proj_end[o] = free
            proj_free[m] = free
        return proj_end, proj_free

    def _makespan_estimate(self):
        proj_end, _ = self._projected_times()
        ends = [e for e in self.end if e is not None] + list(proj_end.values())
        return max([self.t] + ends)

    def step(self, action):
        edge_index = self.state['machine', 'exec', 'job'].edge_index
        total_reward = 0.0
        while True:
            self.num_steps += 1
            sel_mach, sel_job = int(edge_index[0, int(action)]), int(edge_index[1, int(action)])
            _dbg(2, f"  step #{self.num_steps} | t={self.t} | route job {sel_job} -> machine {sel_mach}")
            prev_ms = self._makespan_estimate()
            self._route(sel_job, sel_mach)
            self._advance()
            if all(self.job_done):
                self.mk = round(max(e for e in self.end if e is not None), 2)
                total_reward += prev_ms - self.mk
                _dbg(1, f"  episode DONE after {self.num_steps} steps | makespan={self.mk} | swaps={self.num_swaps}")
                return self.state, total_reward, True, {"current_machine": sel_mach}
            self._build_state()
            total_reward += prev_ms - self._makespan_estimate()
            mask = self.state['machine', 'exec', 'job'].mask
            if int((~mask).sum()) != 1:
                return self.state, total_reward, False, {"current_machine": sel_mach}
            # only one legal action: play it without asking the policy (as src/env.py does)
            action = self.sample()
            edge_index = self.state['machine', 'exec', 'job'].edge_index

    # ── graph ───────────────────────────────────────────────────────────────────

    def _build_state(self):
        M, N = self.num_machines, self.num_operations
        proj_end, proj_free = self._projected_times()
        data = HeteroData()

        routed_live = (INBUF, RUN, HELD, OUTBUF)
        live = [o for o in range(N) if self.status[o] == FUTURE
                or (self.blocking_repr != "none" and self.status[o] in routed_live)]
        idx = {o: i for i, o in enumerate(live)}

        # operations: [is the job's next operation, remaining work]
        op_x = torch.zeros((len(live), 2))
        for o, i in idx.items():
            j = self.op_job[o]
            op_x[i, 0] = float(self.status[o] == FUTURE and self.op_pos[o] == self.job_next[j])
            op_x[i, 1] = self.all_pendings[o]
        data['operation'].x = op_x
        data['operation'].op_ref = torch.LongTensor(live)

        # jobs: [done, ready time, remaining operations, remaining work] (as src/env.py)
        job_x = torch.zeros((self.num_jobs, 4))
        for j in range(self.num_jobs):
            if self.job_done[j] or self._next_op(j) is None:
                job_x[j, 0] = 1
                continue
            k = self.job_next[j]
            ready = 0.0
            if k > 0:
                prev = self.jobs[j][k - 1]
                ready = self.end[prev] if self.end[prev] is not None else proj_end.get(prev, self.t)
            job_x[j, 1] = ready
            job_x[j, 2] = len(self.jobs[j]) - k
            job_x[j, 3] = self.all_pendings[self.jobs[j][k]]
        data['job'].x = job_x

        # machines: [free time, utilisation, free time relative to the earliest machine]
        free = torch.tensor(proj_free, dtype=torch.float)
        mach_x = torch.zeros((M, 3))
        mach_x[:, 0] = free
        mach_x[:, 1] = torch.tensor(self.busy_time) / torch.clamp(free, min=1.0)
        mach_x[:, 2] = free - free.min()
        data['machine'].x = mach_x

        belongs, prec, exec_pairs, exec_attr = [], [], [], []
        for o in live:
            i = idx[o]
            prec.append([i, i])
            if self.status[o] == FUTURE:
                j = self.op_job[o]
                belongs.append([i, j])
                if self.op_pos[o] + 1 < len(self.jobs[j]):
                    prec.append([idx[self.jobs[j][self.op_pos[o] + 1]], i])
                row = self.proc[o]
                for m in np.nonzero(row)[0]:
                    exec_pairs.append([i, int(m)])
                    exec_attr.append([row[m], row[m] / row.sum(), row[m] / self.all_pendings[o], 0.0, 0.0])
            else:
                m = self.assigned[o]
                t = self.proc[o, m]
                exec_pairs.append([i, m])
                exec_attr.append([t, t / self.proc[o].sum(), t / self.all_pendings[o], 1.0, 0.0])
        exec_attr = _attrs(exec_attr)
        exec_index = _edges(exec_pairs)
        for m in range(M):  # column 4: processing time relative to the longest one on m
            sel = exec_index[1] == m if exec_index.numel() else torch.zeros(0, dtype=torch.bool)
            if sel.any():
                exec_attr[sel, 4] = exec_attr[sel, 0] / exec_attr[sel, 0].max()
        data['operation', 'belongs', 'job'].edge_index = _edges(belongs)
        data['operation', 'prec', 'operation'].edge_index = _edges(prec)
        data['operation', 'exec', 'machine'].edge_index = exec_index
        data['operation', 'exec', 'machine'].edge_attr = exec_attr
        data['machine', 'exec', 'operation'].edge_index = exec_index.flip(0)
        data['machine', 'exec', 'operation'].edge_attr = exec_attr.clone()
        data['machine', 'listens', 'machine'].edge_index = _edges([[a, b] for a in range(M) for b in range(M)])
        data['job', 'listens', 'job'].edge_index = _edges(
            [[a, b] for a in range(self.num_jobs) if not job_x[a, 0] for b in range(self.num_jobs)])

        # action edges machine -> job, one per eligible machine of each job's next operation
        self.job_start_machines = torch.full((self.num_jobs, M), SENTINEL)
        self.current_job_proc = torch.zeros((self.num_jobs, M))
        legal = torch.zeros((self.num_jobs, M), dtype=torch.bool)
        action_pairs = []
        for j in range(self.num_jobs):
            o = self._next_op(j)
            if o is None or self.job_done[j]:
                continue
            for m in np.nonzero(self.proc[o])[0]:
                m = int(m)
                action_pairs.append([m, j])
                self.current_job_proc[j, m] = float(self.proc[o, m])
                self.job_start_machines[j, m] = max(self.t, proj_free[m])
                legal[j, m] = self._legal(j, m)
        self._mask_deadlocking_actions(legal)
        data['machine', 'exec', 'job'].edge_index = _edges(action_pairs)

        if self.blocking_repr == "node":
            self._add_buffer_nodes(data, idx)
        elif self.blocking_repr == "dummy":
            self._add_dummy_machines(data, idx)
        self.state = data
        self.calculate_mask(legal)

    _DYNAMIC_FIELDS = ("t", "status", "assigned", "start", "end", "entry", "routed_at", "depart",
                       "in_queue", "out_queue", "running", "held", "busy_time", "job_next",
                       "job_done", "num_blocked", "num_swaps", "max_occupancy")

    def _snapshot(self):
        return {f: copy.deepcopy(getattr(self, f)) for f in self._DYNAMIC_FIELDS}

    def _restore(self, snap):
        for f, v in snap.items():
            setattr(self, f, v)

    def _mask_deadlocking_actions(self, legal):
        """Deadlock avoidance (one-step lookahead): simulate each legal routing up to the next
        decision and forbid it if a deadlock (swap) occurs on the way - unless every legal
        routing leads to one, in which case they all stay and the swap resolves it."""
        if not self.avoid_deadlocks:
            return
        candidates = [tuple(p) for p in legal.nonzero().tolist()]
        bad = []
        for j, m in candidates:
            snap = self._snapshot()
            swaps = self.num_swaps
            self._route(j, m)
            self._advance()
            if self.num_swaps > swaps:
                bad.append((j, m))
            self._restore(snap)
        if bad and len(bad) < len(candidates):
            for j, m in bad:
                legal[j, m] = False
            self.num_masked_deadlock_actions += len(bad)

    def _buffer_information(self, idx):
        """The blocking information, independent of how it is placed in the graph, so the
        node-type and dummy-machine representations carry exactly the same information.
        Buffer b: input buffer of machine b (b < M) or output buffer of machine b - M.
        Returns
            feats: per buffer [capacity, occupancy, occupancy/capacity, is_output, blocked]
            parts: (operation index, buffer) for every part inside a buffer, with attr
                   [in_buffer, priority (1 = next to leave), waiting time, processing time, is_output]
            links: (buffer, its machine), with attr [full, free slots / capacity, is_output, 0, 0]
        """
        M = self.num_machines
        caps = [self.in_cap] * M + [self.out_cap] * M
        queues = self.in_queue + self.out_queue
        feats, part_pairs, part_attr, link_pairs, link_attr = [], [], [], [], []
        for b, q in enumerate(queues):
            is_out = float(b >= M)
            m = b - M if b >= M else b
            occ = len(q)
            cap = max(caps[b], 1)  # capacity 0 (no buffer) is never occupied
            feats.append([caps[b], occ, occ / cap, is_out, float(b >= M and self.held[m] is not None)])
            link_pairs.append([b, m])
            link_attr.append([float(occ >= caps[b]), (caps[b] - occ) / cap, is_out, 0.0, 0.0])
            for pos, o in enumerate(q):
                part_pairs.append([idx[o], b])
                part_attr.append([1.0, 1.0 - pos / cap, self.t - self.entry[o],
                                  0.0 if is_out else float(self.proc[o, m]), is_out])
        return (torch.tensor(feats, dtype=torch.float), _edges(part_pairs), _attrs(part_attr),
                _edges(link_pairs), _attrs(link_attr))

    def _add_buffer_nodes(self, data, idx):
        """Representation 1 - new node type 'buffer': buffers get their own node type, so
        the GNN learns separate parameters for them (their own input projection and their
        own attention per edge type)."""
        M = self.num_machines
        feats, part_index, part_attr, link_index, link_attr = self._buffer_information(idx)
        data['buffer'].x = feats
        data['operation', 'in', 'buffer'].edge_index = part_index
        data['operation', 'in', 'buffer'].edge_attr = part_attr
        data['buffer', 'holds', 'operation'].edge_index = part_index.flip(0)
        data['buffer', 'holds', 'operation'].edge_attr = part_attr.clone()
        data['buffer', 'of', 'machine'].edge_index = link_index
        data['buffer', 'of', 'machine'].edge_attr = link_attr
        data['machine', 'has', 'buffer'].edge_index = link_index.flip(0)
        data['machine', 'has', 'buffer'].edge_attr = link_attr.clone()
        data['buffer', 'self', 'buffer'].edge_index = _edges([[b, b] for b in range(2 * M)])

    def _add_dummy_machines(self, data, idx):
        """Representation 2 - dummy machines: every buffer becomes an extra node of the
        EXISTING 'machine' type (rows M..3M-1: dummy of buffer b = M + b), so buffers and real
        machines share the same parameters and are told apart only by the is_dummy feature.
        Machine features become [free time, utilisation, relative free time, is_dummy,
        capacity, occupancy, occupancy/capacity, is_output, blocked]: real machines have zeros
        in the buffer columns, dummies have zeros in the first three.
        Dummies only get their self-loop in ('machine','listens','machine'): joining the
        machine clique would let every real machine attend to every buffer of every machine,
        while the node-type representation only links a buffer to its own machine."""
        M = self.num_machines
        feats, part_index, part_attr, link_index, link_attr = self._buffer_information(idx)
        real = data['machine'].x
        x = torch.zeros((3 * M, 9))
        x[:M, :3] = real
        x[M:, 3] = 1.0
        x[M:, 4:] = feats
        data['machine'].x = x
        part_index = part_index.clone()
        part_index[1] += M
        link_index = link_index.clone()
        link_index[0] += M
        data['operation', 'in', 'machine'].edge_index = part_index
        data['operation', 'in', 'machine'].edge_attr = part_attr
        data['machine', 'holds', 'operation'].edge_index = part_index.flip(0)
        data['machine', 'holds', 'operation'].edge_attr = part_attr.clone()
        data['machine', 'buffer_of', 'machine'].edge_index = link_index
        data['machine', 'buffer_of', 'machine'].edge_attr = link_attr
        data['machine', 'has_buffer', 'machine'].edge_index = link_index.flip(0)
        data['machine', 'has_buffer', 'machine'].edge_attr = link_attr.clone()
        listens = data['machine', 'listens', 'machine'].edge_index
        dummy_loops = torch.arange(M, 3 * M).repeat(2, 1)
        data['machine', 'listens', 'machine'].edge_index = torch.cat([listens, dummy_loops], dim=1)

    def calculate_mask(self, legal=None):
        """Same per-job top-sel_k rule as src/env.py, restricted to the routings that the
        buffers currently allow."""
        if self.mask_option == 0:
            mask_matrix = self.job_start_machines.clone()
        else:
            mask_matrix = self.job_start_machines + self.current_job_proc
        # +inf, not SENTINEL: on long instances a legal completion time can exceed 10000
        mask_matrix[~legal] = float("inf")
        k = max(1, int(self.sel_k))
        keep = torch.zeros_like(legal)
        for j in range(mask_matrix.shape[0]):
            n_valid = int(legal[j].sum())
            if n_valid:
                _, top = torch.topk(mask_matrix[j], k=min(k, n_valid), largest=False)
                keep[j, top] = True
        edge_index = self.state['machine', 'exec', 'job'].edge_index
        mask = ~keep[edge_index[1], edge_index[0]]
        assert not bool(mask.all()), f"no legal action at t={self.t} (every routing masked)"
        self.state['machine', 'exec', 'job'].mask = mask
        self._last_mask_matrix = mask_matrix
        self._refresh_jm_edges()

    def sample(self):
        mask = self.state['machine', 'exec', 'job'].mask
        return random.choice([i for i in range(len(mask)) if not mask[i]])

    # ── normalisation ───────────────────────────────────────────────────────────

    def _scale_buffer_feats(self, feats):
        # capacities/occupancies divided by the largest capacity; ratios and flags are
        # already in [0, 1]. A fixed scale (not per-state min-max) keeps "half full" meaning
        # half full: min-max would map "every buffer half full" to the same value as "empty".
        feats = feats.clone()
        feats[:, :2] = feats[:, :2] / max(self.in_cap, self.out_cap, 1)
        return (2 * feats - 1).float()

    def normalize_state(self, state):
        raw_machine_x = state['machine'].x
        state = super().normalize_state(state)
        if self.blocking_repr == "none":
            return state
        if self.blocking_repr == "node":
            state['buffer'].x = self._scale_buffer_feats(state['buffer'].x)
            part_types = (('operation', 'in', 'buffer'), ('buffer', 'holds', 'operation'))
            link_types = (('buffer', 'of', 'machine'), ('machine', 'has', 'buffer'))
        else:
            # the parent min-max runs over real AND dummy rows, which would let the dummies'
            # zeros distort the real machines' time columns: redo the machine features here.
            # M is read from the state, not self.num_machines: BOPO normalises its rollout
            # copies' states through the training env, whose current instance may differ
            M = int((raw_machine_x[:, 3] == 0).sum())
            real = raw_machine_x[:M, :3]
            mins, maxs = real.min(dim=0, keepdim=True).values, real.max(dim=0, keepdim=True).values
            x = torch.zeros_like(raw_machine_x)
            x[:M, :3] = 2 * (real - mins) / (maxs - mins + 1e-7) - 1
            x[:, 3] = 2 * raw_machine_x[:, 3] - 1
            x[:, 4:] = self._scale_buffer_feats(raw_machine_x[:, 4:])
            state['machine'].x = x.float()
            part_types = (('operation', 'in', 'machine'), ('machine', 'holds', 'operation'))
            link_types = (('machine', 'buffer_of', 'machine'), ('machine', 'has_buffer', 'machine'))
        for edge_type in part_types:
            e = state[edge_type].edge_attr
            if e.numel() == 0:
                continue
            e = e.clone()
            times = e[:, 2:4]
            mins, maxs = times.min(dim=0).values, times.max(dim=0).values
            e[:, 2:4] = (times - mins) / (maxs - mins + 1e-7)
            state[edge_type].edge_attr = (2 * e - 1).float()
        for edge_type in link_types:
            e = state[edge_type].edge_attr.clone()
            e[:, :3] = 2 * e[:, :3] - 1
            state[edge_type].edge_attr = e.float()
        return state


class FJSPEnvBlockingDummy(FJSPEnvBlocking):
    """Blocking represented with dummy machines (representation "ojmd")."""

    def __init__(self, instances, mask_option=3, sel_k=5, jm_design="edges",
                 in_cap=IN_CAP, out_cap=OUT_CAP, blocking_repr="dummy", avoid_deadlocks=True):
        super().__init__(instances, mask_option, sel_k, jm_design, in_cap, out_cap, blocking_repr, avoid_deadlocks)


class FJSPEnvBlockingNoBuffer(FJSPEnvBlocking):
    """Plain OJM graph on the blocking dynamics (representation "ojm_blk")."""

    def __init__(self, instances, mask_option=3, sel_k=5, jm_design="edges",
                 in_cap=IN_CAP, out_cap=OUT_CAP, blocking_repr="none", avoid_deadlocks=True):
        super().__init__(instances, mask_option, sel_k, jm_design, in_cap, out_cap, blocking_repr, avoid_deadlocks)
