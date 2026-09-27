"""Blocking FJSP with machine unavailability (representations "ojmb_uf", "ojmb_uo", "ojmb_u0").

Problem: the blocking FJSP of src/env_blocking.py (finite input/output buffers, blocked machines,
swaps) plus unavailability windows per machine (src/unavailability.py):
    scheduled  planned maintenance, known from t = 0, non-preemptive (an operation never
               starts if it would cross one);
    breakdown  Weibull failures (increasing failure rate, reliability eta_m per machine) with
               exponential repairs, preempt-resume (the part stays on the machine and continues
               after the repair). Revealed only when they happen.
The realisation of an episode (its "scenario") is sampled once in reset() from a seed, or read
from instance["unavailability"][mode] for the stored validation/test sets.

Interaction with blocking (the rules are identical in src/solver_blocking.py and in
src/metrics/unavailability.py):
    - routing into the input buffer of a machine that is down is legal (legality stays
      capacity-only): the part waits there, and the policy must learn not to trap parts;
    - unloading continues during a window: a part in the output buffer, or a finished part held
      on a blocked machine, can leave when its next operation is routed. A window only stops
      starts (and, for a breakdown, the progress of the running part);
    - a machine that is down cannot start its queued part, so it cannot take part in a swap;
      when every deadlock cycle contains such a machine the state is a WAIT, not a deadlock:
      the clock jumps to the end of the window (self.num_waits);
    - the clock also stops at the time a waiting machine can start (the end of its window).

Information model (nothing the policy, its mask or the look-ahead sees depends on the future):
    - the dynamics run on the true windows (self.true_windows);
    - everything visible is computed from the information view at the current time
      (src/unavailability.py:information_view): the scheduled windows, and the breakdowns that
      have started, with estimated end t + MTTR_m while the repair is in progress;
    - the deadlock look-ahead simulates with that view (self._lookahead), so a routing is never
      masked because of a breakdown that has not happened yet;
    - the ranking of the action mask (top-sel_k earliest completion) is window-BLIND for every
      representation: all window information must arrive through the graph. The imitation
      teacher (expert_action) is window-aware and identical for every representation.

Graph (unavail_repr), on top of the blocking graph of blocking_repr ("node" = ojmb):
    one method, _window_information(), gives per machine its current window, its next known
    window starting within the horizon H, and its reliability; both representations are built
    from it, so they carry exactly the same information and differ only in structure.
    "feat"  ("ojmb_uf") 7 columns appended to the machine features:
            [planned_downtime, breakdown_status, availability_period, next_window_length,
             remaining_downtime, p_fail_H, age/eta]
    "dummy" ("ojmb_uo") every current / next window is an extra 'operation' node pre-assigned to
            its machine (one exec edge each way, column 3 "assigned" = 1, a prec self-loop, no
            job, no action edge, no buffer: it is never routed, queued or swapped). Operation
            features become [is next, remaining work, is_dummy, start_offset, remaining_length,
            planned, breakdown]; the 2 reliability columns [p_fail_H, age/eta] go on the machine.
    "none"  ("ojmb_u0") the blocking graph only, with window-blind projections: the control.
    The machine free time and the action-edge start / completion / idle columns are
    window-aware in "feat" and "dummy" (identical in both) and window-blind in "none".
    Every time feature is divided by H and clipped to [0, 1]; normalize_state maps these
    columns with a fixed 2x - 1 instead of the per-state min-max, which would map "every
    machine down" and "no machine down" to the same value.
"""
import numpy as np
import torch

from src.blocking_config import BLOCKING_CONFIG
from src.env_blocking import RUN, SENTINEL, FJSPEnvBlocking, _attrs, _edges
from src.metrics.schedule import lower_bound
from src.unavailability import (BREAKDOWN, SCHEDULED, earliest_start, information_view, last_renewal,
                                next_window, p_fail, resume_end, sample_scenario, scenario_from_json,
                                window_at)
from src.unavailability_config import UNAVAIL_CONFIG

IN_CAP, OUT_CAP = BLOCKING_CONFIG["in_cap"], BLOCKING_CONFIG["out_cap"]


def _clip01(x):
    return float(min(max(x, 0.0), 1.0))


class FJSPEnvUnavailability(FJSPEnvBlocking):
    # "feat": features on machines | "dummy": dummy operations | "none": window-blind control
    UNAVAIL_REPRS = ("feat", "dummy", "none")
    MACHINE_COLS = {"feat": 7, "dummy": 2, "none": 0}  # appended to the machine features
    DUMMY_OP_COLS = 5                                 # appended to the operation features ("dummy")
    _DYNAMIC_FIELDS = FJSPEnvBlocking._DYNAMIC_FIELDS + ("num_waits",)

    def __init__(self, instances, mask_option=3, sel_k=5, jm_design="edges",
                 in_cap=IN_CAP, out_cap=OUT_CAP, blocking_repr="node", avoid_deadlocks=True,
                 unavail_repr="feat", unavail_mode=None, scenario_seed=None):
        super().__init__(instances, mask_option, sel_k, jm_design, in_cap, out_cap, blocking_repr, avoid_deadlocks)
        if unavail_repr not in self.UNAVAIL_REPRS:
            raise ValueError(f"unavail_repr must be one of {self.UNAVAIL_REPRS}, got {unavail_repr!r}")
        self.unavail_repr = unavail_repr
        self.unavail_mode = unavail_mode or UNAVAIL_CONFIG["mode"]
        self.scenario_seed = UNAVAIL_CONFIG["seed"] if scenario_seed is None else int(scenario_seed)
        self._scenario_rng = np.random.default_rng(self.scenario_seed)
        self._fixed_scenario = None
        self._aware = unavail_repr != "none"
        self._lookahead = False
        self._lookahead_view = None
        self._view_cache = None

    def env_kwargs(self):
        return {**super().env_kwargs(), "unavail_repr": self.unavail_repr,
                "unavail_mode": self.unavail_mode, "scenario_seed": self.scenario_seed}

    # ── scenarios (common random numbers) ──────────────────────────────────────

    def draw_scenario(self):
        """A fresh scenario seed. BOPO draws one per group and gives it to every rollout copy
        (fix_scenario), so the rollouts it compares face the same breakdowns."""
        return int(self._scenario_rng.integers(2 ** 31 - 1))

    def fix_scenario(self, seed):
        """Use this scenario seed in every following reset (None: draw a new one each time)."""
        self._fixed_scenario = seed

    def _load_instance(self, instance):
        super()._load_instance(instance)
        self.lb = lower_bound(instance)
        self.horizon = max(UNAVAIL_CONFIG["horizon_factor"] * self.lb, 1.0)
        stored = (instance.get("unavailability") or {}).get(self.unavail_mode)
        if stored is not None:
            scenario = scenario_from_json(stored)
        else:
            seed = self._fixed_scenario if self._fixed_scenario is not None else self.draw_scenario()
            scenario = sample_scenario(self.operations, self.lb, self.unavail_mode,
                                       np.random.default_rng(seed), UNAVAIL_CONFIG)
        self.true_windows = scenario["windows"]
        self.beta, self.eta = scenario["beta"], scenario["eta"]
        self.mttr = scenario["mttr"] or [0.0] * self.num_machines
        self.num_waits = 0
        self.num_breakdowns = 0
        self._view_cache = None

    # ── windows ────────────────────────────────────────────────────────────────

    def _view(self):
        """Information view at the current time (cached per decision time)."""
        if self._view_cache is None or self._view_cache[0] != self.t:
            self._view_cache = (self.t, information_view(self.true_windows, self.mttr, self.t))
        return self._view_cache[1]

    def _windows(self):
        """Windows the dynamics run on: the true ones, or the information view of the decision
        time while the deadlock look-ahead simulates."""
        return self._lookahead_view if self._lookahead else self.true_windows

    def _earliest(self, m, o, t, windows):
        return earliest_start(t, float(self.proc[o, m]), windows[m])

    # ── dynamics ───────────────────────────────────────────────────────────────

    def _start_idle_machines(self):
        windows = self._windows()
        for m in range(self.num_machines):
            if self.running[m] is None and self.held[m] is None and self.in_queue[m]:
                o = self.in_queue[m][0]
                if self._earliest(m, o, self.t, windows) > self.t:
                    continue  # down, or the part would cross a planned maintenance: it waits
                self.in_queue[m].pop(0)
                p = float(self.proc[o, m])
                self.status[o] = RUN
                self.start[o] = self.t
                self.end[o] = resume_end(self.t, p, windows[m])
                self.busy_time[m] += p
                self.running[m] = o

    def _pending_starts(self, held):
        """Times at which a machine that cannot start its queue head now will be able to:
        idle machines (held=False) or blocked ones (held=True)."""
        windows = self._windows()
        times = []
        for m in range(self.num_machines):
            if self.running[m] is None and (self.held[m] is not None) == held and self.in_queue[m]:
                x = self._earliest(m, self.in_queue[m][0], self.t, windows)
                if x > self.t:
                    times.append(x)
        return times

    def _advance(self):
        while True:
            self._start_idle_machines()
            self._track_occupancy()
            if all(self.job_done):
                return
            if self._any_legal():
                return
            ends = {m: e for m, o in enumerate(self.running)
                    if o is not None and (e := self.end[o]) is not None}
            events = list(ends.values()) + self._pending_starts(held=False)
            if events:
                self.t = min(events)
                for m, e in ends.items():
                    if e <= self.t:
                        self._finish(m)
                continue
            # nothing processing and no legal routing: circular blocking
            if self._swap_cycle() is not None:
                self._swap()
                continue
            # every cycle needs a machine that is down: wait for a window to end
            waits = self._pending_starts(held=True)
            assert waits, f"deadlock without a swap cycle or a window to wait for at t={self.t}"
            self.t = min(waits)
            self.num_waits += 1

    def _swap_cycle(self):
        """As in the blocking env, but only machines that can start their queued part now can
        take part: that start is what frees the slot the predecessor's part moves into."""
        windows = self._windows()
        saved = self.held
        self.held = [o if o is not None and (not self.in_queue[a] or
                                             self._earliest(a, self.in_queue[a][0], self.t, windows) <= self.t)
                     else None for a, o in enumerate(saved)]
        try:
            return super()._swap_cycle()
        finally:
            self.held = saved

    def _mask_deadlocking_actions(self, legal):
        """Deadlock look-ahead on the information view: breakdowns that have not started are
        ignored, one in progress ends at its estimate, and a running part ends when the view
        says so (its true end may include a future breakdown)."""
        if not self.avoid_deadlocks:
            return
        snap = self._snapshot()
        self._lookahead_view = self._view()
        for m, o in enumerate(self.running):
            if o is not None:
                self.end[o] = max(self.t, resume_end(self.start[o], float(self.proc[o, m]), self._lookahead_view[m]))
        self._lookahead = True
        try:
            super()._mask_deadlocking_actions(legal)
        finally:
            self._lookahead = False
            self._restore(snap)

    def _record_schedule(self):
        super()._record_schedule()
        for rec in self.schedule:
            p = float(self.proc[rec["operation"], rec["machine"]])
            rec["interrupted"] = float(rec["end"] - rec["start"] - p)
        self.num_breakdowns = sum(1 for ws in self.true_windows for s, _, k in ws
                                  if k == BREAKDOWN and s < self.mk)

    # ── projections (what the policy may know) ─────────────────────────────────

    def _known_end(self, o):
        if self.status[o] != RUN:
            return self.end[o]
        m = self.assigned[o]
        p = float(self.proc[o, m])
        if not self._aware:
            return max(self.t, self.start[o] + p)
        return max(self.t, resume_end(self.start[o], p, self._view()[m]))

    def _project(self, aware):
        """End of every routed operation and free time of every machine if each machine keeps
        processing its input buffer in FIFO order, with (aware) or without the known windows."""
        view = self._view()
        proj_end = {}
        proj_free = [self.t] * self.num_machines
        for m in range(self.num_machines):
            free = self.t
            o = self.running[m]
            if o is not None:
                p = float(self.proc[o, m])
                free = (max(self.t, resume_end(self.start[o], p, view[m])) if aware
                        else max(self.t, self.start[o] + p))
                proj_end[o] = free
            for o in self.in_queue[m]:
                p = float(self.proc[o, m])
                if aware:
                    free = resume_end(earliest_start(free, p, view[m]), p, view[m])
                else:
                    free += p
                proj_end[o] = free
            proj_free[m] = earliest_start(free, 0.0, view[m]) if aware else free
        return proj_end, proj_free

    def _projected_times(self):
        return self._project(self._aware)

    # ── state ──────────────────────────────────────────────────────────────────

    def _build_state(self):
        self._view_cache = None
        super()._build_state()
        if self.unavail_repr == "feat":
            self._add_window_features(self.state)
        elif self.unavail_repr == "dummy":
            self._add_dummy_operations(self.state)

    def calculate_mask(self, legal=None):
        """Mask ranking on window-blind start times (identical for every representation);
        the action-edge features then get the window-aware ones in "feat"/"dummy"."""
        view = self._view()
        _, free_aware = self._project(True)
        _, free_blind = self._project(False)
        blind = torch.full_like(self.current_job_proc, SENTINEL)
        aware = torch.full_like(self.current_job_proc, SENTINEL)
        for j, m in self.current_job_proc.nonzero().tolist():
            p = float(self.current_job_proc[j, m])
            blind[j, m] = max(self.t, free_blind[m])
            aware[j, m] = earliest_start(max(self.t, free_aware[m]), p, view[m])
        self.job_start_machines = blind
        super().calculate_mask(legal)
        # earliest completion time with and without the windows (the rules differ only there,
        # whatever mask_option ranks the mask by)
        self._expert_matrix = aware + self.current_job_proc
        self._blind_expert_matrix = blind + self.current_job_proc
        if self._aware:
            self.job_start_machines = aware
            self._refresh_jm_edges()

    def _ect_action(self, matrix):
        edge_index = self.state['machine', 'exec', 'job'].edge_index
        values = matrix[edge_index[1], edge_index[0]].clone()
        values[self.state['machine', 'exec', 'job'].mask] = float("inf")
        return int(torch.argmin(values).item())

    def expert_action(self):
        """Window-aware earliest completion time (the imitation teacher, the same for every
        representation)."""
        return self._ect_action(self._expert_matrix)

    def blind_expert_action(self):
        """Window-blind earliest completion time (baseline of the calibration)."""
        return self._ect_action(self._blind_expert_matrix)

    def _window_information(self):
        """The unavailability information, independent of how it is placed in the graph.
        Per machine: (current, next, reliability), where current / next is None or
        [planned, breakdown, start_offset / H, remaining_length / H] of the window containing t
        / of the next known window starting within H, and reliability is
        [P(failure within H | age), age / eta] (zeros when breakdowns are off)."""
        t, H = self.t, self.horizon
        view = self._view()
        info = []
        for m in range(self.num_machines):
            cur = window_at(view[m], t)
            nxt = next_window(view[m], t)
            if nxt is not None and nxt[0] - t > H:
                nxt = None

            def encode(w):
                s, e, kind = w
                begin = max(s, t)
                return [float(kind == SCHEDULED), float(kind == BREAKDOWN),
                        _clip01((begin - t) / H), _clip01((e - begin) / H)]

            reliability = [0.0, 0.0]
            if self.eta is not None:
                age = 0.0 if cur is not None else t - last_renewal(view[m], t)
                reliability = [p_fail(age, self.eta[m], self.beta, H), _clip01(age / self.eta[m])]
            info.append((None if cur is None else encode(cur), None if nxt is None else encode(nxt), reliability))
        return info

    def _append_machine_cols(self, data, cols):
        x = data['machine'].x
        extra = torch.zeros((x.shape[0], cols.shape[1]))
        extra[:self.num_machines] = cols  # dummy machines of ojmd (rows M..3M-1) get zeros
        data['machine'].x = torch.cat([x, extra], dim=1)

    def _add_window_features(self, data):
        """Representation A: [planned_downtime, breakdown_status, availability_period,
        next_window_length, remaining_downtime, p_fail_H, age/eta] on each machine."""
        rows = []
        for cur, nxt, rel in self._window_information():
            rows.append([
                cur[0] if cur else 0.0,        # planned downtime now
                cur[1] if cur else 0.0,        # breakdown now
                nxt[2] if nxt else 1.0,        # availability period: time to the next window / H
                nxt[3] if nxt else 0.0,        # its length / H
                cur[3] if cur else 0.0,        # remaining downtime / H (estimated for a breakdown)
                *rel,
            ])
        self._append_machine_cols(data, torch.tensor(rows, dtype=torch.float))

    def _add_dummy_operations(self, data):
        """Representation B: one 'operation' node per current / next window, pre-assigned to its
        machine. Real operations get zeros in the 5 extra columns."""
        info = self._window_information()
        self._append_machine_cols(data, torch.tensor([rel for _, _, rel in info], dtype=torch.float))
        n_real = data['operation'].x.shape[0]
        dummy_x, pairs, attrs = [], [], []
        for m, (cur, nxt, _) in enumerate(info):
            for w in (cur, nxt):
                if w is None:
                    continue
                planned, breakdown, offset, length = w
                pairs.append([n_real + len(dummy_x), m])
                attrs.append([length * self.horizon, 1.0, 0.0, 1.0, 0.0])
                dummy_x.append([0.0, 0.0, 1.0, offset, length, planned, breakdown])
        x = torch.cat([data['operation'].x, torch.zeros((n_real, self.DUMMY_OP_COLS))], dim=1)
        if dummy_x:
            x = torch.cat([x, torch.tensor(dummy_x, dtype=torch.float)], dim=0)
        data['operation'].x = x
        data['operation'].op_ref = torch.cat([data['operation'].op_ref,
                                              torch.full((len(dummy_x),), -1, dtype=torch.long)])
        if not pairs:
            return
        index, attr = _edges(pairs), _attrs(attrs)
        for edge_type, idx in ((('operation', 'exec', 'machine'), index), (('machine', 'exec', 'operation'), index.flip(0))):
            data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, idx], dim=1)
            data[edge_type].edge_attr = torch.cat([data[edge_type].edge_attr, attr.clone()], dim=0)
        loops = torch.arange(n_real, n_real + len(dummy_x)).repeat(2, 1)
        prec = data['operation', 'prec', 'operation'].edge_index
        data['operation', 'prec', 'operation'].edge_index = torch.cat([prec, loops], dim=1)

    # ── normalisation ──────────────────────────────────────────────────────────

    def normalize_state(self, state):
        """The unavailability columns are already in [0, 1] (times / H, flags, probabilities):
        they are set aside, the blocking normalisation runs on the rest, and they come back as
        2x - 1 (a fixed scale). With dummy operations, the operation min-max is redone on the
        real operations only, so the dummies' zeros do not shift them."""
        k_m = self.MACHINE_COLS[self.unavail_repr]
        k_o = self.DUMMY_OP_COLS if self.unavail_repr == "dummy" else 0
        if not k_m and not k_o:
            return super().normalize_state(state)
        state = state.clone()
        mx = state['machine'].x
        m_extra = mx[:, -k_m:]
        state['machine'].x = mx[:, :-k_m]
        ox = state['operation'].x
        if k_o:
            state['operation'].x = ox[:, :2]
        out = super().normalize_state(state)
        if k_o:
            o_extra = ox[:, 2:]
            dummy = o_extra[:, 0] > 0.5
            base = ox[:, :2]
            norm = torch.full_like(base, -1.0)
            if (~dummy).any():
                real = base[~dummy]
                mins, maxs = real.min(dim=0, keepdim=True).values, real.max(dim=0, keepdim=True).values
                norm[~dummy] = 2 * (real - mins) / (maxs - mins + 1e-7) - 1
            out['operation'].x = torch.cat([norm, 2 * o_extra - 1], dim=1).float()
        out['machine'].x = torch.cat([out['machine'].x, 2 * m_extra.clamp(0, 1) - 1], dim=1).float()
        return out


class FJSPEnvUnavailFeatures(FJSPEnvUnavailability):
    """Unavailability as machine features (representation "ojmb_uf")."""

    def __init__(self, instances, mask_option=3, sel_k=5, jm_design="edges", in_cap=IN_CAP, out_cap=OUT_CAP,
                 blocking_repr="node", avoid_deadlocks=True, unavail_repr="feat", unavail_mode=None, scenario_seed=None):
        super().__init__(instances, mask_option, sel_k, jm_design, in_cap, out_cap, blocking_repr, avoid_deadlocks,
                         unavail_repr, unavail_mode, scenario_seed)


class FJSPEnvUnavailDummyOps(FJSPEnvUnavailability):
    """Unavailability as dummy operations (representation "ojmb_uo")."""

    def __init__(self, instances, mask_option=3, sel_k=5, jm_design="edges", in_cap=IN_CAP, out_cap=OUT_CAP,
                 blocking_repr="node", avoid_deadlocks=True, unavail_repr="dummy", unavail_mode=None, scenario_seed=None):
        super().__init__(instances, mask_option, sel_k, jm_design, in_cap, out_cap, blocking_repr, avoid_deadlocks,
                         unavail_repr, unavail_mode, scenario_seed)


class FJSPEnvUnavailBlind(FJSPEnvUnavailability):
    """Window-blind graph on the same dynamics (representation "ojmb_u0"): the control."""

    def __init__(self, instances, mask_option=3, sel_k=5, jm_design="edges", in_cap=IN_CAP, out_cap=OUT_CAP,
                 blocking_repr="node", avoid_deadlocks=True, unavail_repr="none", unavail_mode=None, scenario_seed=None):
        super().__init__(instances, mask_option, sel_k, jm_design, in_cap, out_cap, blocking_repr, avoid_deadlocks,
                         unavail_repr, unavail_mode, scenario_seed)
