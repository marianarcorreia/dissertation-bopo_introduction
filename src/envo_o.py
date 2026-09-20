import importlib
import os
import random
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

gym = importlib.import_module("gymnasium")

_DBG = int(os.environ.get("FJSP_DEBUG", "0"))


def _dbg(level, *args, **kwargs):
    if _DBG >= level:
        print("[ENV]", *args, **kwargs)


class FJSPEnvOO(gym.Env):
    def __init__(self, instances, mask_option=3, sel_k=5):
        super(FJSPEnvOO, self).__init__()
        if isinstance(instances, dict):
            instances = [instances]
        self.instances = instances
        self.current_instance = 0
        self.mask_option = mask_option
        self.sel_k = sel_k
        self.mk = 0.0
        _dbg(1, f"FJSPEnvOO created | instances={len(self.instances)} | mask_option={mask_option} | sel_k={sel_k}")

    def get_prev_op(self, o_id):
        for job in self.jobs:
            if o_id in job:
                idx = job.index(o_id)
                return None if idx == 0 else job[idx - 1]
        return None

    def generate_instance(self, instance):
        jobs, operations = instance["jobs"], instance["operations"]

        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.operations = operations
        self.num_operations = len(operations)
        self.num_machines = len(instance["operations"][0])

        self.operation_to_job = {}
        for job_id, job_ops in enumerate(self.jobs):
            for op_id in job_ops:
                self.operation_to_job[op_id] = job_id

        self.num_features_oper = 3

        self.data = HeteroData()
        self.data["operation"].x = torch.zeros((self.num_operations, self.num_features_oper), dtype=torch.float)

        precedence_edges = []
        for job_ops in self.jobs:
            for i in range(len(job_ops) - 1):
                precedence_edges.append([job_ops[i], job_ops[i + 1]])
        if precedence_edges:
            self.data["operation", "prec", "operation"].edge_index = torch.LongTensor(precedence_edges).T
            self.data["operation", "prec", "operation"].edge_attr = torch.ones((len(precedence_edges), 5), dtype=torch.float)
        else:
            self.data["operation", "prec", "operation"].edge_index = torch.empty((2, 0), dtype=torch.long)
            self.data["operation", "prec", "operation"].edge_attr = torch.empty((0, 5), dtype=torch.float)
        
        self.all_pendings = []
        for job_ops in self.jobs:
            pending = []
            for op_id in reversed(job_ops):
                op_times = np.array(self.operations[op_id])
                valid = op_times[np.where(op_times != 0)]
                mean_t = float(np.mean(valid)) if len(valid) > 0 else 0.0
                pending.append(mean_t if not pending else mean_t + pending[-1])
            self.all_pendings.extend(list(reversed(pending)))

        self.data["operation", "disj", "operation"].edge_index = torch.empty((2, 0), dtype=torch.long)
        self.data["operation", "disj", "operation"].edge_attr = torch.empty((0, 5), dtype=torch.float)

        for op_id in range(self.num_operations):
            self.data["operation"].x[op_id, 1] = self.all_pendings[op_id]
            self.data["operation"].x[op_id, 2] = 0.0

        # Precompute per-operation static features and machine-compatibility, used every
        # step by _build_dynamic_disj_edges(). These only depend on self.operations (fixed
        # for the whole episode), so computing them once per instance instead of on every
        # (src, dst) pair of every decision step avoids an O(rounds * B * k^2) blow-up of
        # tiny numpy calls that used to dominate BOPO's per-update wall-clock time.
        ops_arr = np.asarray(self.operations, dtype=np.float64)
        self._op_compat = ops_arr > 0
        valid_rows = self._op_compat.any(axis=1)
        masked = np.where(self._op_compat, ops_arr, np.nan)
        self._op_mean = np.zeros(self.num_operations, dtype=np.float32)
        self._op_min = np.zeros(self.num_operations, dtype=np.float32)
        self._op_max = np.ones(self.num_operations, dtype=np.float32)
        if valid_rows.any():
            self._op_mean[valid_rows] = np.nanmean(masked[valid_rows], axis=1)
            self._op_min[valid_rows] = np.nanmin(masked[valid_rows], axis=1)
            self._op_max[valid_rows] = np.nanmax(masked[valid_rows], axis=1)

    def _select_machine_for_operation(self, op_id):
        job_id = self.operation_to_job[op_id]
        earliest_start = float(self.operations_ends[job_id])

        best_machine = None
        best_proc = None
        best_end = float("inf")

        for machine_id, proc_time in enumerate(self.operations[op_id]):
            if proc_time <= 0:
                continue
            start_time = max(float(self.machine_available[machine_id]), earliest_start)
            end_time = start_time + float(proc_time)
            if end_time < best_end:
                best_end = end_time
                best_machine = machine_id
                best_proc = float(proc_time)

        if best_machine is None:
            raise RuntimeError(f"Operation {op_id} has no eligible machine")

        return best_machine, best_proc, best_end

    def _build_dynamic_disj_edges(self):
        available_ops = [op_id for op_id in self.current_operations if op_id != -1 and not self.scheduled_mask[op_id]]
        k = len(available_ops)

        if k <= 1:
            self.state["operation", "disj", "operation"].edge_index = torch.empty((2, 0), dtype=torch.long)
            self.state["operation", "disj", "operation"].edge_attr = torch.empty((0, 5), dtype=torch.float)
            return

        idx = np.asarray(available_ops, dtype=np.int64)
        means = self._op_mean[idx]
        mins = self._op_min[idx]
        maxs = self._op_max[idx]
        compat = self._op_compat[idx]  # (k, num_machines)
        overlap_matrix = compat @ compat.T  # (k, k), >0 where src/dst share a machine

        off_diag = ~np.eye(k, dtype=bool)
        ii, jj = np.nonzero(off_diag)

        edges = np.stack([idx[ii], idx[jj]], axis=0)
        edge_attr = np.stack([
            (overlap_matrix[ii, jj] > 0).astype(np.float32),
            means[ii].astype(np.float32),
            means[jj].astype(np.float32),
            np.abs(mins[ii] - mins[jj]).astype(np.float32),
            np.abs(maxs[ii] - maxs[jj]).astype(np.float32),
        ], axis=1)

        self.state["operation", "disj", "operation"].edge_index = torch.from_numpy(edges).long()
        self.state["operation", "disj", "operation"].edge_attr = torch.from_numpy(edge_attr).float()

    def reset(self, sel_index=None):
        idx = sel_index if sel_index is not None else self.current_instance
        _dbg(1, f"reset() called | instance_index={idx}")

        if sel_index is None:
            self.generate_instance(self.instances[self.current_instance])
            self.current_instance = (self.current_instance + 1) % len(self.instances)
        else:
            self.generate_instance(self.instances[sel_index])

        self.num_steps = 0
        self.mk = 0.0
        self.state: Any = self.data.clone()

        self.current_operations = [self.jobs[job_id][0] for job_id in range(self.num_jobs)]
        self.operations_ends = [0.0] * self.num_jobs
        self.machine_available = [0.0] * self.num_machines
        self.scheduled_mask = [False] * self.num_operations
        self.selected_machine = [-1] * self.num_operations

        self.calculate_next_state()
        self.calculate_mask()

        _dbg(1, f"reset() done | jobs={self.num_jobs} | ops={self.num_operations} | machines={self.num_machines}")
        return self.state

    def calculate_mask(self):
        mask = [True] * self.num_operations
        for op_id in self.current_operations:
            if op_id != -1 and not self.scheduled_mask[op_id]:
                mask[op_id] = False
        self.state["operation"].mask = torch.BoolTensor(mask)

    def expert_action(self):
        """Most-Work-Remaining (MWKR) dispatch heuristic: among the currently available
        operations, pick the one whose job has the most total remaining processing time
        (self.all_pendings). MWKR is a standard job-shop priority rule (favors jobs that
        would otherwise fall behind) and is used as the teacher for BOPO's warm-start
        behavior-cloning phase (src/bopo_utils.py:run_behavior_cloning), which needs a
        non-uniform initial policy to break the self-rewarding cold-start."""
        candidates = [
            op_id for op_id in self.current_operations
            if op_id != -1 and not self.scheduled_mask[op_id]
        ]
        return max(candidates, key=lambda op_id: self.all_pendings[op_id])

    def calculate_next_state(self):
        # x[:,1] (all_pendings) is static per instance and already set once in
        # generate_instance(); x[:,2] (scheduled flag) is updated incrementally in step()
        # the moment an operation is scheduled. Looping over every operation here on every
        # single decision step to re-write values that either never change or changed for
        # exactly one operation was the single most expensive line in the env (profiled at
        # ~30% of BOPO's per-update wall-clock time for no behavioral benefit).
        self.state["operation"].x[:, 0] = 0
        for op_id in self.current_operations:
            if op_id != -1:
                self.state["operation"].x[op_id, 0] = 1

        self._build_dynamic_disj_edges()

    def step(self, action):
        self.num_steps += 1

        if isinstance(action, torch.Tensor):
            sel_operation = int(action.item())
        else:
            sel_operation = int(action)

        if sel_operation < 0 or sel_operation >= self.num_operations:
            raise RuntimeError(f"Invalid action operation id: {sel_operation}")
        if self.state["operation"].mask[sel_operation]:
            raise RuntimeError(f"Invalid masked action for operation id: {sel_operation}")

        sel_job = int(self.operation_to_job[sel_operation])

        prev_makespan = max(self.machine_available) if self.machine_available else 0.0

        sel_machine, proc_time, final_time = self._select_machine_for_operation(sel_operation)
        self.machine_available[sel_machine] = final_time
        self.selected_machine[sel_operation] = sel_machine

        self.operations_ends[sel_job] = final_time
        self.scheduled_mask[sel_operation] = True
        self.state["operation"].x[sel_operation, 2] = 1.0

        job_ops = self.jobs[sel_job]
        curr_index = job_ops.index(sel_operation)
        if curr_index == len(job_ops) - 1:
            self.current_operations[sel_job] = -1
        else:
            next_op_id = job_ops[curr_index + 1]
            self.current_operations[sel_job] = next_op_id

        reward = prev_makespan - max(self.machine_available)

        done = all(self.scheduled_mask)
        if done:
            self.mk = round(max(self.machine_available), 2)
            _dbg(1, f"Episode done | makespan={self.mk} | steps={self.num_steps}")
            return self.state, reward, True, {"current_machine": sel_machine}

        self.calculate_next_state()
        self.calculate_mask()

        total_reward = reward
        done = False
        valid_actions = int((~self.state["operation"].mask).sum().item())
        if valid_actions == 1:
            self.state, extra_reward, done, _ = self.step(self.sample())
            total_reward += extra_reward

        return self.state, total_reward, done, {"current_machine": sel_machine}

    def sample(self):
        valid = [
            idx
            for idx in range(len(self.state["operation"].mask))
            if not self.state["operation"].mask[idx]
        ]
        return random.choice(valid)

    def normalize_state(self, state):
        # .clone() (PyG's per-tensor clone) instead of copy.deepcopy(): this runs every
        # decision round for every active rollout in BOPO's sample_group, and generic
        # deepcopy's recursive python traversal of the whole HeteroData object graph is
        # ~4x slower than cloning the stored tensors directly - it was a major chunk of
        # BOPO's per-update wall-clock time.
        state = state.clone()

        x = state["operation"].x
        mins = x.min(dim=0, keepdim=True).values
        maxs = x.max(dim=0, keepdim=True).values
        state["operation"].x = (2 * (x - mins) / (maxs - mins + 1e-7) - 1).float()

        for edge_type in [
            ("operation", "prec", "operation"),
            ("operation", "disj", "operation"),
        ]:
            if edge_type not in state.edge_types:
                continue
            if state[edge_type].edge_attr.numel() == 0:
                continue
            attrs = state[edge_type].edge_attr
            min_vals = attrs.min(dim=0).values
            max_vals = attrs.max(dim=0).values
            state[edge_type].edge_attr = (2 * (attrs - min_vals) / (max_vals - min_vals + 1e-7) - 1).float()

        return state
