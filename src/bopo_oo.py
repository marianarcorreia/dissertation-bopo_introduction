import copy
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

    def _op_features(self, op_id):
        op_times = [t for t in self.operations[op_id] if t > 0]
        mean_t = float(np.mean(op_times)) if op_times else 0.0
        min_t = float(np.min(op_times)) if op_times else 0.0
        max_t = float(np.max(op_times)) if op_times else 1.0
        return mean_t, min_t, max_t

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
        edges = []
        edge_attr = []

        for i, src in enumerate(available_ops):
            src_mean, src_min, src_max = self._op_features(src)
            for j, dst in enumerate(available_ops):
                if i == j:
                    continue
                dst_mean, dst_min, dst_max = self._op_features(dst)
                overlap = 0
                for m_id, t_src in enumerate(self.operations[src]):
                    if t_src > 0 and self.operations[dst][m_id] > 0:
                        overlap = 1
                        break
                edges.append([src, dst])
                edge_attr.append([
                    float(overlap),
                    src_mean,
                    dst_mean,
                    abs(src_min - dst_min),
                    abs(src_max - dst_max),
                ])

        if edges:
            self.state["operation", "disj", "operation"].edge_index = torch.LongTensor(edges).T
            self.state["operation", "disj", "operation"].edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            self.state["operation", "disj", "operation"].edge_index = torch.empty((2, 0), dtype=torch.long)
            self.state["operation", "disj", "operation"].edge_attr = torch.empty((0, 5), dtype=torch.float)

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
        self.state: Any = copy.deepcopy(self.data)

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

    def calculate_next_state(self):
        self.state["operation"].x[:, 0] = 0
        for op_id in self.current_operations:
            if op_id != -1:
                self.state["operation"].x[op_id, 0] = 1
        for op_id in range(self.num_operations):
            self.state["operation"].x[op_id, 1] = self.all_pendings[op_id]
            self.state["operation"].x[op_id, 2] = 1.0 if self.scheduled_mask[op_id] else 0.0

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
        state = copy.deepcopy(state)

        for i in range(state["operation"].x.shape[1]):
            values = state["operation"].x[:, i]
            state["operation"].x[:, i] = (2 * (values - values.min()) / (values.max() - values.min() + 1e-7) - 1).float()

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
