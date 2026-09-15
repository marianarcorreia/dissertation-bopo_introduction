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


class FJSPEnvMO(gym.Env):
    def __init__(self, instances, mask_option=3, sel_k=5):
        super(FJSPEnvMO, self).__init__()
        if isinstance(instances, dict):
            instances = [instances]
        self.instances = instances
        self.current_instance = 0
        self.mask_option = mask_option
        self.sel_k = sel_k
        self.mk = 0.0
        _dbg(1, f"FJSPEnvMO created | instances={len(self.instances)} | mask_option={mask_option} | sel_k={sel_k}")

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

        self.num_features_oper = 2
        self.num_features_mach = 3

        self.data = HeteroData()
        self.data["operation"].x = torch.zeros((self.num_operations, self.num_features_oper), dtype=torch.float)
        self.data["machine"].x = torch.zeros((self.num_machines, self.num_features_mach), dtype=torch.float)

        precedence_edges = []
        for job_ops in self.jobs:
            for i in range(len(job_ops) - 1):
                precedence_edges.append([job_ops[i], job_ops[i]])
                precedence_edges.append([job_ops[i], job_ops[i + 1]])
            precedence_edges.append([job_ops[-1], job_ops[-1]])
        self.data["operation", "prec", "operation"].edge_index = torch.LongTensor(precedence_edges).T

        machine_edges = []
        for i in range(self.num_machines):
            for j in range(self.num_machines):
                machine_edges.append([i, j])

        self.data["machine", "listens", "machine"].edge_index = torch.LongTensor(machine_edges).T
        self.data["machine", "listens", "machine"].edge_attr = torch.zeros((len(machine_edges), 5), dtype=torch.float) #it is 5 to maintain the same number of features as the exec edges, even if they are not used
        
        self.all_pendings = []
        for job_ops in self.jobs:
            pending = []
            for op_id in reversed(job_ops):
                op_times = np.array(self.operations[op_id])
                valid = op_times[np.where(op_times != 0)]
                mean_t = float(np.mean(valid)) if len(valid) > 0 else 0.0
                pending.append(mean_t if not pending else mean_t + pending[-1])
            self.all_pendings.extend(list(reversed(pending)))

        op_machine_edges = []
        edge_features = []
        for op_id in range(self.num_operations):
            op = self.operations[op_id]
            op_sum = float(np.sum(op))
            for machine_id, proc_time in enumerate(op):
                if proc_time != 0:
                    op_machine_edges.append([op_id, machine_id])
                    ratio_sum = proc_time / op_sum if op_sum != 0 else 0.0
                    ratio_pending = proc_time / self.all_pendings[op_id] if self.all_pendings[op_id] != 0 else 0.0
                    edge_features.append([proc_time, ratio_sum, ratio_pending, 0, 0])

        self.data["operation", "exec", "machine"].edge_index = torch.LongTensor(op_machine_edges).T
        self.data["operation", "exec", "machine"].edge_attr = torch.Tensor(edge_features)

        for op_id in range(self.num_operations):
            self.data["operation"].x[op_id, 1] = self.all_pendings[op_id]

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

        self.job_start_machines = torch.full((self.num_jobs, self.num_machines), 10000.0)
        self.current_job_proc = torch.zeros((self.num_jobs, self.num_machines))

        self.current_operations = [self.jobs[job_id][0] for job_id in range(self.num_jobs)]
        self.operations_ends = [0.0] * self.num_jobs
        self.machines_occupations = [0.0] * self.num_machines

        action_edges = []
        action_features = []
        for job_id in range(self.num_jobs):
            first_op_id = self.jobs[job_id][0]
            op = self.operations[first_op_id]
            op_sum = float(np.sum(op))
            eligible = [(m,t) for m, t in enumerate(op) if t != 0]
            total_gap = sum(t for _, t in eligible)
            den = total_gap if total_gap != 0 else 1.0
            for machine_id, proc_time in eligible:
                action_edges.append([machine_id, first_op_id])
                ratio = proc_time / op_sum if op_sum != 0 else 0.0
                gap_norm = proc_time / den  
                action_features.append([proc_time, ratio, ratio, proc_time, 0])
                self.job_start_machines[job_id, machine_id] = 0
                self.current_job_proc[job_id, machine_id] = float(proc_time)

        self.state["machine", "exec", "operation"].edge_index = torch.LongTensor(action_edges).T
        self.state["machine", "exec", "operation"].edge_attr = torch.Tensor(action_features)

        self.calculate_next_state()
        self.calculate_mask()

        _dbg(1, f"reset() done | jobs={self.num_jobs} | ops={self.num_operations} | machines={self.num_machines}")
        return self.state

    def calculate_mask(self):
        if self.mask_option == 0:
            mask_matrix = self.job_start_machines
        else:
            mask_matrix = self.job_start_machines + self.current_job_proc

        flat_size = int(mask_matrix.numel())
        k = max(1, min(int(self.sel_k), flat_size))
        smallest = torch.unique(torch.topk(torch.flatten(mask_matrix), k=k, largest=False, dim=0).values)

        candidate_pairs = []
        for value in smallest:
            job_indices, machine_indices = (mask_matrix == value).nonzero(as_tuple=True)
            for job_idx, machine_idx in zip(job_indices.tolist(), machine_indices.tolist()):
                op_id = int(self.current_operations[job_idx])
                if op_id != -1:
                    candidate_pairs.append((int(machine_idx), op_id))

        indexes = []
        for pair in candidate_pairs:
            pair_tensor = torch.tensor(pair, dtype=torch.long)
            equal_pairs = self.state["machine", "exec", "operation"].edge_index.T == pair_tensor
            both_equal = np.logical_and(equal_pairs[:, 0], equal_pairs[:, 1])
            indexes.extend([idx for idx, value in enumerate(both_equal) if value == 1])

        result = [True] * self.state["machine", "exec", "operation"].edge_index.shape[1]
        for idx in indexes:
            result[idx] = False

        self.state["machine", "exec", "operation"].mask = torch.BoolTensor(result)

    def calculate_next_state(self):
        self.state["machine"].x[:, 2] = self.state["machine"].x[:, 0] - torch.min(self.state["machine"].x[:, 0])

        self.state["operation"].x[:, 0] = 0
        for op_id in self.current_operations:
            if op_id != -1:
                self.state["operation"].x[op_id, 0] = 1

        for machine_id in range(self.num_machines):
            op_to_machine_mask = self.state["operation", "exec", "machine"].edge_index[1, :] == machine_id

            machine_to_op_mask = self.state["machine", "exec", "operation"].edge_index[0, :] == machine_id

    def step(self, action):
        self.num_steps += 1

        action_pair = self.state["machine", "exec", "operation"].edge_index[:, action]
        sel_operation = int(action_pair[1])
        sel_machine = int(action_pair[0])
        sel_job = int(self.operation_to_job[sel_operation])

        keep_mask = self.state["machine", "exec", "operation"].edge_index[1, :] != sel_operation
        self.state["machine", "exec", "operation"].edge_index = self.state["machine", "exec", "operation"].edge_index[:, keep_mask]
        self.state["machine", "exec", "operation"].edge_attr = self.state["machine", "exec", "operation"].edge_attr[keep_mask]

        prev_makespan = float(torch.max(self.state["machine"].x[:, 0]))

        start_time = max(float(self.state["machine"].x[sel_machine, 0]), float(self.operations_ends[sel_job]))
        proc_time = float(self.operations[sel_operation][sel_machine])
        final_time = start_time + proc_time

        self.state["machine"].x[sel_machine, 0] = final_time
        self.job_start_machines[self.job_start_machines[:, sel_machine] < final_time, sel_machine] = final_time
        self.machines_occupations[sel_machine] += proc_time
        self.state["machine"].x[sel_machine, 1] = self.machines_occupations[sel_machine] / final_time

        self.operations_ends[sel_job] = final_time
        self.job_start_machines[sel_job, :] = 10000
        self.current_job_proc[sel_job, :] = 0

        job_ops = self.jobs[sel_job]
        curr_index = job_ops.index(sel_operation)
        if curr_index == len(job_ops) - 1:
            self.current_operations[sel_job] = -1
        else:
            next_op_id = job_ops[curr_index + 1]
            self.current_operations[sel_job] = next_op_id

            op = np.array(self.operations[next_op_id])
            op_sum = float(np.sum(op))
            total_gap = 0.0
            new_edges = []
            new_features = []
            for machine_id, proc in enumerate(op):
                if proc != 0:
                    gap_value = float(proc) + max(float(self.operations_ends[sel_job]) - float(self.state["machine"].x[machine_id, 0]), 0.0)
                    total_gap += gap_value
                    new_edges.append([machine_id, next_op_id])
                    ratio = float(proc) / op_sum if op_sum != 0 else 0.0
                    span = float(proc) + max(float(self.operations_ends[sel_job]), float(self.state["machine"].x[machine_id, 0]))
                    new_features.append([gap_value, ratio, span])
                    self.job_start_machines[sel_job, machine_id] = max(float(self.operations_ends[sel_job]), float(self.state["machine"].x[machine_id, 0]))
                    self.current_job_proc[sel_job, machine_id] = float(proc)

            den = total_gap if total_gap != 0 else 1.0
            for row in new_features:
                row.append(row[0] / den)
                row.append(0)

            if new_edges:
                self.state["machine", "exec", "operation"].edge_index = torch.concat(
                    [self.state["machine", "exec", "operation"].edge_index, torch.LongTensor(new_edges).T],
                    dim=1,
                )
                self.state["machine", "exec", "operation"].edge_attr = torch.concat(
                    [self.state["machine", "exec", "operation"].edge_attr, torch.Tensor(new_features)]
                )

        reward = prev_makespan - float(torch.max(self.state["machine"].x[:, 0]))

        done = all(op_id == -1 for op_id in self.current_operations)
        if done:
            self.mk = round(float(torch.max(self.state["machine"].x[:, 0])), 2)
            _dbg(1, f"Episode done | makespan={self.mk} | steps={self.num_steps}")
            return self.state, reward, True, {"current_machine": sel_machine}

        op_exec_mask = self.state["operation", "exec", "machine"].edge_index[0, :] != sel_operation
        self.state["operation", "exec", "machine"].edge_index = self.state["operation", "exec", "machine"].edge_index[:, op_exec_mask]
        self.state["operation", "exec", "machine"].edge_attr = self.state["operation", "exec", "machine"].edge_attr[op_exec_mask]

        op_prec_mask = self.state["operation", "prec", "operation"].edge_index[1, :] != sel_operation
        self.state["operation", "prec", "operation"].edge_index = self.state["operation", "prec", "operation"].edge_index[:, op_prec_mask]

        self.calculate_next_state()
        self.calculate_mask()

        total_reward = reward
        done = False
        valid_actions = int((~self.state["machine", "exec", "operation"].mask).sum().item())
        if valid_actions == 1:
            self.state, extra_reward, done, _ = self.step(self.sample())
            total_reward += extra_reward

        return self.state, total_reward, done, {"current_machine": sel_machine}

    def sample(self):
        valid = [
            idx
            for idx in range(len(self.state["machine", "exec", "operation"].mask))
            if not self.state["machine", "exec", "operation"].mask[idx]
        ]
        return random.choice(valid)

    def normalize_state(self, state):
        state = copy.deepcopy(state)

        for i in range(state["operation"].x.shape[1]):
            values = state["operation"].x[:, i]
            state["operation"].x[:, i] = (2 * (values - values.min()) / (values.max() - values.min() + 1e-7) - 1).float()

        for i in range(state["machine"].x.shape[1]):
            values = state["machine"].x[:, i]
            state["machine"].x[:, i] = (2 * (values - values.min()) / (values.max() - values.min() + 1e-7) - 1).float()

        for edge_type in [
            ("operation", "exec", "machine"),
            ("machine", "exec", "operation"),
        ]:
            if state[edge_type].edge_attr.numel() == 0:
                continue
            attrs = state[edge_type].edge_attr
            min_vals = attrs.min(dim=0).values
            max_vals = attrs.max(dim=0).values
            state[edge_type].edge_attr = (2 * (attrs - min_vals) / (max_vals - min_vals + 1e-7) - 1).float()

        return state
