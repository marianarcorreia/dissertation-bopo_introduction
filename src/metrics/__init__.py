"""Evaluation metrics shared by every representation (and, once merged, every constraint branch).

schedule        makespan, lower bound, relative error, scheduling score,
                constraint violation rate per step / feasibility (extensible Constraint checks)
efficiency      memory used (GPU peak, process RSS) and inference time, model size
representation  expressiveness and heterophily of the learned representation
statistics      sample size, 95% CIs, paired Wilcoxon tests, effect sizes
references      CP-SAT reference makespans (load / solve missing)
evaluate        runs a trained policy and computes all of the above per instance
"""
from .schedule import (FJSP_CONSTRAINTS, Constraint, check_schedule, lower_bound, makespan,
                       relative_error, schedule_metrics, scheduling_score)
from .efficiency import MemoryTracker, model_size
from .representation import RepresentationProbe
from .statistics import compare, describe, required_sample_size
from .evaluate import evaluate_agent, summarize

__all__ = [
    "FJSP_CONSTRAINTS", "Constraint", "check_schedule", "lower_bound", "makespan",
    "relative_error", "schedule_metrics", "scheduling_score",
    "MemoryTracker", "model_size", "RepresentationProbe",
    "compare", "describe", "required_sample_size",
    "evaluate_agent", "summarize",
]
