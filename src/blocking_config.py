"""Single source of truth for the blocking experiments (representations ojmb / ojmd / ojm_blk).

Used by src/env_blocking.py (default buffer capacities), src/train.py (training instances),
src/generate_blocking_references.py (validation/test instances + references) and
src/calibrate_blocking.py (which is how these values were chosen - rerun it and update this
file if the problem setting changes; every blocking result must then be regenerated).

Why these instances differ from the other experiments: with the default generator (up to every
machine eligible per operation, no bottleneck) and buffers 5/2, parts never queue, so the
blocking constraint never binds - no blocked events and the same optimum as without blocking.

Calibration (results/blocking_calibration/stage1.csv, stage2.csv; 10 instances per setting):
    chosen: flexibility <= 3, 1 bottleneck (x2.0, in 70% of the operations), 12-15 jobs,
            4-6 machines, buffers in=2 / out=1
    - blocked events on every instance (mean 20.9 per instance, earliest-completion-time rule)
    - a buffer-blind rule loses 21.5% makespan vs unlimited buffers (h_tightness 1.215)
    - the CP-SAT optimum moves only +0.6% (max +5.7%): blocking is avoidable with
      buffer-aware decisions, which is the room the buffer representations have to show value
    - 0.7 deadlock swaps per instance (lowest among the settings where blocking binds)
    alternatives: 1/1 binds harder (h_tightness 1.29-1.39) but needs 1.4-1.7 swaps per instance;
    3/1 and 5/2 bind weakly (1.087, 1.020); pure blocking (x/0) needs ~20 swaps per instance.
    Caveat: at 20 s CP-SAT proved optimality on only 20% of these instances, so the final
    references use a longer time limit (generate_blocking_references --time-limit).

Final references (val/*_blocking.json, 20 + 20 instances, 120 s per CP-SAT model):
    blocking optimum proven on 38/40, non-blocking on 40/40, and the two optima are EQUAL on
    all 40 instances (the +0.6% above came from weaker 20 s incumbents): a buffer-aware schedule
    loses nothing to blocking. The expert rule's gap is 4.1% with unlimited buffers but 19.9%
    with buffers 2/1 (1.9 swaps per instance) - blocking only costs a policy that ignores the
    buffers, which is exactly what the buffer representations are meant to fix.
"""

BLOCKING_CONFIG = {
    # buffer capacities per machine
    "in_cap": 2,
    "out_cap": 1,
    # instance generator (src/generator.py:generate_instance_list keyword arguments)
    "generator": {
        "range_jobs": (12, 15),
        "range_machines": (4, 6),
        "range_op_per_job": (5, 6),
        "max_processing": 100,
        "mas_per_ope_max": 3,
        "n_bottlenecks": 1,
        "bottleneck_prob": 0.7,
        "bottleneck_factor": 2.0,
    },
}
