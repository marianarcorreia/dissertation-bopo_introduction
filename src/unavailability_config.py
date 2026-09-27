"""Single source of truth for the machine-unavailability experiments (representations
ojmb_uf / ojmb_uo / ojmb_u0, src/env_unavailability.py).

Unavailability is always layered on the blocking FJSP: the instances and buffers are those of
src/blocking_config.py. Every length below is a fraction of the instance's solver-free makespan
lower bound LB (src/metrics/schedule.py:lower_bound), so the windows scale with the instance.

Sources (see src/unavailability.py for the sampling):
    scheduled  planned maintenance, known from t = 0. Non-preemptive: an operation never starts
               if it would cross one.
    breakdown  Weibull time to failure with shape beta > 1 (increasing failure rate) and a scale
               eta_m per machine (heterogeneous reliability), exponential repair with mean MTTR_m
               (preempt-resume: the part stays on the machine and continues after the repair).
               A repair and a scheduled maintenance both renew the machine (age = 0).
mode: "scheduled" | "breakdown" | "mixed" selects which sources are active.

Calibration: src/calibrate_unavailability.py (results/unavailability_calibration/, 10 instances
per setting, earliest-completion-time rule ranking by completion time). The values below are the
chosen setting; rerun it and update this file if the setting changes (every unavailability
result must then be regenerated).
    chosen (mixed): scheduled length 0.05-0.12 LB, availability 0.75-0.90, MTTR 0.06-0.12 LB
    - downtime_cost (makespan with / without windows, same rule) 1.201: the windows bind;
      machines are down 15.9% of the time; 9.7 parts per instance wait in the input buffer of a
      machine while it is down (the downtime-induced blocking the representations target)
    - 1.2 swaps and 0.3 waits per instance (swaps unchanged from the blocking setting)
    - breakdown only (same reliability): downtime_cost 1.234; scheduled only: 1.107
    - info_value (window-blind / window-aware makespan of the greedy rule) is only 1.018
      (1.000-1.035 on the whole grid): the two rules pick the same action in almost every
      decision, because the earliest completion over all jobs is rarely the option a window
      delays. Knowing the windows pays only with look-ahead, i.e. for a learned policy - the gap
      between the rule and the CP-SAT reference (src/generate_unavailability_references.py)
      is the room the representations have.
    Longer windows or lower availability bind harder (downtime_cost up to 1.385) but put
    machines down ~19% of the time, which is less realistic.
"""

UNAVAIL_CONFIG = {
    "mode": "mixed",
    # windows are sampled over [0, timeline_factor * LB]; later breakdowns are not generated
    "timeline_factor": 5.0,
    # look-ahead horizon H of the representations (current window + next window starting
    # within H), and scale of every time feature: H = horizon_factor * LB
    "horizon_factor": 0.5,
    "scheduled": {
        "per_machine": (0, 1),       # windows per machine (inclusive range)
        "bottleneck_extra": 1,       # extra window on the most loaded machine
        "length": (0.05, 0.12),      # window length / LB
        "start": (0.05, 0.9),        # window start / LB
    },
    "breakdown": {
        "beta": 2.0,                 # Weibull shape: > 1 = wear-out (failure rate grows with age)
        "availability": (0.75, 0.90),  # per machine: MTBF / (MTBF + MTTR) without maintenance
        "mttr": (0.06, 0.12),        # per machine mean repair time / LB
    },
    # base seed of the training scenarios (validation/test store their own realisation)
    "seed": 20260927,
}
