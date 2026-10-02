# Metrics comparison

- created: 2026-10-02T05:57:55

## Systems

| System | Model | Repr. | Git | Report created | Report file |
|---|---|---|---|---|---|
| gat_L2_selk1:702627640508775.pth | 702627640508775.pth | ojmb | blocking-constraint @ b95be99 (dirty) | 2026-10-01T14:59:55 | results/blocking/metrics/blk350_gat_L2_selk1.json |
| gat_L2_selk1:191510980309327.pth | 191510980309327.pth | ojmd | blocking-constraint @ b95be99 (dirty) | 2026-10-01T14:59:55 | results/blocking/metrics/blk350_gat_L2_selk1.json |
| gat_L2_selk1:658367275525800.pth | 658367275525800.pth | ojmf | blocking-constraint @ b95be99 (dirty) | 2026-10-01T14:59:55 | results/blocking/metrics/blk350_gat_L2_selk1.json |
| gat_L2_selk1:746085911473341.pth | 746085911473341.pth | ojm_blk | blocking-constraint @ b95be99 (dirty) | 2026-10-01T14:59:55 | results/blocking/metrics/blk350_gat_L2_selk1.json |
| gin_L3_selk5:628378467920239.pth | 628378467920239.pth | ojmb | blocking-constraint @ b95be99 (dirty) | 2026-10-02T05:57:46 | results/blocking/metrics/blk350_gin_L3_selk5.json |
| gin_L3_selk5:66658390227729.pth | 66658390227729.pth | ojmd | blocking-constraint @ b95be99 (dirty) | 2026-10-02T05:57:46 | results/blocking/metrics/blk350_gin_L3_selk5.json |
| gin_L3_selk5:909627879422998.pth | 909627879422998.pth | ojmf | blocking-constraint @ b95be99 (dirty) | 2026-10-02T05:57:46 | results/blocking/metrics/blk350_gin_L3_selk5.json |
| gin_L3_selk5:358310777019953.pth | 358310777019953.pth | ojm_blk | blocking-constraint @ b95be99 (dirty) | 2026-10-02T05:57:46 | results/blocking/metrics/blk350_gin_L3_selk5.json |

## val/test_dataset_blocking.json

Mean ± 95% CI half-width.

| System | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Swaps | Blocked time | Interrupted time | Trapped time | Waits | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gat_L2_selk1:702627640508775.pth | 20 | 2069.8 ± 237.5 | 0.1963 ± 0.0619 | 0.1970 ± 0.0623 | 0.8458 ± 0.0443 | 0.0000 ± 0.0000 | 71.3498 ± 6.7075 | - | 442.0 ± 1.6 | 1.0000 ± 0.0000 | 1.0898 ± 0.0138 | 0.3027 ± 0.0111 | 0.3589 ± 0.0047 | 0.3051 ± 0.0086 | 1.4000 ± 0.9651 | 1752.7 ± 508.0 | - | - | - | 1.0000 |
| gat_L2_selk1:191510980309327.pth | 20 | 1983.8 ± 217.4 | 0.1460 ± 0.0524 | 0.1466 ± 0.0524 | 0.8800 ± 0.0378 | 0.0000 ± 0.0000 | 69.8474 ± 4.8039 | - | 458.9 ± 0.7 | 0.9996 ± 0.0004 | 1.0894 ± 0.0132 | 0.2059 ± 0.0062 | 0.3626 ± 0.0046 | 0.2925 ± 0.0071 | 1.0000 ± 0.6958 | 1420.5 ± 425.5 | - | - | - | 1.0000 |
| gat_L2_selk1:658367275525800.pth | 20 | 2036.1 ± 239.1 | 0.1719 ± 0.0624 | 0.1726 ± 0.0629 | 0.8631 ± 0.0427 | 0.0000 ± 0.0000 | 58.0902 ± 4.8631 | - | 469.3 ± 0.7 | 1.0000 ± 0.0000 | 1.0115 ± 0.0034 | 0.2059 ± 0.0086 | 0.3112 ± 0.0045 | 0.2491 ± 0.0056 | 1.0000 ± 0.6791 | 1719.2 ± 532.9 | - | - | - | 1.0000 |
| gat_L2_selk1:746085911473341.pth | 20 | 2060.5 ± 239.5 | 0.1891 ± 0.0563 | 0.1898 ± 0.0568 | 0.8494 ± 0.0411 | 0.0000 ± 0.0000 | 53.8626 ± 4.3445 | - | 481.7 ± 0.7 | 0.9999 ± 0.0003 | 1.0120 ± 0.0037 | 0.2364 ± 0.0078 | 0.3554 ± 0.0076 | 0.3065 ± 0.0083 | 1.4500 ± 0.7357 | 1931.9 ± 661.1 | - | - | - | 1.0000 |
| gin_L3_selk5:628378467920239.pth | 20 | 1936.1 ± 268.7 | 0.1043 ± 0.0653 | 0.1048 ± 0.0651 | 0.9175 ± 0.0468 | 0.0000 ± 0.0000 | 62.2112 ± 2.2446 | - | 465.4 ± 1.7 | 0.9998 ± 0.0003 | 1.0876 ± 0.0120 | 0.3322 ± 0.0148 | 0.3525 ± 0.0048 | 0.3112 ± 0.0082 | 1.0000 ± 0.6958 | 1323.3 ± 583.4 | - | - | - | 1.0000 |
| gin_L3_selk5:66658390227729.pth | 20 | 1916.2 ± 207.1 | 0.1078 ± 0.0467 | 0.1085 ± 0.0469 | 0.9092 ± 0.0359 | 0.0000 ± 0.0000 | 58.8141 ± 1.6590 | - | 515.9 ± 0.5 | 0.9997 ± 0.0003 | 1.0863 ± 0.0128 | 0.2030 ± 0.0071 | 0.3269 ± 0.0051 | 0.2883 ± 0.0063 | 0.7500 ± 0.6420 | 1429.3 ± 468.2 | - | - | - | 1.0000 |
| gin_L3_selk5:909627879422998.pth | 20 | 1895.0 ± 225.9 | 0.0890 ± 0.0422 | 0.0896 ± 0.0425 | 0.9239 ± 0.0335 | 0.0000 ± 0.0000 | 45.8411 ± 0.9499 | - | 532.7 ± 0.6 | 1.0000 ± 0.0000 | 1.0109 ± 0.0026 | 0.2402 ± 0.0087 | 0.3471 ± 0.0046 | 0.2540 ± 0.0072 | 0.8500 ± 0.8355 | 1281.5 ± 424.4 | - | - | - | 1.0000 |
| gin_L3_selk5:358310777019953.pth | 20 | 1898.2 ± 202.8 | 0.0973 ± 0.0517 | 0.0980 ± 0.0523 | 0.9194 ± 0.0397 | 0.0000 ± 0.0000 | 42.8737 ± 0.7125 | - | 550.5 ± 0.4 | 1.0000 ± 0.0000 | 1.0129 ± 0.0030 | 0.2818 ± 0.0085 | 0.3673 ± 0.0054 | 0.3070 ± 0.0058 | 1.0500 ± 0.9171 | 1193.0 ± 345.3 | - | - | - | 1.0000 |

Best mean per metric: makespan → **gin_L3_selk5:909627879422998.pth**, relative_error → **gin_L3_selk5:909627879422998.pth**, relative_error_lb → **gin_L3_selk5:909627879422998.pth**, scheduling_score → **gin_L3_selk5:909627879422998.pth**, feasible → **tie (8 systems)**, violation_rate_per_step → **tie (8 systems)**, time_per_decision_ms → **gin_L3_selk5:358310777019953.pth**

Pairwise tests (alpha = 0.05); winner only when significant.

| Metric | A | B | Test | n | Mean diff (A−B) | p | Effect | Winner |
|---|---|---|---|---|---|---|---|---|
| makespan | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 20 | 86.0000 | 0.1075 | 0.4211 (rank-biserial) | - |
| relative_error | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 20 | 0.0503 | 0.1165 | 0.4105 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 20 | 0.0504 | 0.1165 | 0.4105 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 20 | -0.0342 | 0.0836 | -0.4526 (rank-biserial) | - |
| feasible | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 20 | 1.5023 | 0.5459 | 0.1619 (rank-biserial) | - |
| gpu_peak_mb | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:191510980309327.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 33.7000 | 0.9358 | -0.0211 (rank-biserial) | - |
| relative_error | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 0.0244 | 0.9359 | 0.0211 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 0.0244 | 0.9359 | 0.0211 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | -0.0173 | 0.8721 | -0.0421 (rank-biserial) | - |
| feasible | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 13.2595 | 0.0007 | 0.8095 (rank-biserial) | gat_L2_selk1:658367275525800.pth |
| gpu_peak_mb | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 9.3000 | 0.8276 | 0.0585 (rank-biserial) | - |
| relative_error | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0072 | 0.8789 | 0.0409 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0072 | 0.8789 | 0.0409 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | -0.0036 | 0.8789 | -0.0409 (rank-biserial) | - |
| feasible | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 17.4871 | 0.0001 | 0.9143 (rank-biserial) | gat_L2_selk1:746085911473341.pth |
| gpu_peak_mb | gat_L2_selk1:702627640508775.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 133.7 | 0.0441 | 0.5143 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| relative_error | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0920 | 0.0240 | 0.5714 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| relative_error_lb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0922 | 0.0240 | 0.5714 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| scheduling_score | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | -0.0717 | 0.0172 | -0.6000 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| feasible | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 9.1385 | 0.0003 | 0.8476 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| gpu_peak_mb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 153.7 | 0.0064 | 0.6952 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| relative_error | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0885 | 0.0037 | 0.7143 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| relative_error_lb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0886 | 0.0037 | 0.7143 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| scheduling_score | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -0.0634 | 0.0037 | -0.7143 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| feasible | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 12.5356 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| gpu_peak_mb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 174.8 | 0.0003 | 0.9579 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.1073 | 0.0003 | 0.9579 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error_lb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.1074 | 0.0003 | 0.9579 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| scheduling_score | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | -0.0781 | 0.0003 | -0.9579 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| feasible | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 25.5087 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| gpu_peak_mb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 171.6 | 0.0064 | 0.6762 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| relative_error | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0990 | 0.0095 | 0.6619 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| relative_error_lb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0990 | 0.0095 | 0.6619 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| scheduling_score | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0736 | 0.0094 | -0.6476 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| feasible | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 28.4761 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| gpu_peak_mb | gat_L2_selk1:702627640508775.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | -52.3000 | 0.3603 | -0.2333 (rank-biserial) | - |
| relative_error | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | -0.0259 | 0.5217 | -0.1714 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | -0.0260 | 0.5217 | -0.1714 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 0.0169 | 0.4304 | 0.2095 (rank-biserial) | - |
| feasible | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 20 | 11.7572 | 0.0000 | 1.0000 (rank-biserial) | gat_L2_selk1:658367275525800.pth |
| gpu_peak_mb | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:658367275525800.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | -76.7000 | 0.1589 | -0.3684 (rank-biserial) | - |
| relative_error | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | -0.0431 | 0.1474 | -0.3789 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | -0.0432 | 0.1474 | -0.3789 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0306 | 0.1165 | 0.4105 (rank-biserial) | - |
| feasible | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 15.9848 | 0.0000 | 0.9524 (rank-biserial) | gat_L2_selk1:746085911473341.pth |
| gpu_peak_mb | gat_L2_selk1:191510980309327.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 47.7000 | 0.2772 | 0.2842 (rank-biserial) | - |
| relative_error | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0417 | 0.1474 | 0.3789 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0418 | 0.1474 | 0.3789 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | -0.0374 | 0.1165 | -0.4105 (rank-biserial) | - |
| feasible | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 7.6362 | 0.0005 | 0.8286 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| gpu_peak_mb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 67.6500 | 0.2024 | 0.3333 (rank-biserial) | - |
| relative_error | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0382 | 0.1769 | 0.3524 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0382 | 0.1769 | 0.3524 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -0.0292 | 0.1769 | -0.3524 (rank-biserial) | - |
| feasible | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 11.0333 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| gpu_peak_mb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 88.8500 | 0.0347 | 0.5673 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0570 | 0.0123 | 0.6725 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error_lb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0570 | 0.0123 | 0.6725 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| scheduling_score | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | -0.0439 | 0.0123 | -0.6725 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| feasible | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 24.0063 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| gpu_peak_mb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 85.5500 | 0.1165 | 0.4105 (rank-biserial) | - |
| relative_error | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0487 | 0.0990 | 0.4316 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0486 | 0.0990 | 0.4316 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0393 | 0.0910 | -0.4421 (rank-biserial) | - |
| feasible | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 26.9737 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| gpu_peak_mb | gat_L2_selk1:191510980309327.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | -24.4000 | 0.5699 | -0.1569 (rank-biserial) | - |
| relative_error | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | -0.0172 | 0.5862 | -0.1503 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | -0.0172 | 0.5862 | -0.1503 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0137 | 0.5540 | 0.1634 (rank-biserial) | - |
| feasible | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 20 | 4.2276 | 0.0532 | 0.4952 (rank-biserial) | - |
| gpu_peak_mb | gat_L2_selk1:658367275525800.pth | gat_L2_selk1:746085911473341.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 100.0 | 0.0929 | 0.4286 (rank-biserial) | - |
| relative_error | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0676 | 0.0637 | 0.4762 (rank-biserial) | - |
| relative_error_lb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0678 | 0.0583 | 0.4857 (rank-biserial) | - |
| scheduling_score | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | -0.0544 | 0.0583 | -0.4857 (rank-biserial) | - |
| feasible | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | -4.1210 | 0.0266 | -0.5619 (rank-biserial) | gat_L2_selk1:658367275525800.pth |
| gpu_peak_mb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 120.0 | 0.0242 | 0.5895 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| relative_error | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0641 | 0.0218 | 0.6000 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| relative_error_lb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0642 | 0.0218 | 0.6000 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| scheduling_score | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -0.0461 | 0.0269 | -0.5789 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| feasible | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -0.7239 | 0.1650 | -0.3619 (rank-biserial) | - |
| gpu_peak_mb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 141.2 | 0.0032 | 0.7524 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0829 | 0.0023 | 0.7429 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error_lb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0830 | 0.0023 | 0.7429 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| scheduling_score | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | -0.0608 | 0.0037 | -0.7143 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| feasible | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 12.2491 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| gpu_peak_mb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 137.8 | 0.0062 | 0.7158 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| relative_error | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0746 | 0.0062 | 0.7158 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| relative_error_lb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0746 | 0.0062 | 0.7158 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| scheduling_score | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0563 | 0.0062 | -0.7158 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| feasible | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 15.2165 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| gpu_peak_mb | gat_L2_selk1:658367275525800.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 124.4 | 0.0836 | 0.4526 (rank-biserial) | - |
| relative_error | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0848 | 0.0401 | 0.5368 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| relative_error_lb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0850 | 0.0401 | 0.5368 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| scheduling_score | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | -0.0681 | 0.0442 | -0.5263 (rank-biserial) | gin_L3_selk5:628378467920239.pth |
| feasible | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 20 | -8.3486 | 0.0014 | -0.7714 (rank-biserial) | gat_L2_selk1:746085911473341.pth |
| gpu_peak_mb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:628378467920239.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 144.3 | 0.0090 | 0.7018 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| relative_error | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0813 | 0.0108 | 0.6842 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| relative_error_lb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0814 | 0.0108 | 0.6842 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| scheduling_score | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -0.0598 | 0.0108 | -0.6842 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| feasible | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -4.9515 | 0.0007 | -0.8095 (rank-biserial) | gat_L2_selk1:746085911473341.pth |
| gpu_peak_mb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 165.6 | 0.0016 | 0.8480 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.1001 | 0.0012 | 0.8713 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| relative_error_lb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.1002 | 0.0012 | 0.8713 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| scheduling_score | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | -0.0745 | 0.0012 | -0.8713 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| feasible | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 8.0215 | 0.0000 | 0.9333 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| gpu_peak_mb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 162.2 | 0.0218 | 0.6000 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| relative_error | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0918 | 0.0141 | 0.6421 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| relative_error_lb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0918 | 0.0141 | 0.6421 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| scheduling_score | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0700 | 0.0176 | -0.6211 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| feasible | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 10.9889 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| gpu_peak_mb | gat_L2_selk1:746085911473341.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 19.9500 | 0.9547 | -0.0167 (rank-biserial) | - |
| relative_error | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -0.0035 | 0.8203 | -0.0667 (rank-biserial) | - |
| relative_error_lb | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | -0.0036 | 0.8203 | -0.0667 (rank-biserial) | - |
| scheduling_score | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0083 | 0.8647 | 0.0500 (rank-biserial) | - |
| feasible | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 20 | 3.3971 | 0.0032 | 0.7238 (rank-biserial) | gin_L3_selk5:66658390227729.pth |
| gpu_peak_mb | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:66658390227729.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 41.1500 | 0.7776 | 0.0857 (rank-biserial) | - |
| relative_error | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0153 | 0.8753 | 0.0476 (rank-biserial) | - |
| relative_error_lb | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0152 | 0.9250 | 0.0286 (rank-biserial) | - |
| scheduling_score | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | -0.0064 | 0.9250 | -0.0286 (rank-biserial) | - |
| feasible | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 16.3701 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| gpu_peak_mb | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 37.8500 | 0.6496 | 0.1333 (rank-biserial) | - |
| relative_error | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0070 | 0.8647 | 0.0500 (rank-biserial) | - |
| relative_error_lb | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0068 | 0.8647 | 0.0500 (rank-biserial) | - |
| scheduling_score | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0019 | 0.8647 | -0.0500 (rank-biserial) | - |
| feasible | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 19.3375 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| gpu_peak_mb | gin_L3_selk5:628378467920239.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 21.2000 | 0.3520 | 0.2647 (rank-biserial) | - |
| relative_error | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0188 | 0.2343 | 0.3382 (rank-biserial) | - |
| relative_error_lb | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0188 | 0.2343 | 0.3382 (rank-biserial) | - |
| scheduling_score | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | -0.0147 | 0.2343 | -0.3382 (rank-biserial) | - |
| feasible | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 20 | 12.9730 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:909627879422998.pth |
| gpu_peak_mb | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:909627879422998.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 17.9000 | 0.6092 | 0.1500 (rank-biserial) | - |
| relative_error | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0105 | 0.6909 | 0.1167 (rank-biserial) | - |
| relative_error_lb | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0104 | 0.6909 | 0.1167 (rank-biserial) | - |
| scheduling_score | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0102 | 0.5321 | -0.1833 (rank-biserial) | - |
| feasible | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 15.9404 | 0.0000 | 1.0000 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| gpu_peak_mb | gin_L3_selk5:66658390227729.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |
| makespan | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -3.3000 | 0.8647 | -0.0500 (rank-biserial) | - |
| relative_error | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0083 | 0.7764 | -0.0833 (rank-biserial) | - |
| relative_error_lb | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | -0.0084 | 0.7764 | -0.0833 (rank-biserial) | - |
| scheduling_score | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0045 | 0.7764 | 0.0833 (rank-biserial) | - |
| feasible | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| violation_rate_per_step | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 0.0000 | 1.0000 | 0.0000 (rank-biserial) | - |
| time_per_decision_ms | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 20 | 2.9674 | 0.0000 | 0.9905 (rank-biserial) | gin_L3_selk5:358310777019953.pth |
| gpu_peak_mb | gin_L3_selk5:909627879422998.pth | gin_L3_selk5:358310777019953.pth | wilcoxon (paired) | 0 | - | - | - (rank-biserial) | - |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on each report's in-distribution folder.

| System | Folder | In training range | RE (CP-SAT) | RE (LB) | Score | Feasible | Gap |
|---|---|---|---|---|---|---|---|
| gat_L2_selk1:702627640508775.pth | val/test_dataset_blocking.json | 1.00 | 0.1963 | 0.1970 | 0.8458 | 1.0000 | 0.0000 |
| gat_L2_selk1:191510980309327.pth | val/test_dataset_blocking.json | 1.00 | 0.1460 | 0.1466 | 0.8800 | 1.0000 | 0.0000 |
| gat_L2_selk1:658367275525800.pth | val/test_dataset_blocking.json | 1.00 | 0.1719 | 0.1726 | 0.8631 | 1.0000 | 0.0000 |
| gat_L2_selk1:746085911473341.pth | val/test_dataset_blocking.json | 1.00 | 0.1891 | 0.1898 | 0.8494 | 1.0000 | 0.0000 |
| gin_L3_selk5:628378467920239.pth | val/test_dataset_blocking.json | 1.00 | 0.1043 | 0.1048 | 0.9175 | 1.0000 | 0.0000 |
| gin_L3_selk5:66658390227729.pth | val/test_dataset_blocking.json | 1.00 | 0.1078 | 0.1085 | 0.9092 | 1.0000 | 0.0000 |
| gin_L3_selk5:909627879422998.pth | val/test_dataset_blocking.json | 1.00 | 0.0890 | 0.0896 | 0.9239 | 1.0000 | 0.0000 |
| gin_L3_selk5:358310777019953.pth | val/test_dataset_blocking.json | 1.00 | 0.0973 | 0.0980 | 0.9194 | 1.0000 | 0.0000 |
