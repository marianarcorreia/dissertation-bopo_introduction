# Metrics report: gin_L3_selk5

- created: 2026-10-02T05:57:46
- git: blocking-constraint @ b95be99 (uncommitted changes)
- device: cpu
- models file: candidate_models/blocking/model_params.json

## val/test_dataset_blocking.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Swaps | Blocked time | Interrupted time | Trapped time | Waits | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 628378467920239.pth | ojmb | 20 | 1936.1 ± 268.7 | 0.1043 ± 0.0653 | 0.1048 ± 0.0651 | 0.9175 ± 0.0468 | 0.0000 ± 0.0000 | 62.2112 ± 2.2446 | - | 465.4 ± 1.7 | 0.9998 ± 0.0003 | 1.0876 ± 0.0120 | 0.3322 ± 0.0148 | 0.3525 ± 0.0048 | 0.3112 ± 0.0082 | 1.0000 ± 0.6958 | 1323.3 ± 583.4 | - | - | - | 1.0000 |
| 66658390227729.pth | ojmd | 20 | 1916.2 ± 207.1 | 0.1078 ± 0.0467 | 0.1085 ± 0.0469 | 0.9092 ± 0.0359 | 0.0000 ± 0.0000 | 58.8141 ± 1.6590 | - | 515.9 ± 0.5 | 0.9997 ± 0.0003 | 1.0863 ± 0.0128 | 0.2030 ± 0.0071 | 0.3269 ± 0.0051 | 0.2883 ± 0.0063 | 0.7500 ± 0.6420 | 1429.3 ± 468.2 | - | - | - | 1.0000 |
| 909627879422998.pth | ojmf | 20 | 1895.0 ± 225.9 | 0.0890 ± 0.0422 | 0.0896 ± 0.0425 | 0.9239 ± 0.0335 | 0.0000 ± 0.0000 | 45.8411 ± 0.9499 | - | 532.7 ± 0.6 | 1.0000 ± 0.0000 | 1.0109 ± 0.0026 | 0.2402 ± 0.0087 | 0.3471 ± 0.0046 | 0.2540 ± 0.0072 | 0.8500 ± 0.8355 | 1281.5 ± 424.4 | - | - | - | 1.0000 |
| 358310777019953.pth | ojm_blk | 20 | 1898.2 ± 202.8 | 0.0973 ± 0.0517 | 0.0980 ± 0.0523 | 0.9194 ± 0.0397 | 0.0000 ± 0.0000 | 42.8737 ± 0.7125 | - | 550.5 ± 0.4 | 1.0000 ± 0.0000 | 1.0129 ± 0.0030 | 0.2818 ± 0.0085 | 0.3673 ± 0.0054 | 0.3070 ± 0.0058 | 1.0500 ± 0.9171 | 1193.0 ± 345.3 | - | - | - | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `val/test_dataset_blocking.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 628378467920239.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.1043 | 0.1048 | 0.9175 | 0.0000 |
| 66658390227729.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.1078 | 0.1085 | 0.9092 | 0.0000 |
| 909627879422998.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.0890 | 0.0896 | 0.9239 | 0.0000 |
| 358310777019953.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.0973 | 0.0980 | 0.9194 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| val/test_dataset_blocking.json | makespan | 628378467920239.pth | 66658390227729.pth | 20 | 19.9500 | 0.9547 | no | 0.0830 | -0.0167 | 1141 |
| val/test_dataset_blocking.json | relative_error_lb | 628378467920239.pth | 66658390227729.pth | 20 | -0.0036 | 0.8203 | no | -0.0255 | -0.0667 | 12079 |
| val/test_dataset_blocking.json | scheduling_score | 628378467920239.pth | 66658390227729.pth | 20 | 0.0083 | 0.8647 | no | 0.0830 | 0.0500 | 1141 |
| val/test_dataset_blocking.json | makespan | 628378467920239.pth | 909627879422998.pth | 20 | 41.1500 | 0.7776 | no | 0.1627 | 0.0857 | 298 |
| val/test_dataset_blocking.json | relative_error_lb | 628378467920239.pth | 909627879422998.pth | 20 | 0.0152 | 0.9250 | no | 0.1099 | 0.0286 | 652 |
| val/test_dataset_blocking.json | scheduling_score | 628378467920239.pth | 909627879422998.pth | 20 | -0.0064 | 0.9250 | no | -0.0663 | -0.0286 | 1786 |
| val/test_dataset_blocking.json | makespan | 628378467920239.pth | 358310777019953.pth | 20 | 37.8500 | 0.6496 | no | 0.1606 | 0.1333 | 306 |
| val/test_dataset_blocking.json | relative_error_lb | 628378467920239.pth | 358310777019953.pth | 20 | 0.0068 | 0.8647 | no | 0.0492 | 0.0500 | 3243 |
| val/test_dataset_blocking.json | scheduling_score | 628378467920239.pth | 358310777019953.pth | 20 | -0.0019 | 0.8647 | no | -0.0201 | -0.0500 | 19380 |
| val/test_dataset_blocking.json | makespan | 66658390227729.pth | 909627879422998.pth | 20 | 21.2000 | 0.3520 | no | 0.1553 | 0.2647 | 327 |
| val/test_dataset_blocking.json | relative_error_lb | 66658390227729.pth | 909627879422998.pth | 20 | 0.0188 | 0.2343 | no | 0.2469 | 0.3382 | 130 |
| val/test_dataset_blocking.json | scheduling_score | 66658390227729.pth | 909627879422998.pth | 20 | -0.0147 | 0.2343 | no | -0.2417 | -0.3382 | 136 |
| val/test_dataset_blocking.json | makespan | 66658390227729.pth | 358310777019953.pth | 20 | 17.9000 | 0.6092 | no | 0.1013 | 0.1500 | 767 |
| val/test_dataset_blocking.json | relative_error_lb | 66658390227729.pth | 358310777019953.pth | 20 | 0.0104 | 0.6909 | no | 0.0814 | 0.1167 | 1185 |
| val/test_dataset_blocking.json | scheduling_score | 66658390227729.pth | 358310777019953.pth | 20 | -0.0102 | 0.5321 | no | -0.1075 | -0.1833 | 681 |
| val/test_dataset_blocking.json | makespan | 909627879422998.pth | 358310777019953.pth | 20 | -3.3000 | 0.8647 | no | -0.0171 | -0.0500 | 26702 |
| val/test_dataset_blocking.json | relative_error_lb | 909627879422998.pth | 358310777019953.pth | 20 | -0.0084 | 0.7764 | no | -0.0683 | -0.0833 | 1683 |
| val/test_dataset_blocking.json | scheduling_score | 909627879422998.pth | 358310777019953.pth | 20 | 0.0045 | 0.7764 | no | 0.0491 | 0.0833 | 3251 |
