# Metrics report: gat_L2_selk1

- created: 2026-10-01T14:59:55
- git: blocking-constraint @ b95be99 (uncommitted changes)
- device: cpu
- models file: candidate_models/blocking/model_params.json

## val/test_dataset_blocking.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Swaps | Blocked time | Interrupted time | Trapped time | Waits | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 702627640508775.pth | ojmb | 20 | 2069.8 ± 237.5 | 0.1963 ± 0.0619 | 0.1970 ± 0.0623 | 0.8458 ± 0.0443 | 0.0000 ± 0.0000 | 71.3498 ± 6.7075 | - | 442.0 ± 1.6 | 1.0000 ± 0.0000 | 1.0898 ± 0.0138 | 0.3027 ± 0.0111 | 0.3589 ± 0.0047 | 0.3051 ± 0.0086 | 1.4000 ± 0.9651 | 1752.7 ± 508.0 | - | - | - | 1.0000 |
| 191510980309327.pth | ojmd | 20 | 1983.8 ± 217.4 | 0.1460 ± 0.0524 | 0.1466 ± 0.0524 | 0.8800 ± 0.0378 | 0.0000 ± 0.0000 | 69.8474 ± 4.8039 | - | 458.9 ± 0.7 | 0.9996 ± 0.0004 | 1.0894 ± 0.0132 | 0.2059 ± 0.0062 | 0.3626 ± 0.0046 | 0.2925 ± 0.0071 | 1.0000 ± 0.6958 | 1420.5 ± 425.5 | - | - | - | 1.0000 |
| 658367275525800.pth | ojmf | 20 | 2036.1 ± 239.1 | 0.1719 ± 0.0624 | 0.1726 ± 0.0629 | 0.8631 ± 0.0427 | 0.0000 ± 0.0000 | 58.0902 ± 4.8631 | - | 469.3 ± 0.7 | 1.0000 ± 0.0000 | 1.0115 ± 0.0034 | 0.2059 ± 0.0086 | 0.3112 ± 0.0045 | 0.2491 ± 0.0056 | 1.0000 ± 0.6791 | 1719.2 ± 532.9 | - | - | - | 1.0000 |
| 746085911473341.pth | ojm_blk | 20 | 2060.5 ± 239.5 | 0.1891 ± 0.0563 | 0.1898 ± 0.0568 | 0.8494 ± 0.0411 | 0.0000 ± 0.0000 | 53.8626 ± 4.3445 | - | 481.7 ± 0.7 | 0.9999 ± 0.0003 | 1.0120 ± 0.0037 | 0.2364 ± 0.0078 | 0.3554 ± 0.0076 | 0.3065 ± 0.0083 | 1.4500 ± 0.7357 | 1931.9 ± 661.1 | - | - | - | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `val/test_dataset_blocking.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 702627640508775.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.1963 | 0.1970 | 0.8458 | 0.0000 |
| 191510980309327.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.1460 | 0.1466 | 0.8800 | 0.0000 |
| 658367275525800.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.1719 | 0.1726 | 0.8631 | 0.0000 |
| 746085911473341.pth | val/test_dataset_blocking.json | 12-15 | 4-6 | 1.00 | 0.1891 | 0.1898 | 0.8494 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| val/test_dataset_blocking.json | makespan | 702627640508775.pth | 191510980309327.pth | 20 | 86.0000 | 0.1075 | no | 0.4739 | 0.4211 | 36 |
| val/test_dataset_blocking.json | relative_error_lb | 702627640508775.pth | 191510980309327.pth | 20 | 0.0504 | 0.1165 | no | 0.4811 | 0.4105 | 35 |
| val/test_dataset_blocking.json | scheduling_score | 702627640508775.pth | 191510980309327.pth | 20 | -0.0342 | 0.0836 | no | -0.4955 | -0.4526 | 33 |
| val/test_dataset_blocking.json | makespan | 702627640508775.pth | 658367275525800.pth | 20 | 33.7000 | 0.9358 | no | 0.1148 | -0.0211 | 597 |
| val/test_dataset_blocking.json | relative_error_lb | 702627640508775.pth | 658367275525800.pth | 20 | 0.0244 | 0.9359 | no | 0.1486 | 0.0211 | 357 |
| val/test_dataset_blocking.json | scheduling_score | 702627640508775.pth | 658367275525800.pth | 20 | -0.0173 | 0.8721 | no | -0.1541 | -0.0421 | 332 |
| val/test_dataset_blocking.json | makespan | 702627640508775.pth | 746085911473341.pth | 20 | 9.3000 | 0.8276 | no | 0.0520 | 0.0585 | 2907 |
| val/test_dataset_blocking.json | relative_error_lb | 702627640508775.pth | 746085911473341.pth | 20 | 0.0072 | 0.8789 | no | 0.0723 | 0.0409 | 1501 |
| val/test_dataset_blocking.json | scheduling_score | 702627640508775.pth | 746085911473341.pth | 20 | -0.0036 | 0.8789 | no | -0.0526 | -0.0409 | 2837 |
| val/test_dataset_blocking.json | makespan | 191510980309327.pth | 658367275525800.pth | 20 | -52.3000 | 0.3603 | no | -0.2329 | -0.2333 | 146 |
| val/test_dataset_blocking.json | relative_error_lb | 191510980309327.pth | 658367275525800.pth | 20 | -0.0260 | 0.5217 | no | -0.1937 | -0.1714 | 211 |
| val/test_dataset_blocking.json | scheduling_score | 191510980309327.pth | 658367275525800.pth | 20 | 0.0169 | 0.4304 | no | 0.1799 | 0.2095 | 244 |
| val/test_dataset_blocking.json | makespan | 191510980309327.pth | 746085911473341.pth | 20 | -76.7000 | 0.1589 | no | -0.4130 | -0.3684 | 48 |
| val/test_dataset_blocking.json | relative_error_lb | 191510980309327.pth | 746085911473341.pth | 20 | -0.0432 | 0.1474 | no | -0.4316 | -0.3789 | 44 |
| val/test_dataset_blocking.json | scheduling_score | 191510980309327.pth | 746085911473341.pth | 20 | 0.0306 | 0.1165 | no | 0.4352 | 0.4105 | 43 |
| val/test_dataset_blocking.json | makespan | 658367275525800.pth | 746085911473341.pth | 20 | -24.4000 | 0.5699 | no | -0.1101 | -0.1569 | 649 |
| val/test_dataset_blocking.json | relative_error_lb | 658367275525800.pth | 746085911473341.pth | 20 | -0.0172 | 0.5862 | no | -0.1336 | -0.1503 | 441 |
| val/test_dataset_blocking.json | scheduling_score | 658367275525800.pth | 746085911473341.pth | 20 | 0.0137 | 0.5540 | no | 0.1478 | 0.1634 | 361 |
