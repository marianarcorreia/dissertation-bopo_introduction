# Metrics report: batching_constraint:report_metrics_batching_v2_gin_L3_k5_s43_steps

- created: 2026-10-02T16:09:10
- git: batching_constraint @ 5d4253c6 (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 411825325582948.pth | ojmb_node_v2 | 20 | 403.4 ± 29.3 | 0.1276 ± 0.0385 | 0.1856 ± 0.0721 | 0.8911 ± 0.0290 | 0.0000 ± 0.0000 | 56.6541 ± 13.3447 | 46.8981 ± 0.1922 | 1358.5 ± 1.4 | 0.9999 ± 0.0001 | 1.0179 ± 0.0047 | 0.2763 ± 0.0162 | 0.3748 ± 0.0037 | 0.3914 ± 0.0069 | 1.1127 ± 0.0367 | 0.1844 ± 0.0535 | 1.0000 |
| 681929219451006.pth | ojmb_edge_v2 | 20 | 410.6 ± 29.0 | 0.1476 ± 0.0349 | 0.2063 ± 0.0691 | 0.8749 ± 0.0263 | 0.0000 ± 0.0000 | 37.6700 ± 1.0667 | 72.6118 ± 0.1913 | 1361.4 ± 1.5 | 1.0000 ± 0.0001 | 1.0216 ± 0.0049 | 0.2665 ± 0.0151 | 0.3634 ± 0.0069 | 0.3561 ± 0.0092 | 1.1283 ± 0.0280 | 0.2113 ± 0.0409 | 1.0000 |
| 948172624102402.pth | ojmb_base_v2 | 20 | 414.8 ± 28.9 | 0.1586 ± 0.0275 | 0.2169 ± 0.0603 | 0.8652 ± 0.0203 | 0.0000 ± 0.0000 | 34.0075 ± 0.6546 | 95.4853 ± 0.1913 | 1106.9 ± 166.7 | 1.0000 ± 0.0000 | 1.0256 ± 0.0089 | 0.2341 ± 0.0120 | 0.3497 ± 0.0053 | 0.3679 ± 0.0094 | 1.1029 ± 0.0316 | 0.1702 ± 0.0483 | 1.0000 |
| 537660859795754.pth | ojmb_feat_v2 | 20 | 411.6 ± 31.4 | 0.1492 ± 0.0400 | 0.2070 ± 0.0669 | 0.8745 ± 0.0288 | 0.0000 ± 0.0000 | 35.8348 ± 1.4047 | 118.4 ± 0.2 | 635.9 ± 1.0 | 0.9999 ± 0.0001 | 1.0177 ± 0.0043 | 0.2719 ± 0.0132 | 0.3760 ± 0.0063 | 0.3733 ± 0.0121 | 1.1015 ± 0.0307 | 0.1673 ± 0.0463 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 411825325582948.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1276 | 0.1856 | 0.8911 | 0.0000 |
| 681929219451006.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1476 | 0.2063 | 0.8749 | 0.0000 |
| 948172624102402.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1586 | 0.2169 | 0.8652 | 0.0000 |
| 537660859795754.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1492 | 0.2070 | 0.8745 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 411825325582948.pth | 681929219451006.pth | 20 | -7.1500 | 0.1167 | no | -0.3413 | -0.4000 | 69 |
| data/batching/batching_test_split.json | relative_error_lb | 411825325582948.pth | 681929219451006.pth | 20 | -0.0207 | 0.1054 | no | -0.3563 | -0.4190 | 63 |
| data/batching/batching_test_split.json | scheduling_score | 411825325582948.pth | 681929219451006.pth | 20 | 0.0163 | 0.0897 | no | 0.3938 | 0.4381 | 52 |
| data/batching/batching_test_split.json | makespan | 411825325582948.pth | 948172624102402.pth | 20 | -11.4000 | 0.1589 | no | -0.3417 | -0.3684 | 69 |
| data/batching/batching_test_split.json | relative_error_lb | 411825325582948.pth | 948172624102402.pth | 20 | -0.0313 | 0.1590 | no | -0.3220 | -0.3684 | 77 |
| data/batching/batching_test_split.json | scheduling_score | 411825325582948.pth | 948172624102402.pth | 20 | 0.0260 | 0.1165 | no | 0.3792 | 0.4105 | 56 |
| data/batching/batching_test_split.json | makespan | 411825325582948.pth | 537660859795754.pth | 20 | -8.1500 | 0.1212 | no | -0.2700 | -0.3952 | 109 |
| data/batching/batching_test_split.json | relative_error_lb | 411825325582948.pth | 537660859795754.pth | 20 | -0.0215 | 0.1231 | no | -0.2290 | -0.4000 | 151 |
| data/batching/batching_test_split.json | scheduling_score | 411825325582948.pth | 537660859795754.pth | 20 | 0.0166 | 0.1327 | no | 0.2660 | 0.3905 | 112 |
| data/batching/batching_test_split.json | makespan | 681929219451006.pth | 948172624102402.pth | 20 | -4.2500 | 0.6540 | no | -0.1917 | -0.1143 | 215 |
| data/batching/batching_test_split.json | relative_error_lb | 681929219451006.pth | 948172624102402.pth | 20 | -0.0106 | 0.6742 | no | -0.1613 | -0.1143 | 303 |
| data/batching/batching_test_split.json | scheduling_score | 681929219451006.pth | 948172624102402.pth | 20 | 0.0097 | 0.5459 | no | 0.2074 | 0.1619 | 184 |
| data/batching/batching_test_split.json | makespan | 681929219451006.pth | 537660859795754.pth | 20 | -1.0000 | 0.9702 | no | -0.0326 | -0.0095 | 7407 |
| data/batching/batching_test_split.json | relative_error_lb | 681929219451006.pth | 537660859795754.pth | 20 | -0.0008 | 0.9563 | no | -0.0084 | -0.0190 | 111119 |
| data/batching/batching_test_split.json | scheduling_score | 681929219451006.pth | 537660859795754.pth | 20 | 0.0003 | 0.9563 | no | 0.0054 | -0.0190 | 272720 |
| data/batching/batching_test_split.json | makespan | 948172624102402.pth | 537660859795754.pth | 20 | 3.2500 | 0.3316 | no | 0.0904 | 0.2476 | 963 |
| data/batching/batching_test_split.json | relative_error_lb | 948172624102402.pth | 537660859795754.pth | 20 | 0.0098 | 0.4524 | no | 0.0966 | 0.2000 | 842 |
| data/batching/batching_test_split.json | scheduling_score | 948172624102402.pth | 537660859795754.pth | 20 | -0.0094 | 0.3683 | no | -0.1383 | -0.2381 | 412 |
