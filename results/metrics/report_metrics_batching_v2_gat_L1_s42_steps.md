# Metrics report: batching_constraint:report_metrics_batching_v2_gat_L1_s42_steps

- created: 2026-10-04T19:47:42
- git: batching_constraint @ 7e23c8f4 (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 135502049729475.pth | ojmb_node_v2 | 20 | 405.5 ± 36.9 | 0.1287 ± 0.0491 | 0.1869 ± 0.0779 | 0.8928 ± 0.0364 | 0.0000 ± 0.0000 | 24.0325 ± 0.7526 | 15.1530 ± 0.2386 | 1340.1 ± 1.1 | 0.9998 ± 0.0003 | 1.0189 ± 0.0054 | 0.4101 ± 0.0152 | 0.3795 ± 0.0038 | 0.3918 ± 0.0071 | 1.1311 ± 0.0287 | 0.2184 ± 0.0428 | 1.0000 |
| 61987014244890.pth | ojmb_edge_v2 | 20 | 395.6 ± 35.0 | 0.1010 ± 0.0447 | 0.1586 ± 0.0780 | 0.9144 ± 0.0352 | 0.0000 ± 0.0000 | 18.9194 ± 0.3452 | 18.6602 ± 0.2380 | 1341.3 ± 0.9 | 1.0000 ± 0.0000 | 1.0208 ± 0.0047 | 0.3866 ± 0.0136 | 0.3640 ± 0.0059 | 0.3529 ± 0.0093 | 1.1068 ± 0.0273 | 0.1787 ± 0.0396 | 1.0000 |
| 596128222770811.pth | ojmb_base_v2 | 20 | 404.6 ± 34.7 | 0.1278 ± 0.0490 | 0.1866 ± 0.0806 | 0.8937 ± 0.0373 | 0.0000 ± 0.0000 | 17.1587 ± 0.2662 | 21.7797 ± 0.2379 | 1341.8 ± 0.9 | 0.9998 ± 0.0003 | 1.0260 ± 0.0088 | 0.2780 ± 0.0141 | 0.3656 ± 0.0054 | 0.3605 ± 0.0110 | 1.1211 ± 0.0291 | 0.2039 ± 0.0447 | 1.0000 |
| 611250957417164.pth | ojmb_feat_v2 | 20 | 402.6 ± 37.6 | 0.1196 ± 0.0554 | 0.1780 ± 0.0849 | 0.9020 ± 0.0413 | 0.0000 ± 0.0000 | 18.1370 ± 0.3698 | 24.9089 ± 0.2380 | 1341.3 ± 0.8 | 1.0000 ± 0.0000 | 1.0168 ± 0.0040 | 0.3833 ± 0.0167 | 0.3725 ± 0.0045 | 0.3733 ± 0.0121 | 1.1171 ± 0.0274 | 0.1963 ± 0.0405 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 135502049729475.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1287 | 0.1869 | 0.8928 | 0.0000 |
| 61987014244890.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1010 | 0.1586 | 0.9144 | 0.0000 |
| 596128222770811.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1278 | 0.1866 | 0.8937 | 0.0000 |
| 611250957417164.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1196 | 0.1780 | 0.9020 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 135502049729475.pth | 61987014244890.pth | 20 | 9.9000 | 0.1166 | no | 0.4878 | 0.4000 | 34 |
| data/batching/batching_test_split.json | relative_error_lb | 135502049729475.pth | 61987014244890.pth | 20 | 0.0282 | 0.1327 | no | 0.4637 | 0.3905 | 38 |
| data/batching/batching_test_split.json | scheduling_score | 135502049729475.pth | 61987014244890.pth | 20 | -0.0216 | 0.1231 | no | -0.4546 | -0.4000 | 39 |
| data/batching/batching_test_split.json | makespan | 135502049729475.pth | 596128222770811.pth | 20 | 0.9500 | 0.5626 | no | 0.0383 | 0.1476 | 5346 |
| data/batching/batching_test_split.json | relative_error_lb | 135502049729475.pth | 596128222770811.pth | 20 | 0.0002 | 0.5958 | no | 0.0032 | 0.1429 | 774140 |
| data/batching/batching_test_split.json | scheduling_score | 135502049729475.pth | 596128222770811.pth | 20 | -0.0009 | 0.6742 | no | -0.0168 | -0.1143 | 27938 |
| data/batching/batching_test_split.json | makespan | 135502049729475.pth | 611250957417164.pth | 20 | 2.9500 | 0.5566 | no | 0.1134 | 0.1579 | 612 |
| data/batching/batching_test_split.json | relative_error_lb | 135502049729475.pth | 611250957417164.pth | 20 | 0.0089 | 0.5862 | no | 0.1151 | 0.1462 | 594 |
| data/batching/batching_test_split.json | scheduling_score | 135502049729475.pth | 611250957417164.pth | 20 | -0.0092 | 0.5277 | no | -0.1676 | -0.1696 | 281 |
| data/batching/batching_test_split.json | makespan | 61987014244890.pth | 596128222770811.pth | 20 | -8.9500 | 0.1041 | no | -0.4253 | -0.4143 | 45 |
| data/batching/batching_test_split.json | relative_error_lb | 61987014244890.pth | 596128222770811.pth | 20 | -0.0280 | 0.1054 | no | -0.4436 | -0.4190 | 41 |
| data/batching/batching_test_split.json | scheduling_score | 61987014244890.pth | 596128222770811.pth | 20 | 0.0208 | 0.0973 | no | 0.4658 | 0.4286 | 38 |
| data/batching/batching_test_split.json | makespan | 61987014244890.pth | 611250957417164.pth | 20 | -6.9500 | 0.1117 | no | -0.3338 | -0.4158 | 72 |
| data/batching/batching_test_split.json | relative_error_lb | 61987014244890.pth | 611250957417164.pth | 20 | -0.0194 | 0.1262 | no | -0.3253 | -0.4000 | 76 |
| data/batching/batching_test_split.json | scheduling_score | 61987014244890.pth | 611250957417164.pth | 20 | 0.0124 | 0.1165 | no | 0.3350 | 0.4105 | 71 |
| data/batching/batching_test_split.json | makespan | 596128222770811.pth | 611250957417164.pth | 20 | 2.0000 | 0.4687 | no | 0.0777 | 0.1895 | 1302 |
| data/batching/batching_test_split.json | relative_error_lb | 596128222770811.pth | 611250957417164.pth | 20 | 0.0086 | 0.4445 | no | 0.1133 | 0.2000 | 613 |
| data/batching/batching_test_split.json | scheduling_score | 596128222770811.pth | 611250957417164.pth | 20 | -0.0083 | 0.3760 | no | -0.1544 | -0.2316 | 331 |
