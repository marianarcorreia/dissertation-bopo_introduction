# Metrics report: batching_constraint:report_metrics_batching_gat_L2_s43_steps

- created: 2026-09-29T12:27:36
- git: batching_constraint @ 444ab7cc (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 690574809317532.pth | ojmb_node | 20 | 413.5 ± 34.4 | 0.1530 ± 0.0517 | 0.2135 ± 0.0864 | 0.8747 ± 0.0380 | 0.0000 ± 0.0000 | 53.2465 ± 5.9179 | 30.0024 ± 0.2398 | 2330.1 ± 1.6 | 1.0000 ± 0.0000 | 1.0180 ± 0.0048 | 0.3679 ± 0.0141 | 0.3634 ± 0.0042 | 0.3857 ± 0.0070 | 1.1126 ± 0.0263 | 0.1919 ± 0.0401 | 1.0000 |
| 429848009625277.pth | ojmb_edge | 20 | 401.4 ± 34.5 | 0.1178 ± 0.0465 | 0.1767 ± 0.0820 | 0.9010 ± 0.0353 | 0.0000 ± 0.0000 | 37.5239 ± 0.6934 | 43.7299 ± 0.2389 | 2333.0 ± 1.4 | 1.0000 ± 0.0000 | 1.0187 ± 0.0041 | 0.2926 ± 0.0113 | 0.3549 ± 0.0080 | 0.3485 ± 0.0079 | 1.1183 ± 0.0290 | 0.1994 ± 0.0433 | 1.0000 |
| 751138358325447.pth | ojmb_base | 20 | 410.2 ± 39.2 | 0.1406 ± 0.0595 | 0.2025 ± 0.0979 | 0.8865 ± 0.0433 | 0.0000 ± 0.0000 | 33.9785 ± 0.5097 | 55.9433 ± 0.2389 | 2332.0 ± 1.4 | 0.9998 ± 0.0004 | 1.0248 ± 0.0080 | 0.2836 ± 0.0150 | 0.3557 ± 0.0052 | 0.3673 ± 0.0099 | 1.1190 ± 0.0262 | 0.1977 ± 0.0382 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 690574809317532.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1530 | 0.2135 | 0.8747 | 0.0000 |
| 429848009625277.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1178 | 0.1767 | 0.9010 | 0.0000 |
| 751138358325447.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1406 | 0.2025 | 0.8865 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 690574809317532.pth | 429848009625277.pth | 20 | 12.1500 | 0.0462 | yes | 0.4248 | 0.5211 | 45 |
| data/batching/batching_test_split.json | relative_error_lb | 690574809317532.pth | 429848009625277.pth | 20 | 0.0368 | 0.0442 | yes | 0.4293 | 0.5263 | 44 |
| data/batching/batching_test_split.json | scheduling_score | 690574809317532.pth | 429848009625277.pth | 20 | -0.0263 | 0.0364 | yes | -0.4732 | -0.5474 | 37 |
| data/batching/batching_test_split.json | makespan | 690574809317532.pth | 751138358325447.pth | 20 | 3.2500 | 0.3868 | no | 0.0799 | 0.2263 | 1231 |
| data/batching/batching_test_split.json | relative_error_lb | 690574809317532.pth | 751138358325447.pth | 20 | 0.0110 | 0.4209 | no | 0.0957 | 0.2105 | 858 |
| data/batching/batching_test_split.json | scheduling_score | 690574809317532.pth | 751138358325447.pth | 20 | -0.0118 | 0.3760 | no | -0.1606 | -0.2316 | 306 |
| data/batching/batching_test_split.json | makespan | 429848009625277.pth | 751138358325447.pth | 20 | -8.9000 | 0.5861 | no | -0.2515 | -0.1462 | 126 |
| data/batching/batching_test_split.json | relative_error_lb | 429848009625277.pth | 751138358325447.pth | 20 | -0.0258 | 0.5862 | no | -0.2570 | -0.1462 | 120 |
| data/batching/batching_test_split.json | scheduling_score | 429848009625277.pth | 751138358325447.pth | 20 | 0.0145 | 0.5566 | no | 0.2328 | 0.1579 | 146 |
