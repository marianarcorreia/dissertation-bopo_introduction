# Metrics report: batching_constraint:report_metrics_batching_gat_L2_s42_steps

- created: 2026-09-29T02:17:17
- git: batching_constraint @ 444ab7cc (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 796096899412646.pth | ojmb_node | 20 | 404.6 ± 36.1 | 0.1263 ± 0.0474 | 0.1841 ± 0.0755 | 0.8942 ± 0.0350 | 0.0000 ± 0.0000 | 37.5011 ± 0.9577 | 30.0024 ± 0.2398 | 1355.5 ± 1.4 | 0.9997 ± 0.0004 | 1.0191 ± 0.0054 | 0.3885 ± 0.0163 | 0.3871 ± 0.0039 | 0.3916 ± 0.0075 | 1.1375 ± 0.0259 | 0.2290 ± 0.0372 | 1.0000 |
| 683173606461368.pth | ojmb_edge | 20 | 398.4 ± 35.0 | 0.1085 ± 0.0434 | 0.1662 ± 0.0766 | 0.9078 ± 0.0336 | 0.0000 ± 0.0000 | 27.5300 ± 0.2503 | 43.7299 ± 0.2389 | 1358.8 ± 1.5 | 1.0000 ± 0.0000 | 1.0209 ± 0.0044 | 0.3236 ± 0.0148 | 0.3273 ± 0.0061 | 0.3510 ± 0.0080 | 1.1180 ± 0.0357 | 0.1945 ± 0.0511 | 1.0000 |
| 951042479229448.pth | ojmb_base | 20 | 404.2 ± 36.2 | 0.1259 ± 0.0561 | 0.1853 ± 0.0885 | 0.8968 ± 0.0402 | 0.0000 ± 0.0000 | 24.8534 ± 0.2728 | 55.9433 ± 0.2389 | 1359.2 ± 1.4 | 0.9999 ± 0.0002 | 1.0237 ± 0.0075 | 0.2691 ± 0.0133 | 0.3444 ± 0.0052 | 0.3649 ± 0.0108 | 1.1328 ± 0.0292 | 0.2196 ± 0.0417 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 796096899412646.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1263 | 0.1841 | 0.8942 | 0.0000 |
| 683173606461368.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1085 | 0.1662 | 0.9078 | 0.0000 |
| 951042479229448.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1259 | 0.1853 | 0.8968 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 796096899412646.pth | 683173606461368.pth | 20 | 6.3000 | 0.1840 | no | 0.3097 | 0.3474 | 83 |
| data/batching/batching_test_split.json | relative_error_lb | 796096899412646.pth | 683173606461368.pth | 20 | 0.0179 | 0.1590 | no | 0.3011 | 0.3684 | 88 |
| data/batching/batching_test_split.json | scheduling_score | 796096899412646.pth | 683173606461368.pth | 20 | -0.0136 | 0.1712 | no | -0.3150 | -0.3579 | 81 |
| data/batching/batching_test_split.json | makespan | 796096899412646.pth | 951042479229448.pth | 20 | 0.4000 | 0.5382 | no | 0.0131 | 0.1699 | 45832 |
| data/batching/batching_test_split.json | relative_error_lb | 796096899412646.pth | 951042479229448.pth | 20 | -0.0011 | 0.4925 | no | -0.0124 | 0.1895 | 50963 |
| data/batching/batching_test_split.json | scheduling_score | 796096899412646.pth | 951042479229448.pth | 20 | -0.0026 | 0.4348 | no | -0.0501 | -0.2157 | 3130 |
| data/batching/batching_test_split.json | makespan | 683173606461368.pth | 951042479229448.pth | 20 | -5.9000 | 0.8721 | no | -0.1658 | -0.0421 | 287 |
| data/batching/batching_test_split.json | relative_error_lb | 683173606461368.pth | 951042479229448.pth | 20 | -0.0190 | 0.9039 | no | -0.1767 | -0.0316 | 253 |
| data/batching/batching_test_split.json | scheduling_score | 683173606461368.pth | 951042479229448.pth | 20 | 0.0110 | 0.8405 | no | 0.1624 | 0.0526 | 299 |
