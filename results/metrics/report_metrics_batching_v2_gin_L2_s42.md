# Metrics report: batching_constraint:report_metrics_batching_v2_gin_L2_s42

- created: 2026-10-04T09:07:13
- git: batching_constraint @ 7e23c8f4 (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 87572601051817.pth | ojmb_node_v2 | 20 | 396.4 ± 31.3 | 0.1070 ± 0.0441 | 0.1647 ± 0.0770 | 0.9092 ± 0.0341 | 0.0000 ± 0.0000 | 26.0365 ± 0.6594 | 32.0837 ± 0.1922 | 1357.6 ± 1.7 | 1.0000 ± 0.0000 | 1.0188 ± 0.0056 | 0.3140 ± 0.0146 | 0.3816 ± 0.0041 | 0.3873 ± 0.0061 | 1.1158 ± 0.0291 | 0.1952 ± 0.0447 | 1.0000 |
| 656861295438343.pth | ojmb_edge_v2 | 20 | 402.4 ± 32.7 | 0.1238 ± 0.0551 | 0.1820 ± 0.0830 | 0.8983 ± 0.0398 | 0.0000 ± 0.0000 | 20.5536 ± 0.3923 | 47.5406 ± 0.1913 | 1359.4 ± 1.1 | 1.0000 ± 0.0000 | 1.0206 ± 0.0043 | 0.2520 ± 0.0118 | 0.3508 ± 0.0062 | 0.3525 ± 0.0069 | 1.1112 ± 0.0310 | 0.1895 ± 0.0472 | 1.0000 |
| 607799136706282.pth | ojmb_base_v2 | 20 | 394.4 ± 28.1 | 0.1017 ± 0.0288 | 0.1584 ± 0.0658 | 0.9104 ± 0.0235 | 0.0000 ± 0.0000 | 18.5698 ± 0.2011 | 61.2958 ± 0.1913 | 1360.1 ± 1.0 | 1.0000 ± 0.0000 | 1.0261 ± 0.0088 | 0.2582 ± 0.0136 | 0.3673 ± 0.0051 | 0.3587 ± 0.0074 | 1.1124 ± 0.0270 | 0.1915 ± 0.0422 | 1.0000 |
| 255641086077630.pth | ojmb_feat_v2 | 20 | 396.6 ± 31.5 | 0.1061 ± 0.0397 | 0.1629 ± 0.0701 | 0.9088 ± 0.0310 | 0.0000 ± 0.0000 | 19.5221 ± 0.3754 | 75.0559 ± 0.1913 | 1361.4 ± 1.9 | 1.0000 ± 0.0000 | 1.0176 ± 0.0044 | 0.2555 ± 0.0105 | 0.3636 ± 0.0050 | 0.3630 ± 0.0115 | 1.1067 ± 0.0299 | 0.1791 ± 0.0428 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 87572601051817.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1070 | 0.1647 | 0.9092 | 0.0000 |
| 656861295438343.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1238 | 0.1820 | 0.8983 | 0.0000 |
| 607799136706282.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1017 | 0.1584 | 0.9104 | 0.0000 |
| 255641086077630.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1061 | 0.1629 | 0.9088 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 87572601051817.pth | 656861295438343.pth | 20 | -5.9000 | 0.7170 | no | -0.1726 | 0.0947 | 265 |
| data/batching/batching_test_split.json | relative_error_lb | 87572601051817.pth | 656861295438343.pth | 20 | -0.0173 | 0.7475 | no | -0.1670 | 0.0842 | 283 |
| data/batching/batching_test_split.json | scheduling_score | 87572601051817.pth | 656861295438343.pth | 20 | 0.0109 | 0.6874 | no | 0.1627 | -0.1053 | 298 |
| data/batching/batching_test_split.json | makespan | 87572601051817.pth | 607799136706282.pth | 20 | 2.0500 | 0.8092 | no | 0.0869 | 0.0632 | 1041 |
| data/batching/batching_test_split.json | relative_error_lb | 87572601051817.pth | 607799136706282.pth | 20 | 0.0063 | 0.8092 | no | 0.0907 | 0.0632 | 955 |
| data/batching/batching_test_split.json | scheduling_score | 87572601051817.pth | 607799136706282.pth | 20 | -0.0012 | 0.9039 | no | -0.0240 | -0.0316 | 13665 |
| data/batching/batching_test_split.json | makespan | 87572601051817.pth | 255641086077630.pth | 20 | -0.1500 | 0.5134 | no | -0.0049 | 0.1667 | 328324 |
| data/batching/batching_test_split.json | relative_error_lb | 87572601051817.pth | 255641086077630.pth | 20 | 0.0018 | 0.5459 | no | 0.0204 | 0.1619 | 18879 |
| data/batching/batching_test_split.json | scheduling_score | 87572601051817.pth | 255641086077630.pth | 20 | 0.0004 | 0.5706 | no | 0.0059 | -0.1524 | 228765 |
| data/batching/batching_test_split.json | makespan | 656861295438343.pth | 607799136706282.pth | 20 | 7.9500 | 0.3139 | no | 0.2503 | 0.2632 | 127 |
| data/batching/batching_test_split.json | relative_error_lb | 656861295438343.pth | 607799136706282.pth | 20 | 0.0236 | 0.2954 | no | 0.2417 | 0.2737 | 136 |
| data/batching/batching_test_split.json | scheduling_score | 656861295438343.pth | 607799136706282.pth | 20 | -0.0121 | 0.3760 | no | -0.2013 | -0.2316 | 195 |
| data/batching/batching_test_split.json | makespan | 656861295438343.pth | 255641086077630.pth | 20 | 5.7500 | 0.9479 | no | 0.1561 | 0.0175 | 324 |
| data/batching/batching_test_split.json | relative_error_lb | 656861295438343.pth | 255641086077630.pth | 20 | 0.0191 | 0.9479 | no | 0.1732 | 0.0175 | 263 |
| data/batching/batching_test_split.json | scheduling_score | 656861295438343.pth | 255641086077630.pth | 20 | -0.0106 | 0.9479 | no | -0.1566 | -0.0175 | 322 |
| data/batching/batching_test_split.json | makespan | 607799136706282.pth | 255641086077630.pth | 20 | -2.2000 | 0.6811 | no | -0.0956 | 0.1048 | 861 |
| data/batching/batching_test_split.json | relative_error_lb | 607799136706282.pth | 255641086077630.pth | 20 | -0.0045 | 0.6477 | no | -0.0657 | 0.1238 | 1819 |
| data/batching/batching_test_split.json | scheduling_score | 607799136706282.pth | 255641086077630.pth | 20 | 0.0015 | 0.6742 | no | 0.0319 | -0.1143 | 7710 |
