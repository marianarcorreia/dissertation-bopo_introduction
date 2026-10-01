# Metrics report: batching_constraint:report_metrics_batching_v2_gin_L3_k5_s42_steps

- created: 2026-09-30T19:38:30
- git: batching_constraint @ edc9dfae (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 61370701217751.pth | ojmb_node_v2 | 20 | 408.7 ± 27.4 | 0.1429 ± 0.0233 | 0.1999 ± 0.0553 | 0.8765 ± 0.0173 | 0.0000 ± 0.0000 | 37.9801 ± 0.9355 | 46.8981 ± 0.1922 | 1358.1 ± 1.0 | 0.9999 ± 0.0001 | 1.0190 ± 0.0054 | 0.2861 ± 0.0170 | 0.4055 ± 0.0046 | 0.3993 ± 0.0073 | 1.1387 ± 0.0317 | 0.2279 ± 0.0461 | 1.0000 |
| 415574885060126.pth | ojmb_edge_v2 | 20 | 413.8 ± 26.5 | 0.1582 ± 0.0287 | 0.2171 ± 0.0652 | 0.8656 ± 0.0210 | 0.0000 ± 0.0000 | 28.4209 ± 0.3661 | 72.6118 ± 0.1913 | 1361.6 ± 1.5 | 0.9998 ± 0.0001 | 1.0227 ± 0.0054 | 0.2632 ± 0.0159 | 0.3562 ± 0.0062 | 0.3521 ± 0.0081 | 1.1473 ± 0.0304 | 0.2410 ± 0.0442 | 1.0000 |
| 839658720056486.pth | ojmb_base_v2 | 20 | 412.8 ± 29.2 | 0.1542 ± 0.0359 | 0.2131 ± 0.0692 | 0.8700 ± 0.0269 | 0.0000 ± 0.0000 | 25.9962 ± 0.2773 | 95.4853 ± 0.1913 | 1362.6 ± 2.6 | 0.9999 ± 0.0001 | 1.0264 ± 0.0086 | 0.2350 ± 0.0120 | 0.3737 ± 0.0060 | 0.3607 ± 0.0093 | 1.1171 ± 0.0288 | 0.1971 ± 0.0441 | 1.0000 |
| 81917570931129.pth | ojmb_feat_v2 | 20 | 418.9 ± 30.7 | 0.1703 ± 0.0356 | 0.2279 ± 0.0588 | 0.8578 ± 0.0250 | 0.0000 ± 0.0000 | 27.1287 ± 0.4485 | 118.4 ± 0.2 | 1362.7 ± 1.3 | 0.9999 ± 0.0001 | 1.0201 ± 0.0043 | 0.2741 ± 0.0118 | 0.3680 ± 0.0065 | 0.3785 ± 0.0111 | 1.0903 ± 0.0303 | 0.1496 ± 0.0448 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 61370701217751.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1429 | 0.1999 | 0.8765 | 0.0000 |
| 415574885060126.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1582 | 0.2171 | 0.8656 | 0.0000 |
| 839658720056486.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1542 | 0.2131 | 0.8700 | 0.0000 |
| 81917570931129.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1703 | 0.2279 | 0.8578 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 61370701217751.pth | 415574885060126.pth | 20 | -5.0500 | 0.3316 | no | -0.2458 | -0.2476 | 131 |
| data/batching/batching_test_split.json | relative_error_lb | 61370701217751.pth | 415574885060126.pth | 20 | -0.0172 | 0.2943 | no | -0.2930 | -0.2762 | 93 |
| data/batching/batching_test_split.json | scheduling_score | 61370701217751.pth | 415574885060126.pth | 20 | 0.0109 | 0.3488 | no | 0.2571 | 0.2476 | 120 |
| data/batching/batching_test_split.json | makespan | 61370701217751.pth | 839658720056486.pth | 20 | -4.1000 | 0.8091 | no | -0.1508 | -0.0632 | 347 |
| data/batching/batching_test_split.json | relative_error_lb | 61370701217751.pth | 839658720056486.pth | 20 | -0.0133 | 0.7782 | no | -0.1687 | -0.0737 | 277 |
| data/batching/batching_test_split.json | scheduling_score | 61370701217751.pth | 839658720056486.pth | 20 | 0.0065 | 0.9039 | no | 0.1192 | 0.0316 | 554 |
| data/batching/batching_test_split.json | makespan | 61370701217751.pth | 81917570931129.pth | 20 | -10.2000 | 0.0893 | no | -0.4317 | -0.4561 | 44 |
| data/batching/batching_test_split.json | relative_error_lb | 61370701217751.pth | 81917570931129.pth | 20 | -0.0281 | 0.0936 | no | -0.4154 | -0.4503 | 47 |
| data/batching/batching_test_split.json | scheduling_score | 61370701217751.pth | 81917570931129.pth | 20 | 0.0188 | 0.0778 | no | 0.3961 | 0.4737 | 52 |
| data/batching/batching_test_split.json | makespan | 415574885060126.pth | 839658720056486.pth | 20 | 0.9500 | 0.6406 | no | 0.0376 | 0.1190 | 5544 |
| data/batching/batching_test_split.json | relative_error_lb | 415574885060126.pth | 839658720056486.pth | 20 | 0.0040 | 0.6215 | no | 0.0561 | 0.1333 | 2492 |
| data/batching/batching_test_split.json | scheduling_score | 415574885060126.pth | 839658720056486.pth | 20 | -0.0044 | 0.6215 | no | -0.0857 | -0.1333 | 1071 |
| data/batching/batching_test_split.json | makespan | 415574885060126.pth | 81917570931129.pth | 20 | -5.1500 | 0.3504 | no | -0.1810 | -0.2381 | 241 |
| data/batching/batching_test_split.json | relative_error_lb | 415574885060126.pth | 81917570931129.pth | 20 | -0.0108 | 0.3884 | no | -0.1355 | -0.2286 | 429 |
| data/batching/batching_test_split.json | scheduling_score | 415574885060126.pth | 81917570931129.pth | 20 | 0.0079 | 0.3884 | no | 0.1477 | 0.2286 | 361 |
| data/batching/batching_test_split.json | makespan | 839658720056486.pth | 81917570931129.pth | 20 | -6.1000 | 0.4440 | no | -0.1721 | -0.1952 | 267 |
| data/batching/batching_test_split.json | relative_error_lb | 839658720056486.pth | 81917570931129.pth | 20 | -0.0148 | 0.5217 | no | -0.1477 | -0.1714 | 361 |
| data/batching/batching_test_split.json | scheduling_score | 839658720056486.pth | 81917570931129.pth | 20 | 0.0123 | 0.4749 | no | 0.1839 | 0.1905 | 234 |
