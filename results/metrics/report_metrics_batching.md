# Metrics report: batching_constraint:report_metrics_batching

- created: 2026-09-28T05:07:22
- git: batching_constraint @ 452b569e (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 477356094558498.pth | ojmb_node | 20 | 401.6 ± 32.7 | 0.1204 ± 0.0423 | 0.1790 ± 0.0781 | 0.8979 ± 0.0326 | 0.0000 ± 0.0000 | 39.9787 ± 0.6770 | 30.0024 ± 0.2398 | 1354.9 ± 1.7 | 0.9999 ± 0.0002 | 1.0185 ± 0.0052 | 0.3787 ± 0.0146 | 0.3764 ± 0.0043 | 0.3959 ± 0.0057 | 1.1348 ± 0.0334 | 0.2211 ± 0.0447 | 1.0000 |
| 852735153700113.pth | ojmb_edge | 20 | 407.6 ± 36.6 | 0.1345 ± 0.0522 | 0.1954 ± 0.0899 | 0.8893 ± 0.0396 | 0.0000 ± 0.0000 | 29.8407 ± 0.3413 | 43.7299 ± 0.2389 | 1358.3 ± 1.1 | 1.0000 ± 0.0000 | 1.0202 ± 0.0044 | 0.2922 ± 0.0125 | 0.3392 ± 0.0066 | 0.3500 ± 0.0084 | 1.1066 ± 0.0269 | 0.1839 ± 0.0434 | 1.0000 |
| 541188441970801.pth | ojmb_base | 20 | 396.9 ± 32.3 | 0.1067 ± 0.0415 | 0.1645 ± 0.0765 | 0.9088 ± 0.0324 | 0.0000 ± 0.0000 | 26.7466 ± 0.2896 | 55.9433 ± 0.2389 | 1358.3 ± 1.3 | 0.9999 ± 0.0002 | 1.0273 ± 0.0089 | 0.2820 ± 0.0158 | 0.3349 ± 0.0046 | 0.3640 ± 0.0104 | 1.1385 ± 0.0297 | 0.2256 ± 0.0410 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 477356094558498.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1204 | 0.1790 | 0.8979 | 0.0000 |
| 852735153700113.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1345 | 0.1954 | 0.8893 | 0.0000 |
| 541188441970801.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1067 | 0.1645 | 0.9088 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 477356094558498.pth | 852735153700113.pth | 20 | -6.0000 | 0.4208 | no | -0.2182 | -0.2105 | 166 |
| data/batching/batching_test_split.json | relative_error_lb | 477356094558498.pth | 852735153700113.pth | 20 | -0.0163 | 0.4445 | no | -0.2013 | -0.2000 | 195 |
| data/batching/batching_test_split.json | scheduling_score | 477356094558498.pth | 852735153700113.pth | 20 | 0.0086 | 0.4445 | no | 0.1430 | 0.2000 | 385 |
| data/batching/batching_test_split.json | makespan | 477356094558498.pth | 541188441970801.pth | 20 | 4.7000 | 0.1903 | no | 0.2519 | 0.3421 | 125 |
| data/batching/batching_test_split.json | relative_error_lb | 477356094558498.pth | 541188441970801.pth | 20 | 0.0145 | 0.2122 | no | 0.2540 | 0.3263 | 123 |
| data/batching/batching_test_split.json | scheduling_score | 477356094558498.pth | 541188441970801.pth | 20 | -0.0109 | 0.2122 | no | -0.2477 | -0.3263 | 129 |
| data/batching/batching_test_split.json | makespan | 852735153700113.pth | 541188441970801.pth | 20 | 10.7000 | 0.0878 | no | 0.4804 | 0.4853 | 36 |
| data/batching/batching_test_split.json | relative_error_lb | 852735153700113.pth | 541188441970801.pth | 20 | 0.0309 | 0.0787 | no | 0.4752 | 0.5000 | 36 |
| data/batching/batching_test_split.json | scheduling_score | 852735153700113.pth | 541188441970801.pth | 20 | -0.0195 | 0.0787 | no | -0.4374 | -0.5000 | 43 |
