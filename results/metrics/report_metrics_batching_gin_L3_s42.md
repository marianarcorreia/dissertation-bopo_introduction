# Metrics report: batching_constraint:report_metrics_batching_gin_L3_s42

- created: 2026-09-28T14:25:28
- git: batching_constraint @ 4197e439 (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 39108826801431.pth | ojmb_node | 20 | 400.0 ± 30.5 | 0.1172 ± 0.0419 | 0.1744 ± 0.0716 | 0.9004 ± 0.0325 | 0.0000 ± 0.0000 | 47.7796 ± 1.2556 | 46.8568 ± 0.1918 | 1358.2 ± 1.6 | 1.0000 ± 0.0000 | 1.0188 ± 0.0051 | 0.2656 ± 0.0121 | 0.3799 ± 0.0033 | 0.3804 ± 0.0066 | 1.1301 ± 0.0300 | 0.2189 ± 0.0442 | 1.0000 |
| 941200189550968.pth | ojmb_edge | 20 | 401.4 ± 33.7 | 0.1190 ± 0.0472 | 0.1782 ± 0.0844 | 0.9002 ± 0.0357 | 0.0000 ± 0.0000 | 36.7976 ± 1.1012 | 72.5442 ± 0.1909 | 1360.7 ± 0.9 | 1.0000 ± 0.0000 | 1.0212 ± 0.0046 | 0.2339 ± 0.0110 | 0.3570 ± 0.0068 | 0.3510 ± 0.0086 | 1.0815 ± 0.0249 | 0.1416 ± 0.0393 | 1.0000 |
| 982671064340682.pth | ojmb_base | 20 | 406.7 ± 35.3 | 0.1325 ± 0.0495 | 0.1935 ± 0.0899 | 0.8901 ± 0.0379 | 0.0000 ± 0.0000 | 34.2008 ± 1.1919 | 95.3943 ± 0.1909 | 1361.5 ± 1.1 | 1.0000 ± 0.0000 | 1.0260 ± 0.0090 | 0.2192 ± 0.0115 | 0.3412 ± 0.0039 | 0.3654 ± 0.0096 | 1.1096 ± 0.0273 | 0.1860 ± 0.0428 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 39108826801431.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1172 | 0.1744 | 0.9004 | 0.0000 |
| 941200189550968.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1190 | 0.1782 | 0.9002 | 0.0000 |
| 982671064340682.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1325 | 0.1935 | 0.8901 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 39108826801431.pth | 941200189550968.pth | 20 | -1.3500 | 0.8960 | no | -0.0433 | -0.0333 | 4186 |
| data/batching/batching_test_split.json | relative_error_lb | 39108826801431.pth | 941200189550968.pth | 20 | -0.0038 | 0.9854 | no | -0.0403 | -0.0095 | 4839 |
| data/batching/batching_test_split.json | scheduling_score | 39108826801431.pth | 941200189550968.pth | 20 | 0.0002 | 0.9563 | no | 0.0028 | 0.0190 | 1032192 |
| data/batching/batching_test_split.json | makespan | 39108826801431.pth | 982671064340682.pth | 20 | -6.7000 | 0.4780 | no | -0.2484 | -0.1810 | 129 |
| data/batching/batching_test_split.json | relative_error_lb | 39108826801431.pth | 982671064340682.pth | 20 | -0.0192 | 0.5217 | no | -0.2333 | -0.1714 | 146 |
| data/batching/batching_test_split.json | scheduling_score | 39108826801431.pth | 982671064340682.pth | 20 | 0.0102 | 0.6215 | no | 0.1752 | 0.1333 | 257 |
| data/batching/batching_test_split.json | makespan | 941200189550968.pth | 982671064340682.pth | 20 | -5.3500 | 0.2351 | no | -0.2213 | -0.3105 | 162 |
| data/batching/batching_test_split.json | relative_error_lb | 941200189550968.pth | 982671064340682.pth | 20 | -0.0154 | 0.2772 | no | -0.2108 | -0.2842 | 178 |
| data/batching/batching_test_split.json | scheduling_score | 941200189550968.pth | 982671064340682.pth | 20 | 0.0101 | 0.3341 | no | 0.1796 | 0.2526 | 245 |
