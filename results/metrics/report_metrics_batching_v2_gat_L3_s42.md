# Metrics report: batching_constraint:report_metrics_batching_v2_gat_L3_s42

- created: 2026-10-03T16:57:52
- git: batching_constraint @ c5a62f6c (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 38030729443917.pth | ojmb_node_v2 | 20 | 406.7 ± 35.0 | 0.1331 ± 0.0523 | 0.1919 ± 0.0819 | 0.8904 ± 0.0391 | 0.0000 ± 0.0000 | 54.2316 ± 0.8174 | 44.8592 ± 0.2402 | 1355.4 ± 1.7 | 1.0000 ± 0.0000 | 1.0154 ± 0.0053 | 0.4122 ± 0.0129 | 0.3896 ± 0.0045 | 0.3916 ± 0.0060 | 1.1037 ± 0.0339 | 0.1728 ± 0.0496 | 1.0000 |
| 603361675692974.pth | ojmb_edge_v2 | 20 | 399.5 ± 30.2 | 0.1158 ± 0.0393 | 0.1734 ± 0.0721 | 0.9009 ± 0.0311 | 0.0000 ± 0.0000 | 40.4255 ± 0.3519 | 68.8698 ± 0.2393 | 1358.3 ± 1.3 | 1.0000 ± 0.0000 | 1.0178 ± 0.0042 | 0.2624 ± 0.0114 | 0.3356 ± 0.0070 | 0.3404 ± 0.0081 | 1.0934 ± 0.0299 | 0.1544 ± 0.0448 | 1.0000 |
| 57920169926805.pth | ojmb_base_v2 | 20 | 392.5 ± 29.6 | 0.0959 ± 0.0348 | 0.1526 ± 0.0690 | 0.9163 ± 0.0282 | 0.0000 ± 0.0000 | 36.9290 ± 0.9203 | 90.2236 ± 0.2392 | 1358.7 ± 1.2 | 0.9996 ± 0.0004 | 1.0210 ± 0.0093 | 0.2129 ± 0.0106 | 0.3657 ± 0.0058 | 0.3597 ± 0.0099 | 1.1140 ± 0.0314 | 0.1897 ± 0.0485 | 1.0000 |
| 255475133880190.pth | ojmb_feat_v2 | 20 | 398.4 ± 33.0 | 0.1104 ± 0.0398 | 0.1673 ± 0.0702 | 0.9054 ± 0.0314 | 0.0000 ± 0.0000 | 40.9034 ± 1.0715 | 111.6 ± 0.2 | 1359.8 ± 2.2 | 0.9990 ± 0.0018 | 1.0142 ± 0.0040 | 0.2754 ± 0.0133 | 0.3638 ± 0.0055 | 0.3670 ± 0.0108 | 1.1130 ± 0.0362 | 0.1871 ± 0.0533 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 38030729443917.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1331 | 0.1919 | 0.8904 | 0.0000 |
| 603361675692974.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1158 | 0.1734 | 0.9009 | 0.0000 |
| 57920169926805.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.0959 | 0.1526 | 0.9163 | 0.0000 |
| 255475133880190.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1104 | 0.1673 | 0.9054 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 38030729443917.pth | 603361675692974.pth | 20 | 7.2000 | 0.3043 | no | 0.2570 | 0.2684 | 120 |
| data/batching/batching_test_split.json | relative_error_lb | 38030729443917.pth | 603361675692974.pth | 20 | 0.0185 | 0.4445 | no | 0.2123 | 0.2000 | 176 |
| data/batching/batching_test_split.json | scheduling_score | 38030729443917.pth | 603361675692974.pth | 20 | -0.0106 | 0.3144 | no | -0.1751 | -0.2632 | 258 |
| data/batching/batching_test_split.json | makespan | 38030729443917.pth | 57920169926805.pth | 20 | 14.2000 | 0.0175 | yes | 0.5503 | 0.6211 | 27 |
| data/batching/batching_test_split.json | relative_error_lb | 38030729443917.pth | 57920169926805.pth | 20 | 0.0393 | 0.0158 | yes | 0.4983 | 0.6316 | 33 |
| data/batching/batching_test_split.json | scheduling_score | 38030729443917.pth | 57920169926805.pth | 20 | -0.0260 | 0.0176 | yes | -0.4558 | -0.6211 | 39 |
| data/batching/batching_test_split.json | makespan | 38030729443917.pth | 255475133880190.pth | 20 | 8.2500 | 0.3378 | no | 0.2705 | 0.2573 | 109 |
| data/batching/batching_test_split.json | relative_error_lb | 38030729443917.pth | 255475133880190.pth | 20 | 0.0247 | 0.3720 | no | 0.2669 | 0.2398 | 112 |
| data/batching/batching_test_split.json | scheduling_score | 38030729443917.pth | 255475133880190.pth | 20 | -0.0151 | 0.2860 | no | -0.2426 | -0.2865 | 135 |
| data/batching/batching_test_split.json | makespan | 603361675692974.pth | 57920169926805.pth | 20 | 7.0000 | 0.1975 | no | 0.3267 | 0.3286 | 75 |
| data/batching/batching_test_split.json | relative_error_lb | 603361675692974.pth | 57920169926805.pth | 20 | 0.0208 | 0.2471 | no | 0.3304 | 0.2952 | 73 |
| data/batching/batching_test_split.json | scheduling_score | 603361675692974.pth | 57920169926805.pth | 20 | -0.0154 | 0.2162 | no | -0.3291 | -0.3238 | 74 |
| data/batching/batching_test_split.json | makespan | 603361675692974.pth | 255475133880190.pth | 20 | 1.0500 | 0.6789 | no | 0.0445 | 0.1111 | 3965 |
| data/batching/batching_test_split.json | relative_error_lb | 603361675692974.pth | 255475133880190.pth | 20 | 0.0061 | 0.7439 | no | 0.0908 | 0.0877 | 954 |
| data/batching/batching_test_split.json | scheduling_score | 603361675692974.pth | 255475133880190.pth | 20 | -0.0045 | 0.7439 | no | -0.0910 | -0.0877 | 949 |
| data/batching/batching_test_split.json | makespan | 57920169926805.pth | 255475133880190.pth | 20 | -5.9500 | 0.1613 | no | -0.2709 | -0.3571 | 108 |
| data/batching/batching_test_split.json | relative_error_lb | 57920169926805.pth | 255475133880190.pth | 20 | -0.0147 | 0.1893 | no | -0.2188 | -0.3429 | 165 |
| data/batching/batching_test_split.json | scheduling_score | 57920169926805.pth | 255475133880190.pth | 20 | 0.0109 | 0.1769 | no | 0.2125 | 0.3524 | 175 |
