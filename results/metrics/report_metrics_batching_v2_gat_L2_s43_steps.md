# Metrics report: batching_constraint:report_metrics_batching_v2_gat_L2_s43_steps

- created: 2026-10-01T10:00:12
- git: batching_constraint @ edc9dfae (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 896415652061583.pth | ojmb_node_v2 | 20 | 414.3 ± 45.6 | 0.1498 ± 0.0789 | 0.2115 ± 0.1100 | 0.8853 ± 0.0528 | 0.0000 ± 0.0000 | 44.6004 ± 1.1812 | 30.0437 ± 0.2402 | 1354.3 ± 1.5 | 0.9999 ± 0.0002 | 1.0181 ± 0.0055 | 0.4086 ± 0.0202 | 0.3983 ± 0.0048 | 0.3915 ± 0.0048 | 1.1085 ± 0.0290 | 0.1814 ± 0.0439 | 1.0000 |
| 353679720639032.pth | ojmb_edge_v2 | 20 | 403.4 ± 33.3 | 0.1256 ± 0.0465 | 0.1843 ± 0.0787 | 0.8948 ± 0.0360 | 0.0000 ± 0.0000 | 33.2519 ± 0.2719 | 43.7975 ± 0.2393 | 1358.6 ± 1.0 | 1.0000 ± 0.0000 | 1.0177 ± 0.0041 | 0.3709 ± 0.0153 | 0.3334 ± 0.0058 | 0.3486 ± 0.0084 | 1.1167 ± 0.0277 | 0.1941 ± 0.0419 | 1.0000 |
| 17360939543554.pth | ojmb_base_v2 | 20 | 407.4 ± 37.8 | 0.1345 ± 0.0562 | 0.1954 ± 0.0947 | 0.8902 ± 0.0407 | 0.0000 ± 0.0000 | 30.1483 ± 0.2692 | 56.0342 ± 0.2392 | 1358.6 ± 1.7 | 1.0000 ± 0.0000 | 1.0251 ± 0.0090 | 0.3110 ± 0.0152 | 0.3573 ± 0.0053 | 0.3645 ± 0.0095 | 1.1223 ± 0.0263 | 0.2060 ± 0.0385 | 1.0000 |
| 482729522553108.pth | ojmb_feat_v2 | 20 | 404.1 ± 35.3 | 0.1246 ± 0.0471 | 0.1840 ± 0.0832 | 0.8958 ± 0.0368 | 0.0000 ± 0.0000 | 31.2203 ± 0.3019 | 68.2806 ± 0.2393 | 1357.7 ± 0.8 | 0.9999 ± 0.0003 | 1.0160 ± 0.0041 | 0.3279 ± 0.0168 | 0.3735 ± 0.0053 | 0.3701 ± 0.0104 | 1.1103 ± 0.0318 | 0.1867 ± 0.0493 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 896415652061583.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1498 | 0.2115 | 0.8853 | 0.0000 |
| 353679720639032.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1256 | 0.1843 | 0.8948 | 0.0000 |
| 17360939543554.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1345 | 0.1954 | 0.8902 | 0.0000 |
| 482729522553108.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1246 | 0.1840 | 0.8958 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 896415652061583.pth | 353679720639032.pth | 20 | 10.8500 | 0.2145 | no | 0.2713 | 0.3333 | 108 |
| data/batching/batching_test_split.json | relative_error_lb | 896415652061583.pth | 353679720639032.pth | 20 | 0.0272 | 0.2860 | no | 0.2313 | 0.2865 | 148 |
| data/batching/batching_test_split.json | scheduling_score | 896415652061583.pth | 353679720639032.pth | 20 | -0.0095 | 0.3958 | no | -0.1365 | -0.2281 | 423 |
| data/batching/batching_test_split.json | makespan | 896415652061583.pth | 17360939543554.pth | 20 | 6.9000 | 0.5868 | no | 0.2148 | 0.1421 | 172 |
| data/batching/batching_test_split.json | relative_error_lb | 896415652061583.pth | 17360939543554.pth | 20 | 0.0162 | 0.6874 | no | 0.1716 | 0.1053 | 268 |
| data/batching/batching_test_split.json | scheduling_score | 896415652061583.pth | 17360939543554.pth | 20 | -0.0049 | 0.7475 | no | -0.0871 | -0.0842 | 1037 |
| data/batching/batching_test_split.json | makespan | 896415652061583.pth | 482729522553108.pth | 20 | 10.2500 | 0.4438 | no | 0.2447 | 0.2000 | 133 |
| data/batching/batching_test_split.json | relative_error_lb | 896415652061583.pth | 482729522553108.pth | 20 | 0.0275 | 0.4939 | no | 0.2305 | 0.1789 | 149 |
| data/batching/batching_test_split.json | scheduling_score | 896415652061583.pth | 482729522553108.pth | 20 | -0.0105 | 0.4445 | no | -0.1490 | -0.2000 | 355 |
| data/batching/batching_test_split.json | makespan | 353679720639032.pth | 17360939543554.pth | 20 | -3.9500 | 0.2952 | no | -0.1290 | -0.2737 | 473 |
| data/batching/batching_test_split.json | relative_error_lb | 353679720639032.pth | 17360939543554.pth | 20 | -0.0110 | 0.2954 | no | -0.1190 | -0.2737 | 556 |
| data/batching/batching_test_split.json | scheduling_score | 353679720639032.pth | 17360939543554.pth | 20 | 0.0047 | 0.2598 | no | 0.0803 | 0.2947 | 1219 |
| data/batching/batching_test_split.json | makespan | 353679720639032.pth | 482729522553108.pth | 20 | -0.6000 | 0.9107 | no | -0.0248 | -0.0286 | 12758 |
| data/batching/batching_test_split.json | relative_error_lb | 353679720639032.pth | 482729522553108.pth | 20 | 0.0003 | 0.9563 | no | 0.0043 | 0.0190 | 433079 |
| data/batching/batching_test_split.json | scheduling_score | 353679720639032.pth | 482729522553108.pth | 20 | -0.0010 | 0.9273 | no | -0.0175 | -0.0286 | 25492 |
| data/batching/batching_test_split.json | makespan | 17360939543554.pth | 482729522553108.pth | 20 | 3.3500 | 0.3437 | no | 0.1187 | 0.2614 | 559 |
| data/batching/batching_test_split.json | relative_error_lb | 17360939543554.pth | 482729522553108.pth | 20 | 0.0113 | 0.3318 | no | 0.1289 | 0.2680 | 474 |
| data/batching/batching_test_split.json | scheduling_score | 17360939543554.pth | 482729522553108.pth | 20 | -0.0057 | 0.3812 | no | -0.0870 | -0.2418 | 1038 |
