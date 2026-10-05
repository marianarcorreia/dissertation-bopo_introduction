# Metrics report: batching_constraint:report_metrics_batching_v2_gin_L3_s42

- created: 2026-10-04T01:20:47
- git: batching_constraint @ 7e23c8f4 (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 502767954072923.pth | ojmb_node_v2 | 20 | 400.2 ± 37.8 | 0.1133 ± 0.0557 | 0.1717 ± 0.0861 | 0.9070 ± 0.0408 | 0.0000 ± 0.0000 | 33.5397 ± 0.6167 | 46.8981 ± 0.1922 | 1358.2 ± 2.0 | 1.0000 ± 0.0000 | 1.0180 ± 0.0049 | 0.3064 ± 0.0147 | 0.3914 ± 0.0049 | 0.3863 ± 0.0089 | 1.1008 ± 0.0337 | 0.1692 ± 0.0508 | 1.0000 |
| 315686338266069.pth | ojmb_edge_v2 | 20 | 400.3 ± 34.9 | 0.1151 ± 0.0500 | 0.1739 ± 0.0836 | 0.9043 ± 0.0386 | 0.0000 ± 0.0000 | 25.7064 ± 0.2649 | 72.6118 ± 0.1913 | 1360.3 ± 1.5 | 1.0000 ± 0.0000 | 1.0207 ± 0.0045 | 0.2700 ± 0.0120 | 0.3380 ± 0.0053 | 0.3589 ± 0.0094 | 1.1092 ± 0.0239 | 0.1851 ± 0.0353 | 1.0000 |
| 670714460629214.pth | ojmb_base_v2 | 20 | 399.9 ± 33.9 | 0.1139 ± 0.0440 | 0.1716 ± 0.0756 | 0.9035 ± 0.0339 | 0.0000 ± 0.0000 | 23.5044 ± 0.2288 | 95.4853 ± 0.1913 | 1361.7 ± 1.8 | 1.0000 ± 0.0000 | 1.0264 ± 0.0089 | 0.1952 ± 0.0081 | 0.3535 ± 0.0060 | 0.3623 ± 0.0086 | 1.1316 ± 0.0319 | 0.2180 ± 0.0452 | 1.0000 |
| 835769155240958.pth | ojmb_feat_v2 | 20 | 394.6 ± 29.6 | 0.1013 ± 0.0358 | 0.1576 ± 0.0664 | 0.9120 ± 0.0285 | 0.0000 ± 0.0000 | 24.4125 ± 0.3047 | 118.4 ± 0.2 | 1361.0 ± 1.4 | 0.9999 ± 0.0003 | 1.0180 ± 0.0044 | 0.2306 ± 0.0109 | 0.3697 ± 0.0059 | 0.3647 ± 0.0116 | 1.1167 ± 0.0276 | 0.1956 ± 0.0401 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 502767954072923.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1133 | 0.1717 | 0.9070 | 0.0000 |
| 315686338266069.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1151 | 0.1739 | 0.9043 | 0.0000 |
| 670714460629214.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1139 | 0.1716 | 0.9035 | 0.0000 |
| 835769155240958.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1013 | 0.1576 | 0.9120 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 502767954072923.pth | 315686338266069.pth | 20 | -0.0500 | 0.9518 | no | -0.0019 | 0.0158 | 2266254 |
| data/batching/batching_test_split.json | relative_error_lb | 502767954072923.pth | 315686338266069.pth | 20 | -0.0022 | 0.9679 | no | -0.0293 | -0.0105 | 9121 |
| data/batching/batching_test_split.json | scheduling_score | 502767954072923.pth | 315686338266069.pth | 20 | 0.0028 | 0.9679 | no | 0.0565 | -0.0105 | 2461 |
| data/batching/batching_test_split.json | makespan | 502767954072923.pth | 670714460629214.pth | 20 | 0.4000 | 0.7781 | no | 0.0145 | -0.0737 | 37110 |
| data/batching/batching_test_split.json | relative_error_lb | 502767954072923.pth | 670714460629214.pth | 20 | 0.0001 | 0.7782 | no | 0.0016 | -0.0737 | 3092747 |
| data/batching/batching_test_split.json | scheduling_score | 502767954072923.pth | 670714460629214.pth | 20 | 0.0035 | 0.8092 | no | 0.0629 | 0.0632 | 1985 |
| data/batching/batching_test_split.json | makespan | 502767954072923.pth | 835769155240958.pth | 20 | 5.6500 | 0.7227 | no | 0.1616 | -0.0905 | 302 |
| data/batching/batching_test_split.json | relative_error_lb | 502767954072923.pth | 835769155240958.pth | 20 | 0.0141 | 0.7562 | no | 0.1480 | -0.0857 | 360 |
| data/batching/batching_test_split.json | scheduling_score | 502767954072923.pth | 835769155240958.pth | 20 | -0.0050 | 0.7012 | no | -0.0808 | 0.1048 | 1203 |
| data/batching/batching_test_split.json | makespan | 315686338266069.pth | 670714460629214.pth | 20 | 0.4500 | 0.9132 | no | 0.0183 | -0.0292 | 23451 |
| data/batching/batching_test_split.json | relative_error_lb | 315686338266069.pth | 670714460629214.pth | 20 | 0.0024 | 0.8789 | no | 0.0332 | -0.0409 | 7119 |
| data/batching/batching_test_split.json | scheduling_score | 315686338266069.pth | 670714460629214.pth | 20 | 0.0007 | 0.9133 | no | 0.0131 | 0.0292 | 45482 |
| data/batching/batching_test_split.json | makespan | 315686338266069.pth | 835769155240958.pth | 20 | 5.7000 | 0.4723 | no | 0.2391 | 0.1930 | 139 |
| data/batching/batching_test_split.json | relative_error_lb | 315686338266069.pth | 835769155240958.pth | 20 | 0.0163 | 0.3720 | no | 0.2422 | 0.2398 | 135 |
| data/batching/batching_test_split.json | scheduling_score | 315686338266069.pth | 835769155240958.pth | 20 | -0.0078 | 0.4460 | no | -0.1576 | -0.2047 | 317 |
| data/batching/batching_test_split.json | makespan | 670714460629214.pth | 835769155240958.pth | 20 | 5.2500 | 0.4327 | no | 0.2482 | 0.2105 | 129 |
| data/batching/batching_test_split.json | relative_error_lb | 670714460629214.pth | 835769155240958.pth | 20 | 0.0140 | 0.4460 | no | 0.2435 | 0.2047 | 134 |
| data/batching/batching_test_split.json | scheduling_score | 670714460629214.pth | 835769155240958.pth | 20 | -0.0085 | 0.4724 | no | -0.2124 | -0.1930 | 176 |
