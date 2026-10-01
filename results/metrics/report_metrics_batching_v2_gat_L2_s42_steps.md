# Metrics report: batching_constraint:report_metrics_batching_v2_gat_L2_s42_steps

- created: 2026-09-30T05:18:48
- git: batching_constraint @ edc9dfae (uncommitted changes)
- device: cuda
- models file: candidate_models/model_params.json

## data/batching/batching_test_split.json

Mean ± 95% CI half-width.

| Model | Repr. | n | Makespan | RE (CP-SAT) | RE (LB) | Score | Viol./step | ms/decision | GPU peak MB | RSS MB | Action distinct | Structure gain | Eff. rank | Heterophily (emb) | Heterophily (feat) | Batch size | Batched ops | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 814023052835086.pth | ojmb_node_v2 | 20 | 406.2 ± 35.5 | 0.1318 ± 0.0542 | 0.1922 ± 0.0898 | 0.8919 ± 0.0405 | 0.0000 ± 0.0000 | 44.7717 ± 0.9827 | 30.0437 ± 0.2402 | 1354.5 ± 1.3 | 1.0000 ± 0.0000 | 1.0182 ± 0.0052 | 0.3944 ± 0.0160 | 0.3807 ± 0.0037 | 0.3899 ± 0.0044 | 1.0997 ± 0.0230 | 0.1714 ± 0.0363 | 1.0000 |
| 801911654859808.pth | ojmb_edge_v2 | 20 | 403.2 ± 37.6 | 0.1224 ± 0.0603 | 0.1817 ± 0.0915 | 0.9009 ± 0.0430 | 0.0000 ± 0.0000 | 33.5258 ± 0.3671 | 43.7975 ± 0.2393 | 1357.1 ± 1.8 | 1.0000 ± 0.0000 | 1.0196 ± 0.0043 | 0.3181 ± 0.0141 | 0.3359 ± 0.0063 | 0.3556 ± 0.0088 | 1.1150 ± 0.0314 | 0.1941 ± 0.0487 | 1.0000 |
| 935685383141427.pth | ojmb_base_v2 | 20 | 400.1 ± 33.8 | 0.1155 ± 0.0460 | 0.1736 ± 0.0790 | 0.9029 ± 0.0359 | 0.0000 ± 0.0000 | 30.1510 ± 0.3378 | 56.0342 ± 0.2392 | 1356.5 ± 0.8 | 0.9999 ± 0.0002 | 1.0244 ± 0.0088 | 0.2834 ± 0.0149 | 0.3642 ± 0.0053 | 0.3625 ± 0.0105 | 1.1243 ± 0.0312 | 0.2083 ± 0.0469 | 1.0000 |
| 149005691968570.pth | ojmb_feat_v2 | 20 | 396.8 ± 36.3 | 0.1034 ± 0.0489 | 0.1626 ± 0.0885 | 0.9134 ± 0.0373 | 0.0000 ± 0.0000 | 31.2608 ± 0.4317 | 68.2806 ± 0.2393 | 1356.5 ± 0.9 | 1.0000 ± 0.0000 | 1.0157 ± 0.0041 | 0.2893 ± 0.0128 | 0.3637 ± 0.0062 | 0.3703 ± 0.0114 | 1.1144 ± 0.0247 | 0.1937 ± 0.0367 | 1.0000 |

## Generalization to unseen sizes

Gap = RE (LB) on the folder minus RE (LB) on `data/batching/batching_test_split.json`.

| Model | Folder | Jobs | Machines | In training range | RE (CP-SAT) | RE (LB) | Score | Gap |
|---|---|---|---|---|---|---|---|---|
| 814023052835086.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1318 | 0.1922 | 0.8919 | 0.0000 |
| 801911654859808.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1224 | 0.1817 | 0.9009 | 0.0000 |
| 935685383141427.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1155 | 0.1736 | 0.9029 | 0.0000 |
| 149005691968570.pth | data/batching/batching_test_split.json | 8-10 | 5-10 | 1.00 | 0.1034 | 0.1626 | 0.9134 | 0.0000 |

## Paired comparisons between models

Wilcoxon signed-rank on the same instances; negative mean diff = model A lower.

| Folder | Metric | Model A | Model B | n | Mean diff | p | Significant | Cohen's d | Rank-biserial | n needed |
|---|---|---|---|---|---|---|---|---|---|---|
| data/batching/batching_test_split.json | makespan | 814023052835086.pth | 801911654859808.pth | 20 | 3.0000 | 0.5867 | no | 0.1227 | 0.1544 | 523 |
| data/batching/batching_test_split.json | relative_error_lb | 814023052835086.pth | 801911654859808.pth | 20 | 0.0105 | 0.5014 | no | 0.1390 | 0.1912 | 408 |
| data/batching/batching_test_split.json | scheduling_score | 814023052835086.pth | 801911654859808.pth | 20 | -0.0090 | 0.4380 | no | -0.2111 | -0.2206 | 178 |
| data/batching/batching_test_split.json | makespan | 814023052835086.pth | 935685383141427.pth | 20 | 6.1500 | 0.7794 | no | 0.1922 | 0.0714 | 214 |
| data/batching/batching_test_split.json | relative_error_lb | 814023052835086.pth | 935685383141427.pth | 20 | 0.0187 | 0.7285 | no | 0.1945 | 0.0952 | 209 |
| data/batching/batching_test_split.json | scheduling_score | 814023052835086.pth | 935685383141427.pth | 20 | -0.0110 | 0.8124 | no | -0.1618 | -0.0667 | 301 |
| data/batching/batching_test_split.json | makespan | 814023052835086.pth | 149005691968570.pth | 20 | 9.4500 | 0.2658 | no | 0.3028 | 0.3072 | 87 |
| data/batching/batching_test_split.json | relative_error_lb | 814023052835086.pth | 149005691968570.pth | 20 | 0.0296 | 0.2097 | no | 0.3165 | 0.3464 | 80 |
| data/batching/batching_test_split.json | scheduling_score | 814023052835086.pth | 149005691968570.pth | 20 | -0.0215 | 0.1359 | no | -0.3394 | -0.4118 | 70 |
| data/batching/batching_test_split.json | makespan | 801911654859808.pth | 935685383141427.pth | 20 | 3.1500 | 0.9679 | no | 0.0927 | 0.0105 | 915 |
| data/batching/batching_test_split.json | relative_error_lb | 801911654859808.pth | 935685383141427.pth | 20 | 0.0082 | 0.9679 | no | 0.0788 | 0.0105 | 1265 |
| data/batching/batching_test_split.json | scheduling_score | 801911654859808.pth | 935685383141427.pth | 20 | -0.0020 | 0.9359 | no | -0.0301 | 0.0211 | 8681 |
| data/batching/batching_test_split.json | makespan | 801911654859808.pth | 149005691968570.pth | 20 | 6.4500 | 0.4487 | no | 0.1778 | 0.2092 | 250 |
| data/batching/batching_test_split.json | relative_error_lb | 801911654859808.pth | 149005691968570.pth | 20 | 0.0191 | 0.4074 | no | 0.1725 | 0.2288 | 265 |
| data/batching/batching_test_split.json | scheduling_score | 801911654859808.pth | 149005691968570.pth | 20 | -0.0125 | 0.3560 | no | -0.1898 | -0.2549 | 219 |
| data/batching/batching_test_split.json | makespan | 935685383141427.pth | 149005691968570.pth | 20 | 3.3000 | 0.3435 | no | 0.1743 | 0.2614 | 260 |
| data/batching/batching_test_split.json | relative_error_lb | 935685383141427.pth | 149005691968570.pth | 20 | 0.0109 | 0.3560 | no | 0.1823 | 0.2549 | 238 |
| data/batching/batching_test_split.json | scheduling_score | 935685383141427.pth | 149005691968570.pth | 20 | -0.0105 | 0.2659 | no | -0.2448 | -0.3072 | 132 |
