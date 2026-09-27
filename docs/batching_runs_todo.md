# Batching runs: to do

Branch `batching_constraint`. Three batching representations, trained one after the other
(the GPU has 4 GB and parallel trainings have been OOM-killed before), then one test report.

Settings for every run: GAT, 2 layers, 3 heads, hidden 128, `sel_k = 1`, `mask_option = 1`,
lr 4.7e-3 (cosine decay to 20%), 350 BOPO steps, 200 warm-start steps, B = 32, K = 10,
30 training instances renewed every 200 steps, validation every 25 steps on 20 instances,
action edges `jm_design = edges`, seed 42. The rest is the v2 setup of `run_diagnostic_job.py`.

## Estimated time

Measured on this machine (GTX 1650 Ti) with exactly these settings, 5 BOPO steps and
20 warm-start steps per representation, then extrapolated (200 warm-start steps, 350 BOPO
steps, 14 validations + 1 final test evaluation):

| Run | Warm-start step | BOPO step | Validation | Estimated total |
|---|---|---|---|---|
| ojmb_node | 3.6 s | 18.4 s | 49 s | ~2.2 h |
| ojmb_edge | 2.7 s | 16.8 s | 37 s | ~1.9 h |
| ojmb_base | 3.1 s | 16.9 s | 33 s | ~2.0 h |
| Test report (3 models x 20 test instances) | | | | ~15 min |
| **Total** | | | | **~6.3 h** (roughly 5-7.5 h) |

The BOPO step time can drift as the policy changes, so allow about ±20%.

## Steps

- [ ] 1. Train the family node (Representation A), ~2.2 h
  ```
  python run_diagnostic_job.py --run-name batching_gat_L2_ojmb_node_s42 --representation ojmb_node --gnn-type gat --num-layers 2 --sel-k 1 --mask-option 1 --lr 4.7e-3 --max-episodes 350 --validation-size 20 --jm-design edges --seed 42
  ```
- [ ] 2. Train the batch edge (Representation B), ~1.9 h
  ```
  python run_diagnostic_job.py --run-name batching_gat_L2_ojmb_edge_s42 --representation ojmb_edge --gnn-type gat --num-layers 2 --sel-k 1 --mask-option 1 --lr 4.7e-3 --max-episodes 350 --validation-size 20 --jm-design edges --seed 42
  ```
- [ ] 3. Train the baseline (no batching information), ~2.0 h
  ```
  python run_diagnostic_job.py --run-name batching_gat_L2_ojmb_base_s42 --representation ojmb_base --gnn-type gat --num-layers 2 --sel-k 1 --mask-option 1 --lr 4.7e-3 --max-episodes 350 --validation-size 20 --jm-design edges --seed 42
  ```
- [ ] 4. Read the best checkpoint of each run: `best_model_path` in
  `results/batching_gat_L2_<rep>_s42/run_summary.json`. If it is `null`, validation never
  improved and that run has no model to test.
- [ ] 5. Test every metric, including the isomorphism / expressiveness metrics, on the
  held-out test split (the 20 instances not used for validation), ~15 min
  ```
  python run_metrics.py --run-name report_metrics_batching --models-file candidate_models/model_params.json --models <node>.pth <edge>.pth <base>.pth --folders data/batching/batching_test_split.json
  ```
  Writes `results/metrics/report_metrics_batching.json`, `.md` (tables, paired Wilcoxon
  tests) and `.png` (bar chart of every metric, `src/metrics/report.py:plot_metric_bars`).
  Redraw the chart from a saved report with
  `python -m src.metrics.report results/metrics/report_metrics_batching.json`.
- [ ] 6. Check the report: feasibility 1.0 for all three, then compare relative error,
  scheduling score and the isomorphism panels (distinct action scores, distinct node
  embeddings, structure gain).

## Notes

- One seed is not enough to rank the representations. Earlier FJSP runs with 2-layer GAT
  scored 0.188 / 0.104 / 0.106 on test with seeds 42 / 43 / 44. Repeat steps 1-3 with
  `--seed 43` and `--seed 44` (another ~12 h) before drawing conclusions.
- lr 4.7e-3 is almost 10x the v2 value (5e-4). If the validation gap curve
  (`results/<run>/`) oscillates or diverges, that is the first thing to check.
