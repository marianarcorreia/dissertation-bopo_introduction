# Runs to do (main branch)

All runs: GAT, seed 42, hidden_channels 128 and heads 3 (train() defaults), `--mode train_test`.

- [ ] **1** — ojm · L1 · sel_k 1 · mask 1 · logp mean · exclude greedy · 30 episodes
  ```
  python main.py --mode train_test --gnn-type gat --representation ojm --num-layers 1 --sel-k 1 --mask-option 1 --logp-norm mean --exclude-greedy --max-episodes 30 --seed 42 --run-name gat_ojm_L1_selk1_mask1_mean_exg_s42
  ```
- [ ] **2** — ojm · L2 · sel_k 10 · mask 1 · logp mean · exclude greedy · 30 episodes
  ```
  python main.py --mode train_test --gnn-type gat --representation ojm --num-layers 2 --sel-k 10 --mask-option 1 --logp-norm mean --exclude-greedy --max-episodes 30 --seed 42 --run-name gat_ojm_L2_selk10_mask1_mean_exg_s42
  ```
- [ ] **3** — ojm · L2 · sel_k 1 · mask 0 · logp sum · keep greedy · 400 episodes
  ```
  python main.py --mode train_test --gnn-type gat --representation ojm --num-layers 2 --sel-k 1 --mask-option 0 --logp-norm sum --no-exclude-greedy --max-episodes 400 --seed 42 --run-name gat_ojm_L2_selk1_mask0_sum_noexg_s42
  ```
- [ ] **4** — ojm · L2 · sel_k 10 · mask 0 · logp sum · keep greedy · 400 episodes
  ```
  python main.py --mode train_test --gnn-type gat --representation ojm --num-layers 2 --sel-k 10 --mask-option 0 --logp-norm sum --no-exclude-greedy --max-episodes 400 --seed 42 --run-name gat_ojm_L2_selk10_mask0_sum_noexg_s42
  ```
- [ ] **5** — **om** · L2 · sel_k 1 · mask 1 · logp sum · keep greedy · 400 episodes
  ```
  python main.py --mode train_test --gnn-type gat --representation om --num-layers 2 --sel-k 1 --mask-option 1 --logp-norm sum --no-exclude-greedy --max-episodes 400 --seed 42 --run-name gat_om_L2_selk1_mask1_sum_noexg_s42
  ```
