#!/usr/bin/env bash
# Runs the five RUNS_TODO.md runs one after another (GPU is too small to run them in parallel).
cd "$(dirname "$0")"
PY="D:/Dissertação/dissertation-bopo_introduction/venv/Scripts/python.exe"
COMMON="--mode train_test --gnn-type gat --seed 42 --no-dashboard"

run() {
    local name=$1; shift
    echo "[TODO] $(date '+%F %T') START $name"
    "$PY" -u main.py $COMMON --run-name "$name" "$@" > "sweep_logs/$name.log" 2>&1
    echo "[TODO] $(date '+%F %T') END $name exit=$?"
}

run gat_ojm_L1_selk1_mask1_mean_exg_s42    --representation ojm --num-layers 1 --sel-k 1  --mask-option 1 --logp-norm mean --exclude-greedy    --max-episodes 30
run gat_ojm_L2_selk10_mask1_mean_exg_s42   --representation ojm --num-layers 2 --sel-k 10 --mask-option 1 --logp-norm mean --exclude-greedy    --max-episodes 30
run gat_ojm_L2_selk1_mask0_sum_noexg_s42   --representation ojm --num-layers 2 --sel-k 1  --mask-option 0 --logp-norm sum  --no-exclude-greedy --max-episodes 400
run gat_ojm_L2_selk10_mask0_sum_noexg_s42  --representation ojm --num-layers 2 --sel-k 10 --mask-option 0 --logp-norm sum  --no-exclude-greedy --max-episodes 400
run gat_om_L2_selk1_mask1_sum_noexg_s42    --representation om  --num-layers 2 --sel-k 1  --mask-option 1 --logp-norm sum  --no-exclude-greedy --max-episodes 400
echo "[TODO] all runs finished"
