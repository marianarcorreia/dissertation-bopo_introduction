#!/bin/bash
# The 4 blocking representations with GAT, seed 42, 350 episodes: L3, sel_k 1, mask_option 1,
# jm_design edges, logp_norm sum, greedy kept in the loss (--no-exclude-greedy). Two lanes of two
# runs in parallel; when both finish, the 4 best checkpoints are tested on blocking_test_instances.
cd "C:/Users/maria/Desktop/dissertation-bopo_introduction"
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
TAG=blk350_gat_L3_selk1_mask1_edges_sum_incgreedy
MODELS_FILE=models/blocking/model_params_blk350_L3_edges.json

lane() {
    name=$1; shift
    for rep in "$@"; do
        echo "=== [$name] TRAIN ${TAG}_${rep}_s42 | $(date '+%F %T') ==="
        $PY main.py --mode train --run-name ${TAG}_${rep} --representation $rep \
            --gnn-type gat --num-layers 3 --sel-k 1 --mask-option 1 --jm-design edges \
            --logp-norm sum --no-exclude-greedy --max-episodes 350 --seeds 42 --no-dashboard
        echo "=== [$name] exit code $? | $(date '+%F %T') ==="
    done
    echo "=== [$name] LANE DONE | $(date '+%F %T') ==="
}

echo "queue started $(date '+%F %T')"
lane A ojmb ojmd > logs/blk350_L3_edges_s42_laneA.log 2>&1 &
lane B ojmf ojm_blk > logs/blk350_L3_edges_s42_laneB.log 2>&1 &
wait
echo "both lanes done $(date '+%F %T'); building $MODELS_FILE"

# one entry per run: the best checkpoint recorded in its run_summary.json
$PY - "$TAG" "$MODELS_FILE" <<'EOF'
import json, os, sys
tag, out = sys.argv[1], sys.argv[2]
registry = {e["name"]: e for e in json.load(open("candidate_models/blocking/model_params.json"))}
entries = []
for rep in ("ojmb", "ojmd", "ojmf", "ojm_blk"):
    summary_path = f"results/blocking/{tag}_{rep}_s42/run_summary.json"
    if not os.path.exists(summary_path):
        print(f"missing {summary_path}, skipping {rep}")
        continue
    name = os.path.basename(json.load(open(summary_path))["best_model_path"])
    entries.append(registry[name])
    print(f"{rep}: {name}")
json.dump(entries, open(out, "w"), indent=2)
EOF

echo "=== TEST | $(date '+%F %T') ==="
$PY main.py --mode test --run-name test_blk350_L3_edges --models-file $MODELS_FILE \
    --representation blocking --source-folder val --folders blocking_test_instances --workers 4 --no-dashboard
echo "=== TEST exit code $? | queue finished $(date '+%F %T') ==="
