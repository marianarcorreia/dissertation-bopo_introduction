#!/bin/bash
# The 8 blocking runs from runs_not_run.txt (seed 42, L3, sel_k 1, blk350 settings), two lanes
# in parallel (GAT, GIN). The blk350 seed-1/2 pipeline (run_blk350_seeds.sh) was paused for
# them; its two training processes are resumed when both lanes have finished.
cd "C:/Users/maria/Desktop/dissertation-bopo_introduction"
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
PAUSED_PIDS="17896 44600"
REPS="ojmb ojmf ojmd ojm_blk"

lane() {
    gnn=$1
    for rep in $REPS; do
        echo "=== [$gnn] TRAIN blk350_${gnn}_L3_selk1_${rep}_s42 | $(date '+%F %T') ==="
        $PY main.py --mode train --run-name blk350_${gnn}_L3_selk1_${rep} --representation $rep \
            --gnn-type $gnn --sel-k 1 --num-layers 3 --max-episodes 350 --seeds 42 --no-dashboard
        echo "=== [$gnn] exit code $? | $(date '+%F %T') ==="
    done
    echo "=== [$gnn] LANE DONE | $(date '+%F %T') ==="
}

echo "queue started $(date '+%F %T')"
lane gat > logs/blk350_L3_s42_gat.log 2>&1 &
lane gin > logs/blk350_L3_s42_gin.log 2>&1 &
wait
echo "both lanes done $(date '+%F %T'); resuming paused seed runs: $PAUSED_PIDS"
for pid in $PAUSED_PIDS; do
    powershell -NoProfile -Command "Add-Type 'using System;using System.Runtime.InteropServices;public static class R{[DllImport(\"ntdll.dll\")]public static extern int NtResumeProcess(IntPtr h);}'; 'resume $pid -> status ' + [R]::NtResumeProcess((Get-Process -Id $pid).Handle)"
done
echo "queue finished $(date '+%F %T')"
