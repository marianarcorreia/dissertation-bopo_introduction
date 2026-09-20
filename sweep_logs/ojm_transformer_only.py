import os, sys, time
sys.path.insert(0, os.getcwd())
from src.train import train

t0 = time.time()
run_name = "convergence_check_ojm_transformer"
print(f"=== START {run_name} @ {time.strftime('%H:%M:%S')} ===", flush=True)
try:
    train(
        max_episodes=150, new_freq=75, n_cases=20,
        mask_option=1, sel_k=2, B=24, K=8, use_greedy=True,
        lr=5e-4, hidden_channels=128, num_layers=2, heads=3,
        j_max=10, j_min=8, m_max=10, m_min=5, op_max=6, max_processing=100,
        validation_freq=20, validation_size=30,
        run_name=run_name, representation="ojm", gnn_type="transformer",
        warm_start_steps=150,
    )
    print(f"=== DONE {run_name} in {time.time()-t0:.1f}s ===", flush=True)
except Exception as exc:
    print(f"=== FAILED {run_name} after {time.time()-t0:.1f}s: {exc!r} ===", flush=True)
    raise
