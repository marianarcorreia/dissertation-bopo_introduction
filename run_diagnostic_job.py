"""One-off training job for the post-sweep diagnostic follow-ups (depth ablation,
extra seeds). Mirrors the exact hyperparameters used for the final 9-combination
sweep (sweep_logs/v2_*.log) so results are comparable, varying only what the specific
follow-up is testing (num_layers and/or seed). Writes to results/<run-name>/, never
to results/convergence_check_<rep>_<backbone>/, so the sweep results the README's
section 7 is built on are never touched.

Mask: --sel-k defaults to 1. NOTE the v2 sweep was NOT uniform here - its om/ojm runs
used sel_k=2 while oo used sel_k=1 (see candidate_models/model_params.json), so v2 om/ojm
results are not directly comparable to sel_k=1 runs (the ojm depth ablation, the reseeds).

The BOPO loss defaults to the fixed variant (per-decision mean log-likelihood, greedy
rollout kept out of the preference pairs - see src/bopo_utils.py:bopo_group_loss). Pass
--logp-norm sum --greedy-in-loss to reproduce the loss the v2 sweep actually used.

The batching representations (ojmb_node / ojmb_edge / ojmb_base, src/env_batching.py) use
the same setup. Their dataset has 40 instances, so pass --validation-size 20 (20 validation +
20 test) and --jm-design edges (their action-edge features are rebuilt every decision).
--lr and --max-episodes override the v2 values (5e-4, 400) when given.

Usage:
    python run_diagnostic_job.py --run-name ablation_ojm_gat_layers3 \
        --representation ojm --gnn-type gat --num-layers 3 --seed 42
    python run_diagnostic_job.py --run-name batching_ojmb_node_gat_s42 --representation ojmb_node \
        --gnn-type gat --jm-design edges --validation-size 20 --lr 4.7e-3 --max-episodes 350
"""
import argparse

from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--representation", required=True,
                    choices=["oo", "om", "ojm", "ojmb_node", "ojmb_edge", "ojmb_base"])
parser.add_argument("--gnn-type", required=True, choices=["gat", "gin", "transformer"])
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sel-k", type=int, default=1)
parser.add_argument("--mask-option", type=int, default=1)
parser.add_argument("--logp-norm", default="mean", choices=["mean", "sum"])
parser.add_argument("--greedy-in-loss", action="store_true",
                    help="Keep the greedy rollout in the preference pairs (pre-fix behavior).")
parser.add_argument("--jm-design", default="baseline", choices=["baseline", "edges", "attn"],
                    help="ojm-based only: job-machine action edge design (see src/env.py:FJSSPEnv.JM_DESIGNS).")
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--max-episodes", type=int, default=400)
parser.add_argument("--validation-size", type=int, default=40)
parser.add_argument("--step-metrics", type=int, default=0,
                    help="Evaluate every src/metrics metric after every step on this many validation instances (0 = off).")
args = parser.parse_args()

train(
    run_name=args.run_name,
    representation=args.representation,
    gnn_type=args.gnn_type,
    num_layers=args.num_layers,
    seed=args.seed,
    sel_k=args.sel_k,
    mask_option=args.mask_option,
    logp_norm=args.logp_norm,
    exclude_greedy_from_loss=not args.greedy_in_loss,
    jm_design=args.jm_design,
    # everything below matches sweep_logs/v2_*.log exactly (the loss options and, for om/ojm, sel_k above do not
    # by default; lr, max_episodes and validation_size only when left at their defaults)
    max_episodes=args.max_episodes,
    new_freq=200,
    n_cases=30,
    B=32,
    K=10,
    use_greedy=True,
    lr=args.lr,
    hidden_channels=128,
    heads=3,
    validation_freq=25,
    validation_size=args.validation_size,
    step_metrics_size=args.step_metrics,
    warm_start_steps=200,
    lr_min_ratio=0.2,
    checkpoint_smooth_window=3,
)
