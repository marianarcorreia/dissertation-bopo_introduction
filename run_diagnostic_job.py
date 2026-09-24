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

Usage:
    python run_diagnostic_job.py --run-name ablation_ojm_gat_layers3 \
        --representation ojm --gnn-type gat --num-layers 3 --seed 42
"""
import argparse

from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--representation", required=True, choices=["oo", "om", "ojm"])
parser.add_argument("--gnn-type", required=True, choices=["gat", "gin", "transformer"])
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sel-k", type=int, default=1)
parser.add_argument("--mask-option", type=int, default=1)
parser.add_argument("--logp-norm", default="mean", choices=["mean", "sum"])
parser.add_argument("--greedy-in-loss", action="store_true",
                    help="Keep the greedy rollout in the preference pairs (pre-fix behavior).")
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
    # everything below matches sweep_logs/v2_*.log exactly (the loss options and, for om/ojm, sel_k above do not by default)
    max_episodes=400,
    new_freq=200,
    n_cases=30,
    B=32,
    K=10,
    use_greedy=True,
    lr=0.0005,
    hidden_channels=128,
    heads=3,
    validation_freq=25,
    validation_size=40,
    warm_start_steps=200,
    lr_min_ratio=0.2,
    checkpoint_smooth_window=3,
)
