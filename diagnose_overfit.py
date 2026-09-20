"""
BOPO overfit sanity check.

Trains BOPO on a SINGLE fixed, small FJSP instance (n_cases=1, the training pool is
never regenerated) for many updates. This isolates one question: can the pipeline
learn at all?

  - If entropy collapses and the group's best makespan drops toward optimal on this
    one memorized instance, the algorithm/gradient path is sound - a real-run plateau
    (see results/*_oo/) is a training-budget / hyperparameter problem, not a bug.
  - If gradients are ~0 the whole run, that's a real bug (loss not producing signal,
    something detached, masking wiping out every valid action, etc).
  - If gradients are healthy but nothing moves, that points at the loss shape / LR
    rather than a broken gradient path.

Usage:
    python diagnose_overfit.py
    python diagnose_overfit.py --representation om --gnn-type transformer
    python diagnose_overfit.py --quick          # ~1-2 min smoke test
    python diagnose_overfit.py --max-episodes 800 --lr 1e-3
"""

import argparse
import json
import os
from datetime import datetime

from src.train import train

REP_DEFAULT_SEL_K = {"oo": 1, "om": 2, "ojm": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--representation", default="oo", choices=["oo", "om", "ojm"])
    parser.add_argument("--gnn-type", default="gat", choices=["gat", "gin", "transformer"])
    parser.add_argument("--max-episodes", type=int, default=300, help="Number of BOPO updates.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--B", type=int, default=32, help="Rollouts sampled per update.")
    parser.add_argument("--K", type=int, default=8, help="Best-vs-K-1-worst comparisons per update.")
    parser.add_argument("--use-greedy", action="store_true", default=True)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4, help="Fixed number of jobs in the single instance.")
    parser.add_argument("--machines", type=int, default=3, help="Fixed number of machines in the single instance.")
    parser.add_argument("--op-max", type=int, default=5, help="Max ops/job (train()'s internal floor is 5).")
    parser.add_argument("--max-processing", type=int, default=20)
    parser.add_argument("--mask-option", type=int, default=1)
    parser.add_argument("--sel-k", type=int, default=None,
                         help="Defaults to 1 for oo (ignored anyway), 2 for om/ojm so the "
                              "policy actually gets to choose between machines instead of "
                              "silently collapsing to a fixed greedy heuristic.")
    parser.add_argument("--validation-freq", type=int, default=50)
    parser.add_argument("--validation-size", type=int, default=5)
    parser.add_argument("--warm-start-steps", type=int, default=0,
                         help="train()'s behavior-cloning warm start now defaults to 200 "
                              "steps, but this script's whole point is isolating whether "
                              "BOPO's own pairwise loss can learn from scratch - so it "
                              "defaults to 0 here. Pass >0 to instead check whether warm "
                              "start + BOPO together move faster than either alone.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quick", action="store_true",
                         help="Small/fast smoke test: 50 updates, tiny instance, B=16/K=4.")
    return parser.parse_args()


def window_mean(values, n, from_end):
    values = [v for v in values if v is not None]
    if not values:
        return None
    window = values[-n:] if from_end else values[:n]
    return sum(window) / len(window)


def load_json(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def report(run_dir):
    episode_metrics = load_json(os.path.join(run_dir, "episode_metrics.json"))
    update_metrics = load_json(os.path.join(run_dir, "update_metrics.json"))

    if not episode_metrics:
        print(f"[DIAGNOSE] No episode_metrics.json found under {run_dir} - training did not produce output.")
        return

    n = max(1, min(10, len(episode_metrics) // 4))

    entropy = [e.get("action_entropy_max_entropy_normalized") for e in episode_metrics]
    makespan = [e.get("makespan") for e in episode_metrics]
    loss = [e.get("actor_loss") for e in episode_metrics]
    grad_norm = [u.get("actor_grad_norm") for u in update_metrics]

    entropy_start, entropy_end = window_mean(entropy, n, False), window_mean(entropy, n, True)
    ms_start, ms_end = window_mean(makespan, n, False), window_mean(makespan, n, True)
    loss_start, loss_end = window_mean(loss, n, False), window_mean(loss, n, True)
    grad_mean = window_mean(grad_norm, len(grad_norm), True)
    grad_start, grad_end = window_mean(grad_norm, n, False), window_mean(grad_norm, n, True)

    entropy_drop = None if entropy_start in (None, 0) else (entropy_start - entropy_end) / entropy_start
    ms_improve = None if ms_start in (None, 0) else (ms_start - ms_end) / ms_start

    entropy_line = f"  action_entropy_max_entropy_normalized : start={entropy_start:.4f}  end={entropy_end:.4f}"
    if entropy_drop is not None:
        entropy_line += f"  (drop={entropy_drop*100:.1f}%)"

    ms_line = f"  group best makespan                   : start={ms_start:.2f}  end={ms_end:.2f}"
    if ms_improve is not None:
        ms_line += f"  (improve={ms_improve*100:.1f}%)"

    print()
    print("=" * 70)
    print(f"[DIAGNOSE] Overfit sanity check report | run_dir={run_dir}")
    print(f"[DIAGNOSE] {len(episode_metrics)} updates completed, averaging over first/last {n} episodes")
    print("=" * 70)
    print(entropy_line)
    print(ms_line)
    print(f"  actor_loss                             : start={loss_start:.4f}  end={loss_end:.4f}")
    if grad_norm and grad_norm[0] is not None:
        print(f"  actor_grad_norm                        : start={grad_start:.6g}  end={grad_end:.6g}  mean={grad_mean:.6g}")
    else:
        print("  actor_grad_norm                        : not available (re-run after pulling the grad-norm logging patch)")
    print("-" * 70)

    grad_near_zero = grad_mean is not None and grad_mean < 1e-6
    if grad_near_zero:
        print("VERDICT: SUSPECT BUG - actor_grad_norm is ~0 throughout the run.")
        print("  The loss is not producing a learning signal at all. Check that:")
        print("   - logp_total in sample_group() keeps requires_grad=True end to end")
        print("     (no stray .detach()/no_grad() on the sampled, non-greedy rollouts)")
        print("   - the mask isn't leaving zero or one valid action at every decision step")
        print("     for this instance size (softmax over a single unmasked logit has zero")
        print("     gradient wrt that logit)")
        print("   - select_pairs() isn't returning best_idx == every worse_idx (degenerate group)")
    elif entropy_drop is not None and ms_improve is not None and entropy_drop > 0.15 and ms_improve > 0.05:
        print("VERDICT: PASS - the pipeline can learn.")
        print("  Entropy collapsed and the memorized instance's makespan improved meaningfully.")
        print("  A real-run plateau is most likely a training-budget / hyperparameter issue")
        print("  (see the earlier diagnostic plan: scale max_episodes 10-50x, tune lr/B/K,")
        print("  raise sel_k for om/ojm), not a correctness bug.")
    elif ms_improve is not None and ms_improve > 0.05:
        print("VERDICT: PARTIAL - makespan improved without much entropy collapse.")
        print("  Consistent with the algorithm working but slowly. Try more updates and/or")
        print("  a higher LR before suspecting a bug.")
    else:
        print("VERDICT: INCONCLUSIVE / CONCERNING - gradients are non-zero but neither")
        print("  entropy nor makespan moved on a single memorized instance over this many")
        print("  updates. Before concluding there's a deeper bug, try: a higher LR (this")
        print("  script defaults to 5e-4), more --max-episodes, or an even smaller instance")
        print("  (--jobs 3 --machines 2). If it still won't move, inspect the raw actor")
        print("  logits (add a print of action_probs before softmax in Policy.act_batch)")
        print("  to see if they're saturating or staying near-identical across updates.")
    print("=" * 70)


def main():
    args = parse_args()

    if args.quick:
        args.max_episodes = 50
        args.jobs, args.machines, args.op_max, args.max_processing = 3, 2, 5, 10
        args.B, args.K = 16, 4
        args.validation_freq, args.validation_size = 25, 3

    sel_k = args.sel_k if args.sel_k is not None else REP_DEFAULT_SEL_K[args.representation]
    run_name = args.run_name or f"diagnose_overfit_{args.representation}_{args.gnn_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("results", run_name)

    print("[DIAGNOSE] Overfit sanity check")
    print(f"[DIAGNOSE] representation={args.representation} | gnn_type={args.gnn_type} | "
          f"jobs={args.jobs} | machines={args.machines} | op_max={args.op_max} | "
          f"max_processing={args.max_processing} | sel_k={sel_k}")
    print(f"[DIAGNOSE] max_episodes={args.max_episodes} | lr={args.lr} | B={args.B} | K={args.K}")
    print(f"[DIAGNOSE] Training pool is a single instance (n_cases=1, new_freq=max_episodes+1) "
          f"so it is never regenerated.")

    train(
        max_episodes=args.max_episodes,
        new_freq=args.max_episodes + 1,
        n_cases=1,
        mask_option=args.mask_option,
        sel_k=sel_k,
        B=args.B,
        K=args.K,
        use_greedy=args.use_greedy,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        j_max=args.jobs,
        j_min=args.jobs,
        m_max=args.machines,
        m_min=args.machines,
        op_max=args.op_max,
        max_processing=args.max_processing,
        validation_freq=args.validation_freq,
        validation_size=args.validation_size,
        warm_start_steps=args.warm_start_steps,
        run_name=run_name,
        representation=args.representation,
        gnn_type=args.gnn_type,
    )

    report(run_dir)


if __name__ == "__main__":
    main()
