import math

import torch

# Shared, representation-agnostic pieces of BOPO (self-rewarding preference optimization).
# Each ppo*.py file keeps its own env-specific rollout/masking code (mirroring how the
# existing PPO files already duplicate that logic per representation); only the loss and
# the pair-selection rule are identical across representations, so they live here once.


def select_pairs(makespans, K):
    """
    Self-label a group of B sampled solutions of the same instance into comparison pairs,
    without any external reference solution (BOPO FJSP/sampling.py:sample_training_pair).

    Args:
        makespans: 1D tensor of shape (B,) with the final makespan of each sampled rollout.
        K: number of solutions to use for the comparison (best + up to K-1 worse ones).

    Returns:
        best_idx: index (int) of the best (lowest-makespan) rollout in the group.
        worse_idx: LongTensor of indices to compare the best rollout against.
    """
    B = makespans.shape[0]
    K = max(2, min(int(K), B))
    order = torch.argsort(makespans)
    stride = max(1, B // K)
    spread = order[::stride][:K]
    return int(spread[0].item()), spread[1:]


def sro_loss(logp_better, logp_worse, ms_better, ms_worse, eps=1e-7):
    """
    Self-Rewarding Optimization pairwise loss (BOPO FJSP/sampling.py:SROLoss), applied to a
    single (better, worse) pair instead of a batch of pairs, so it can be accumulated per
    representation-specific rollout loop.

    Args:
        logp_better / logp_worse: scalar tensors (grad-enabled), mean per-decision
            log-likelihood of the better/worse trajectory under the current policy.
        ms_better / ms_worse: python floats, final makespans of the two trajectories.

    Returns:
        Scalar loss tensor for this pair.
    """
    ms_factor = float(ms_worse) / (float(ms_better) + eps)
    return -torch.log(torch.sigmoid(ms_factor * (logp_better - logp_worse)) + eps)


def summarize_action_entropy(entropies, valid_counts):
    """
    Aggregate the per-decision-step entropy of Policy.act_batch's action distribution
    over one BOPO update() call (all rounds, all B rollouts in the group).

    Args:
        entropies: Shannon entropy (nats) of the softmax distribution actually sampled
            from at each decision step - masked-out actions already get -inf logits
            before that softmax, so this is already restricted to valid actions.
        valid_counts: number of unmasked (valid) actions available at that same step.

    Returns:
        dict of scalar floats, empty if no decision steps were recorded.
    """
    if not entropies:
        return {}

    random_refs = [math.log(c) if c > 1 else 0.0 for c in valid_counts]
    normalized = [(e / r) if r > 1e-8 else 1.0 for e, r in zip(entropies, random_refs)]
    effective_counts = [math.exp(e) for e in entropies]
    effective_ratios = [(ec / c) if c > 0 else 0.0 for ec, c in zip(effective_counts, valid_counts)]
    uncertainty_gaps = [r - e for r, e in zip(random_refs, entropies)]

    def mean(values):
        return sum(values) / len(values)

    action_entropy = mean(entropies)
    return {
        "action_entropy": action_entropy,
        "action_entropy_random_ref": mean(random_refs),
        "action_entropy_max_entropy_normalized": mean(normalized),
        "action_entropy_effective_action_count": mean(effective_counts),
        "action_entropy_effective_action_ratio": mean(effective_ratios),
        "action_entropy_uncertainty_gap": mean(uncertainty_gaps),
        # Distribution is already restricted to valid actions (masked logits are -inf
        # before softmax), so there is no separate symmetry correction to apply here -
        # this collapses to action_entropy.
        "action_entropy_symmetry_corrected": action_entropy,
    }
