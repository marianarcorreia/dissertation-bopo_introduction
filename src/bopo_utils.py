import math
from src.gat import GAT
import torch

# Shared, representation-agnostic pieces of BOPO (self-rewarding preference optimization).
# Each ppo*.py file keeps its own env-specific rollout/masking code (mirroring how the
# existing PPO files already duplicate that logic per representation); only the loss and
# the pair-selection rule are identical across representations, so they live here once.


def select_pairs(makespans, K):
    """
    Self-label a group of B sampled solutions of the same instance into comparison pairs,
    without any external reference solution (BOPO FJSP/sampling.py:sample_training_pair).

    Anchors the best (lowest-makespan) rollout against the actual K-1 WORST rollouts in
    the group, rather than a quantile spread. Early in training the policy is close to
    uniform, so log-prob differences between trajectories are tiny and noisy; comparing
    against the worst available trajectories maximizes ms_worse/ms_better for each pair,
    which is exactly the factor that scales the gradient in sro_loss() - i.e. it gives
    the strongest, least noisy learning signal available from this sampled group.

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
    best_idx = int(order[0].item())
    worse_idx = order[-(K - 1):]
    return best_idx, worse_idx


def sro_loss(logp_better, logp_worse, ms_better, ms_worse, eps=1e-7, max_ms_factor=10.0):
    """
    Self-Rewarding Optimization pairwise loss (BOPO FJSP/sampling.py:SROLoss), applied to a
    single (better, worse) pair instead of a batch of pairs, so it can be accumulated per
    representation-specific rollout loop.

    ms_factor = ms_worse/ms_better scales how much log-prob separation is required: at
    Delta=logp_better-logp_worse=0 (the near-uniform-policy start of training), the loss
    gradient wrt Delta is -ms_factor/2, so pairs with a bigger quality gap get a
    proportionally bigger gradient - the most informative, least noisy comparisons drive
    the biggest updates early on. It is clamped to max_ms_factor to avoid instability if a
    sampled rollout has a pathologically small makespan (ms_better -> 0).

    Args:
        logp_better / logp_worse: scalar tensors (grad-enabled), per-trajectory
            log-likelihood of the better/worse trajectory under the current policy.
        ms_better / ms_worse: python floats, final makespans of the two trajectories.

    Returns:
        Scalar loss tensor for this pair.
    """
    ms_factor = float(ms_worse) / (float(ms_better) + eps)
    ms_factor = min(ms_factor, max_ms_factor)
    return -torch.log(torch.sigmoid(ms_factor * (logp_better - logp_worse)) + eps)


# A pair whose scaled log-prob margin ms_factor*(logp_better - logp_worse) exceeds this
# has sigmoid > 0.993, i.e. its SRO loss (and gradient) is effectively zero.
SRO_SATURATION_MARGIN = 5.0


def bopo_group_loss(logp, makespans, K, greedy_idx=None, exclude_greedy=False):
    """
    Build the (best vs. K-1 worst) preference pairs for one sampled group and return the
    mean SRO loss, plus diagnostics for whether that loss still carries any gradient.

    exclude_greedy: when the group contains a greedy (argmax) rollout (greedy_idx), leave it
    out of the pairs. The greedy trajectory is by construction the policy's most likely
    one, so whenever it is also the best rollout its pairs are already ranked correctly by
    a wide margin and the loss saturates to ~0 - measured on the ojm sweep, 60-80% of
    transformer updates had loss < 1e-3. Its makespan is still reported by the caller.

    Args:
        logp: 1D tensor (B,), grad-enabled trajectory log-likelihoods (sum or per-decision
            mean, see BOPO.logp_norm).
        makespans: 1D tensor (B,), final makespan of each rollout.
        K: number of solutions per comparison (best + up to K-1 worse ones).
        greedy_idx: index of the greedy rollout in the group, or None if there is none.
        exclude_greedy: see above.

    Returns:
        (loss, diagnostics dict of python floats).
    """
    candidates = torch.arange(makespans.shape[0], device=makespans.device)
    if exclude_greedy and greedy_idx is not None and makespans.shape[0] > 2:
        candidates = candidates[candidates != greedy_idx]

    best_local, worse_local = select_pairs(makespans[candidates].detach(), K)
    best_idx = int(candidates[best_local])
    worse_idx = candidates[worse_local].tolist()

    pair_losses = []
    margins = []
    for w in worse_idx:
        ms_b, ms_w = float(makespans[best_idx]), float(makespans[w])
        pair_losses.append(sro_loss(logp[best_idx], logp[w], ms_b, ms_w))
        ms_factor = min(ms_w / (ms_b + 1e-7), 10.0)
        margins.append(ms_factor * float((logp[best_idx] - logp[w]).detach()))
    loss = torch.stack(pair_losses).mean()

    diagnostics = {
        "sro_margin_mean": sum(margins) / len(margins),
        "sro_saturated_frac": sum(m > SRO_SATURATION_MARGIN for m in margins) / len(margins),
    }
    if greedy_idx is not None:
        diagnostics["greedy_is_best"] = float(float(makespans[greedy_idx]) <= float(makespans.min()))
    return loss, diagnostics


def compute_actor_grad_norm(actor):
    """
    Total L2 norm of the actor's parameter gradients after loss.backward(), before
    optimizer.step(). Logged per BOPO update to tell apart two failure modes that look
    identical from the loss/entropy curves alone: gradients near zero (a real bug -
    vanishing gradients, dead ReLUs, a masked-out loss term, etc.) vs. healthy gradients
    that simply haven't had enough update steps yet to move the policy off-uniform.

    Args:
        actor: the policy's actor module, right after loss.backward().

    Returns:
        Python float, the L2 norm of all actor gradients combined (0.0 if none set).
    """
    total = 0.0
    for p in actor.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def run_behavior_cloning(env, policy, optimizer, instance_index):
    """
    One teacher-forced behavior-cloning update for BOPO's warm-start phase (see
    src/train.py's warm_start_steps). Rolls out a full episode by following the env's
    own expert_action() heuristic (teacher forcing: the EXPERT's action is what actually
    gets played into env.step(), not the policy's own action, so a still-untrained
    policy's mistakes never compound into a corrupted trajectory), and fits the policy's
    masked action distribution to that expert action at every decision step via
    cross-entropy (implemented as -log_prob(expert_action)).

    Why this exists: BOPO bootstraps entirely from its own sampled trajectories, with no
    external reference solution. If the initial network is close to uniform, sampled
    trajectories' log-likelihood barely varies with actual solution quality (all actions
    are ~equally likely regardless of outcome), so the pairwise SRO loss has almost no
    gradient to work with - empirically, action_entropy_max_entropy_normalized stayed
    above 0.99 for 100 straight training episodes on results/*_oo. A short supervised
    warm start against a simple dispatch heuristic breaks that symmetry before BOPO
    takes over, giving it a non-uniform starting policy to actually refine.

    Every representation's env implements expert_action() (envo_o.py: Most-Work-
    Remaining; env.py/envheterogeneosmo.py: the same earliest-completion-time criterion
    already used to build the action mask) and every representation's Policy implements
    action_distribution(state) (masked softmax over a single, non-batched graph, with
    gradient) - this function only depends on that shared interface, not on any
    representation-specific state layout.

    Args:
        env: a representation's env instance (already constructed with its instance
            pool); reset()/step() are called directly, mutating env's own state - safe
            because BOPO's sample_group() always rolls out on FRESH env copies built
            from type(env), never on this exact object.
        policy: the BOPO agent's Policy module (its actor's parameters are what get
            trained here, via the SAME optimizer as the main BOPO loop).
        optimizer: the BOPO agent's optimizer.
        instance_index: which of env.instances to warm-start on for this call.

    Returns:
        (avg_loss, steps): mean per-decision cross-entropy loss (python float) and the
        number of decision steps taken in the teacher-forced episode.
    """
    device = next(policy.parameters()).device
    state = env.reset(sel_index=instance_index)
    losses = []

    while True:
        norm_state = env.normalize_state(state).to(device)
        dist = policy.action_distribution(norm_state)
        expert = env.expert_action()
        losses.append(-dist.log_prob(torch.tensor(expert, device=device)))
        state, _, done, _ = env.step(expert)
        if done:
            break

    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item()), len(losses)


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

    # random_refs = log(n_valid): the entropy of a UNIFORM distribution over the same
    # number of valid actions - the reference needed to tell whether a given raw entropy
    # value is "close to uniform" or not (a raw entropy of 2.0 means something different
    # at n_valid=5 vs n_valid=50).
    random_refs = [math.log(c) if c > 1 else 0.0 for c in valid_counts]
    # normalized = action_entropy / random_ref: this is THE headline signal for whether
    # the policy is actually learning anything (1.0 = indistinguishable from uniform
    # random, 0.0 = fully deterministic) - see the entropy-collapse diagnosis discussed
    # in conversation.
    normalized = [(e / r) if r > 1e-8 else 1.0 for e, r in zip(entropies, random_refs)]
    # effective_counts = exp(entropy): "effective number of equally-likely choices" -
    # kept alongside the normalized ratio above because it's a more intuitive framing
    # (e.g. "9 effective choices out of 12 valid" reads more directly than "0.92
    # normalized entropy"), not because it carries different information.
    effective_counts = [math.exp(e) for e in entropies]

    def mean(values):
        return sum(values) / len(values)

    return {
        "action_entropy": mean(entropies),
        "action_entropy_random_ref": mean(random_refs),
        "action_entropy_max_entropy_normalized": mean(normalized),
        "action_entropy_effective_action_count": mean(effective_counts),
    }
