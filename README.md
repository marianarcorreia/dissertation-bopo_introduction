# Graph Representations for DRL-based FJSP — Study Document

## 1. Representations

### 1.1 Disjunctive Graph — $G^{disj} = \{O, C, D_t\}$

The disjunctive graph is the classical OR scheduling graph adapted for DRL.
It contains only **operation nodes** O, connected by two families of edges:

- **Conjunctive edges C** (precedence): directed arcs enforcing job order.
  Operation O_i → O_j means O_j cannot start before O_i finishes.
- **Disjunctive edges D_t** (machine sequences): undirected arcs between
  pairs of unscheduled operations competing for the same machine. These
  encode machine conflict directly in the operation-to-operation topology.
  Subscript _t indicates they are dynamic: arcs are removed as operations
  get scheduled and machine conflicts are resolved.

**Inductive bias**: machine conflict is structurally explicit. Two operations
competing for the same resource are directly connected, so the GNN can reason
about conflict through local neighborhood aggregation without needing a
machine node as intermediary.

**Strengths**:
- Machine contention is a first-class topological signal, not an aggregated feature
- Natural substrate for sequence-dependent setup times (SDST): setup costs
  can be placed as edge features on disjunctive arcs, since those arcs already
  represent potential consecutive machine sequences
- Well-studied in the JSP literature — benchmarks exist for comparison

**Weaknesses**:
- Dense and dynamic: the initial disjunctive arc count is O(|O|²) in the worst
  case (all operations eligible for all machines); for flexible problems this
  grows further with machine flexibility
- Dynamic edge removal at every step means the GNN's receptive field changes
  structurally during an episode, which can destabilize training
- Action space is over operation pairs or operation-machine pairs, which is
  larger than the action space of representations that use a job node to
  summarize pending work
- No explicit job context: job-level signals (remaining workload, job completion
  time) must be computed bottom-up over potentially many message-passing hops

**Key reference**: classical disjunctive graph (Adams et al., 1988, https://doi.org/10.1287/mnsc.34.3.391); DRL adaptation in multiple recent works including Zhang et al. 2020 (https://doi.org/10.48550/arXiv.2010.12367).

---

### 1.2 Heterogeneous Operation-Machine Graph — G^OM = {O, M, C, D_t}

The O-M graph introduces a dedicated **machine node** type M, shifting conflict
representation from the operation-operation topology to the machine node itself.

- **Conjunctive edges C**: same precedence arcs as G^disj (O → O)
- **Disjunctive edges D_t**: now connect **operations to eligible machines**
  (O → M, or bidirectional). A machine node aggregates all operations currently
  eligible to run on it, making machine load and contention implicitly computable
  from the machine node's neighborhood.

**Inductive bias**: machine nodes act as conflict bottlenecks. Rather than
pairwise operation conflict, contention is mediated through the machine node
aggregating eligible operations. This is a fundamentally different information
routing: conflict is learned as a machine-centric feature, not a pairwise
operation signal.

**Strengths**:
- Reduced edge count compared to G^disj: O-M edges scale as O(|O| × average_flexibility)
  rather than O(|O|²)
- The machine node becomes a natural locus for machine-level features (utilization,
  availability time, queue length), keeping operation nodes lighter
- Action space is over (operation, machine) pairs, directly indexable from the
  O-M edge set
- More stable graph topology: disjunctive edges in G^disj are removed per-step,
  while O-M eligibility edges only disappear when an operation is scheduled or
  a machine becomes ineligible — less structural churn per step

**Weaknesses**:
- Job context is absent: information about job-level state (remaining operations,
  job completion time) must be encoded in operation node features rather than
  aggregated through dedicated job nodes
- Precedence information is local to operation pairs; multi-hop job state
  requires deeper message-passing layers to propagate
- Less natural for SDST: setup costs depend on consecutive machine sequences,
  but in G^OM there are no operation-operation edges on the machine side —
  SDST would require adding new edge types or modifying the machine node features

**Key reference**: Song et al. 2023 (https://doi.org/10.1109/TII.2022.3189725, https://github.com/songwenas12/fjsp-drl).

---

### 1.3 Heterogeneous Operation-Job-Machine Graph — G^OJM = {O, M, J, C, D_t, E_t}

The O-J-M graph adds a **job node** type J as a hierarchical aggregator,
introducing a richer set of both conjunctive and disjunctive edges:

- **Conjunctive edges C**: two subtypes
  - O → O: same precedence arcs as before
  - O → J: each operation communicates upward to its parent job node,
    enabling the job node to aggregate state across all its operations
- **Disjunctive edges D_t**: two subtypes
  - O → M: operation-machine eligibility (same as G^OM)
  - J → M (edge set E_t): **the action edge** — connects the job node
    (representing the job's next available operation) to eligible machines.
    This is a dense, summarized action space where one action = assign the
    next operation of job J to machine M.

Additional lateral edges are typically included:
- M → M: machine-to-machine communication for load balancing signals
- J → J: job-to-job communication for global schedule state

**Inductive bias**: job nodes act as hierarchical summarizers. A single message
from a job node to a machine carries aggregated information about the job's
full remaining workload, its current completion time, and its urgency — information
that would require several hops to compute in G^disj or G^OM. The J-M action
edge makes the action space job-centric rather than operation-centric.

**Strengths**:
- Compact action space: |actions| = |eligible J-M pairs| ≤ |jobs| × |machines|,
  which is smaller than operation-level action spaces for problems with many
  operations per job
- Job-level context is explicitly routed to machine nodes via J-M edges, enabling
  the policy to reason about job urgency without relying on deep message passing
- The critic can pool over job nodes for value estimation, using well-aggregated
  job state rather than raw operation features
- Naturally accommodates worker constraints: a worker node type can be added
  in parallel to the machine node with the same J-W edge pattern

**Weaknesses**:
- Larger encoder: more node types and edge types mean more parameter sets in
  the hetero GNN, which can slow convergence on small datasets
- The job node may conflate job identity with job state if not carefully designed,
  causing the encoder to partially memorize job indices rather than generalize
  to new instance sizes
- More complex graph reconstruction at each step: both operation and job nodes
  must be updated consistently after each scheduling decision

**Key reference**: Echeverria et al. 2025 (current implementation baseline, https://doi.org/10.1016/j.engappai.2024.109488, https://github.com/Echever/RL-for-FJSSP/tree/main).

---

## 2. Representation Comparison

| Property | G^disj | G^OM | G^OJM |
|---|---|---|---|
| Node types | O | O, M | O, M, J |
| Conflict encoding | Explicit O-O arcs | Implicit via M aggregation | Implicit via M aggregation + J context |
| Action space size | O(\|O\|²) or O(\|O\|×\|M\|) | O(\|O\|×\|M\|) | O(\|J\|×\|M\|) |
| Edge dynamism | High (disj arcs removed per step) | Medium (O-M edges removed on schedule) | Medium (J-M edges updated per step) |
| Job context | Implicit (multi-hop) | Implicit (feature-encoded) | Explicit (J node) |
| Natural fit for SDST | High (O-O arc carries sequence) | Low (requires new arc type) | Medium (can add O-O on machine side) |
| Natural fit for workers | Low (requires redesign) | Medium (add W node parallel to M) | High (add W node parallel to M with J-W edges) |
| Encoder parameter count | Low | Medium | High |

---

## 3. Implementation Requirements

### 3.1 Schema Independence

Each representation is a self-contained graph schema. Node feature
dimensionalities and edge attribute dimensionalities are defined per
representation and differ between them. PyG's `HeteroData` and `to_hetero`
handle variable schemas natively — the encoder rebuilds itself from `metadata()`
at initialization, so there is no constraint forcing shared dimensionalities.

Information that is delegated to dedicated node types in richer representations
must be absorbed into operation node features in simpler ones. Specifically:
- In G^disj: machine availability, job workload, and job precedence context all
  become operation node features
- In G^OM: job workload and job precedence context become operation node features;
  machine-level features live in machine nodes as intended

The only shared interface across representations is the **action edge**: a set
of (resource, candidate) pairs with a boolean mask, where the policy selects
an index. This edge must exist in the graph data structure regardless of
representation, as it is the contract with the PPO loop.

### 3.2 G^disj Specific

- No machine or job nodes. The graph contains operation nodes only.
- Disjunctive arcs must be fully rebuilt at each step over unscheduled operations
  only. Profile this cost early — it dominates step time for large instances.
- The critic must pool over operation nodes. This requires adjusting the value
  estimation head relative to the G^OJM baseline.
- Conjunctive arcs should be strictly directed without self-loops.

### 3.3 G^OM Specific

- No job nodes. The action edge connects operations directly to machines.
- Precedence is not represented topologically; it must be compensated by
  including position-in-job and remaining-ops-count in operation features.
- M-M edges should be included following Song et al. 2023.

### 3.4 G^OJM Specific (Current Baseline)

Before running ablations, document explicitly:
- The role of self-loops in the O-O precedence edges
- What features J nodes carry and when they are updated within a step
- Whether J-J and M-M edges are active and what information they carry
- How the J-M action edge is repopulated after each scheduling decision

This documentation is the reference for understanding what G^disj and G^OM
lose by not having job nodes.
---

## 4. Intermediate Research Questions

These are questions this experimental stage should produce evidence toward,
even if not definitively answer. They guide what to log and how to analyze it.

### Q1 — Conflict Representation and Policy Learning Speed

*Does making machine conflict topologically explicit (G^disj) accelerate early
learning compared to representations where conflict is implicit?*

Rationale: in G^disj, the GNN can detect conflict between two operations from
a single message-passing hop. In G^OM and G^OJM, conflict is inferred from
the machine node's aggregated neighborhood. If explicit conflict accelerates
credit assignment, G^disj should show faster initial makespan improvement.
However, if the dense dynamic topology introduces noise, the opposite may occur.

Observable proxy: slope of validation makespan vs. training episodes in the
first 30% of training, before any representation has converged.

### Q2 — Job Context and Value Estimation Quality

*Does the explicit job node in G^OJM lead to lower value loss compared to
representations where job state must be inferred?*

Rationale: the critic estimates future makespan from the current graph state.
If job-level context (remaining work, urgency) is easy to read from the graph,
value estimation should be more accurate earlier in training. G^OJM provides
a dedicated job aggregation channel; G^OM and G^disj do not.

Observable proxy: mean squared error of critic predictions vs. actual episode
return, tracked separately from policy loss across training.

### Q3 — Graph Density and Computational Scalability

*How does edge count per step scale with instance size for each representation,
and does this predict wall-clock training time?*

Rationale: G^disj has quadratic worst-case edge growth; G^OM and G^OJM are
linear in operations × machines. If G^disj's edge count grows substantially
faster, it may be impractical for real-world instance sizes even if it learns
faster per episode.

Observable proxy: edge count per step as a function of (n_jobs × n_ops_per_job)
across the training distribution. Also: GNN forward pass time per step.

### Q4 — Action Space Compactness and Exploration Efficiency

*Does the compact J-M action space in G^OJM lead to more efficient exploration
compared to the larger O-M or O-O action spaces?*

Rationale: a smaller action space requires fewer samples to explore, all else
equal. G^OJM's action space is bounded by |J|×|M|; G^OM's by |O|×|M|.
If policy entropy converges faster for G^OJM, this supports the hypothesis that
action space compactness matters for learning efficiency.

Observable proxy: per-episode action entropy over training. Lower entropy
earlier = faster policy convergence. But also watch for premature entropy
collapse, which indicates the policy is overconfident before it is good.

### Q5 — Representation Stability Under Dynamic Graph Changes

*Does the structural dynamism of G^disj (arc removal per step) increase
training variance compared to the more stable G^OM and G^OJM topologies?*

Rationale: the GNN's effective receptive field changes at every step in G^disj
because disjunctive arcs are removed as scheduling progresses. This means the
same policy network sees structurally different neighborhoods for the same
operation depending on how many other operations have been scheduled. G^OM and
G^OJM have more stable topologies — edges are removed but node connectivity
patterns are more predictable.

Observable proxy: variance of episode reward across seeds and across episodes
at the same training point. Higher variance for G^disj would support the instability
hypothesis.

### Q6 — Representation Extensibility for Future Constraints

*Which representation provides the most natural extension point for sequence-
dependent setup times (SDST) and worker constraints?*

This is not measured experimentally in this stage — it is a design question to
be argued from the structural analysis. The answer should be written as a
hypothesis before experiments start, so it can be confirmed or contradicted when
those constraints are added.

---

## 5. Training, Validation, and Testing protocols

### 5.1. Training

Each episode consists of one complete scheduling sequence on one randomly generated instance. Instances drawn are drawn from a fixed training pool that is periodically refreshed.

- **Training pool**: $N$ randomly generated instances per pool
- **Pool refresh**: Every $N$ episodes, regenerate the pool with new random instances
- **Policy update**: One PPO update (training) every $K$ episodes (collect $K$ rollouts/solutions, then update)
- **Instance size**: Fixed per experiment (e.g., 6Jx5M for small runs)

### 5.2. Validation

Validation is performed periodically (every 10 episodes) on a fixed dataset of instances, which are solved offline with Constraint Programming, for computing the optimality gap (error to best solution). A fixed validation instance dataset was generated (val/instances) and solved using ORtools (src/solver.py, invoked from src/generate_val.py). The resulting solutions are kept in val/solutions, storing both the solution and an approximate of the optimal value of the objective function ("indicators" > "best_objective_bound"). The main metric to record during validation is the average validation gap, defined as the relative distance to optimal solution (average across validation instances). For an instance $i$, assuming a DRL performance of $C_{max}^{DRL}$, and a known objective bound $C_{max}^{ref}$, the optimality gap is computed as $C_{max}^{DRL}/C_{max}^{ref} - 1$. The average validation gap consists of the average (and standard deviation) of this expression across all validation instances. In principle, the learning model will be as good as the decrease in validation gap.


### 5.3. Testing

Testing is performed once at the end of training on a held-out test set never seen during training or validation. Three test conditions:

- **In-distribution**: same instance size as training 
- **Out-of-distribution**: larger instances, e.g., 10J×8M when trained on 6J×5M, to measure structural generalization
- **Benchmark**: diverse datasets comprising the addressed problems in the literature (reproducibility)

## 6. Design of Experiments

**Visualization**
Create a dashboard that allows for visualizing the main KPIs across training and validation (gap, makespan, loss, etc.). Use matplotlib and seaborn or plotly with pastel colorscale, and whitegrid style. 

### 6.1 Experimental Conditions

**Controlled variables** (identical across all runs):
- Training instance distribution: same generator, same size range (to be specified)
- PPO hyperparameters: lr, gamma, K_epochs, eps_clip, batch_size — fixed across representations
- GNN architecture: same hidden_channels, num_layers, heads — only metadata changes
- Validation set: same fixed instances across all representations

**Independent variable**: graph representation ∈ {G^disj, G^OM, G^OJM}

**Runs table**:

| Run ID | Representation | Instance size  | Episodes | Purpose |
|---|---|---|---|---|---|
| R1.x | G^OJM | small (6J×5M) | 500 | Baseline learning curve |
| R2.x | G^OM | small (6J×5M) | 500 | Learning curve comparison |
| R3.x | G^disj | small (6J×5M) | 500 | Learning curve comparison |
| R4.x | G^OJM | medium (10J×8M) | 300 | Size generalization signal |
| R5.x | G^OM | medium (10J×8M) | 300 | Size generalization signal |
| R6.x | G^disj | medium (10J×8M) | 300 | Size generalization signal |
| R7 | — | small→medium | — | Train small, test medium (all 3 reps) |

Total runs: 6 training runs + 3 cross-size eval passes.

Note: 500 episodes is insufficient for convergence on 10×8 instances — runs R4–R6 are explicitly for early-trajectory analysis only, not final quality comparison.

### 6.2 KPIs

#### Primary KPIs (policy quality)

**Makespan Gap** = (RL makespan / OR-Tools makespan) - 1

The primary quality metric. Measured on the fixed validation set at checkpoints:
episode 10, 20, 30, etc. Report mean ± std across validation instances per representation.
A gap of 0 = matches the OR-Tools 15s solution. A gap of 0.1 = 10% worse.

*Why this over raw makespan*: instance difficulty varies; the gap normalizes
across instances of different sizes. 

**Learning Curve Slope (LCS)** = linear regression slope of mean validation
makespan gap vs. episode number, computed over episodes 1–200.

Measures how fast each representation learns, independent of final quality.
A steeper (more negative) slope = faster improvement per episode. This is the
primary metric for comparing learning efficiency when convergence is not reached.

#### Secondary KPIs (training dynamics)

**Value Loss** = mean squared error between critic output and discounted return.

Tracked per update step. Measures whether the graph structure supports credit
assignment — a representation that produces more informative state embeddings
should yield lower value loss earlier. Reported as a learning curve alongside
the makespan gap curve.

**Action Entropy** = entropy of the policy's action distribution at each step,
averaged over all steps in an episode.

Two signals:
1. *Initial entropy* (episode 1–10): should be near log(|action_space|) — a
   sanity check that the policy starts unbiased
2. *Entropy trajectory*: rate of entropy decrease. Fast decrease = policy
   specializing quickly (good if makespan also improves; bad if it collapses
   prematurely). Compare across representations.

**Policy Loss** = PPO clipped surrogate loss, tracked per update.

Separates policy gradient signal quality from value estimation quality. If
value loss is high but policy loss is low for a given representation, the
representation supports action selection better than state valuation.

#### Computational KPIs

**Episode Wall Time** = total seconds per episode (reset + all steps + update).

Decomposes into:
- *Graph construction time*: time to build/update graph state per step
- *GNN forward time*: time for policy evaluation per step

These are particularly important for G^disj, where graph reconstruction at
each step involves rebuilding disjunctive arcs. Report as a box plot across
episodes (early vs. late in episode, since arc count decreases as ops are scheduled).

**Edge Count per Step** = total number of edges across all edge types in the
graph, at each scheduling step within an episode.

Averaged over episodes, plotted as a function of scheduling progress (step 0 = 
start, step N = last operation). This reveals:
- How graph density evolves within an episode (G^disj should thin out; others are stable)
- How edge count scales with instance size (compare small vs. medium runs)

Report as: mean edge count at step 0 (full graph), step N/2 (halfway), step N (end).

**Parameter Count** = total trainable parameters in the actor and critic networks.

Computed once per representation after initialization. G^OJM is expected to have
more parameters due to additional node and edge type embeddings. This is a
confound for quality comparisons — representations with more parameters have
higher capacity.

#### Generalization KPIs (from R7)

**Out-of-distribution Makespan Gap** = same gap metric but evaluated on medium
instances after training only on small instances.

Measures structural generalizability. A representation whose inductive bias
captures the combinatorial structure of FJSP (rather than overfitting to
instance size) should generalize better. This is the single most informative
result for the long-term research question.

### 6.3 Analysis Plan

**For Q1 (conflict and learning speed)**: compare LCS across R1–R3. Plot learning
curves for all three on the same axes. Perform a paired t-test on LCS values
across seeds to assess whether differences are significant.

**For Q2 (job context and value estimation)**: plot value loss curves for R1–R3
on the same axes. Compute correlation between value loss at episode 100 and
makespan gap at episode 500 across seeds — a high correlation would mean value
estimation quality is a good predictor of final policy quality.

**For Q3 (density and scalability)**: plot edge count vs. step for one representative
episode per representation. Then plot mean episode wall time vs. instance size
(using small vs. medium runs). Fit a scaling function to identify whether G^disj
wall time grows super-linearly with instance size.

**For Q4 (action space and exploration)**: plot action entropy over training for R1–R3.
Annotate with the theoretical action space size for each representation on the
specific instances used.

**For Q5 (stability)**: plot makespan gap variance across seeds (the standard deviation
band on the learning curve is the signal). Also plot value loss variance. G^disj
instability hypothesis predicts higher variance in both.

**For Q6 (extensibility)**: no experimental analysis — produce a written design
assessment comparing how each of SDST and worker constraints would modify the
graph schema for each representation. This is a qualitative deliverable for the
presentation.

### 6.4 Presentation Deliverables

The presentation should deliver:
1. Learning curve figure: makespan gap mean ± std vs. episodes for all 3 representations on small instances (R1–R3)
2. Edge count profile: edge count per step for all 3 representations on one canonical instance
3. Scalability plot: episode wall time vs. instance size (small vs. medium, all 3)
4. Value loss comparison: critic MSE curves for R1–R3
5. Constraint extensibility table: qualitative assessment of G1 vs G2 for SDST and workers
6. One slide of open questions from Q1–Q6, without answers
#   d i s s e r t a t i o n - b o p o _ i n t r o d u c t i o n  
 