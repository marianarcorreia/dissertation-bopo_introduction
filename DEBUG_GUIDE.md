# Debug & Workflow Tracing Guide

All verbose output is controlled by the **`FJSP_DEBUG`** environment variable.  
Prefixes distinguish the source: `[MAIN]`, `[VAL]`, `[TRAIN]`, `[ENV]`, `[BOPO]`.

The training algorithm is **BOPO** (self-rewarding preference/ranking optimisation, no
critic): each training step samples `B` parallel trajectories of the same instance,
self-labels pairs by relative makespan, and optimises a pairwise ranking loss — see
`src/bopo_utils.py` and the `BOPO` class in `src/bopo.py` / `src/bopomo.py` / `src/ppoo_o.py`.

| Level | What is shown |
|-------|---------------|
| `0`   | Default — only always-on `[MAIN]` / `[TRAIN]` / `[VAL]` banners + per-step loss |
| `1`   | + env lifecycle (create/reset/done), BOPO init, update summary, model save/load |
| `2`   | + every env step (job/machine chosen, reward, active edges), action selection detail |
| `3`   | + GNN forward shapes, sampled-group makespan stats, per-round update detail |

All commands below use a **tiny problem** that fits in one terminal window:
- 2 training steps, 3 training instances, 3–4 jobs, 2–3 machines, ≤3 ops/job.

---

## 1 — Full workflow at a glance (level 0)

Bare summary: validation generation → training loop → final validation.

```powershell
$env:FJSP_DEBUG = "0"
python -c "
from src.generate_val import generate_val
from src.train import train
generate_val(2)
train(max_episodes=2, new_freq=1, n_cases=3,
      j_min=3, j_max=4, m_min=2, m_max=3, op_max=3, B=4, K=2, lr=0.001)
"
```

---

## 2 — Environment objects & lifecycle (level 1)

See `FJSSPEnv.__init__`, `generate_instance`, `reset`, and episode completion.

```powershell
$env:FJSP_DEBUG = "1"
python -c "
import json, os
from src.generate_val import generate_val
generate_val(2)
from src.env import FJSSPEnv
from src.generator import generate_instance_list
from src.parsedata import get_data, parse

instances = []
for raw in generate_instance_list(n_cases=2, range_jobs=(3,4),
        range_machines=(2,3), range_op_per_job=(2,3), max_processing=10):
    jobs, ops, info, mx = get_data(parse(raw))
    instances.append({'jobs': jobs, 'operations': ops, 'maximum': mx, 'num_machines': info['machinesNb']})

env = FJSSPEnv(instances, mask_option=1, sel_k=1)
state = env.reset()
print('Node types:', list(state.node_types))
print('Edge types:', list(state.edge_types))
print('job.x shape:', state['job'].x.shape)
print('operation.x shape:', state['operation'].x.shape)
print('machine.x shape:', state['machine'].x.shape)
print('machine->job edges:', state['machine','exec','job'].edge_index.shape)
print('mask:', state['machine','exec','job'].mask)
"
```

---

## 3 — Step-by-step environment actions (level 2)

Trace every scheduling decision: chosen job/machine, reward, active-edge count, mask recalculation.

```powershell
$env:FJSP_DEBUG = "2"
python -c "
from src.generate_val import generate_val
generate_val(1)
from src.env import FJSSPEnv
from src.generator import generate_instance_list
from src.parsedata import get_data, parse

instances = []
for raw in generate_instance_list(n_cases=1, range_jobs=(3,3),
        range_machines=(2,2), range_op_per_job=(2,2), max_processing=8):
    jobs, ops, info, mx = get_data(parse(raw))
    instances.append({'jobs': jobs, 'operations': ops, 'maximum': mx, 'num_machines': info['machinesNb']})

env = FJSSPEnv(instances, mask_option=1, sel_k=1)
state = env.reset()
done = False
while not done:
    action = env.sample()          # random valid action
    state, reward, done, info = env.step(action)
print('Final makespan:', env.mk)
"
```

---

## 4 — BOPO initialisation & update (level 1)

Shows `Policy` construction, `BOPO.__init__` hyperparams, and update loss summary.

```powershell
$env:FJSP_DEBUG = "1"
python -c "
from src.generate_val import generate_val
generate_val(1)
from src.train import train
train(max_episodes=4, new_freq=1, n_cases=2,
      j_min=3, j_max=4, m_min=2, m_max=3, op_max=3,
      B=4, K=2, lr=0.001, hidden_channels=32, num_layers=1, heads=2)
"
```

---

## 5 — GNN + Actor forward passes (level 3)

Traces tensor shapes through the `GAT` message-passing layers and the actor linear head
(BOPO has no critic, so there is only one head to trace).

```powershell
$env:FJSP_DEBUG = "3"
python -c "
from src.generate_val import generate_val
generate_val(1)
from src.env import FJSSPEnv
from src.generator import generate_instance_list
from src.parsedata import get_data, parse
from src.bopo import BOPO

instances = []
for raw in generate_instance_list(n_cases=1, range_jobs=(3,3),
        range_machines=(2,2), range_op_per_job=(2,2), max_processing=8):
    jobs, ops, info, mx = get_data(parse(raw))
    instances.append({'jobs': jobs, 'operations': ops, 'maximum': mx, 'num_machines': info['machinesNb']})

env = FJSSPEnv(instances, mask_option=1, sel_k=1)
state = env.reset()
metadata = state.metadata()

agent = BOPO(0.001, env, metadata, hidden_channels=16, num_layers=1, heads=2, B=4, K=2)

# Sample one group of B parallel trajectories of the same instance — triggers
# GAT.forward once per decision round (batched across the still-active copies).
mean_logp, makespans, rounds, entropy_stats = agent.sample_group(0)
print('Rounds:', rounds, '| makespans:', makespans.tolist())
"
```

---

## 6 — Full training cycle detail (level 2)

Combines training loop + per-step env tracing + BOPO update detail — best for a complete picture.

```powershell
$env:FJSP_DEBUG = "2"
python -c "
from src.generate_val import generate_val
generate_val(2)
from src.train import train
train(max_episodes=4, new_freq=2, n_cases=2,
      j_min=3, j_max=3, m_min=2, m_max=2, op_max=2,
      B=4, K=2, lr=0.001, hidden_channels=16, num_layers=1, heads=2)
"
```

---

## 7 — Validation phase only (level 1)

Load training artefacts and re-run only the validation loop.

```powershell
$env:FJSP_DEBUG = "1"
python -c "
import json, torch
from src.env import FJSSPEnv
from src.bopo import BOPO

with open('val/validation_set.json') as f:
    val_set = json.load(f)
with open('val/validation_results.json') as f:
    val_res = json.load(f)

env = FJSSPEnv(val_set, mask_option=1, sel_k=1)
state = env.reset()
metadata = state.metadata()
agent = BOPO(0.001, env, metadata, hidden_channels=32, num_layers=1, heads=2, B=4, K=2)

for i, inst in enumerate(val_set):
    s = env.reset()
    for q in range(1, 10**6):
        a = agent.select_action(s, 2, q)
        s, _, done, _ = env.step(a)
        if done:
            gap = env.mk / int(inst['score']) - 1
            print(f'  val[{i}] makespan={env.mk}  optimal={inst[\"score\"]}  gap={gap:.4f}')
            break
"
```

---

## Quick reference

```powershell
# Disable all extra output (production-like)
$env:FJSP_DEBUG = "0"

# Environment + training lifecycle only
$env:FJSP_DEBUG = "1"

# Add per-step action/reward trace
$env:FJSP_DEBUG = "2"

# Add GNN tensor shapes + reward statistics
$env:FJSP_DEBUG = "3"

# Remove the variable (resets to 0)
Remove-Item Env:FJSP_DEBUG
```
