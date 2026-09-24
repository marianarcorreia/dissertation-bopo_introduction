from typing import List, Optional, cast
import torch
import torch.nn as nn
from torch_geometric.nn import Linear, to_hetero #camadas do GNN
from torch_geometric.data import HeteroData, Batch
from torch.distributions import Categorical #dist. de probab. para ações
import random
import os
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from src.bopo_utils import bopo_group_loss, summarize_action_entropy, compute_actor_grad_norm
from src.gat import GAT
from src.gine import GINModel
from src.transformer import TransformerModel

# Controlled by FJSP_DEBUG (same variable as env.py)
# 1=BOPO lifecycle  2=+forward/action  3=+update internals
_DBG = int(os.environ.get("FJSP_DEBUG", "0"))

def _dbg(level, *args, **kwargs):
    if _DBG >= level:
        print("[BOPO]", *args, **kwargs)

device = torch.device('cpu')

if(torch.cuda.is_available()) and random.random()<1:
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    _dbg(1, f"Device set to: {torch.cuda.get_device_name(device)}")
else:
    _dbg(1, "Device set to: cpu")

#BOPO não tem critic: o único modelo treinado é o ator, que atribui um score a cada aresta
#maquina->job candidata. Sem baseline/valor de estado, porque a loss do BOPO (SROLoss)
#compara diretamente a log-likelihood de trajetórias completas, em vez de usar uma vantagem.
class ActorModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_layers = 2, heads = 3, gnn_type = 'gat'):
        super().__init__()
        _dbg(1, f"  ActorModel.__init__ | hidden={hidden_channels} | layers={num_layers} | heads={heads} | gnn_type={gnn_type}")
        #modelo gnn homogeneo, apenas processa um tipo de no e de aresta
        if gnn_type == 'gat':
            self.gnn = GAT(hidden_channels, out_channels, num_layers=num_layers, heads=heads)
        elif gnn_type == 'gin':
            self.gnn = GINModel(hidden_channels, out_channels, num_layers=num_layers, heads=heads)
        elif gnn_type == 'transformer':
            self.gnn = TransformerModel(hidden_channels, out_channels, num_layers=num_layers, heads=heads)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type!r} (expected 'gat', 'gin' or 'transformer')")
        #to_hetero converte o modelo homogeneo para um modelo heterogeneo
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')
        #um score por aresta
        self.lin3 = Linear(-1, 1)

    #passa o grafo para o gnn het, embeddings atualizados, e depois processa os embeddings para calcular os scores das ações.
    def forward(self, data: HeteroData):
        _dbg(3, "  ActorModel.forward")
        res = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        #concatena os embeddings da maq., features da aresta e emb. do job
        x_src, x_dst = res['machine'][data.edge_index_dict[('machine','exec','job')][0]], res['job'][data.edge_index_dict[('machine','exec','job')][1]]
        edge_feat = torch.cat([x_src,  data.edge_attr_dict[('machine','exec','job')], x_dst], dim=-1)
        res = self.lin3(edge_feat)
        _dbg(3, f"    actor logits shape={res.shape}")
        return res


class Policy(nn.Module):
    def __init__(self, metadata, hidden_channels=128, num_layers=2, heads = 3, gnn_type = 'gat'):
        super(Policy, self).__init__()
        _dbg(1, f"Policy.__init__ | hidden={hidden_channels} | layers={num_layers} | heads={heads} | gnn_type={gnn_type}")
        self.gnn_type = gnn_type
        self.actor = ActorModel(hidden_channels, 32, metadata, num_layers, heads, gnn_type=gnn_type)
        self.metadata = metadata
        self.soft = torch.nn.Softmax(dim=0)

    def forward(self):
        raise NotImplementedError

    #distribuição de ações mascarada para um único grafo (não em batch), com gradiente -
    #usada por act() (amostragem) e pela fase de warm-start (behavior cloning, ver
    #src/bopo_utils.py:run_behavior_cloning) que ajusta a política a uma ação especialista
    #por cross-entropy antes do BOPO propriamente dito começar.
    def action_distribution(self, state):
        action_probs = self.actor(state).T[0]
        action_probs[state[('machine','exec','job')].mask] = float("-inf")
        action_probs = self.soft(action_probs)
        return Categorical(action_probs)

    #seleciona uma ação para um único grafo (não em batch) - usado em teste/validação
    def act(self, state, sample, num):
        dist = self.action_distribution(state)

        if sample == 0:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs)

        action_logprob = dist.log_prob(action)
        _dbg(2, f"  act() | step={num} | sample_mode={sample} | chosen_edge={int(action)} | logprob={float(action_logprob):.4f} | n_valid={int((~state[('machine','exec','job')].mask).sum())}")

        return action.detach(), action_logprob.detach()

    #seleciona uma ação por grafo para um batch de B grafos da MESMA instância (rollout
    #paralelo do BOPO). Ao contrário de act(), NÃO faz detach - o gradiente tem de fluir
    #até este forward pass, porque a SROLoss usa diretamente a log-likelihood aqui calculada.
    def act_batch(self, batched_state, greedy_flags=None):
        logits = self.actor(batched_state).T[0]
        row, _ = batched_state[('machine', 'exec', 'job')].edge_index
        batch_index = batched_state["machine"].batch[row]
        mask = batched_state[('machine', 'exec', 'job')].mask

        num_graphs = int(batched_state["job"].batch.max().item()) + 1
        actions = []
        logprobs = []
        entropies = []
        valid_counts = []
        for i in range(num_graphs):
            mask_i = mask[batch_index == i]
            probs_i = logits[batch_index == i]
            probs_i[mask_i] = float("-inf")
            probs_i = self.soft(probs_i)
            dist = Categorical(probs_i)
            if greedy_flags is not None and greedy_flags[i]:
                action_i = torch.argmax(probs_i)
            else:
                action_i = dist.sample()
            actions.append(action_i)
            logprobs.append(dist.log_prob(action_i))
            entropies.append(dist.entropy())
            valid_counts.append(int((~mask_i).sum().item()))

        return torch.stack(actions), torch.stack(logprobs), torch.stack(entropies), valid_counts


class BOPO:
    def __init__(self, lr, env, metadata, hidden_channels=128, num_layers=2, heads=3,
                 B=16, K=8, use_greedy=True, gnn_type='gat',
                 logp_norm='mean', exclude_greedy_from_loss=True):
        _dbg(1, f"BOPO.__init__ | lr={lr} | B={B} | K={K} | use_greedy={use_greedy} | hidden={hidden_channels} | layers={num_layers} | heads={heads} | gnn_type={gnn_type} | logp_norm={logp_norm} | exclude_greedy_from_loss={exclude_greedy_from_loss}")

        self.env = env
        self.metadata = metadata
        self.B = B
        self.K = K
        self.use_greedy = use_greedy
        if logp_norm not in ("mean", "sum"):
            raise ValueError(f"logp_norm must be 'mean' or 'sum', got {logp_norm!r}")
        self.logp_norm = logp_norm
        self.exclude_greedy_from_loss = exclude_greedy_from_loss

        self.policy = Policy(metadata, hidden_channels, num_layers, heads, gnn_type=gnn_type).to(device)
        self.optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr)

    #inferência de uma única trajetória (usada por test_model/run_validation) - mantém a
    #mesma assinatura do PPO para não obrigar a alterações nos callers.
    def select_action(self, state, sample, num):
        with torch.no_grad():
            state = self.env.normalize_state(state)
            state = state.to(device)
            action, _ = self.policy.act(state, sample, num)
        return action

    #amostra B trajetorias paralelas da MESMA instância, avançando cada cópia do ambiente
    #passo a passo mas fazendo UM forward pass do GNN em batch por passo. O gradiente é
    #mantido ao longo de todo o rollout - a SROLoss usa diretamente estas log-probs.
    def sample_group(self, instance_index):
        env_cls = type(self.env)
        rollout_envs = [env_cls(self.env.instances, self.env.mask_option, self.env.sel_k) for _ in range(self.B)]
        states = [e.reset(sel_index=instance_index) for e in rollout_envs]

        active = list(range(self.B))
        # Keep the accumulated log-probabilities typed as optional tensors until
        # each rollout has taken its first action.
        logp_sum: List[Optional[torch.Tensor]] = [None] * self.B
        logp_count = [0] * self.B
        rounds = 0
        all_entropies = []
        all_valid_counts = []

        while active:
            norm_states = [self.env.normalize_state(states[i]).to(device) for i in active]
            batched = Batch.from_data_list(norm_states)

            greedy_flags = None
            if self.use_greedy and 0 in active:
                greedy_flags = [False] * len(active)
                greedy_flags[active.index(0)] = True

            actions, logprobs, entropies, valid_counts = self.policy.act_batch(batched, greedy_flags)

            still_active = []
            for j, i in enumerate(active):
                lp = logprobs[j]
                prev = logp_sum[i]
                logp_sum[i] = lp if prev is None else prev + lp
                logp_count[i] += 1
                all_entropies.append(float(entropies[j].item()))
                all_valid_counts.append(valid_counts[j])
                next_state, _, done, _ = rollout_envs[i].step(actions[j])
                states[i] = next_state
                if not done:
                    still_active.append(i)
            active = still_active
            rounds += 1

        assert all(count > 0 for count in logp_count), "every rollout must take at least one decision step"
        makespans = torch.tensor([e.mk for e in rollout_envs], dtype=torch.float32, device=device)
        # Trajectory log-likelihood, summed over decisions and then (logp_norm="mean", the
        # default) divided by each rollout's own decision count. The raw sum spans ~45
        # decisions, so best-vs-worst differences quickly exceed what the SRO sigmoid can
        # use: on the ojm sweep 60-80% of transformer updates had loss < 1e-3 and a median
        # actor grad norm of ~1e-6..1e-3. The per-decision mean keeps the margin in the
        # sigmoid's useful range; logp_norm="sum" restores the previous behavior.
        logp_total = torch.stack([cast(torch.Tensor, logp_sum[i]) for i in range(self.B)])
        if self.logp_norm == "mean":
            logp_total = logp_total / torch.tensor(logp_count, dtype=logp_total.dtype, device=logp_total.device)
        entropy_stats = summarize_action_entropy(all_entropies, all_valid_counts)
        _dbg(2, f"  sample_group | instance={instance_index} | rounds={rounds} | makespans min/mean/max={float(makespans.min()):.2f}/{float(makespans.mean()):.2f}/{float(makespans.max()):.2f}")
        return logp_total, makespans, rounds, entropy_stats

    #um passo de treino do BOPO: amostra o grupo de B soluções, auto-rotula pares
    #(melhor vs. K-1 piores) e otimiza a SROLoss (loss de ranking, sem critic/vantagem).
    def update(self, instance_index):
        logp_total, makespans, rounds, entropy_stats = self.sample_group(instance_index)

        # rollout 0 is the greedy one whenever use_greedy (see sample_group)
        greedy_idx = 0 if self.use_greedy else None
        loss, loss_stats = bopo_group_loss(logp_total, makespans, self.K, greedy_idx, self.exclude_greedy_from_loss)
        entropy_stats.update(loss_stats)

        self.optimizer.zero_grad()
        loss.backward()
        entropy_stats["actor_grad_norm"] = compute_actor_grad_norm(self.policy.actor)
        self.optimizer.step()

        loss_val = float(loss.item())
        best_ms = float(makespans.min().item())
        _dbg(1, f"BOPO.update() done | instance={instance_index} | loss={loss_val:.6f} | best_makespan={best_ms:.2f} | rounds={rounds}")
        return loss_val, best_ms, rounds, entropy_stats

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
        _dbg(1, f"BOPO.save() | path={checkpoint_path}")

    def _build_compatible_state_dict(self, state_dict):
        model_state = self.policy.state_dict()
        compatible_state = {}
        unexpected_keys = []
        shape_mismatch_keys = []

        for key, value in state_dict.items():
            model_value = model_state.get(key)
            if model_value is None:
                unexpected_keys.append(key)
                continue
            if isinstance(model_value, (UninitializedParameter, UninitializedBuffer)):
                compatible_state[key] = value
                continue
            if model_value.shape != value.shape:
                shape_mismatch_keys.append((key, tuple(value.shape), tuple(model_value.shape)))
                continue
            compatible_state[key] = value

        missing_keys = [key for key in model_state.keys() if key not in compatible_state]
        return compatible_state, missing_keys, unexpected_keys, shape_mismatch_keys

    #compatível com checkpoints antigos do PPO: estes guardavam pesos de actor.* e critic.*;
    #aqui só existe actor.*, por isso as chaves critic.* ficam em unexpected_keys e são
    #ignoradas, enquanto os pesos do actor (mesma arquitetura) continuam a carregar normalmente.
    def load(self, checkpoint_path):
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        compatible_state, missing_keys, unexpected_keys, shape_mismatch_keys = self._build_compatible_state_dict(state_dict)

        self.policy.load_state_dict(compatible_state, strict=False)

        if missing_keys or unexpected_keys or shape_mismatch_keys:
            _dbg(
                1,
                f"BOPO.load() compat | path={checkpoint_path} | loaded={len(compatible_state)} "
                f"| missing={len(missing_keys)} | unexpected={len(unexpected_keys)} | shape_mismatch={len(shape_mismatch_keys)}"
            )
            if _DBG >= 2:
                if unexpected_keys:
                    _dbg(2, f"  unexpected sample: {unexpected_keys[:5]}")
                if missing_keys:
                    _dbg(2, f"  missing sample: {missing_keys[:5]}")
                if shape_mismatch_keys:
                    mismatch_sample = [
                        f"{key}: ckpt{src_shape}!=model{dst_shape}"
                        for key, src_shape, dst_shape in shape_mismatch_keys[:3]
                    ]
                    _dbg(2, f"  shape mismatch sample: {mismatch_sample}")
        _dbg(1, f"BOPO.load() | path={checkpoint_path}")
