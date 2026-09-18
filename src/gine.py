import os
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, Linear #camadas do GNN

# Controlled by FJSP_DEBUG (same variable as env.py / bopo*.py)
_DBG = int(os.environ.get("FJSP_DEBUG", "0"))

def _dbg(level, *args, **kwargs):
    if _DBG >= level:
        print("[BOPO]", *args, **kwargs)


class GINEConvSafe(GINEConv):
    def forward(self, x, edge_index, edge_attr=None, size=None):
        if edge_attr is None:
            assert self.lin is not None, "GINEConvSafe requires edge_dim to be set"
            num_edges = edge_index.size(1)
            ref = x[0] if isinstance(x, tuple) else x
            edge_attr = ref.new_zeros((num_edges, self.lin.in_channels))
        return super().forward(x, edge_index, edge_attr=edge_attr, size=size)


#GIN
class GINModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers = 2, heads = 2):
        super().__init__()
        self.lin1 = Linear(-1, 8) #camada linear inicial, -1 porque significa que se infere automaticamente o tamanho de entrada, e 8 apenas pq dava.
        self.tanh = nn.Tanh() #função de ativação
        self.num_layers = num_layers

        # GAT/TransformerConv both default to concat=True, so their actual per-layer
        # width is heads*hidden_channels; this used to ignore `heads` entirely and stay
        # at hidden_channels, making GIN's actor 6-12x smaller than GAT/Transformer at
        # the "same" hidden_channels in a side-by-side comparison (measured: 103K vs
        # 616K/1.2M params for the oo representation) - not a fair comparison, and the
        # most likely reason GIN scored worst across every representation in the
        # convergence sweep.
        width = hidden_channels * heads

        #GINEConv (variante do GIN com suporte a edge features) precisa de um MLP próprio
        #por camada e do in_channels concreto (não -1) para poder projetar as 5 features
        #das arestas para a dimensão dos nós antes de as somar às mensagens.
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        in_dim = 8
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.Linear(width, width),
            )
            conv = GINEConvSafe(mlp, edge_dim=5, train_eps=True) #edge_dim=5 -> features das arestas, projetadas para in_dim dentro do GINEConv
            self.convs.append(conv)
            # GINEConv's message aggregation defaults to unnormalized SUM (unlike GAT/
            # TransformerConv's softmax-normalized attention), so a node's embedding
            # magnitude scales with how many neighbors it has. In the oo representation
            # specifically, disjunctive edges connect every currently-available
            # operation to every other one, and that count swings from 1 up to
            # num_jobs-1 over an episode as jobs finish - an extra, representation-
            # specific source of embedding-scale noise on top of GIN's usual
            # sensitivity here. LayerNorm after every conv layer renormalizes each
            # node's embedding regardless of neighbor count - the same fix the original
            # GIN paper (Xu et al., 2019) uses (BatchNorm there; LayerNorm here since
            # graphs of very different sizes get batched together for BOPO's parallel
            # rollouts, where BatchNorm's batch statistics would be inconsistent).
            self.norms.append(nn.LayerNorm(width))
            in_dim = width

    def forward(self, x, edge_index, edge_attr_dict):
        _dbg(3, f"  GIN.forward | x.shape={x.shape}")
        x = self.lin1(x) #features projetadas para a dimensão 8
        x = self.tanh(x) #função de ativação - introduz não linearidade e mantem valores entre -1 e 1
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr_dict)
            x = norm(x)
        #ativação final - após todas as camadas de conv. Embeding adequados para serem usados pelo ator.
        x = self.tanh(x)
        _dbg(3, f"  GIN.forward done | out.shape={x.shape}")
        return x
