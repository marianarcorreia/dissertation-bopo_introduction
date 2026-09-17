import os
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, Linear #camadas do GNN

# Controlled by FJSP_DEBUG (same variable as env.py / bopo*.py)
_DBG = int(os.environ.get("FJSP_DEBUG", "0"))

def _dbg(level, *args, **kwargs):
    if _DBG >= level:
        print("[BOPO]", *args, **kwargs)


#GIN
class GINModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers = 2, heads = 2):
        super().__init__()
        self.lin1 = Linear(-1, 8) #camada linear inicial, -1 porque significa que se infere automaticamente o tamanho de entrada, e 8 apenas pq dava.
        self.tanh = nn.Tanh() #função de ativação
        self.num_layers = num_layers

        #GINEConv (variante do GIN com suporte a edge features) precisa de um MLP próprio
        #por camada e do in_channels concreto (não -1) para poder projetar as 5 features
        #das arestas para a dimensão dos nós antes de as somar às mensagens.
        self.convs = torch.nn.ModuleList()
        in_dim = 8
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            conv = GINEConv(mlp, edge_dim=5, train_eps=True) #edge_dim=5 -> features das arestas, projetadas para in_dim dentro do GINEConv
            self.convs.append(conv)
            in_dim = hidden_channels

    def forward(self, x, edge_index, edge_attr_dict):
        _dbg(3, f"  GIN.forward | x.shape={x.shape}")
        x = self.lin1(x) #features projetadas para a dimensão 8
        x = self.tanh(x) #função de ativação - introduz não linearidade e mantem valores entre -1 e 1
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
        #ativação final - após todas as camadas de conv. Embeding adequados para serem usados pelo ator.
        x = self.tanh(x)
        _dbg(3, f"  GIN.forward done | out.shape={x.shape}")
        return x
