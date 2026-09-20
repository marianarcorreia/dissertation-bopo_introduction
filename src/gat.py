import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, Linear #camadas do GNN

# Controlled by FJSP_DEBUG (same variable as env.py / bopo*.py)
_DBG = int(os.environ.get("FJSP_DEBUG", "0"))

def _dbg(level, *args, **kwargs):
    if _DBG >= level:
        print("[BOPO]", *args, **kwargs)

#GAT
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers = 2, heads = 2):
        super().__init__()
        self.lin1 = Linear(-1, 8) #camada linear inicial, -1 porque significa que se infere automaticamente o tamanho de entrada, e 8 apenas pq dava.
        self.s = torch.nn.Softmax(dim=0) #softmax para as atenções
        self.tanh = nn.Tanh() #função de ativação
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(-1, hidden_channels, add_self_loops=False, edge_dim=5, heads = heads) #edge_dim são as features das arestas, os self loops estão desativados porque estes já estão explicitos no grafo
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr_dict):
        _dbg(3, f"  GAT.forward | x.shape={x.shape}")
        x = self.lin1(x) #features projetadas para a dimensão 8
        x = self.tanh(x) #função de ativação - introduz não linearidade e mantem valores entre -1 e 1
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
        #ativação final - após todas as camadas de conv. Embeding adequados para serem usados pelo ator.
        x = self.tanh(x)
        _dbg(3, f"  GAT.forward done | out.shape={x.shape}")
        return x
