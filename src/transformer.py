import os
import torch
import torch.nn as nn
from typing import Any
from torch_geometric.nn import TransformerConv, Linear #camadas do GNN

# Controlled by FJSP_DEBUG (same variable as env.py / bopo*.py)
_DBG = int(os.environ.get("FJSP_DEBUG", "0"))

def _dbg(level, *args, **kwargs):
    if _DBG >= level:
        print("[BOPO]", *args, **kwargs)


class TransformerConvSafe(TransformerConv):
    #tal como GINEConvSafe (gine.py): alguns tipos de aresta (ex.: operation-prec-operation)
    #não têm edge_attr definido em env.py, pelo que o to_hetero passa None. TransformerConv
    #faz assert edge_attr is not None quando edge_dim está definido, por isso preenchemos
    #com zeros em vez de deixar o assert rebentar.
    def forward(
        self, x, edge_index, edge_attr=None, return_attention_weights=None
    ) -> Any:
        if edge_attr is None:
            assert self.lin_edge is not None, "TransformerConvSafe requires edge_dim to be set"
            num_edges = edge_index.size(1)
            ref = x[0] if isinstance(x, tuple) else x
            edge_attr = ref.new_zeros((num_edges, self.lin_edge.in_channels))
        return super().forward(x, edge_index, edge_attr=edge_attr,
                                return_attention_weights=return_attention_weights)


#Transformer (graph transformer - atenção multi-cabeça no estilo query/key/value, com
#as edge features somadas diretamente aos vetores de key/value antes do score de atenção)
class TransformerModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers = 2, heads = 2):
        super().__init__()
        self.lin1 = Linear(-1, 8) #camada linear inicial, -1 porque significa que se infere automaticamente o tamanho de entrada, e 8 apenas pq dava.
        self.tanh = nn.Tanh() #função de ativação
        self.num_layers = num_layers

        #TransformerConv suporta edge_dim nativamente (tal como o GATv2Conv), por isso não
        #precisa de um MLP próprio por camada como o GINEConv. concat=True (default) para
        #manter a mesma lógica do GAT: cada camada expande a dimensão para heads*hidden_channels.
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConvSafe(-1, hidden_channels, heads=heads, edge_dim=5)
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr_dict):
        _dbg(3, f"  Transformer.forward | x.shape={x.shape}")
        x = self.lin1(x) #features projetadas para a dimensão 8
        x = self.tanh(x) #função de ativação - introduz não linearidade e mantem valores entre -1 e 1
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
        #ativação final - após todas as camadas de conv. Embeding adequados para serem usados pelo ator.
        x = self.tanh(x)
        _dbg(3, f"  Transformer.forward done | out.shape={x.shape}")
        return x
