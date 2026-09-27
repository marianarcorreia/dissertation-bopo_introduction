"""HeGMN - Heterogeneous Graph Matching Network (Sang et al. 2025, arXiv:2503.08739).

Port of model/HGMN.py, model/RGIN.py and model/layers.py from
https://github.com/alvinsang1906/HeGMN (MIT), kept layer-for-layer so results are
comparable with the paper. Only non-functional changes: one `hidden_dim` instead of the
repo's mix of `hidden_dim`/`RGCN_hidden_dim`, the cross-attention type mask vectorized
(the original fills it with a triple Python loop), and device taken from the inputs.
One functional change, see CrossAttention: the second attention map is masked in the
orientation the paper describes.

Pipeline for a pair (G1, G2):
  1. RGIN encoder - GIN reinvented for heterogeneous graphs: per-relation (basis-
     decomposed) message weights, per-relation mean normalization, (1+eps)*x self term.
  2. Graph-level match: mean-pooled graph embeddings h1, h2.
  3. Node-level match: multi-head cross-attention between the two graphs' nodes, masked
     so only nodes of the SAME type attend to each other -> alignment (self-attention
     over the flattened similarity maps) -> CNN pooling.
  4. FCL head -> sigmoid similarity, trained with MSE against exp(-nHGED).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, inits
from torch_geometric.utils import one_hot, scatter


DEFAULT_CONFIG = {
    # model/config.yml -> HGMN (node_match on = the full two-tier model from the paper)
    "hidden_dim": 128,
    "num_heads": 8,
    "pool_first_dim": 32,
    "pool_second_dim": 64,
    "pool_third_dim": 1,
    "pool_forth_dim": 256,
    "node_match": True,
    "dropout": 0.1,
    # dataset-dependent, filled in from the graphs
    "num_features": None,
    "num_etypes": None,
    "max_nums": None,
}


class RGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases=4):
        super().__init__(aggr="add")
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_relations = num_relations
        self._num_bases = num_bases

        self._weight = nn.Parameter(torch.empty(num_bases, in_channels, in_channels))
        self._comp = nn.Parameter(torch.empty(num_relations, num_bases))
        self._root = nn.Parameter(torch.empty(in_channels, in_channels))
        self._bias = nn.Parameter(torch.empty(in_channels))
        self._eps = nn.Parameter(torch.Tensor([0]))

        self._mlp = nn.Sequential(nn.Linear(in_channels, out_channels),
                                  nn.LayerNorm(out_channels),
                                  nn.ReLU(),
                                  nn.Linear(out_channels, out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        super().reset_parameters()
        inits.glorot(self._weight)
        inits.glorot(self._comp)
        inits.glorot(self._root)
        inits.zeros(self._bias)

    def message(self, x_j, edge_type):
        weight = (self._comp @ self._weight.view(self._num_bases, -1)).view(
            self._num_relations, self._in_channels, self._in_channels)
        return torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, inputs, edge_type, index, dim_size=None):
        # mean over each relation separately, then summed over relations
        norm = one_hot(edge_type, self._num_relations, dtype=inputs.dtype)
        norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
        norm = torch.gather(norm, 1, edge_type.view(-1, 1))
        norm = 1. / norm.clamp_(1.)
        return scatter(norm * inputs, index, dim=self.node_dim, dim_size=dim_size)

    def forward(self, x, edge_index, edge_type):
        out = self.propagate(edge_index, x=x, edge_type=edge_type, size=(x.size(0), x.size(0)))
        out = out + x @ self._root + self._bias
        out = out + (1 + self._eps) * x
        return self._mlp(out)


class RGIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["hidden_dim"]
        self._conv1 = RGINConv(config["num_features"], d, config["num_etypes"])
        self._conv2 = RGINConv(d, d, config["num_etypes"])
        self._conv3 = RGINConv(d, d, config["num_etypes"])

    def forward(self, x, edge_index, edge_type):
        u = F.relu(self._conv1(x, edge_index, edge_type))
        u = F.relu(self._conv2(u, edge_index, edge_type))
        return self._conv3(u, edge_index, edge_type)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._in_dim = config["hidden_dim"]
        self._num_heads = config["num_heads"]
        self._scale = self._in_dim ** -0.5
        self._q = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._k = nn.Linear(self._in_dim, self._num_heads * self._in_dim)

    def forward(self, emb_1, emb_2, node_type_1, node_type_2):
        h, d = self._num_heads, self._in_dim
        q_1 = self._q(emb_1).view(-1, h, d).transpose(-2, -3)
        q_2 = self._q(emb_2).view(-1, h, d).transpose(-2, -3)
        k_1 = self._k(emb_1).view(-1, h, d).transpose(-2, -3).transpose(-1, -2)
        k_2 = self._k(emb_2).view(-1, h, d).transpose(-2, -3).transpose(-1, -2)
        a_1 = torch.matmul(q_1, k_2) * self._scale
        a_2 = torch.matmul(q_2, k_1).transpose(-1, -2) * self._scale

        # only nodes of the same type are matched; padded rows/cols stay 0
        n, n_1, n_2 = emb_1.size(0), node_type_1.numel(), node_type_2.numel()
        same = torch.zeros(n, n, dtype=a_1.dtype, device=a_1.device)
        same[:n_1, :n_2] = (node_type_1.view(-1, 1) == node_type_2.view(1, -1)).to(a_1.dtype)
        # a_2 is already transposed back to [graph-1 node, graph-2 node], so it takes the
        # same mask as a_1. (The original masks a_2 in the un-transposed order, i.e. with
        # same.t(), which pairs graph-1 node j with graph-2 node j's type - it contradicts
        # the paper's "only same-type nodes interact", so the paper's version is used.)
        a_1 = a_1 * same
        a_2 = a_2 * same
        return torch.cat([a_1, a_2])  # [2*heads, max_nums, max_nums]


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._in_dim = config["max_nums"] ** 2
        self._num_heads = 4
        self._scale = self._in_dim ** -0.5
        self._q = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._k = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._v = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._dropout = nn.Dropout(config["dropout"])
        self._output = nn.Linear(self._num_heads * self._in_dim, self._in_dim)
        for lin in (self._q, self._k, self._v, self._output):
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, x):
        q = self._q(x).view(-1, self._num_heads, self._in_dim).transpose(0, 1) * self._scale
        k = self._k(x).view(-1, self._num_heads, self._in_dim).transpose(0, 1).transpose(-1, -2)
        v = self._v(x).view(-1, self._num_heads, self._in_dim).transpose(0, 1)
        a = self._dropout(torch.softmax(torch.matmul(q, k), dim=2))
        y = a.matmul(v).transpose(-2, -3).contiguous().view(-1, self._num_heads * self._in_dim)
        return self._output(y)


class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._l1 = nn.Linear(config["max_nums"] ** 2, config["hidden_dim"])
        self._l2 = nn.Linear(config["hidden_dim"], config["max_nums"] ** 2)

    def forward(self, x):
        return self._l2(F.gelu(self._l1(x)))


class Alignment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._max_nodes = config["max_nums"]
        self._norm = nn.LayerNorm(self._max_nodes ** 2)
        self._dropout = nn.Dropout(config["dropout"])
        self._self_attention = MultiHeadAttention(config)
        self._ffn = FeedForwardNetwork(config)

    def forward(self, x):
        h = x.shape[0]
        x = x.view(h, -1)
        x = x + self._dropout(self._self_attention(self._norm(x)))
        x = x + self._dropout(self._ffn(self._norm(x)))
        return x.view(h, self._max_nodes, self._max_nodes)


class Pooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._cnn1 = nn.Conv2d(config["num_heads"] * 2, config["pool_first_dim"], (3, 3))
        self._pooling = nn.AdaptiveAvgPool2d((7, 7))
        self._cnn2 = nn.Conv2d(config["pool_first_dim"], config["pool_second_dim"], (3, 3))
        self._cnn3 = nn.Conv2d(config["pool_second_dim"], config["pool_third_dim"], (3, 3))
        self._cnn4 = nn.Conv2d(config["pool_third_dim"], config["pool_forth_dim"], (3, 3))
        for cnn in (self._cnn1, self._cnn2, self._cnn3, self._cnn4):
            nn.init.xavier_uniform_(cnn.weight)

    def forward(self, sim_mat):
        out = F.leaky_relu(self._cnn1(sim_mat), 0.3)
        out = self._pooling(out)
        out = F.leaky_relu(self._cnn2(out), 0.3)
        out = F.leaky_relu(self._cnn3(out), 0.3)
        return F.leaky_relu(self._cnn4(out), 0.3).view(1, -1)  # [1, pool_forth_dim]


class MatchAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._align = Alignment(config)
        self._pooling = Pooling(config)

    def forward(self, mat):
        return self._pooling(self._align(mat))


class FCL(nn.Module):
    """Four fully connected layers -> sigmoid score."""
    def __init__(self, in_dim, factor):
        super().__init__()
        self._fc1 = nn.Linear(in_dim, in_dim // factor)
        self._fc2 = nn.Linear(in_dim // factor, in_dim // (factor * 2))
        self._fc3 = nn.Linear(in_dim // (factor * 2), in_dim // (factor * 4))
        self._score = nn.Linear(in_dim // (factor * 4), 1)

    def forward(self, h):
        h = F.relu(self._fc1(h))
        h = F.relu(self._fc2(h))
        h = F.relu(self._fc3(h))
        return torch.sigmoid(self._score(h)).view(-1)


class HeGMN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._nconv = RGIN(config)
        if config["node_match"]:
            # the Pooling CNN needs a >= 3x3 map after cnn1, and FCL's input width assumes
            # pool_forth_dim == 2 * hidden_dim (graph-level h is 2 * hidden_dim)
            assert config["max_nums"] >= 3, "max_nums must be >= 3"
            assert config["pool_forth_dim"] == 2 * config["hidden_dim"]
            self._nca = CrossAttention(config)
            self._nmatch = MatchAttention(config)
            self._fcl = FCL(config["hidden_dim"] * 4, 4)
        else:
            self._fcl = FCL(config["hidden_dim"] * 2, 2)

    def _pad(self, u):
        return torch.cat([u, u.new_zeros(self.config["max_nums"] - u.size(0), u.size(1))])

    def forward(self, g1, g2):
        u_1 = self._nconv(g1.x, g1.edge_index, g1.edge_type)
        u_2 = self._nconv(g2.x, g2.edge_index, g2.edge_type)
        h = torch.cat([u_1.mean(dim=0, keepdim=True), u_2.mean(dim=0, keepdim=True)], 1)
        if not self.config["node_match"]:
            return self._fcl(h)
        simmat = self._nca(self._pad(u_1), self._pad(u_2), g1.node_type, g2.node_type)
        return self._fcl(torch.cat([h, self._nmatch(simmat)], 1))
