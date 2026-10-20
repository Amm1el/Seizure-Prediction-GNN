"""Models: spatial encoder, Deep Graph Infomax, temporal network, fusion, and baselines.

Poster steps:
    * "Deep Graph Infomax (DGI) is used for unsupervised learning of node embeddings"
        -> ``GCNEncoder`` + ``BatchedDGI``
    * "another GNN-based temporal model enhances feature extraction"
        -> ``TCN`` run over the sequence of graph embeddings produced by the GNN encoder
    * "fuse spatial and temporal features"
        -> ``FusionClassifier``
    * "ablation studies ... over single-GNN models"
        -> ``SpatialOnly`` and ``TemporalOnly``

Every classifier takes a ``SequenceBatch`` (see ``data.py``) and returns logits of shape (B,).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import DeepGraphInfomax, GCNConv, global_mean_pool

from .data import SequenceBatch

# --------------------------------------------------------------------------- spatial encoder


class GCNEncoder(nn.Module):
    """Graph convolutional encoder: node features -> node embeddings.

    Each layer is a weighted GCN convolution (Kipf & Welling, 2017) followed by PReLU,
    the activation used in the DGI paper. Edge weights are the connectivity strengths.

    Shapes: x (N, F), edge_index (2, E), edge_weight (E,) -> (N, hidden).
    ``batch`` is accepted and ignored so the encoder fits DGI's calling convention.
    """

    def __init__(self, in_dim: int, hidden: int = 32, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * num_layers
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.acts = nn.ModuleList(nn.PReLU(hidden) for _ in range(num_layers))
        self.out_dim = hidden

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x, edge_index, edge_weight))
        return x


# --------------------------------------------------------------------------- Deep Graph Infomax


def shuffle_corruption(x, edge_index, edge_weight=None, batch=None):
    """DGI corruption: shuffle node features across nodes, keep the graph structure.

    The encoder then sees "wrong" features on the right graph, and the discriminator
    learns to tell real (node, graph summary) pairs from corrupted ones.
    """
    return x[torch.randperm(x.size(0), device=x.device)], edge_index, edge_weight, batch


def mean_summary(z, x, edge_index, edge_weight=None, batch=None):
    """Graph summary s = sigmoid(mean of node embeddings), one per graph, broadcast to its nodes.

    Returns shape (N, hidden) so each node is compared with the summary of its own graph.
    """
    if batch is None:
        batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return torch.sigmoid(global_mean_pool(z, batch))[batch]


class BatchedDGI(DeepGraphInfomax):
    """``torch_geometric.nn.DeepGraphInfomax`` adapted to mini-batches of many small graphs.

    PyG's version assumes one big graph with a single summary vector. Our data is
    thousands of 16-node window graphs, so the summary is computed per graph
    (``mean_summary``) and the bilinear discriminator scores each node against its own
    graph's summary: D(z_i, s_g) = sigmoid(z_i^T W s_g). The loss is PyG's unchanged.

    Call as ``pos_z, neg_z, s = dgi(x, edge_index, edge_weight, batch=batch)`` then
    ``dgi.loss(pos_z, neg_z, s)``.
    """

    def __init__(self, encoder: GCNEncoder):
        super().__init__(encoder.out_dim, encoder, summary=mean_summary, corruption=shuffle_corruption)

    def discriminate(self, z, summary, sigmoid: bool = True):
        value = ((z @ self.weight) * summary).sum(dim=-1)
        return torch.sigmoid(value) if sigmoid else value


# --------------------------------------------------------------------------- temporal network


class CausalConvBlock(nn.Module):
    """One TCN residual block: two causal dilated Conv1d layers with ReLU and dropout.

    "Causal" means the output at time t depends only on inputs at times <= t, done by
    padding on the left only. Dilation d spaces the kernel taps d steps apart, so
    stacking blocks with d = 1, 2, 4, ... grows the receptive field exponentially.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x: (B, channels, T)
        h = self.drop(torch.relu(self.conv1(nn.functional.pad(x, (self.left_pad, 0)))))
        h = self.drop(torch.relu(self.conv2(nn.functional.pad(h, (self.left_pad, 0)))))
        return torch.relu(h + self.residual(x))


class TCN(nn.Module):
    """Temporal Convolutional Network over a sequence of feature vectors.

    Why a TCN rather than a GRU: it is parallel over time, its receptive field is
    explicit (with kernel 2 and dilations 1, 2, 4 it sees 1 + 2*(1+2+4) = 15 steps, more
    than our sequence length), and causal padding guarantees no look-ahead.

    In the fusion model the inputs are per-window graph embeddings from the GNN encoder,
    so the temporal branch is "GNN-based" as the poster describes: GNN per time step,
    TCN across time steps.

    Shapes: (B, T, in_dim) -> (B, hidden), the representation at the last time step.
    """

    def __init__(self, in_dim: int, hidden: int = 32, levels: int = 3, kernel_size: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks, ch = [], in_dim
        for i in range(levels):
            blocks.append(CausalConvBlock(ch, hidden, kernel_size, dilation=2**i, dropout=dropout))
            ch = hidden
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden

    def forward(self, seq):
        h = self.net(seq.transpose(1, 2))  # (B, hidden, T)
        return h[:, :, -1]


def _mlp_head(in_dim: int, hidden: int = 32, dropout: float = 0.2) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


# --------------------------------------------------------------------------- classifiers


class FusionClassifier(nn.Module):
    """DGI spatial features + TCN temporal features -> seizure logit.

    For each of the T windows, the (DGI-pretrained) GCN encoder embeds every node and the
    node embeddings are mean-pooled into one graph embedding g_t. Then:
        spatial  = g_T                    (structure of the most recent brain network)
        temporal = TCN(g_1, ..., g_T)     (how the network evolved)
        logit    = MLP([spatial ; temporal])

    Args:
        encoder: a ``GCNEncoder``, usually the one trained inside ``BatchedDGI``.
        freeze_encoder: if True, the encoder keeps its self-supervised weights and only
            the TCN and head are trained; if False it is fine-tuned with the labels.
    """

    def __init__(self, encoder: GCNEncoder, tcn_hidden: int = 32, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.tcn = TCN(encoder.out_dim, tcn_hidden)
        self.head = _mlp_head(encoder.out_dim + tcn_hidden)

    def graph_embeddings(self, batch: SequenceBatch) -> torch.Tensor:
        """Pooled encoder output per window. Returns shape (B, T, hidden)."""
        steps = []
        for g in batch.graphs:
            z = self.encoder(g.x, g.edge_index, g.edge_weight)
            steps.append(global_mean_pool(z, g.batch))
        return torch.stack(steps, dim=1)

    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                emb = self.graph_embeddings(batch)
        else:
            emb = self.graph_embeddings(batch)
        spatial = emb[:, -1]
        temporal = self.tcn(emb)
        return self.head(torch.cat([spatial, temporal], dim=-1)).squeeze(-1)


class SpatialOnly(nn.Module):
    """Ablation baseline: a single GNN on the last window only (no time, no DGI).

    GCN encoder -> mean pool -> MLP, trained end to end from labels.
    """

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden)
        self.head = _mlp_head(hidden)

    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        g = batch.graphs[-1]
        z = self.encoder(g.x, g.edge_index, g.edge_weight)
        return self.head(global_mean_pool(z, g.batch)).squeeze(-1)


class TemporalOnly(nn.Module):
    """Ablation baseline: a TCN over handcrafted per-window features, no graph at all.

    Input is ``batch.feats`` of shape (B, T, C * F): every channel's node features for
    each window, concatenated. No edges, so no connectivity information.
    """

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.tcn = TCN(in_dim, hidden)
        self.head = _mlp_head(hidden)

    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        return self.head(self.tcn(batch.feats)).squeeze(-1)
