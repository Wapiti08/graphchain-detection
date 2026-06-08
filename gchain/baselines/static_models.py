"""GraphSAGE / RGCN encoders + edge decoder for static link-prediction baselines."""
from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import torch.nn as nn

from gchain.baselines.static_graph import StaticGraphData

if TYPE_CHECKING:  # pragma: no cover
    import torch


Variant = Literal["graphsage", "rgcn"]


class _EdgeDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        msg_dim: int,
        num_etypes: int,
    ) -> None:
        super().__init__()
        et_emb = max(8, hidden_dim // 4)
        msg_emb = max(8, hidden_dim // 4)
        self.etype_emb = nn.Embedding(num_etypes, et_emb)
        self.msg_proj = nn.Linear(msg_dim, msg_emb)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + et_emb + msg_emb, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        msg: "torch.Tensor",
        etype: "torch.Tensor",
    ) -> "torch.Tensor":
        import torch

        h = torch.cat(
            [
                z[src],
                z[dst],
                self.msg_proj(msg),
                self.etype_emb(etype.clamp_min(0)),
            ],
            dim=-1,
        )
        return self.mlp(h).squeeze(-1)


class StaticLinkModel(nn.Module):
    """Shared train/infer API for homogeneous GraphSAGE and RGCN."""

    def __init__(
        self,
        graph: StaticGraphData,
        *,
        variant: Variant,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        import torch
        from torch_geometric.nn import RGCNConv, SAGEConv

        super().__init__()
        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        in_dim = int(graph.node_feat.size(-1))
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

        conv_cls = SAGEConv if variant == "graphsage" else RGCNConv
        self.convs = nn.ModuleList()
        for layer in range(int(num_layers)):
            if variant == "graphsage":
                self.convs.append(conv_cls(hidden_dim, hidden_dim))
            else:
                self.convs.append(
                    conv_cls(
                        hidden_dim,
                        hidden_dim,
                        num_relations=int(graph.num_relations),
                    )
                )

        # Decoder uses original (forward) relation ids only.
        num_decoder_etypes = int(graph.train_etype.max().item()) + 1 if graph.train_etype.numel() else 1
        self.decoder = _EdgeDecoder(hidden_dim, int(graph.msg_dim), num_decoder_etypes)
        self.register_buffer("conv_edge_index", graph.conv_edge_index.long(), persistent=False)
        self.register_buffer("conv_edge_type", graph.conv_edge_type.long(), persistent=False)
        self._graph = graph

    def encode(
        self,
        x: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

        g = self._graph
        z = self.input_proj(x if x is not None else g.node_feat)
        z = F.relu(z)
        for conv in self.convs:
            if self.variant == "graphsage":
                z = conv(z, self.conv_edge_index)
            else:
                z = conv(z, self.conv_edge_index, self.conv_edge_type)
            z = F.relu(z)
            z = self.dropout(z)
        return z

    def edge_logits(
        self,
        z: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        msg: "torch.Tensor",
        etype: "torch.Tensor",
    ) -> "torch.Tensor":
        return self.decoder(z, src, dst, msg, etype)

    def anomaly_scores(
        self,
        z: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        msg: "torch.Tensor",
        etype: "torch.Tensor",
    ) -> "torch.Tensor":
        """Higher = more anomalous (aligned with TGN -log p style)."""
        import torch

        logits = self.edge_logits(z, src, dst, msg, etype)
        prob = torch.sigmoid(logits).clamp_min(1e-12)
        return (-torch.log(prob)).detach()


def build_model(
    graph: StaticGraphData,
    *,
    variant: Variant,
    hidden_dim: int,
    num_layers: int,
) -> "StaticLinkModel":
    return StaticLinkModel(
        graph,
        variant=variant,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
    )
