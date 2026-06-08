"""Build static graphs from TGN streams (train prefix) for GraphSAGE / RGCN baselines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from gchain.train.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class StaticGraphData:
    """Homogeneous multi-relational graph over global node ids (TGN layout)."""

    num_nodes: int
    num_relations: int
    msg_dim: int
    train_edge_index: "torch.Tensor"  # [2, E]
    train_etype: "torch.Tensor"  # [E]
    train_msg: "torch.Tensor"  # [E, D]
    node_feat: "torch.Tensor"  # [N, F]
    conv_edge_index: "torch.Tensor"  # [2, E'] bidirectional for message passing
    conv_edge_type: "torch.Tensor"  # [E'] for RGCN (forward + reverse rel offset)


def _hetero_offsets(data: "HeteroData") -> Tuple[Dict[str, int], Dict[str, int]]:
    node_offsets: Dict[str, int] = {}
    num_nodes_by_type: Dict[str, int] = {}
    total = 0
    for nt in sorted(data.node_types):
        node_offsets[nt] = total
        n = int(data[nt].num_nodes)
        num_nodes_by_type[nt] = n
        total += n
    return node_offsets, num_nodes_by_type


def _node_attr_vector(attrs: Dict[str, Any]) -> List[float]:
    """Lightweight numeric sketch of raw node attrs (no external encoders)."""
    out: List[float] = []
    for key in (
        "is_lolbin",
        "path_sensitivity",
        "parent_depth",
        "port_sensitivity",
        "is_system_path",
        "is_temp_path",
    ):
        v = attrs.get(key)
        if isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(0.0)
    return out


def _hetero_node_features(
    data: "HeteroData",
    *,
    num_nodes: int,
    node_offsets: Dict[str, int],
    num_nodes_by_type: Dict[str, int],
) -> "torch.Tensor":
    import torch

    num_nt = max(1, len(node_offsets))
    attr_dim = 6
    feat = torch.zeros((num_nodes, num_nt + attr_dim), dtype=torch.float32)
    for nt in sorted(node_offsets):
        off = node_offsets[nt]
        n = num_nodes_by_type[nt]
        if n <= 0:
            continue
        nt_idx = sorted(node_offsets).index(nt)
        feat[off : off + n, nt_idx] = 1.0
        raw_list = getattr(data[nt], "node_attrs_raw", None) or []
        for local_i in range(min(n, len(raw_list))):
            attrs = raw_list[local_i] or {}
            feat[off + local_i, num_nt : num_nt + attr_dim] = torch.tensor(
                _node_attr_vector(attrs), dtype=torch.float32
            )
    return feat


def _degree_and_relation_hist(
    *,
    num_nodes: int,
    num_relations: int,
    src: "torch.Tensor",
    dst: "torch.Tensor",
    etype: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    out_deg = torch.zeros(num_nodes, dtype=torch.float32)
    rel_hist = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    for i in range(int(src.numel())):
        s = int(src[i].item())
        d = int(dst[i].item())
        r = int(etype[i].item())
        out_deg[s] += 1.0
        in_deg[d] += 1.0
        if 0 <= r < num_relations:
            rel_hist[s, r] += 1.0
    deg = torch.stack([torch.log1p(in_deg), torch.log1p(out_deg)], dim=-1)
    rel_norm = rel_hist / rel_hist.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.cat([deg, rel_norm], dim=-1)


def load_hetero_data(full_pt: Path) -> Optional["HeteroData"]:
    if not full_pt.is_file():
        return None
    import torch

    obj = torch.load(full_pt, map_location="cpu", weights_only=False)
    data = obj.get("data")
    return data


def build_static_graph(
    st: Stream,
    train_end: int,
    *,
    full_pt: Optional[Path] = None,
) -> StaticGraphData:
    """Materialize train-prefix graph tensors aligned with the TGN stream."""
    import torch

    train_end = max(0, min(int(train_end), int(st.src.numel())))
    if train_end == 0:
        raise ValueError("train_end must be > 0 for static GNN training")

    src = st.src[:train_end].long()
    dst = st.dst[:train_end].long()
    etype = st.etype[:train_end].long()
    msg = st.msg[:train_end].float()

    num_nodes = int(max(st.src.max().item(), st.dst.max().item())) + 1
    num_relations = int(etype.max().item()) + 1 if etype.numel() else 1
    msg_dim = int(st.msg.size(-1))

    hetero = load_hetero_data(full_pt) if full_pt is not None else None
    if hetero is not None:
        offsets, sizes = _hetero_offsets(hetero)
        hetero_feat = _hetero_node_features(
            hetero, num_nodes=num_nodes, node_offsets=offsets, num_nodes_by_type=sizes
        )
        struct_feat = _degree_and_relation_hist(
            num_nodes=num_nodes,
            num_relations=num_relations,
            src=src,
            dst=dst,
            etype=etype,
        )
        node_feat = torch.cat([hetero_feat, struct_feat], dim=-1)
    else:
        node_feat = _degree_and_relation_hist(
            num_nodes=num_nodes,
            num_relations=num_relations,
            src=src,
            dst=dst,
            etype=etype,
        )

    train_edge_index = torch.stack([src, dst], dim=0)
    # Bidirectional edges for GraphSAGE; duplicate reverse relations for RGCN.
    rev_src, rev_dst = dst, src
    rev_etype = etype + num_relations
    conv_edge_index = torch.cat([train_edge_index, torch.stack([rev_src, rev_dst], dim=0)], dim=1)
    conv_edge_type = torch.cat([etype, rev_etype], dim=0)
    num_relations_conv = num_relations * 2

    return StaticGraphData(
        num_nodes=num_nodes,
        num_relations=num_relations_conv,
        msg_dim=msg_dim,
        train_edge_index=train_edge_index,
        train_etype=etype,
        train_msg=msg,
        node_feat=node_feat,
        conv_edge_index=conv_edge_index,
        conv_edge_type=conv_edge_type,
    )
