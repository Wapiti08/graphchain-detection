from __future__ import annotations

from typing import Dict, Optional, Tuple


def build_models(
    *,
    num_nodes: int,
    num_etypes: int,
    raw_msg_dim: int,
    memory_dim: int,
    time_dim: int,
    etype_dim: int,
    use_stage: bool,
    stage_hidden_dim: int,
    num_stage_classes: int,
    device: "object",
) -> Tuple["object", "object", "object", Optional["object"]]:
    import torch.nn as nn
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory

    etype_emb = nn.Embedding(int(num_etypes), int(etype_dim)).to(device)
    memory = TGNMemory(
        num_nodes=int(num_nodes),
        raw_msg_dim=int(raw_msg_dim),
        memory_dim=int(memory_dim),
        time_dim=int(time_dim),
        message_module=IdentityMessage(raw_msg_dim=int(raw_msg_dim), memory_dim=int(memory_dim), time_dim=int(time_dim)),
        aggregator_module=LastAggregator(),
    ).to(device)
    link_pred = nn.Sequential(
        nn.Linear(2 * int(memory_dim) + int(raw_msg_dim), int(memory_dim)),
        nn.ReLU(),
        nn.Linear(int(memory_dim), 1),
    ).to(device)

    stage_pred = None
    if bool(use_stage):
        stage_inp_dim = 2 * int(memory_dim) + int(raw_msg_dim)
        stage_pred = nn.Sequential(
            nn.Linear(stage_inp_dim, int(stage_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(stage_hidden_dim), int(num_stage_classes)),
        ).to(device)
    return memory, link_pred, etype_emb, stage_pred


def build_stage_labels(
    *,
    streams: Dict[str, "object"],
    repo_root: "object",
    ioc_type_to_stage_idx: "object",
    load_ioc_type_to_stage: "object",
) -> Dict[str, "object"]:
    import torch

    stage_map = load_ioc_type_to_stage(repo_root)
    out: Dict[str, "object"] = {}
    for sc in sorted(streams.keys()):
        st = streams[sc]
        it_tup = getattr(st, "ioc_type", None)
        n = int(st.src.numel())
        labels = torch.zeros(n, dtype=torch.long)
        if it_tup is not None:
            for idx_e in range(n):
                labels[idx_e] = ioc_type_to_stage_idx(str(it_tup[idx_e]), stage_map)
        out[sc] = labels
    return out

