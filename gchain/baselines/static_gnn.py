"""Static GraphSAGE / RGCN baselines: train on prefix, score tail via link anomaly."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal, Optional, Set, Tuple

from gchain.baselines.static_graph import build_static_graph
from gchain.baselines.static_models import StaticLinkModel, Variant, build_model
from gchain.train.streams import Stream

VariantArg = Literal["graphsage", "rgcn", "static_gnn"]


@dataclass
class StaticGNNConfig:
    variant: Variant = "graphsage"
    epochs: int = 25
    hidden_dim: int = 128
    num_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    neg_ratio: float = 1.0
    train_batch_size: int = 8192
    infer_batch_size: int = 16384
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 0
    full_pt: Optional[Path] = None


def _resolve_variant(variant: str) -> Variant:
    v = str(variant).lower()
    if v in ("graphsage", "static_gnn"):
        return "graphsage"
    if v == "rgcn":
        return "rgcn"
    raise ValueError(f"unsupported static GNN variant: {variant!r}")


def _set_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))


def _train_edge_set(
    src,
    dst,
    etype,
) -> Set[Tuple[int, int, int]]:
    out: Set[Tuple[int, int, int]] = set()
    for i in range(int(src.numel())):
        out.add((int(src[i].item()), int(dst[i].item()), int(etype[i].item())))
    return out


def _sample_negatives(
    *,
    num_samples: int,
    num_nodes: int,
    num_relations: int,
    pos_src,
    pos_dst,
    pos_etype,
    seen: Set[Tuple[int, int, int]],
    rng: random.Random,
) -> Tuple[List[int], List[int], List[int]]:
    src_out: List[int] = []
    dst_out: List[int] = []
    et_out: List[int] = []
    tries = 0
    max_tries = max(1000, num_samples * 50)
    while len(src_out) < num_samples and tries < max_tries:
        tries += 1
        j = rng.randrange(int(pos_src.numel()))
        s = rng.randrange(num_nodes)
        d = rng.randrange(num_nodes)
        r = int(pos_etype[j].item())
        if s == d:
            continue
        key = (s, d, r)
        if key in seen:
            continue
        src_out.append(s)
        dst_out.append(d)
        et_out.append(r)
    return src_out, dst_out, et_out


def train_static_model(
    st: Stream,
    *,
    train_end: int,
    config: StaticGNNConfig,
) -> StaticLinkModel:
    import torch
    import torch.nn.functional as F

    _set_seed(config.seed)
    graph = build_static_graph(st, train_end, full_pt=config.full_pt)
    model = build_model(
        graph,
        variant=_resolve_variant(config.variant),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    device = torch.device(config.device)
    model = model.to(device)
    graph_node_feat = graph.node_feat.to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    pos_src = graph.train_edge_index[0]
    pos_dst = graph.train_edge_index[1]
    pos_etype = graph.train_etype
    pos_msg = graph.train_msg
    seen = _train_edge_set(pos_src, pos_dst, pos_etype)
    rng = random.Random(int(config.seed))
    num_rel_decoder = int(pos_etype.max().item()) + 1 if pos_etype.numel() else 1

    n_pos = int(pos_src.numel())
    batch_pos = max(1, min(int(config.train_batch_size), n_pos))

    for _ in range(int(config.epochs)):
        model.train()
        perm = torch.randperm(n_pos)
        for start in range(0, n_pos, batch_pos):
            idx = perm[start : start + batch_pos]
            b_src = pos_src[idx].to(device)
            b_dst = pos_dst[idx].to(device)
            b_etype = pos_etype[idx].to(device)
            b_msg = pos_msg[idx].to(device)

            n_neg = max(1, int(float(idx.numel()) * float(config.neg_ratio)))
            neg_s, neg_d, neg_r = _sample_negatives(
                num_samples=n_neg,
                num_nodes=graph.num_nodes,
                num_relations=num_rel_decoder,
                pos_src=pos_src,
                pos_dst=pos_dst,
                pos_etype=pos_etype,
                seen=seen,
                rng=rng,
            )
            if not neg_s:
                continue
            neg_src = torch.tensor(neg_s, dtype=torch.long, device=device)
            neg_dst = torch.tensor(neg_d, dtype=torch.long, device=device)
            neg_etype = torch.tensor(neg_r, dtype=torch.long, device=device)
            neg_msg = torch.zeros((len(neg_s), graph.msg_dim), dtype=torch.float32, device=device)

            z = model.encode(graph_node_feat)
            pos_logits = model.edge_logits(z, b_src, b_dst, b_msg, b_etype)
            neg_logits = model.edge_logits(z, neg_src, neg_dst, neg_msg, neg_etype)
            y_pos = torch.ones_like(pos_logits)
            y_neg = torch.zeros_like(neg_logits)
            loss = F.binary_cross_entropy_with_logits(pos_logits, y_pos) + F.binary_cross_entropy_with_logits(
                neg_logits, y_neg
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _infer_tail_scores(
    model: StaticLinkModel,
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
    config: StaticGNNConfig,
) -> List[float]:
    import torch

    device = torch.device(config.device)
    graph = model._graph
    z = model.encode(graph.node_feat.to(device))

    tail_start = max(int(tail_start), int(train_end))
    n_tail = int(st.src.numel()) - tail_start
    if n_tail <= 0:
        return []

    out: List[float] = []
    bs = max(1, int(config.infer_batch_size))
    with torch.no_grad():
        for start in range(tail_start, int(st.src.numel()), bs):
            end = min(int(st.src.numel()), start + bs)
            src = st.src[start:end].to(device)
            dst = st.dst[start:end].to(device)
            etype = st.etype[start:end].to(device)
            msg = st.msg[start:end].to(device)
            scores = model.anomaly_scores(z, src, dst, msg, etype)
            out.extend(float(x) for x in scores.cpu().tolist())
    return out


def _normalize_config(
    variant: str,
    config: Optional[StaticGNNConfig],
) -> StaticGNNConfig:
    cfg = config or StaticGNNConfig()
    return StaticGNNConfig(
        variant=_resolve_variant(variant if variant != "static_gnn" else cfg.variant),
        epochs=cfg.epochs,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        neg_ratio=cfg.neg_ratio,
        train_batch_size=cfg.train_batch_size,
        infer_batch_size=cfg.infer_batch_size,
        dropout=cfg.dropout,
        device=cfg.device,
        seed=cfg.seed,
        full_pt=cfg.full_pt,
    )


def train_and_infer_tail(
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
    variant: str = "graphsage",
    config: Optional[StaticGNNConfig] = None,
) -> Tuple[List[float], StaticLinkModel, StaticGNNConfig]:
    """Train once; return tail scores and the fitted model."""
    cfg = _normalize_config(variant, config)
    model = train_static_model(st, train_end=train_end, config=cfg)
    scores = _infer_tail_scores(
        model, st, train_end=train_end, tail_start=tail_start, config=cfg
    )
    return scores, model, cfg


def static_gnn_scores(
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
    variant: str = "graphsage",
    config: Optional[StaticGNNConfig] = None,
) -> List[float]:
    scores, _, _ = train_and_infer_tail(
        st,
        train_end=train_end,
        tail_start=tail_start,
        variant=variant,
        config=config,
    )
    return scores


def infer_fn_for_model(
    model: StaticLinkModel,
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
    config: StaticGNNConfig,
) -> Callable[[], List[float]]:
    def _fn() -> List[float]:
        return _infer_tail_scores(
            model, st, train_end=train_end, tail_start=tail_start, config=config
        )

    return _fn


def score_fn_for_latency(
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
    variant: str = "graphsage",
    config: Optional[StaticGNNConfig] = None,
) -> Callable[[], List[float]]:
    """Train once; benchmark inference-only on subsequent calls."""
    _, model, cfg = train_and_infer_tail(
        st,
        train_end=train_end,
        tail_start=tail_start,
        variant=variant,
        config=config,
    )
    return infer_fn_for_model(
        model, st, train_end=train_end, tail_start=tail_start, config=cfg
    )
