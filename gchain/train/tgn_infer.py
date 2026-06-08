"""TGN tail inference + latency benchmarking (FuseChain deploy eval path)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from gchain.eval.attack_reconstruct import NUM_STAGE_CLASSES
from gchain.eval.latency import ms_per_1k_edges
from gchain.train.modeling import build_models, load_training_checkpoint
from gchain.train.neg_sampling import build_neg_pools, build_time_pools, inbatch_neg_dst, sample_window_neg_dst
from gchain.train.split import time_split_idx
from gchain.train.streams import Stream, load_stream_from_tgn_pt


@dataclass(frozen=True)
class TGNInferConfig:
    train_frac: float = 0.7
    batch_size: int = 512
    memory_dim: int = 64
    time_dim: int = 32
    etype_dim: int = 16
    warmup: bool = True
    neg_sampling: str = "random"
    neg_window_seconds: int = 3600
    neg_window_max_cands: int = 4096
    device: str = "cpu"


def resolve_checkpoint(run_dir: Path) -> Optional[Path]:
    for name in ("best_ckpt_joint.pt", "best_ckpt.pt", "best_ckpt_holdout.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def infer_config_from_checkpoint(ckpt_path: Path) -> TGNInferConfig:
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    return TGNInferConfig(
        train_frac=float(cfg.get("train_frac", 0.7)),
        batch_size=int(cfg.get("batch_size", 512)),
        memory_dim=int(cfg.get("memory_dim", 64)),
        time_dim=int(cfg.get("time_dim", 32)),
        etype_dim=int(cfg.get("etype_dim", 16)),
        warmup=bool(cfg.get("warmup", True)),
        neg_sampling=str(cfg.get("neg_sampling", "random")),
        neg_window_seconds=int(cfg.get("neg_window_seconds", 3600)),
        neg_window_max_cands=int(cfg.get("neg_window_max_cands", 4096)),
        device=str(cfg.get("device", "cpu")),
    )


def _resolve_device(device: str) -> "object":
    import torch

    dev = str(device or "cpu").strip().lower()
    if dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_runtime(
    st: Stream,
    ckpt_path: Path,
    cfg: TGNInferConfig,
    *,
    device_override: Optional[str] = None,
) -> Tuple["object", "object", "object", Optional["object"], "object", TGNInferConfig, int]:
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    device = _resolve_device(device_override or cfg.device)

    num_nodes = int(max(st.src.max().item(), st.dst.max().item())) + 1 if st.src.numel() else 0
    num_etypes = int(st.etype.max().item()) + 1 if st.etype.numel() else 1
    raw_msg_dim = int(st.msg.size(-1)) + int(cfg.etype_dim)
    use_stage = ckpt.get("stage_pred") is not None

    memory, link_pred, etype_emb, stage_pred = build_models(
        num_nodes=num_nodes,
        num_etypes=num_etypes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=int(cfg.memory_dim),
        time_dim=int(cfg.time_dim),
        etype_dim=int(cfg.etype_dim),
        use_stage=use_stage,
        stage_hidden_dim=int(dict(ckpt.get("config") or {}).get("stage_hidden_dim", 128)),
        num_stage_classes=NUM_STAGE_CLASSES,
        device=device,
    )
    load_training_checkpoint(
        ckpt_path,
        memory=memory,
        link_pred=link_pred,
        etype_emb=etype_emb,
        stage_pred=stage_pred,
        device=device,
    )
    memory.eval()
    link_pred.eval()
    etype_emb.eval()
    if stage_pred is not None:
        stage_pred.eval()

    assoc = torch.empty(num_nodes, dtype=torch.long, device=device).fill_(-1)
    return memory, link_pred, etype_emb, stage_pred, assoc, cfg, num_nodes


def _sample_neg(
    true_dst: "torch.Tensor",
    e: "torch.Tensor",
    pools: Dict[int, "torch.Tensor"],
    *,
    num_nodes: int,
    neg_sampling: str,
) -> "torch.Tensor":
    import torch

    if neg_sampling == "pool" and int(e.item()) in pools and pools[int(e.item())].numel() > 1:
        pool = pools[int(e.item())].to(true_dst.device)
        j = torch.randint(0, int(pool.numel()), (1,), device=true_dst.device)
        neg = pool[j].view_as(true_dst)
        if int(neg.item()) == int(true_dst.item()):
            neg = pool[(j + 1) % int(pool.numel())].view_as(true_dst)
        return neg
    neg = torch.randint(0, num_nodes, true_dst.size(), device=true_dst.device)
    if true_dst.numel() == 1 and int(neg.item()) == int(true_dst.item()):
        neg = (neg + 1) % num_nodes
    return neg


def score_tail_edges(
    st: Stream,
    *,
    ckpt_path: Path,
    cfg: Optional[TGNInferConfig] = None,
    device: Optional[str] = None,
) -> int:
    """
    Run FuseChain deploy eval: optional prefix warmup, then score tail edges.
    Returns the number of tail edges scored (for latency normalization).
    """
    import torch

    infer_cfg = cfg or infer_config_from_checkpoint(ckpt_path)
    memory, link_pred, etype_emb, stage_pred, assoc, infer_cfg, num_nodes = _build_runtime(
        st, ckpt_path, infer_cfg, device_override=device
    )
    dev = assoc.device

    src = st.src.to(dev)
    dst = st.dst.to(dev)
    t = st.t.to(dev)
    msg = st.msg.to(dev)
    etype = st.etype.to(dev)

    split_idx = time_split_idx(int(src.numel()), float(infer_cfg.train_frac))
    n_tail = int(src.numel()) - int(split_idx)
    if n_tail <= 0:
        return 0

    pools = (
        build_neg_pools(dst, etype, split_idx)
        if str(infer_cfg.neg_sampling) == "pool"
        else {}
    )
    time_pools = (
        build_time_pools(dst, t, etype, split_idx)
        if str(infer_cfg.neg_sampling) == "window"
        else {}
    )

    memory.reset_state()
    with torch.no_grad():
        if bool(infer_cfg.warmup) and split_idx > 0:
            for i in range(0, split_idx, int(infer_cfg.batch_size)):
                memory.detach()
                j = min(split_idx, i + int(infer_cfg.batch_size))
                s = src[i:j]
                d = dst[i:j]
                tt = t[i:j]
                m = msg[i:j]
                e = etype[i:j]
                eemb = etype_emb(e)
                raw_msg = torch.cat([m, eemb], dim=-1)
                memory.update_state(s, d, tt, raw_msg.detach())

        for i in range(split_idx, int(src.numel()), int(infer_cfg.batch_size)):
            memory.detach()
            j = min(int(src.numel()), i + int(infer_cfg.batch_size))
            s = src[i:j]
            d = dst[i:j]
            tt = t[i:j]
            m = msg[i:j]
            e = etype[i:j]

            if str(infer_cfg.neg_sampling) == "inbatch":
                neg_d = inbatch_neg_dst(d)
            elif str(infer_cfg.neg_sampling) == "window":
                neg_d = torch.stack(
                    [
                        sample_window_neg_dst(
                            d[k : k + 1],
                            tt[k : k + 1],
                            e[k : k + 1],
                            time_pools,
                            window_seconds=int(infer_cfg.neg_window_seconds),
                            max_cands=int(infer_cfg.neg_window_max_cands),
                        ).view(())
                        for k in range(int(d.numel()))
                    ]
                )
            else:
                neg_d = torch.stack(
                    [
                        _sample_neg(
                            d[k : k + 1],
                            e[k : k + 1],
                            pools,
                            num_nodes=num_nodes,
                            neg_sampling=str(infer_cfg.neg_sampling),
                        ).view(())
                        for k in range(int(d.numel()))
                    ]
                )

            eemb = etype_emb(e)
            raw_msg = torch.cat([m, eemb], dim=-1)

            n_id = torch.unique(torch.cat([s, d, neg_d], dim=0))
            assoc[n_id] = torch.arange(n_id.size(0), device=dev)
            z, _ = memory(n_id)

            z_s = z[assoc[s]]
            z_d = z[assoc[d]]
            pos_inp = torch.cat([z_s, z_d, raw_msg], dim=-1)
            pos_logit = link_pred(pos_inp).view(-1)
            pos_prob = torch.sigmoid(pos_logit)
            _ = (-torch.log(pos_prob.clamp_min(1e-12))).detach()

            if stage_pred is not None:
                _ = stage_pred(pos_inp)

            memory.update_state(s, d, tt, raw_msg.detach())

    return n_tail


def tail_score_latency_fn(
    st: Stream,
    *,
    ckpt_path: Path,
    cfg: Optional[TGNInferConfig] = None,
    device: Optional[str] = None,
) -> Callable[[], int]:
    """Callable that re-runs warmup + tail scoring (for ms_per_1k_edges)."""

    def _fn() -> int:
        return score_tail_edges(st, ckpt_path=ckpt_path, cfg=cfg, device=device)

    return _fn


def benchmark_tail_latency_ms_per_1k(
    st: Stream,
    *,
    ckpt_path: Path,
    cfg: Optional[TGNInferConfig] = None,
    device: Optional[str] = None,
    warmup: int = 1,
    repeats: int = 3,
) -> Tuple[float, int]:
    infer_cfg = cfg or infer_config_from_checkpoint(ckpt_path)
    n_tail = int(st.src.numel()) - time_split_idx(int(st.src.numel()), float(infer_cfg.train_frac))
    if n_tail <= 0:
        return float("nan"), 0

    def _timed() -> List[int]:
        return [score_tail_edges(st, ckpt_path=ckpt_path, cfg=infer_cfg, device=device)]

    lat = ms_per_1k_edges(_timed, n_tail_edges=n_tail, warmup=warmup, repeats=repeats)
    return lat, n_tail
