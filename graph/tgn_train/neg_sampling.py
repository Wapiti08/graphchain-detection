from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def build_neg_pools(
    dst: "torch.Tensor",  # [E] int64
    etype: "torch.Tensor",  # [E] int64
    split_idx: int,
) -> Dict[int, "torch.Tensor"]:
    import torch

    pools: Dict[int, "torch.Tensor"] = {}
    for e in torch.unique(etype[:split_idx]).tolist():
        ei = int(e)
        mask = (etype[:split_idx] == ei)
        pools[ei] = torch.unique(dst[:split_idx][mask])
    return pools


def inbatch_neg_dst(dst: "torch.Tensor") -> "torch.Tensor":
    """Return a per-edge negative destination by permuting within batch."""
    import torch

    n = int(dst.numel())
    if n <= 1:
        return dst.clone()
    perm = torch.randperm(n, device=dst.device)
    neg = dst[perm]
    eq = neg.eq(dst)
    if bool(eq.any().item()):
        neg2 = torch.roll(neg, shifts=1, dims=0)
        neg = torch.where(eq, neg2, neg)
    return neg


def build_time_pools(
    dst: "torch.Tensor",  # [E]
    t: "torch.Tensor",  # [E]
    etype: "torch.Tensor",  # [E]
    split_idx: int,
) -> Dict[int, Tuple["torch.Tensor", "torch.Tensor"]]:
    """Per-etype (t_sorted, dst_sorted) pools built from the train prefix."""
    import torch

    pools: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]] = {}
    if int(split_idx) <= 0:
        return pools
    for e in torch.unique(etype[:split_idx]).tolist():
        ei = int(e)
        mask = (etype[:split_idx] == ei)
        tt = t[:split_idx][mask]
        dd = dst[:split_idx][mask]
        if int(tt.numel()) == 0:
            continue
        order = torch.argsort(tt)
        pools[ei] = (tt[order], dd[order])
    return pools


def sample_window_neg_dst(
    true_dst: "torch.Tensor",  # scalar
    true_t: "torch.Tensor",  # scalar
    e: "torch.Tensor",  # scalar
    time_pools: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]],
    *,
    window_seconds: int,
    max_cands: int,
) -> "torch.Tensor":
    """Sample a hard negative dst from same etype within a time window."""
    import torch

    ei = int(e.item())
    if ei not in time_pools:
        return true_dst.clone()
    pool_t, pool_d = time_pools[ei]
    if int(pool_t.numel()) <= 1:
        return true_dst.clone()

    w = int(max(0, window_seconds))
    center = int(true_t.item())
    lo_t = center - w
    hi_t = center + w
    lo = int(torch.searchsorted(pool_t, torch.tensor(lo_t, device=pool_t.device), right=False).item())
    hi = int(torch.searchsorted(pool_t, torch.tensor(hi_t, device=pool_t.device), right=True).item())
    if hi - lo <= 1:
        return true_dst.clone()
    cand = pool_d[lo:hi]
    if int(max_cands) > 0 and int(cand.numel()) > int(max_cands):
        j0 = torch.randint(0, int(cand.numel() - int(max_cands) + 1), (1,), device=cand.device).item()
        cand = cand[int(j0) : int(j0) + int(max_cands)]

    j = torch.randint(0, int(cand.numel()), (1,), device=cand.device)
    neg = cand[j].view_as(true_dst)
    if int(neg.item()) == int(true_dst.item()):
        neg = cand[(j + 1) % int(cand.numel())].view_as(true_dst)
    return neg

