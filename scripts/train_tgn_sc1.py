from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/validate a TGN on SynthChain sc1 (time-split).")
    p.add_argument("--tgn-pt", type=str, default="artifacts/graphs/synthchain_sc1.tgn.pt")
    p.add_argument("--full-pt", type=str, default="artifacts/graphs/synthchain_sc1.full.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-frac", type=float, default=0.7, help="Train on earliest fraction; validate on the rest.")
    p.add_argument("--memory-dim", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--etype-dim", type=int, default=16)
    p.add_argument("--hard-neg", action="store_true", help="Sample negatives from same etype dst pool (harder than uniform).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="artifacts/tgn_runs/sc1")
    return p.parse_args()


def _time_split(t: "torch.Tensor", train_frac: float) -> int:
    ''' split according to timestamp in order to build the pipeline from past to future

    '''
    import torch

    if t.numel() == 0:
        return 0
    k = int(math.floor(float(train_frac) * float(t.numel())))
    k = max(1, min(int(t.numel()) - 1, k))
    # stream is already sorted by time, so split by index.
    return k


def _ensure_stream(tgn_path: Path, full_path: Path) -> Dict[str, "torch.Tensor"]:
    import torch

    if tgn_path.exists():
        return torch.load(tgn_path, weights_only=True)

    if not full_path.exists():
        raise SystemExit(
            f"Missing both `{tgn_path}` and `{full_path}`. Run:\n"
            f"  python scripts/generate_graph.py --dataset synthchain --scenario sc1 --export-tgn\n"
        )

    blob = torch.load(full_path, weights_only=False)
    data = blob["data"]
    from graph import hetero_to_tgn_event_stream

    stream = hetero_to_tgn_event_stream(data, cat_hash_buckets=8, include_meta=False)
    tgn_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"src": stream.src, "dst": stream.dst, "t": stream.t, "msg": stream.msg, "etype": stream.etype}, tgn_path)
    return {"src": stream.src, "dst": stream.dst, "t": stream.t, "msg": stream.msg, "etype": stream.etype}


def _build_neg_pools(
    src: "torch.Tensor",  # [E] int64 (unused in current pooling strategy)
    dst: "torch.Tensor",  # [E] int64
    etype: "torch.Tensor",  # [E] int64
    split_idx: int,
) -> Dict[int, "torch.Tensor"]:
    """
    Build negative sampling pools from the *training* prefix [0:split_idx].

    Current strategy: pool by edge type only:
      pools[etype_id] = unique(dst) seen for that etype in training.
    """
    import torch

    pools: Dict[int, "torch.Tensor"] = {}
    for e in torch.unique(etype[:split_idx]).tolist():
        mask = (etype[:split_idx] == int(e))
        vals = torch.unique(dst[:split_idx][mask])
        pools[int(e)] = vals
    return pools


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory

    torch.manual_seed(int(args.seed))

    tgn_path = (repo_root / args.tgn_pt).resolve()
    full_path = (repo_root / args.full_pt).resolve()
    stream = _ensure_stream(tgn_path, full_path)

    src = stream["src"].long()
    dst = stream["dst"].long()
    # PyG's TGNMemory stores last_update as int64 timestamps.
    # Use integer seconds/steps to avoid dtype mismatches.
    t = stream["t"].long()
    msg = stream["msg"].float()
    etype = stream["etype"].long()

    num_nodes = int(torch.max(torch.stack([src.max(), dst.max()])).item()) + 1 if src.numel() else 0
    num_etypes = int(etype.max().item()) + 1 if etype.numel() else 1
    raw_msg_dim = int(msg.size(-1)) + int(args.etype_dim)

    split_idx = _time_split(t, float(args.train_frac))
    neg_pools = _build_neg_pools(src, dst, etype, split_idx) if args.hard_neg else {}

    etype_emb = nn.Embedding(num_etypes, int(args.etype_dim))
    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=int(args.memory_dim),
        time_dim=int(args.time_dim),
        message_module=IdentityMessage(raw_msg_dim=raw_msg_dim, memory_dim=int(args.memory_dim), time_dim=int(args.time_dim)),
        aggregator_module=LastAggregator(),
    )

    link_pred = nn.Sequential(
        nn.Linear(2 * int(args.memory_dim) + raw_msg_dim, int(args.memory_dim)),
        nn.ReLU(),
        nn.Linear(int(args.memory_dim), 1),
    )

    params = list(memory.parameters()) + list(link_pred.parameters()) + list(etype_emb.parameters())
    opt = torch.optim.Adam(params, lr=float(args.lr))

    assoc = torch.empty(num_nodes, dtype=torch.long).fill_(-1)

    def sample_neg(true_dst: "torch.Tensor", e: "torch.Tensor") -> "torch.Tensor":
        if args.hard_neg and int(e.item()) in neg_pools and neg_pools[int(e.item())].numel() > 1:
            pool = neg_pools[int(e.item())]
            j = torch.randint(0, int(pool.numel()), (1,), device=true_dst.device)
            neg = pool[j].view_as(true_dst)
            if int(neg.item()) == int(true_dst.item()):
                neg = pool[(j + 1) % int(pool.numel())].view_as(true_dst)
            return neg
        neg = torch.randint(0, num_nodes, true_dst.size(), device=true_dst.device)
        if true_dst.numel() == 1 and int(neg.item()) == int(true_dst.item()):
            neg = (neg + 1) % num_nodes
        return neg

    def step_range(lo: int, hi: int, train: bool) -> Tuple[float, float]:
        total_loss = 0.0
        correct = 0.0
        count = 0.0

        if train:
            memory.train()
            link_pred.train()
            etype_emb.train()
        else:
            memory.eval()
            link_pred.eval()
            etype_emb.eval()

        for i in range(lo, hi, int(args.batch_size)):
            # Detach any history carried in memory/message store from prior batches.
            # Otherwise autograd can span across batches and trigger:
            # "Trying to backward through the graph a second time".
            memory.detach()

            j = min(hi, i + int(args.batch_size))
            s = src[i:j]
            d = dst[i:j]
            tt = t[i:j]
            m = msg[i:j]
            e = etype[i:j]

            neg_d = torch.stack([sample_neg(d[k : k + 1], e[k : k + 1]).view(()) for k in range(int(d.numel()))])
            neg_d = neg_d.to(d.device)

            eemb = etype_emb(e)
            raw_msg = torch.cat([m, eemb], dim=-1)

            n_id = torch.unique(torch.cat([s, d, neg_d], dim=0))
            assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
            z, _ = memory(n_id)

            z_s = z[assoc[s]]
            z_d = z[assoc[d]]
            z_neg = z[assoc[neg_d]]

            pos_inp = torch.cat([z_s, z_d, raw_msg], dim=-1)
            neg_inp = torch.cat([z_s, z_neg, raw_msg], dim=-1)

            pos_logit = link_pred(pos_inp).view(-1)
            neg_logit = link_pred(neg_inp).view(-1)

            y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=0)
            logit = torch.cat([pos_logit, neg_logit], dim=0)
            loss = F.binary_cross_entropy_with_logits(logit, y)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                prob = torch.sigmoid(logit)
                pred = (prob > 0.5).float()
                correct += float((pred == y).sum().item())
                count += float(y.numel())
                total_loss += float(loss.item()) * float(y.numel())

            # Important: update memory *after* computing logits
            # Also detach raw_msg so memory updates don't keep the computation graph.
            memory.update_state(s, d, tt, raw_msg.detach())

        return (total_loss / max(1.0, count)), (correct / max(1.0, count))

    # Build run dir
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize memory on train segment once (no grad) for stability.
    with torch.no_grad():
        memory.reset_state()

    for ep in range(1, int(args.epochs) + 1):
        train_loss, train_acc = step_range(0, split_idx, train=True)
        with torch.no_grad():
            val_loss, val_acc = step_range(split_idx, int(src.numel()), train=False)
        print(f"epoch {ep:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

    # Save a lightweight checkpoint (weights only)
    torch.save(
        {
            "memory": memory.state_dict(),
            "link_pred": link_pred.state_dict(),
            "etype_emb": etype_emb.state_dict(),
            "config": vars(args),
            "split_idx": split_idx,
        },
        out_dir / "ckpt.pt",
    )
    print(f"Saved: {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()

