from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

EVAL_TAIL_CSV_FIELDS = [
    "scenario",
    "t",
    "etype",
    "src",
    "dst",
    "score",
    "is_ioc",
    "source_file",
    "row_idx",
    "ioc_type",
    "pred_stage",
    "pred_stage_prob",
]


@dataclass(frozen=True)
class Stream:
    src: "torch.Tensor"  # [E] int64
    dst: "torch.Tensor"  # [E] int64
    t: "torch.Tensor"  # [E] int64 (for TGNMemory last_update)
    msg: "torch.Tensor"  # [E, D] float32
    etype: "torch.Tensor"  # [E] int64
    y_ioc: Optional["torch.Tensor"] = None  # [E] int64 (0/1)
    y_ioc_line: Optional["torch.Tensor"] = None  # [E] int64 (0/1)
    row_idx: Optional["torch.Tensor"] = None  # [E] int64 (-1 if unknown)
    source_file: Optional[Tuple[str, ...]] = None
    ioc_type: Optional[Tuple[str, ...]] = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/validate a TGN on SynthChain scenarios (time-split; optional holdout scenario)."
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="sc1,sc2,sc3,sc4,sc5,sc6,sc7",
        help="Comma-separated scenarios to use as the scenario universe.",
    )
    p.add_argument(
        "--holdout",
        type=str,
        default="",
        help="If set (e.g. sc3), train on all other scenarios and test on holdout.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Within each scenario, train on earliest fraction; validate on the rest.",
    )
    p.add_argument("--memory-dim", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--etype-dim", type=int, default=16)
    p.add_argument(
        "--neg-sampling",
        type=str,
        default="random",
        choices=["random", "pool", "inbatch", "window"],
        help=(
            "Negative sampling strategy for self-supervised link prediction. "
            "random: dst ~ Uniform[0,num_nodes); "
            "pool: sample dst from per-etype pool built on train prefix (similar to --hard-neg); "
            "inbatch: use other dst within the same batch (harder); "
            "window: sample dst from same etype within a time window in the train prefix (hard, distribution-aware)."
        ),
    )
    p.add_argument(
        "--neg-window-seconds",
        type=int,
        default=3600,
        help="Time window (seconds) for --neg-sampling window (uses train prefix pools).",
    )
    p.add_argument(
        "--neg-window-max-cands",
        type=int,
        default=4096,
        help="Cap the number of candidates considered per edge for --neg-sampling window (0 = no cap).",
    )
    # Backward-compatible alias (older scripts may pass --hard-neg).
    p.add_argument(
        "--hard-neg",
        action="store_true",
        help="DEPRECATED: use --neg-sampling pool instead.",
    )
    p.add_argument(
        "--train-only-benign",
        action="store_true",
        help=(
            "During training (prefix only), exclude IOC-labeled edges from loss and memory updates "
            "when y_ioc is available. This makes the self-supervised objective focus on normal behavior."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graphs-dir", type=str, default="artifacts/graphs")
    p.add_argument("--out", type=str, default="artifacts/tgn_runs/synthchain_multi")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Force device. Use cpu if CUDA driver is old.",
    )
    p.add_argument(
        "--auto-generate",
        action="store_true",
        help="If a scenario graph/stream is missing, try generating it via scripts/generate_graph.py.",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="For evaluation, warm up memory on the scenario prefix (no-grad) before scoring the tail.",
    )
    p.add_argument(
        "--eval-ioc",
        action="store_true",
        help="On evaluation tails, compute AUROC/AUPRC using msg[is_ioc] as label.",
    )
    p.add_argument(
        "--save-scores",
        action="store_true",
        help="Save per-event scores for evaluation tails to a CSV under --out.",
    )
    p.add_argument(
        "--save-scores-each-epoch",
        action="store_true",
        help="Save per-event tail scores for every epoch (eval_tail_scores_epochXXX.csv).",
    )
    p.add_argument(
        "--select-metric",
        type=str,
        default="auprc",
        help=(
            "Metric for best checkpoint / early stopping (requires --eval-ioc): "
            "auprc, auroc, p_at (uses --select-p-at-k), or p_at_<K> e.g. p_at_100."
        ),
    )
    p.add_argument(
        "--select-p-at-k",
        type=int,
        default=100,
        help="K for --select-metric p_at (ignored for p_at_<K> or auprc/auroc).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if --select-metric does not improve for this many epochs (0 = disabled). Requires --eval-ioc.",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement on --select-metric to reset early-stopping patience.",
    )
    p.add_argument(
        "--topk",
        type=str,
        default="10,50,100,500",
        help="Comma-separated K values for top-K IOC hit reporting on eval tail scores.",
    )
    p.add_argument(
        "--eval-alert-window",
        type=int,
        default=3600,
        help="Same as aggregate_alerts --window when reporting alert-rate / precision-in-flagged.",
    )
    p.add_argument("--eval-alert-quantile", type=float, default=0.99)
    p.add_argument("--eval-alert-min-events", type=int, default=3)
    p.add_argument("--eval-alert-topk-events", type=int, default=0, help="0 = use quantile.")
    p.add_argument(
        "--no-eval-alert-dedupe",
        action="store_true",
        help="Disable dedupe for alert metrics (matches aggregate_alerts --no-dedupe).",
    )
    p.add_argument(
        "--lambda-ioc-rank",
        type=float,
        default=0.0,
        help="If >0, add margin ranking loss so IOC edges get higher anomaly score than non-IOC in same batch (train prefix only).",
    )
    p.add_argument(
        "--ioc-rank-margin",
        type=float,
        default=0.5,
        help="Margin for IOC ranking loss (anomaly score = -log sigmoid(pos_logit)).",
    )
    p.add_argument(
        "--lambda-stage",
        type=float,
        default=0.0,
        help="If >0, add stage classification CE loss on IOC edges with known stage labels (train prefix).",
    )
    p.add_argument(
        "--stage-hidden-dim",
        type=int,
        default=64,
        help="Hidden dim of the stage classifier MLP head.",
    )
    p.add_argument(
        "--aux-supervision",
        type=str,
        default="train_only",
        choices=["train_only", "off"],
        help=(
            "IOC rank / stage CE auxiliary losses: train_only = only on --train scenarios "
            "(holdout scenario never gets aux gradients; per-scenario run uses that scenario). "
            "off = pure self-supervised link prediction."
        ),
    )
    args = p.parse_args()
    if bool(getattr(args, "hard_neg", False)):
        args.neg_sampling = "pool"
    return args


def _parse_scenarios(s: str) -> List[str]:
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok)
    return out


def _time_split_idx(num_events: int, train_frac: float) -> int:
    if num_events <= 1:
        return 0
    k = int(math.floor(float(train_frac) * float(num_events)))
    return max(1, min(num_events - 1, k))


def _load_stream_from_tgn_pt(path: Path) -> Stream:
    import torch

    blob = torch.load(path, weights_only=True)
    sf = blob.get("source_file")
    it = blob.get("ioc_type")
    return Stream(
        src=blob["src"].long(),
        dst=blob["dst"].long(),
        t=blob["t"].long(),
        msg=blob["msg"].float(),
        etype=blob["etype"].long(),
        y_ioc=(blob.get("y_ioc").long() if blob.get("y_ioc") is not None else None),
        y_ioc_line=(blob.get("y_ioc_line").long() if blob.get("y_ioc_line") is not None else None),
        row_idx=(blob.get("row_idx").long() if blob.get("row_idx") is not None else None),
        source_file=(tuple(str(x) for x in sf) if sf is not None else None),
        ioc_type=(tuple(str(x) for x in it) if it is not None else None),
    )


def _num_nodes_in_stream(st: Stream) -> int:
    import torch

    if st.src.numel() == 0:
        return 0
    return int(torch.max(torch.stack([st.src.max(), st.dst.max()])).item()) + 1


def _offset_stream_nodes(st: Stream, base: int) -> Stream:
    if base == 0:
        return st
    return Stream(
        src=st.src + int(base),
        dst=st.dst + int(base),
        t=st.t,
        msg=st.msg,
        etype=st.etype,
        y_ioc=st.y_ioc,
        y_ioc_line=st.y_ioc_line,
        row_idx=st.row_idx,
        source_file=st.source_file,
        ioc_type=st.ioc_type,
    )


def _ensure_scenario_stream(
    *,
    repo_root: Path,
    graphs_dir: Path,
    scenario: str,
    auto_generate: bool,
) -> Tuple[Path, Stream]:
    tgn_path = graphs_dir / f"synthchain_{scenario}.tgn.pt"
    if tgn_path.exists():
        return tgn_path, _load_stream_from_tgn_pt(tgn_path)

    if not auto_generate:
        raise SystemExit(
            f"Missing `{tgn_path}`. Generate it via:\n"
            f"  python scripts/generate_graph.py --dataset synthchain --scenario {scenario} --export-tgn\n"
        )

    # Try to generate via the current python executable.
    env = os.environ.copy()
    # Many servers have broken CUDA driver; this prevents torch from probing GPUs.
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "generate_graph.py"),
        "--dataset",
        "synthchain",
        "--scenario",
        scenario,
        "--export-tgn",
        "--out",
        str(graphs_dir.relative_to(repo_root)),
    ]
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)

    if not tgn_path.exists():
        raise SystemExit(f"Auto-generation finished but `{tgn_path}` was not created.")
    return tgn_path, _load_stream_from_tgn_pt(tgn_path)


def _regenerate_scenario_stream(*, repo_root: Path, graphs_dir: Path, scenario: str) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "generate_graph.py"),
        "--dataset",
        "synthchain",
        "--scenario",
        scenario,
        "--export-tgn",
        "--out",
        str(graphs_dir.relative_to(repo_root)),
    ]
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def _build_neg_pools(
    dst: "torch.Tensor",  # [E] int64
    etype: "torch.Tensor",  # [E] int64
    split_idx: int,
) -> Dict[int, "torch.Tensor"]:
    import torch

    pools: Dict[int, "torch.Tensor"] = {}
    for e in torch.unique(etype[:split_idx]).tolist():
        e = int(e)
        mask = (etype[:split_idx] == e)
        pools[e] = torch.unique(dst[:split_idx][mask])
    return pools


def _build_time_pools(
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


def _sample_window_neg_dst(
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
    # pool_t is sorted.
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


def _inbatch_neg_dst(dst: "torch.Tensor") -> "torch.Tensor":
    """Return a per-edge negative destination by permuting within batch."""
    import torch

    n = int(dst.numel())
    if n <= 1:
        return dst.clone()
    perm = torch.randperm(n, device=dst.device)
    neg = dst[perm]
    # Avoid trivial self-match; if equal, shift by 1.
    eq = neg.eq(dst)
    if bool(eq.any().item()):
        neg2 = torch.roll(neg, shifts=1, dims=0)
        neg = torch.where(eq, neg2, neg)
    return neg


def _roc_auc(y_true: List[int], y_score: List[float]) -> float:
    # Returns AUROC in [0,1]. If undefined (only one class), returns NaN.
    if not y_true:
        return float("nan")
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(len(y_true)), key=lambda i: y_score[i], reverse=True)
    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    auc = 0.0
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # trapezoid area
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return float(auc)


def _pr_auc(y_true: List[int], y_score: List[float]) -> float:
    # Returns area under precision-recall curve. If undefined, returns NaN.
    if not y_true:
        return float("nan")
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return float("nan")

    order = sorted(range(len(y_true)), key=lambda i: y_score[i], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(1, (tp + fp))
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def _parse_topk(s: str) -> List[int]:
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
        except Exception:
            continue
        if k > 0:
            out.append(k)
    return sorted(set(out))


def _resolve_p_at_k(select_metric: str, select_p_at_k: int) -> Optional[int]:
    """Return K when selection uses tail p@K; None for auprc/auroc."""
    sm = str(select_metric).strip().lower()
    if sm == "p_at":
        return max(1, int(select_p_at_k))
    if sm.startswith("p_at_"):
        try:
            return max(1, int(sm[5:]))
        except ValueError:
            return None
    return None


def _selection_metric_label(select_metric: str, select_p_at_k: int) -> str:
    pk = _resolve_p_at_k(select_metric, select_p_at_k)
    if pk is not None:
        return f"p_at_{pk}"
    return str(select_metric)


def _selection_score(
    select_metric: str,
    select_p_at_k: int,
    *,
    cur_auroc: float,
    cur_auprc: float,
    tail_eval: Dict[str, float],
) -> float:
    pk = _resolve_p_at_k(select_metric, select_p_at_k)
    if pk is not None:
        return float(tail_eval.get(f"p_at_{pk}", float("nan")))
    sm = str(select_metric).strip().lower()
    if sm == "auprc":
        return float(cur_auprc)
    if sm == "auroc":
        return float(cur_auroc)
    return float("nan")


def _topk_ioc_hits(eval_rows: List[Dict[str, object]], ks: Sequence[int]) -> Dict[int, int]:
    rows = sorted(eval_rows, key=lambda r: float(r["score"]), reverse=True)
    out: Dict[int, int] = {}
    for k in ks:
        kk = min(k, len(rows))
        out[k] = int(sum(int(rows[i]["is_ioc"]) for i in range(kk)))
    return out


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # If user forces CPU, prevent torch from probing CUDA.
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from graph.alert_eval import tail_alert_metrics
    from graph.attack_reconstruct import (
        NUM_STAGE_CLASSES,
        IDX_TO_STAGE,
        ioc_type_to_stage_idx,
        load_ioc_type_to_stage,
    )
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory

    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    graphs_dir = (repo_root / args.graphs_dir).resolve()
    graphs_dir.mkdir(parents=True, exist_ok=True)

    scenario_universe = _parse_scenarios(args.scenarios)
    if not scenario_universe:
        raise SystemExit("--scenarios is empty")

    if int(args.early_stop_patience) > 0 and not bool(args.eval_ioc):
        print("warning: --early-stop-patience is ignored without --eval-ioc.", flush=True)

    train_scenarios_set: set[str]
    if args.holdout:
        holdout = args.holdout.strip()
        train_scenarios = [s for s in scenario_universe if s != holdout]
        test_scenarios = [holdout]
    else:
        # Joint or single-scenario: train/val lists match --scenarios.
        train_scenarios = list(scenario_universe)
        test_scenarios = list(scenario_universe)
    train_scenarios_set = set(train_scenarios)

    if args.holdout:
        eval_protocol = "loso_holdout"
    elif len(test_scenarios) == 1 and set(train_scenarios) == {test_scenarios[0]}:
        eval_protocol = "per_scenario"
    else:
        eval_protocol = "joint_multi"

    use_aux_supervision = str(args.aux_supervision) != "off"

    streams: Dict[str, Stream] = {}
    for sc in sorted(set(train_scenarios + test_scenarios)):
        _, st = _ensure_scenario_stream(
            repo_root=repo_root,
            graphs_dir=graphs_dir,
            scenario=sc,
            auto_generate=bool(args.auto_generate),
        )
        streams[sc] = st

    # Ensure all scenarios have identical msg dimensions.
    msg_dims = {sc: int(streams[sc].msg.size(-1)) for sc in streams}
    uniq_dims = sorted(set(msg_dims.values()))
    if len(uniq_dims) > 1:
        if not bool(args.auto_generate):
            raise SystemExit(
                "Inconsistent msg dimensions across scenarios (likely stale *.tgn.pt files). "
                f"Found dims: {msg_dims}. Re-export with --export-tgn for all scenarios, "
                "or rerun with --auto-generate."
            )
        # Regenerate scenarios to match the *current* feature projection.
        # After feature changes, newly exported streams often have a smaller dim;
        # prefer the minimum dim as the target (assumes newer projection removes/merges fields).
        target_dim = min(uniq_dims)
        for sc, d in msg_dims.items():
            if d != target_dim:
                _regenerate_scenario_stream(repo_root=repo_root, graphs_dir=graphs_dir, scenario=sc)
                streams[sc] = _load_stream_from_tgn_pt(graphs_dir / f"synthchain_{sc}.tgn.pt")

        msg_dims2 = {sc: int(streams[sc].msg.size(-1)) for sc in streams}
        if len(set(msg_dims2.values())) > 1:
            raise SystemExit(f"Still inconsistent msg dimensions after regeneration: {msg_dims2}")

    # IMPORTANT: each scenario stream uses its own 0..N-1 node id space.
    # Offset per scenario to make node ids disjoint across scenarios.
    scenario_base: Dict[str, int] = {}
    base = 0
    for sc in sorted(streams.keys()):
        scenario_base[sc] = base
        base += _num_nodes_in_stream(streams[sc])
    for sc in list(streams.keys()):
        streams[sc] = _offset_stream_nodes(streams[sc], scenario_base[sc])

    # Determine global sizes across all included (offset) streams.
    all_src = torch.cat([streams[sc].src for sc in streams], dim=0)
    all_dst = torch.cat([streams[sc].dst for sc in streams], dim=0)
    all_etype = torch.cat([streams[sc].etype for sc in streams], dim=0)
    num_nodes = int(torch.max(torch.stack([all_src.max(), all_dst.max()])).item()) + 1 if all_src.numel() else 0
    num_etypes = int(all_etype.max().item()) + 1 if all_etype.numel() else 1
    raw_msg_dim = int(next(iter(streams.values())).msg.size(-1)) + int(args.etype_dim)

    etype_emb = nn.Embedding(num_etypes, int(args.etype_dim)).to(device)
    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=int(args.memory_dim),
        time_dim=int(args.time_dim),
        message_module=IdentityMessage(raw_msg_dim=raw_msg_dim, memory_dim=int(args.memory_dim), time_dim=int(args.time_dim)),
        aggregator_module=LastAggregator(),
    ).to(device)
    link_pred = nn.Sequential(
        nn.Linear(2 * int(args.memory_dim) + raw_msg_dim, int(args.memory_dim)),
        nn.ReLU(),
        nn.Linear(int(args.memory_dim), 1),
    ).to(device)

    stage_pred: Optional[nn.Module] = None
    use_stage = float(args.lambda_stage) > 0.0
    if use_stage:
        stage_inp_dim = 2 * int(args.memory_dim) + raw_msg_dim
        stage_pred = nn.Sequential(
            nn.Linear(stage_inp_dim, int(args.stage_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(args.stage_hidden_dim), NUM_STAGE_CLASSES),
        ).to(device)

    # Build per-edge stage label tensors (only meaningful when stage head is used).
    _ioc_type_to_stage_map = load_ioc_type_to_stage(repo_root)
    y_stage_per_sc: Dict[str, "torch.Tensor"] = {}
    for sc in sorted(streams.keys()):
        st = streams[sc]
        it_tup = getattr(st, "ioc_type", None)
        n = int(st.src.numel())
        labels = torch.zeros(n, dtype=torch.long)
        if it_tup is not None:
            for idx_e in range(n):
                labels[idx_e] = ioc_type_to_stage_idx(str(it_tup[idx_e]), _ioc_type_to_stage_map)
        y_stage_per_sc[sc] = labels

    all_params = list(memory.parameters()) + list(link_pred.parameters()) + list(etype_emb.parameters())
    if stage_pred is not None:
        all_params += list(stage_pred.parameters())
    opt = torch.optim.Adam(all_params, lr=float(args.lr))
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device).fill_(-1)

    def sample_neg(true_dst: "torch.Tensor", e: "torch.Tensor", pools: Dict[int, "torch.Tensor"]) -> "torch.Tensor":
        if str(args.neg_sampling) == "pool" and int(e.item()) in pools and pools[int(e.item())].numel() > 1:
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

    def run_one_scenario(
        sc: str,
        *,
        train: bool,
        prefix_only: bool,
        collect_eval: bool = False,
    ) -> Tuple[float, float, Optional[List[Dict[str, object]]], Optional[Tuple[float, float]]]:
        st = streams[sc]
        src = st.src.to(device)
        dst = st.dst.to(device)
        t = st.t.to(device)
        msg = st.msg.to(device)
        etype = st.etype.to(device)
        y_ioc = getattr(st, "y_ioc", None)
        if y_ioc is not None:
            y_ioc = y_ioc.to(device)
        row_idx_cpu = getattr(st, "row_idx", None)
        source_file = getattr(st, "source_file", None)
        ioc_type = getattr(st, "ioc_type", None)

        split_idx = _time_split_idx(int(src.numel()), float(args.train_frac))
        lo, hi = (0, split_idx) if prefix_only else (split_idx, int(src.numel()))

        pools = _build_neg_pools(dst, etype, split_idx) if str(args.neg_sampling) == "pool" else {}
        time_pools = (
            _build_time_pools(dst, t, etype, split_idx)
            if str(args.neg_sampling) == "window"
            else {}
        )

        total_loss = 0.0
        correct = 0.0
        count = 0.0
        rows: Optional[List[Dict[str, object]]] = [] if collect_eval else None
        y_true: List[int] = []
        y_score: List[float] = []

        y_stage_sc = y_stage_per_sc.get(sc)

        if train:
            memory.train()
            link_pred.train()
            etype_emb.train()
            if stage_pred is not None:
                stage_pred.train()
        else:
            memory.eval()
            link_pred.eval()
            etype_emb.eval()
            if stage_pred is not None:
                stage_pred.eval()

        # Prevent leakage across scenarios.
        memory.reset_state()

        # Evaluation warmup: use prefix to update memory (no loss computed).
        if (not train) and (not prefix_only) and bool(args.warmup) and split_idx > 0:
            with torch.no_grad():
                w_lo, w_hi = 0, split_idx
                for i in range(w_lo, w_hi, int(args.batch_size)):
                    memory.detach()
                    j = min(w_hi, i + int(args.batch_size))
                    s = src[i:j]
                    d = dst[i:j]
                    tt = t[i:j]
                    m = msg[i:j]
                    e = etype[i:j]
                    eemb = etype_emb(e)
                    raw_msg = torch.cat([m, eemb], dim=-1)
                    memory.update_state(s, d, tt, raw_msg.detach())

        for i in range(lo, hi, int(args.batch_size)):
            memory.detach()

            j = min(hi, i + int(args.batch_size))
            s = src[i:j]
            d = dst[i:j]
            tt = t[i:j]
            m = msg[i:j]
            e = etype[i:j]

            if str(args.neg_sampling) == "inbatch":
                neg_d = _inbatch_neg_dst(d)
            elif str(args.neg_sampling) == "window":
                neg_d = torch.stack(
                    [
                        _sample_window_neg_dst(
                            d[k : k + 1],
                            tt[k : k + 1],
                            e[k : k + 1],
                            time_pools,
                            window_seconds=int(args.neg_window_seconds),
                            max_cands=int(args.neg_window_max_cands),
                        ).view(())
                        for k in range(int(d.numel()))
                    ]
                )
            else:
                neg_d = torch.stack(
                    [sample_neg(d[k : k + 1], e[k : k + 1], pools).view(()) for k in range(int(d.numel()))]
                )
            eemb = etype_emb(e)
            raw_msg = torch.cat([m, eemb], dim=-1)

            n_id = torch.unique(torch.cat([s, d, neg_d], dim=0))
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, _ = memory(n_id)

            z_s = z[assoc[s]]
            z_d = z[assoc[d]]
            z_neg = z[assoc[neg_d]]

            pos_inp = torch.cat([z_s, z_d, raw_msg], dim=-1)
            neg_inp = torch.cat([z_s, z_neg, raw_msg], dim=-1)

            pos_logit = link_pred(pos_inp).view(-1)
            neg_logit = link_pred(neg_inp).view(-1)

            benign_mask: Optional["torch.Tensor"] = None
            if (
                bool(args.train_only_benign)
                and train
                and prefix_only
                and (y_ioc is not None)
            ):
                benign_mask = (y_ioc[i:j] == 0)

            if benign_mask is not None:
                if int(benign_mask.sum().item()) == 0:
                    # Still update memory below (skipping IOC edges too) and continue.
                    loss = None
                else:
                    pos_logit_eff = pos_logit[benign_mask]
                    neg_logit_eff = neg_logit[benign_mask]
                    y = torch.cat([torch.ones_like(pos_logit_eff), torch.zeros_like(neg_logit_eff)], dim=0)
                    logit = torch.cat([pos_logit_eff, neg_logit_eff], dim=0)
                    loss = F.binary_cross_entropy_with_logits(logit, y)
            else:
                y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=0)
                logit = torch.cat([pos_logit, neg_logit], dim=0)
                loss = F.binary_cross_entropy_with_logits(logit, y)

            aux_active = (
                use_aux_supervision
                and train
                and (sc in train_scenarios_set)
            )
            if loss is not None and aux_active and float(args.lambda_ioc_rank) > 0.0 and y_ioc is not None:
                yb = y_ioc[i:j].float()
                pos_prob = torch.sigmoid(pos_logit).view(-1)
                score_anom = -torch.log(pos_prob.clamp_min(1e-12))
                mask_i = yb > 0.5
                mask_n = yb < 0.5
                if int(mask_i.sum().item()) > 0 and int(mask_n.sum().item()) > 0:
                    si = score_anom[mask_i]
                    sn = score_anom[mask_n]
                    idx = torch.randint(0, int(sn.numel()), (int(si.numel()),), device=device)
                    pair_sn = sn[idx]
                    margin = float(args.ioc_rank_margin)
                    loss_ioc = F.relu(margin - (si - pair_sn)).mean()
                    loss = loss + float(args.lambda_ioc_rank) * loss_ioc

            if loss is not None and aux_active and use_stage and stage_pred is not None and y_stage_sc is not None:
                stage_labels_batch = y_stage_sc[i:j].to(device)
                has_label = stage_labels_batch > 0  # 0 = "none" / unlabeled
                if int(has_label.sum().item()) > 0:
                    stage_logits = stage_pred(pos_inp)
                    loss_stage = F.cross_entropy(
                        stage_logits[has_label], stage_labels_batch[has_label]
                    )
                    loss = loss + float(args.lambda_stage) * loss_stage

            if train and loss is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                if benign_mask is not None:
                    # For training-only-benign, track metrics on the same effective edges.
                    if int(benign_mask.sum().item()) > 0:
                        prob = torch.sigmoid(logit)
                        pred = (prob > 0.5).float()
                        correct += float((pred == y).sum().item())
                        count += float(y.numel())
                        assert loss is not None
                        total_loss += float(loss.item()) * float(y.numel())
                else:
                    prob = torch.sigmoid(logit)
                    pred = (prob > 0.5).float()
                    correct += float((pred == y).sum().item())
                    count += float(y.numel())
                    total_loss += float(loss.item()) * float(y.numel())

                if collect_eval:
                    # score only the positive (real) edges
                    pos_prob = torch.sigmoid(pos_logit)
                    pos_score = (-torch.log(pos_prob.clamp_min(1e-12))).detach().cpu()
                    # label: prefer exported y_ioc; fallback to 0 if unavailable.
                    if y_ioc is not None:
                        lbl = y_ioc[i:j].to(torch.int64).detach().cpu()
                    else:
                        lbl = torch.zeros((int(pos_score.numel()),), dtype=torch.int64)
                    ri_sl = row_idx_cpu[i:j] if row_idx_cpu is not None else None
                    sf_sl = source_file[i:j] if source_file is not None else None
                    it_sl = ioc_type[i:j] if ioc_type is not None else None

                    batch_pred_stages: List[str] = []
                    batch_pred_probs: List[float] = []
                    if stage_pred is not None:
                        stage_logits = stage_pred(pos_inp)
                        stage_probs = torch.softmax(stage_logits, dim=-1)
                        stage_cls = torch.argmax(stage_probs, dim=-1).detach().cpu()
                        stage_max_p = stage_probs.max(dim=-1).values.detach().cpu()
                        for k in range(int(pos_score.numel())):
                            cidx = int(stage_cls[k].item())
                            batch_pred_stages.append(IDX_TO_STAGE.get(cidx, "none"))
                            batch_pred_probs.append(float(stage_max_p[k].item()))
                    else:
                        for k in range(int(pos_score.numel())):
                            batch_pred_stages.append("")
                            batch_pred_probs.append(0.0)

                    for k in range(int(pos_score.numel())):
                        y_true.append(int(lbl[k].item()))
                        y_score.append(float(pos_score[k].item()))
                        assert rows is not None
                        ridx = int(ri_sl[k].item()) if ri_sl is not None else -1
                        sf = str(sf_sl[k]) if sf_sl is not None else ""
                        ityp = str(it_sl[k]) if it_sl is not None else ""
                        rows.append(
                            {
                                "scenario": sc,
                                "t": int(tt[k].item()),
                                "etype": int(e[k].item()),
                                "src": int(s[k].item()),
                                "dst": int(d[k].item()),
                                "score": float(pos_score[k].item()),
                                "is_ioc": int(lbl[k].item()),
                                "source_file": sf,
                                "row_idx": ridx,
                                "ioc_type": ityp,
                                "pred_stage": batch_pred_stages[k],
                                "pred_stage_prob": f"{batch_pred_probs[k]:.4f}",
                            }
                        )

            if benign_mask is not None:
                # Exclude IOC-labeled edges from memory updates during training.
                if int(benign_mask.sum().item()) > 0:
                    memory.update_state(s[benign_mask], d[benign_mask], tt[benign_mask], raw_msg[benign_mask].detach())
            else:
                memory.update_state(s, d, tt, raw_msg.detach())

        metrics = None
        if collect_eval and bool(args.eval_ioc):
            metrics = (_roc_auc(y_true, y_score), _pr_auc(y_true, y_score))

        return (total_loss / max(1.0, count)), (correct / max(1.0, count)), rows, metrics

    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = _parse_topk(args.topk)
    select_p_at_k = _resolve_p_at_k(args.select_metric, int(args.select_p_at_k))
    if select_p_at_k is not None:
        if not bool(args.eval_ioc):
            raise SystemExit("--select-metric p_at* requires --eval-ioc.")
        if select_p_at_k not in topks:
            topks = sorted(set(topks) | {select_p_at_k})
            print(
                f"note: added K={select_p_at_k} to --topk for checkpoint selection "
                f"(now {topks}).",
                flush=True,
            )
    elif str(args.select_metric).strip().lower() not in ("auprc", "auroc"):
        raise SystemExit(
            f"Unknown --select-metric {args.select_metric!r}; use auprc, auroc, p_at, or p_at_<K>."
        )
    metric_label = _selection_metric_label(args.select_metric, int(args.select_p_at_k))

    best_metric = float("-inf")
    best_epoch: Optional[int] = None
    best_auroc_at_best: float = float("nan")
    best_auprc_at_best: float = float("nan")
    best_ckpt_path = out_dir / ("best_ckpt_holdout.pt" if args.holdout else "best_ckpt_joint.pt")
    best_scores_path = out_dir / "best_eval_tail_scores.csv"

    last_tr_loss = last_tr_acc = last_va_loss = last_va_acc = float("nan")
    last_auroc = last_auprc = float("nan")
    last_tail_eval: Dict[str, float] = {}
    best_tail_eval: Dict[str, float] = {}

    es_best = float("-inf")
    es_patience = 0
    last_completed_epoch = 0

    for ep in range(1, int(args.epochs) + 1):
        last_completed_epoch = int(ep)
        train_losses = []
        train_accs = []
        for sc in train_scenarios:
            tl, ta, _, _ = run_one_scenario(sc, train=True, prefix_only=True, collect_eval=False)
            train_losses.append(tl)
            train_accs.append(ta)
        tr_loss = float(sum(train_losses) / max(1, len(train_losses)))
        tr_acc = float(sum(train_accs) / max(1, len(train_accs)))

        with torch.no_grad():
            val_losses = []
            val_accs = []
            eval_rows: List[Dict[str, object]] = []
            eval_aurocs: List[float] = []
            eval_auprcs: List[float] = []
            for sc in test_scenarios:
                # In holdout mode, evaluate on the full scenario tail (future portion).
                vl, va, rows, metrics = run_one_scenario(
                    sc, train=False, prefix_only=False, collect_eval=bool(args.save_scores or args.eval_ioc)
                )
                val_losses.append(vl)
                val_accs.append(va)
                if rows is not None:
                    eval_rows.extend(rows)
                if metrics is not None:
                    auroc, auprc = metrics
                    if not math.isnan(auroc):
                        eval_aurocs.append(float(auroc))
                    if not math.isnan(auprc):
                        eval_auprcs.append(float(auprc))
            va_loss = float(sum(val_losses) / max(1, len(val_losses)))
            va_acc = float(sum(val_accs) / max(1, len(val_accs)))

        extra = ""
        cur_auroc = float("nan")
        cur_auprc = float("nan")
        if bool(args.eval_ioc) and (eval_aurocs or eval_auprcs):
            cur_auroc = float(sum(eval_aurocs) / max(1, len(eval_aurocs))) if eval_aurocs else float("nan")
            cur_auprc = float(sum(eval_auprcs) / max(1, len(eval_auprcs))) if eval_auprcs else float("nan")
            extra = f" | AUROC {cur_auroc:.3f} AUPRC {cur_auprc:.3f}"

        if topks and eval_rows:
            hits = _topk_ioc_hits(eval_rows, topks)
            extra += " | " + " ".join([f"top{k}={hits[k]}" for k in topks])

        epoch_tail_eval: Dict[str, float] = {}
        if bool(args.eval_ioc) and eval_rows:
            epoch_tail_eval = tail_alert_metrics(
                eval_rows,
                topks=topks,
                alert_window=int(args.eval_alert_window),
                alert_quantile=float(args.eval_alert_quantile),
                alert_min_events=int(args.eval_alert_min_events),
                alert_topk_events=int(args.eval_alert_topk_events),
                dedupe=not bool(args.no_eval_alert_dedupe),
            )
            last_tail_eval = dict(epoch_tail_eval)
            extra += (
                f" | pf={epoch_tail_eval['precision_in_flagged']:.3f}"
                f" ar={epoch_tail_eval['alerts_per_tail_event']:.4f}"
                f" fr={epoch_tail_eval['flagged_rate']:.3f}"
            )
            for k in topks:
                pk = epoch_tail_eval.get(f"p_at_{int(k)}", float("nan"))
                extra += f" p@{int(k)}={pk:.3f}"
        print(
            f"epoch {ep:03d} | train({len(train_scenarios)}) loss {tr_loss:.4f} acc {tr_acc:.3f} "
            f"| val({len(test_scenarios)}) loss {va_loss:.4f} acc {va_acc:.3f}{extra}"
        )

        last_tr_loss, last_tr_acc = float(tr_loss), float(tr_acc)
        last_va_loss, last_va_acc = float(va_loss), float(va_acc)
        last_auroc, last_auprc = float(cur_auroc), float(cur_auprc)

        if bool(args.save_scores) or bool(args.save_scores_each_epoch):
            csv_path = out_dir / ("eval_tail_scores.csv" if bool(args.save_scores) else f"eval_tail_scores_epoch{ep:03d}.csv")
            with csv_path.open("w", newline="") as f:
                w = DictWriter(f, fieldnames=EVAL_TAIL_CSV_FIELDS)
                w.writeheader()
                for r in eval_rows:
                    w.writerow(r)

        # Best checkpoint selection (requires eval-ioc)
        if bool(args.eval_ioc):
            cur = _selection_score(
                args.select_metric,
                int(args.select_p_at_k),
                cur_auroc=cur_auroc,
                cur_auprc=cur_auprc,
                tail_eval=epoch_tail_eval,
            )
            if not math.isnan(cur) and cur > best_metric:
                best_metric = float(cur)
                best_epoch = int(ep)
                best_auroc_at_best = float(cur_auroc)
                best_auprc_at_best = float(cur_auprc)
                ckpt_dict = {
                        "memory": memory.state_dict(),
                        "link_pred": link_pred.state_dict(),
                        "etype_emb": etype_emb.state_dict(),
                        "config": vars(args),
                        "train_scenarios": train_scenarios,
                        "test_scenarios": test_scenarios,
                        "scenario_base": scenario_base,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "metric_name": metric_label,
                    }
                if stage_pred is not None:
                    ckpt_dict["stage_pred"] = stage_pred.state_dict()
                torch.save(ckpt_dict, best_ckpt_path)
                if eval_rows:
                    with best_scores_path.open("w", newline="") as f:
                        w = DictWriter(f, fieldnames=EVAL_TAIL_CSV_FIELDS)
                        w.writeheader()
                        for r in eval_rows:
                            w.writerow(r)
                if epoch_tail_eval:
                    best_tail_eval = dict(epoch_tail_eval)

            # Early stopping on validation tail metric (same as --select-metric)
            if int(args.early_stop_patience) > 0:
                cur_es = _selection_score(
                    args.select_metric,
                    int(args.select_p_at_k),
                    cur_auroc=cur_auroc,
                    cur_auprc=cur_auprc,
                    tail_eval=epoch_tail_eval,
                )
                if not math.isnan(cur_es):
                    if cur_es > es_best + float(args.early_stop_min_delta):
                        es_best = float(cur_es)
                        es_patience = 0
                    else:
                        es_patience += 1
                        if es_patience >= int(args.early_stop_patience):
                            print(
                                f"early_stop: {metric_label} did not improve by "
                                f">{args.early_stop_min_delta} for {args.early_stop_patience} epochs "
                                f"(best_seen={es_best:.4f}). Stopping at epoch {ep:03d}."
                            )
                            break

    final_ckpt: Dict[str, object] = {
            "memory": memory.state_dict(),
            "link_pred": link_pred.state_dict(),
            "etype_emb": etype_emb.state_dict(),
            "config": vars(args),
            "train_scenarios": train_scenarios,
            "test_scenarios": test_scenarios,
            "scenario_base": scenario_base,
    }
    if stage_pred is not None:
        final_ckpt["stage_pred"] = stage_pred.state_dict()
    torch.save(
        final_ckpt,
        out_dir / ("ckpt_holdout.pt" if args.holdout else "ckpt_joint.pt"),
    )
    if best_epoch is not None:
        print(f"Best by {metric_label}: epoch {best_epoch} = {best_metric:.4f}")
    if int(args.early_stop_patience) > 0 and last_completed_epoch < int(args.epochs):
        print(f"Completed {last_completed_epoch}/{int(args.epochs)} epochs (early stopping).")

    def _jf(x: object) -> object:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    best_metric_out: Optional[float] = None
    if best_epoch is not None and not math.isinf(float(best_metric)):
        best_metric_out = float(best_metric)

    primary_scenario = (
        str(args.holdout).strip()
        if args.holdout
        else (test_scenarios[0] if len(test_scenarios) == 1 else "")
    )

    summary = {
        "eval_protocol": eval_protocol,
        "scenario": primary_scenario,
        "holdout": str(args.holdout) if args.holdout else "",
        "scenarios": list(scenario_universe),
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "aux_supervision": str(args.aux_supervision),
        "lambda_ioc_rank": float(args.lambda_ioc_rank),
        "lambda_stage": float(args.lambda_stage),
        "epochs": int(args.epochs),
        "epochs_completed": int(last_completed_epoch),
        "early_stopped": bool(last_completed_epoch < int(args.epochs)),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "seed": int(args.seed),
        "select_metric": str(args.select_metric),
        "select_metric_label": metric_label,
        "select_p_at_k": int(args.select_p_at_k),
        "best_epoch": best_epoch,
        "best_metric": best_metric_out,
        "best_auroc": _jf(best_auroc_at_best),
        "best_auprc": _jf(best_auprc_at_best),
        "last_train_loss": _jf(last_tr_loss),
        "last_train_acc": _jf(last_tr_acc),
        "last_val_loss": _jf(last_va_loss),
        "last_val_acc": _jf(last_va_acc),
        "last_auroc": _jf(last_auroc),
        "last_auprc": _jf(last_auprc),
        "last_tail_eval": {k: _jf(v) for k, v in last_tail_eval.items()},
        "best_tail_eval": {k: _jf(v) for k, v in best_tail_eval.items()},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

