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


def _ioc_type_str_from_stream(st: "object", idx: int, *, stage_supervision: str) -> str:
    mode = str(stage_supervision or "gt_ioc").strip().lower()
    if mode == "gt_ioc":
        tup = getattr(st, "ioc_type", None)
    elif mode in {"rule", "rule_high"}:
        tup = getattr(st, "rule_ioc_type", None)
    else:
        raise ValueError(f"Unknown stage_supervision: {stage_supervision!r}")
    if tup is None:
        return ""
    return str(tup[idx] or "").strip()


def build_stage_labels(
    *,
    streams: Dict[str, "object"],
    repo_root: "object",
    ioc_type_to_stage_idx: "object",
    load_stage_map: "object",
    stage_supervision: str = "gt_ioc",
) -> Dict[str, "object"]:
    """
    Per-edge stage class indices (0 = none).

    stage_supervision:
      - gt_ioc: map GT ioc_type from stream (eval upper-bound style; uses IOC GT at export)
      - rule: map rule_ioc_type where y_rule=1
      - rule_high: map rule_ioc_type where y_rule_high=1 (deployment-aligned default)

    Network IOC types (attack_ip, …) never receive stage CE for rule/rule_high.
    """
    import torch

    from gchain.train.stage_policy import is_ioc_type_stage_eligible

    mode = str(stage_supervision or "gt_ioc").strip().lower()
    repo_s = str(repo_root)
    if mode not in {"gt_ioc", "rule", "rule_high"}:
        raise ValueError(f"Unknown stage_supervision: {mode!r}")

    stage_map = load_stage_map(repo_root)
    out: Dict[str, "object"] = {}
    for sc in sorted(streams.keys()):
        st = streams[sc]
        n = int(st.src.numel())
        labels = torch.zeros(n, dtype=torch.long)

        if mode in {"rule", "rule_high"}:
            rit = getattr(st, "rule_ioc_type", None)
            if rit is None:
                raise SystemExit(
                    f"[{sc}] stage_supervision={mode!r} requires rule_ioc_type in *.tgn.pt. "
                    "Regenerate graphs: python -m gchain.pipeline --dataset synthchain "
                    f"--scenario {sc} --export-tgn"
                )

        for idx_e in range(n):
            it = _ioc_type_str_from_stream(st, idx_e, stage_supervision=mode)
            if mode in {"rule", "rule_high"}:
                if not it or not is_ioc_type_stage_eligible(it, project_root=repo_s):
                    labels[idx_e] = 0
                    continue
            labels[idx_e] = int(ioc_type_to_stage_idx(it, stage_map))

        if mode == "rule":
            y_rule = getattr(st, "y_rule", None)
            if y_rule is not None:
                labels = labels * (y_rule > 0).long()
        elif mode == "rule_high":
            y_rh = getattr(st, "y_rule_high", None)
            if y_rh is not None:
                labels = labels * (y_rh > 0).long()
            else:
                raise SystemExit(
                    f"[{sc}] stage_supervision=rule_high requires y_rule_high in *.tgn.pt "
                    "(regenerate graphs with weak_supervision_rules v2)."
                )

        out[sc] = labels
    return out


def freeze_ssl_backbone(
    memory: "object",
    link_pred: "object",
    etype_emb: "object",
) -> None:
    """Stop gradient updates on TGN memory / link scorer / etype embedding."""
    for module in (memory, link_pred, etype_emb):
        for param in module.parameters():
            param.requires_grad = False


def load_training_checkpoint(
    path: "object",
    *,
    memory: "object",
    link_pred: "object",
    etype_emb: "object",
    stage_pred: Optional["object"] = None,
    device: "object",
) -> None:
    import torch

    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    memory.load_state_dict(ckpt["memory"])
    link_pred.load_state_dict(ckpt["link_pred"])
    etype_emb.load_state_dict(ckpt["etype_emb"])
    if stage_pred is not None and ckpt.get("stage_pred") is not None:
        stage_pred.load_state_dict(ckpt["stage_pred"])


def count_stage_eligible_rule_labels(
    st: "object",
    labels: "object",
    *,
    stage_supervision: str,
) -> int:
    mode = str(stage_supervision or "gt_ioc").strip().lower()
    if mode == "rule":
        y_mask = getattr(st, "y_rule", None)
    elif mode == "rule_high":
        y_mask = getattr(st, "y_rule_high", None)
    else:
        return int((labels > 0).sum().item())
    if y_mask is None:
        return int((labels > 0).sum().item())
    import torch

    return int(((labels > 0) & (y_mask > 0)).sum().item())


def _stage_eligible_counts_per_scenario(
    streams: Dict[str, "object"],
    *,
    stage_supervision: str,
    repo_root: "object",
    ioc_type_to_stage_idx: "object",
    load_stage_map: "object",
) -> Dict[str, int]:
    mode = str(stage_supervision or "gt_ioc").strip().lower()
    labels_per_sc = build_stage_labels(
        streams=streams,
        repo_root=repo_root,
        ioc_type_to_stage_idx=ioc_type_to_stage_idx,
        load_stage_map=load_stage_map,
        stage_supervision=mode,
    )
    out: Dict[str, int] = {}
    for sc, labels in labels_per_sc.items():
        out[sc] = count_stage_eligible_rule_labels(
            streams[sc], labels, stage_supervision=mode
        )
    return out


def resolve_stage_supervision(
    streams: Dict[str, "object"],
    *,
    stage_supervision: str,
    repo_root: "object",
    ioc_type_to_stage_idx: "object",
    load_stage_map: "object",
) -> Tuple[str, bool]:
    """
    Pick an effective stage supervision mode for training.

    For weak-rule modes, network IOC types (attack_ip, suspicious_port, …) are excluded
    from stage CE. When ``rule_high`` yields zero stage-eligible labels (common on sc1
    where high-confidence hits are mostly CONNECT/alert-shaped), fall back to ``rule``
    (medium+high). If still zero, disable the stage head and train SSL-only.
    """
    import sys

    mode = str(stage_supervision or "gt_ioc").strip().lower()
    if mode not in {"rule", "rule_high"}:
        return mode, True

    counts = _stage_eligible_counts_per_scenario(
        streams,
        stage_supervision=mode,
        repo_root=repo_root,
        ioc_type_to_stage_idx=ioc_type_to_stage_idx,
        load_stage_map=load_stage_map,
    )
    if all(n > 0 for n in counts.values()):
        return mode, True

    for sc, n in sorted(counts.items()):
        if n == 0:
            print(
                f"[{sc}] No stage-eligible {mode} labels after supervision_policy "
                "(network IOC types excluded from stage CE).",
                file=sys.stderr,
            )

    if mode == "rule_high":
        rule_counts = _stage_eligible_counts_per_scenario(
            streams,
            stage_supervision="rule",
            repo_root=repo_root,
            ioc_type_to_stage_idx=ioc_type_to_stage_idx,
            load_stage_map=load_stage_map,
        )
        if any(n > 0 for n in rule_counts.values()):
            for sc, n in sorted(rule_counts.items()):
                if n > 0:
                    print(
                        f"[{sc}] Falling back to stage_supervision=rule "
                        f"({n} stage-eligible weak-rule labels).",
                        file=sys.stderr,
                    )
            return "rule", True

    print(
        "[train] Disabling stage head (lambda_stage ignored): no stage-eligible weak-rule "
        "labels for rule or rule_high. SSL link detection will still train. "
        "Use --lambda-stage 0 to silence this, or --stage-supervision gt_ioc for ablations.",
        file=sys.stderr,
    )
    return mode, False


def validate_stage_supervision_streams(
    streams: Dict[str, "object"],
    *,
    stage_supervision: str,
    repo_root: "object",
    ioc_type_to_stage_idx: "object",
    load_stage_map: "object",
) -> None:
    """Backward-compatible strict check (prefer resolve_stage_supervision in train_loop)."""
    _, use_stage = resolve_stage_supervision(
        streams,
        stage_supervision=stage_supervision,
        repo_root=repo_root,
        ioc_type_to_stage_idx=ioc_type_to_stage_idx,
        load_stage_map=load_stage_map,
    )
    if not use_stage and str(stage_supervision or "").strip().lower() in {"rule", "rule_high"}:
        raise SystemExit(
            "No stage-eligible weak-rule labels. See resolve_stage_supervision warnings above."
        )
