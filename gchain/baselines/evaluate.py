"""Unified baseline evaluation: detection + reconstruction + latency."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from gchain.baselines.freq_rarity import freq_rarity_scores, score_fn_for_latency as freq_latency_fn
from gchain.baselines.path_lof import path_lof_scores, score_fn_for_latency as path_lof_latency_fn
from gchain.baselines.random_baseline import random_scores, score_fn_for_latency as random_latency_fn
from gchain.baselines.score_rows import tail_score_rows
from gchain.baselines.static_gnn import (
    StaticGNNConfig,
    infer_fn_for_model,
    train_and_infer_tail,
)
from gchain.baselines.telemetry import (
    TelemetryKind,
    filter_stream_indices,
    scenario_supports_telemetry,
    subset_stream,
)
from gchain.eval.alert_eval import precision_at_k_all
from gchain.eval.attack_reconstruct import (
    build_line_to_ioc_type,
    evaluate_reconstruction,
    ioc_log_source_files_for_scenario,
    load_ioc_type_to_stage,
    load_json,
)
from gchain.eval.latency import ms_per_1k_edges
from gchain.train.metrics import pr_auc, roc_auc
from gchain.train.split import time_split_idx
from gchain.train.streams import Stream, load_stream_from_tgn_pt

DEFAULT_TOPKS = (10, 50, 100, 500)
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_IOC_GT = "data/SynthChain/iocs/ioc_ground_truth.json"
DEFAULT_STAGE_GT_DIR = "artifacts/stage_gt"


def _is_static_gnn(method: str) -> bool:
    return str(method).lower() in ("graphsage", "rgcn", "static_gnn")


def _score_tail(
    method: str,
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
    seed: int,
    static_gnn_config: Optional[StaticGNNConfig] = None,
) -> List[float]:
    m = str(method).lower()
    if m == "freq_rarity":
        return freq_rarity_scores(st, train_end=train_end, tail_start=tail_start)
    if m == "path_lof":
        return path_lof_scores(st, train_end=train_end, tail_start=tail_start)
    if m == "random":
        return random_scores(st, tail_start=tail_start, seed=seed)
    if _is_static_gnn(m):
        cfg = static_gnn_config or StaticGNNConfig(seed=seed)
        scores, _, _ = train_and_infer_tail(
            st,
            train_end=train_end,
            tail_start=tail_start,
            variant=m,
            config=cfg,
        )
        return scores
    raise ValueError(f"unknown baseline method: {method!r}")


def _latency_fn(
    method: str,
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
    seed: int,
    static_gnn_config: Optional[StaticGNNConfig] = None,
) -> Callable[[], List[float]]:
    m = str(method).lower()
    if m == "freq_rarity":
        return freq_latency_fn(st, train_end=train_end, tail_start=tail_start)
    if m == "path_lof":
        return path_lof_latency_fn(st, train_end=train_end, tail_start=tail_start)
    if m == "random":
        return random_latency_fn(st, tail_start=tail_start, seed=seed)
    if _is_static_gnn(m):
        cfg = static_gnn_config or StaticGNNConfig(seed=seed)
        return static_gnn_latency_fn(
            st,
            train_end=train_end,
            tail_start=tail_start,
            variant=m,
            config=cfg,
        )
    raise ValueError(f"unknown baseline method: {method!r}")


def _detection_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    y_true = [int(r.get("is_ioc", 0)) for r in rows]
    y_score = [float(r.get("score", 0.0)) for r in rows]
    p_at = precision_at_k_all(list(rows), [500])
    return {
        "auroc": roc_auc(y_true, y_score),
        "auprc": pr_auc(y_true, y_score),
        "p_at_500": float(p_at.get(500, float("nan"))),
        "n_tail": int(len(rows)),
        "n_ioc_tail": int(sum(y_true)),
    }


def _recon_at_k(recon: Mapping[str, Any], k: int = 500) -> Dict[str, float]:
    by_k = (recon.get("by_k") or {})
    entry = by_k.get(str(int(k))) or {}
    return {
        "stage_recall": float(entry.get("stage_recall", float("nan"))),
        "ordered_stage_recall_lcs": float(
            entry.get("ordered_stage_recall_lcs", float("nan"))
        ),
    }


def evaluate_scenario(
    *,
    method: str,
    scenario: str,
    tgn_pt: Path,
    repo_root: Path,
    telemetry: TelemetryKind = "full",
    train_frac: float = DEFAULT_TRAIN_FRAC,
    topks: Sequence[int] = DEFAULT_TOPKS,
    stage_gt_dir: Path | str = DEFAULT_STAGE_GT_DIR,
    ioc_gt: Path | str = DEFAULT_IOC_GT,
    seed: int = 0,
    run_reconstruction: bool = True,
    static_gnn_config: Optional[StaticGNNConfig] = None,
    full_pt: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one method on one scenario; returns JSON-serializable metrics dict."""
    if not scenario_supports_telemetry(scenario, telemetry):
        return {
            "method": method,
            "telemetry": telemetry,
            "scenario": scenario,
            "status": "na",
            "reason": f"scenario {scenario} has no {telemetry} telemetry",
        }

    st_full = load_stream_from_tgn_pt(tgn_pt)
    keep = filter_stream_indices(st_full, telemetry)
    st = subset_stream(st_full, keep)
    n = int(st.src.numel())
    if n < 2:
        return {
            "method": method,
            "telemetry": telemetry,
            "scenario": scenario,
            "status": "na",
            "reason": "too few edges after telemetry filter",
            "n_edges": n,
        }

    split = time_split_idx(n, float(train_frac))
    gnn_cfg: Optional[StaticGNNConfig] = None
    if _is_static_gnn(method):
        gnn_cfg = static_gnn_config or StaticGNNConfig(seed=seed)
        if full_pt is not None:
            gnn_cfg = StaticGNNConfig(
                variant=gnn_cfg.variant,
                epochs=gnn_cfg.epochs,
                hidden_dim=gnn_cfg.hidden_dim,
                num_layers=gnn_cfg.num_layers,
                lr=gnn_cfg.lr,
                weight_decay=gnn_cfg.weight_decay,
                neg_ratio=gnn_cfg.neg_ratio,
                train_batch_size=gnn_cfg.train_batch_size,
                infer_batch_size=gnn_cfg.infer_batch_size,
                dropout=gnn_cfg.dropout,
                device=gnn_cfg.device,
                seed=gnn_cfg.seed,
                full_pt=full_pt,
            )
    infer_only: Optional[Callable[[], List[float]]] = None
    if _is_static_gnn(method):
        assert gnn_cfg is not None
        scores, gnn_model, gnn_cfg = train_and_infer_tail(
            st,
            train_end=split,
            tail_start=split,
            variant=method,
            config=gnn_cfg,
        )
        infer_only = infer_fn_for_model(
            gnn_model, st, train_end=split, tail_start=split, config=gnn_cfg
        )
    else:
        scores = _score_tail(
            method,
            st,
            train_end=split,
            tail_start=split,
            seed=seed,
        )
    rows = tail_score_rows(st, scenario=scenario, tail_start=split, scores=scores)
    detection = _detection_metrics(rows)

    recon_summary: Dict[str, Any] = {}
    if run_reconstruction:
        stage_gt_path = (repo_root / stage_gt_dir / f"{scenario}.stages_gt.json").resolve()
        ioc_gt_path = (repo_root / ioc_gt).resolve()
        stages_gt = load_json(stage_gt_path)
        ioc_type_to_stage = load_ioc_type_to_stage(repo_root)
        ioc_gt_data = load_json(ioc_gt_path)
        line_to_type = build_line_to_ioc_type(ioc_gt_data, scenario)
        recon = evaluate_reconstruction(
            rows=rows,
            stages_gt=stages_gt,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            ioc_log_source_files=ioc_log_source_files_for_scenario(scenario),
            run_alert_reconstruction=False,
        )
        recon_summary = {
            "at_500": _recon_at_k(recon, 500),
            "by_k": recon.get("by_k"),
        }

    n_tail = n - split
    if _is_static_gnn(method):
        assert infer_only is not None
        lat = ms_per_1k_edges(infer_only, n_tail_edges=n_tail, warmup=0)
    else:
        lat = ms_per_1k_edges(
            _latency_fn(method, st, train_end=split, tail_start=split, seed=seed),
            n_tail_edges=n_tail,
        )

    out: Dict[str, Any] = {
        "method": method,
        "telemetry": telemetry,
        "scenario": scenario,
        "status": "ok",
        "tgn_pt": str(tgn_pt),
        "train_frac": float(train_frac),
        "split_idx": int(split),
        "n_edges": n,
        "detection": detection,
        "reconstruction": recon_summary,
        "latency_ms_per_1k": lat,
    }
    if gnn_cfg is not None:
        out["static_gnn"] = {
            "variant": gnn_cfg.variant,
            "epochs": gnn_cfg.epochs,
            "hidden_dim": gnn_cfg.hidden_dim,
            "num_layers": gnn_cfg.num_layers,
            "device": gnn_cfg.device,
            "full_pt": str(gnn_cfg.full_pt) if gnn_cfg.full_pt else "",
        }
    return out


def evaluate_from_score_rows(
    *,
    method: str,
    scenario: str,
    rows: Sequence[Mapping[str, Any]],
    repo_root: Path,
    telemetry: TelemetryKind = "full",
    topks: Sequence[int] = DEFAULT_TOPKS,
    stage_gt_dir: Path | str = DEFAULT_STAGE_GT_DIR,
    ioc_gt: Path | str = DEFAULT_IOC_GT,
    latency_ms_per_1k: Optional[float] = None,
) -> Dict[str, Any]:
    """Metrics from precomputed score rows (e.g. FuseChain train output)."""
    detection = _detection_metrics(rows)
    stage_gt_path = (repo_root / stage_gt_dir / f"{scenario}.stages_gt.json").resolve()
    ioc_gt_path = (repo_root / ioc_gt).resolve()
    stages_gt = load_json(stage_gt_path)
    ioc_type_to_stage = load_ioc_type_to_stage(repo_root)
    ioc_gt_data = load_json(ioc_gt_path)
    line_to_type = build_line_to_ioc_type(ioc_gt_data, scenario)
    recon = evaluate_reconstruction(
        rows=list(rows),
        stages_gt=stages_gt,
        ioc_type_to_stage=ioc_type_to_stage,
        line_to_type=line_to_type,
        topks=topks,
        ioc_log_source_files=ioc_log_source_files_for_scenario(scenario),
        run_alert_reconstruction=False,
    )
    return {
        "method": method,
        "telemetry": telemetry,
        "scenario": scenario,
        "status": "ok",
        "detection": detection,
        "reconstruction": {"at_500": _recon_at_k(recon, 500), "by_k": recon.get("by_k")},
        "latency_ms_per_1k": latency_ms_per_1k,
    }


def write_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
