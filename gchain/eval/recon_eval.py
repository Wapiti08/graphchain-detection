"""Orchestrate attack-chain reconstruction metrics (multiple eval modes)."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from gchain.eval.recon_alerts import evaluate_alert_reconstruction
from gchain.eval.recon_pools import filter_rows_to_ioc_log_sources
from gchain.eval.recon_stages import ordered_stage_sequence
from gchain.eval.recon_topk import (
    dedupe_rows_by_endpoint_pair,
    eval_topk_group_cap,
    eval_topk_group_cap_adaptive,
    eval_topk_modes,
    eval_topk_source_quota,
)


RECONSTRUCTION_NOTES: Dict[str, str] = {
    "by_k": (
        "Primary (paper): global top-K by anomaly score; stages from IOC edges only "
        "(rule/line GT on is_ioc=1)."
    ),
    "by_k_pred_stage": (
        "Scheme 1: global top-K; stages from model pred_stage on all edges in top-K. "
        "Requires training with --lambda-stage > 0 and pred_stage in scores CSV."
    ),
    "by_alert_pred_stage": (
        "Scheme 2: aggregate_alerts-style clustering on high-score events; "
        "stages from pred_stage votes within each alert (all edges in cluster)."
    ),
    "by_alert_rule": (
        "Scheme 2 (eval aid): same alert clusters; stages from line/ioc_type on all "
        "cluster edges (no is_ioc filter). Still uses GT line map where available."
    ),
    "by_k_ioc_pool_upper_bound": (
        "Upper bound only: top-K among IOC-labeled edges by score — not for primary reporting."
    ),
    "by_k_ioc_log_sources": (
        "Ablation: global top-K among edges from logs with has_ioc=True in synthchain_sources "
        "(eval-only candidate pool; does not change graph training). Primary metric remains by_k."
    ),
    "by_k_pair_dedupe": (
        "Ablation: dedupe to max score per (scenario, etype, src, dst) then global top-K and IOC stage "
        "labeling (reduces repeated DEPEND/INJECT hub edges). Primary metric remains by_k."
    ),
    "by_k_source_quota": (
        "Ablation: select top-K with per-source_file minimum quotas (eval-only), then stage labeling on IOC edges. "
        "Primary metric remains by_k."
    ),
    "by_k_group_cap": (
        "Ablation: select top-K with a per-group maximum count cap (eval-only) to prevent a single repeated pattern "
        "from dominating top-K (softer than strict pair dedupe)."
    ),
    "by_k_group_cap_adaptive": (
        "Ablation: adaptive group-cap that only caps groups that dominate the score head "
        "(detected from a probe head of size K' = probe_mult * K)."
    ),
}


def has_pred_stage(rows: Sequence[Dict[str, Any]]) -> bool:
    for r in rows[:200]:
        ps = str(r.get("pred_stage") or "").strip()
        if ps and ps != "none":
            return True
    return False


def evaluate_reconstruction(
    *,
    rows: Sequence[Dict[str, Any]],
    stages_gt: Mapping[str, Any],
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    topks: Sequence[int] = (10, 50, 100),
    pred_min_prob: float = 0.0,
    pred_min_count: int = 1,
    include_ioc_pool_upper_bound: bool = False,
    ioc_log_source_files: Optional[Set[str]] = None,
    run_alert_reconstruction: bool = True,
    alert_window: int = 3600,
    alert_quantile: float = 0.99,
    alert_min_events: int = 3,
    alert_topk_events: int = 0,
    alert_dedupe: bool = True,
    source_file_min_quota: Optional[Mapping[str, int]] = None,
    source_quota_exclude_etypes: Optional[Set[int]] = None,
    source_quota_pair_dedupe: bool = True,
    group_cap_key: str = "",
    group_cap_max: int = 0,
    group_cap_pair_dedupe: bool = True,
    group_cap_adaptive_key: str = "",
    group_cap_adaptive_probe_mult: int = 5,
    group_cap_adaptive_hot_threshold: int = 20,
    group_cap_adaptive_hot_cap: int = 10,
    group_cap_adaptive_pair_dedupe: bool = True,
) -> Dict[str, Any]:
    obs = set((stages_gt.get("observable") or {}).get("stages") or [])
    sem = set((stages_gt.get("semantic") or {}).get("stages") or [])
    gt_order = ordered_stage_sequence(obs)

    metrics: Dict[str, Any] = {
        "observable_stages": sorted(obs),
        "semantic_stages": sorted(sem),
        "unobserved_semantic": sorted(sem - obs),
        "reconstruction_notes": dict(RECONSTRUCTION_NOTES),
        "by_k": eval_topk_modes(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            use_predicted=False,
            pred_min_prob=0.0,
            pred_min_count=1,
            ioc_ranked=False,
        ),
    }

    if ioc_log_source_files is not None:
        cand_rows, pool_meta = filter_rows_to_ioc_log_sources(rows, ioc_log_source_files)
        metrics["candidate_pool_ioc_log_sources"] = pool_meta
        metrics["by_k_ioc_log_sources"] = eval_topk_modes(
            cand_rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            use_predicted=False,
            pred_min_prob=0.0,
            pred_min_count=int(pred_min_count),
            ioc_ranked=False,
        )

    _, pair_pool_meta = dedupe_rows_by_endpoint_pair(rows)
    metrics["candidate_pool_pair_dedupe"] = pair_pool_meta
    metrics["by_k_pair_dedupe"] = eval_topk_modes(
        rows,
        obs=obs,
        gt_order=gt_order,
        ioc_type_to_stage=ioc_type_to_stage,
        line_to_type=line_to_type,
        topks=topks,
        use_predicted=False,
        pred_min_prob=0.0,
        pred_min_count=int(pred_min_count),
        ioc_ranked=False,
        endpoint_pair_dedupe=True,
    )

    if source_file_min_quota:
        quota, meta = eval_topk_source_quota(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            source_file_min_quota=source_file_min_quota,
            exclude_etypes=source_quota_exclude_etypes,
            endpoint_pair_dedupe=bool(source_quota_pair_dedupe),
        )
        metrics["candidate_pool_source_quota"] = meta
        metrics["by_k_source_quota"] = quota

    if str(group_cap_key).strip() and int(group_cap_max) > 0:
        cap, meta = eval_topk_group_cap(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            cap_key=str(group_cap_key).strip(),
            cap_max=int(group_cap_max),
            endpoint_pair_dedupe=bool(group_cap_pair_dedupe),
        )
        metrics["candidate_pool_group_cap"] = meta
        metrics["by_k_group_cap"] = cap

    if str(group_cap_adaptive_key).strip() and int(group_cap_adaptive_hot_cap) > 0:
        cap, meta = eval_topk_group_cap_adaptive(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            cap_key=str(group_cap_adaptive_key).strip(),
            probe_mult=int(group_cap_adaptive_probe_mult),
            hot_threshold=int(group_cap_adaptive_hot_threshold),
            hot_cap_max=int(group_cap_adaptive_hot_cap),
            endpoint_pair_dedupe=bool(group_cap_adaptive_pair_dedupe),
        )
        metrics["candidate_pool_group_cap_adaptive"] = meta
        metrics["by_k_group_cap_adaptive"] = cap

    if include_ioc_pool_upper_bound:
        ub = eval_topk_modes(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            use_predicted=False,
            pred_min_prob=0.0,
            pred_min_count=int(pred_min_count),
            ioc_ranked=True,
        )
        metrics["by_k_ioc_pool_upper_bound"] = ub
        metrics["by_k_ioc_ranked"] = ub

    if has_pred_stage(rows):
        pred_k = eval_topk_modes(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            use_predicted=True,
            pred_min_prob=float(pred_min_prob),
            pred_min_count=int(pred_min_count),
        )
        metrics["by_k_pred_stage"] = pred_k
        metrics["by_k_predicted"] = pred_k
    else:
        metrics["by_k_pred_stage"] = {
            "_note": RECONSTRUCTION_NOTES["by_k_pred_stage"],
            "available": False,
        }

    if run_alert_reconstruction:
        metrics["by_alert_rule"] = evaluate_alert_reconstruction(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            use_predicted=False,
            pred_min_prob=0.0,
            pred_min_count=int(pred_min_count),
            alert_window=int(alert_window),
            alert_quantile=float(alert_quantile),
            alert_min_events=int(alert_min_events),
            alert_topk_events=int(alert_topk_events),
            alert_dedupe=bool(alert_dedupe),
        )
        if has_pred_stage(rows):
            metrics["by_alert_pred_stage"] = evaluate_alert_reconstruction(
                rows,
                obs=obs,
                gt_order=gt_order,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=True,
                pred_min_prob=float(pred_min_prob),
                pred_min_count=int(pred_min_count),
                alert_window=int(alert_window),
                alert_quantile=float(alert_quantile),
                alert_min_events=int(alert_min_events),
                alert_topk_events=int(alert_topk_events),
                alert_dedupe=bool(alert_dedupe),
            )
        else:
            metrics["by_alert_pred_stage"] = {
                "_note": RECONSTRUCTION_NOTES["by_alert_pred_stage"],
                "available": False,
            }

    return metrics
