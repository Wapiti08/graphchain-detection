"""Alert-cluster based stage reconstruction (aggregate_alerts rules)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from gchain.eval.alert_eval import (
    ScoredEvent,
    cluster_connected,
    dedupe_events,
    rows_to_events,
    select_high_score,
    time_windows,
)
from gchain.eval.recon_stages import recon_scores, stage_for_edge, stage_for_edge_predicted
from gchain.eval.recon_topk import rows_by_event_key


def cluster_rows_into_alerts(
    rows: Sequence[Dict[str, Any]],
    *,
    window: int,
    score_quantile: float,
    min_events: int,
    topk_events: int,
    dedupe: bool,
) -> List[List[Dict[str, Any]]]:
    by_sc: Dict[str, List[ScoredEvent]] = {}
    for e in rows_to_events(rows):
        by_sc.setdefault(e.scenario, []).append(e)
    row_by_key = rows_by_event_key(rows)
    alerts: List[List[Dict[str, Any]]] = []
    for _sc, evs in sorted(by_sc.items()):
        if not evs:
            continue
        if dedupe:
            evs = dedupe_events(evs)
        kept = select_high_score(
            evs,
            topk_events=int(topk_events),
            score_quantile=float(score_quantile),
        )
        for win in time_windows(kept, int(window)):
            for cl in cluster_connected(win):
                if len(cl) < int(min_events):
                    continue
                edge_rows: List[Dict[str, Any]] = []
                for e in cl:
                    key = (e.scenario, e.t, e.etype, e.src, e.dst)
                    r = row_by_key.get(key)
                    if r is not None:
                        edge_rows.append(r)
                if edge_rows:
                    alerts.append(edge_rows)
    return alerts


def stage_edge_counts(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    use_predicted: bool,
    pred_min_prob: float,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in edge_rows:
        if use_predicted:
            st = stage_for_edge_predicted(r, min_prob=float(pred_min_prob))
        else:
            st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
        if st:
            counts[st] = counts.get(st, 0) + 1
    return counts


def stages_from_alerts(
    alerts: Sequence[Sequence[Mapping[str, Any]]],
    *,
    use_predicted: bool,
    pred_min_prob: float,
    pred_min_count: int,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
) -> Set[str]:
    minc = max(1, int(pred_min_count))
    out: Set[str] = set()
    for edge_rows in alerts:
        counts = stage_edge_counts(
            edge_rows,
            use_predicted=use_predicted,
            pred_min_prob=float(pred_min_prob),
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
        )
        for st, n in counts.items():
            if int(n) >= minc:
                out.add(st)
    return out


def evaluate_alert_reconstruction(
    rows: Sequence[Dict[str, Any]],
    *,
    obs: Set[str],
    gt_order: List[str],
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    use_predicted: bool,
    pred_min_prob: float,
    pred_min_count: int,
    alert_window: int,
    alert_quantile: float,
    alert_min_events: int,
    alert_topk_events: int,
    alert_dedupe: bool,
) -> Dict[str, Any]:
    alerts = cluster_rows_into_alerts(
        rows,
        window=int(alert_window),
        score_quantile=float(alert_quantile),
        min_events=int(alert_min_events),
        topk_events=int(alert_topk_events),
        dedupe=bool(alert_dedupe),
    )
    pred = stages_from_alerts(
        alerts,
        use_predicted=use_predicted,
        pred_min_prob=float(pred_min_prob),
        pred_min_count=int(pred_min_count),
        ioc_type_to_stage=ioc_type_to_stage,
        line_to_type=line_to_type,
    )
    scores = recon_scores(pred, obs, gt_order)
    alert_summaries: List[Dict[str, Any]] = []
    for i, edge_rows in enumerate(alerts[:50]):
        counts = stage_edge_counts(
            edge_rows,
            use_predicted=use_predicted,
            pred_min_prob=float(pred_min_prob),
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
        )
        ts = [int(float(r["t"])) for r in edge_rows]
        alert_summaries.append(
            {
                "alert_index": i,
                "num_events": len(edge_rows),
                "t_start": min(ts) if ts else 0,
                "t_end": max(ts) if ts else 0,
                "stage_votes": dict(sorted(counts.items())),
                "stages_included": sorted(
                    [s for s, n in counts.items() if int(n) >= max(1, int(pred_min_count))]
                ),
            }
        )
    return {
        "alert_params": {
            "window": int(alert_window),
            "score_quantile": float(alert_quantile),
            "min_events": int(alert_min_events),
            "topk_events": int(alert_topk_events),
            "dedupe": bool(alert_dedupe),
        },
        "num_alerts": len(alerts),
        **scores,
        "alerts": alert_summaries,
    }
