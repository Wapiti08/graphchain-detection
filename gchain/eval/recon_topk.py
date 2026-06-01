"""Top-K edge ranking and stage collection for chain reconstruction."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from gchain.eval.alert_eval import dedupe_events, rows_to_events
from gchain.eval.recon_stages import (
    lcs_length,
    ordered_stage_sequence,
    stage_for_edge,
    stage_for_edge_predicted,
)


def rows_by_event_key(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, int, int, int, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int, int, int, int], Dict[str, Any]] = {}
    for r in rows:
        try:
            key = (
                str(r["scenario"]),
                int(float(r["t"])),
                int(r["etype"]),
                int(r["src"]),
                int(r["dst"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = dict(r)
    return out


def dedupe_rows_by_endpoint_pair(
    rows: Iterable[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    best: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    n_in = 0
    for r in rows:
        n_in += 1
        try:
            key = (str(r["scenario"]), int(r["etype"]), int(r["src"]), int(r["dst"]))
            score = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
        prev = best.get(key)
        if prev is None:
            best[key] = dict(r)
            continue
        prev_score = float(prev["score"])
        if score > prev_score or (
            score == prev_score and int(r.get("is_ioc", 0)) > int(prev.get("is_ioc", 0))
        ):
            best[key] = dict(r)
    kept = list(best.values())
    return kept, {
        "n_input_rows": int(n_in),
        "n_pair_deduped_rows": int(len(kept)),
        "n_collapsed_rows": int(n_in - len(kept)),
    }


def topk_edges(
    rows: Iterable[Dict[str, Any]],
    k: int,
    *,
    endpoint_pair_dedupe: bool = False,
) -> List[Dict[str, Any]]:
    if endpoint_pair_dedupe:
        pool, _ = dedupe_rows_by_endpoint_pair(rows)
        pool.sort(key=lambda r: float(r["score"]), reverse=True)
        return [dict(r) for r in pool[: max(0, int(k))]]

    evs = dedupe_events(rows_to_events(rows))
    evs.sort(key=lambda e: e.score, reverse=True)
    top = evs[: max(0, int(k))]
    out: List[Dict[str, Any]] = []
    for e in top:
        out.append(
            {
                "scenario": e.scenario,
                "t": e.t,
                "etype": e.etype,
                "src": e.src,
                "dst": e.dst,
                "score": e.score,
                "is_ioc": e.is_ioc,
                "source_file": "",
                "row_idx": -1,
                "ioc_type": "",
            }
        )
    by_key = rows_by_event_key(rows)
    for i, r in enumerate(out):
        k2 = (r["scenario"], r["t"], r["etype"], r["src"], r["dst"])
        src = by_key.get(k2)
        if src:
            out[i] = dict(src)
    return out


def topk_ioc_edges(
    rows: Iterable[Dict[str, Any]],
    k: int,
    *,
    endpoint_pair_dedupe: bool = False,
) -> List[Dict[str, Any]]:
    ioc_rows = [dict(r) for r in rows if int(r.get("is_ioc", 0)) == 1]
    return topk_edges(ioc_rows, k, endpoint_pair_dedupe=endpoint_pair_dedupe)


def stages_from_topk(
    rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    ioc_only: bool = True,
    use_predicted: bool = False,
    pred_min_prob: float = 0.0,
    pred_min_count: int = 1,
    endpoint_pair_dedupe: bool = False,
) -> Set[str]:
    counts: Dict[str, int] = {}
    for r in topk_edges(rows, k, endpoint_pair_dedupe=endpoint_pair_dedupe):
        if use_predicted:
            st = stage_for_edge_predicted(r, min_prob=float(pred_min_prob))
        else:
            if ioc_only and int(r.get("is_ioc", 0)) != 1:
                continue
            st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
        if st:
            counts[st] = counts.get(st, 0) + 1
    minc = max(1, int(pred_min_count))
    return {s for s, c in counts.items() if int(c) >= minc}


def stages_from_ioc_topk(
    rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    pred_min_count: int = 1,
    endpoint_pair_dedupe: bool = False,
) -> Set[str]:
    counts: Dict[str, int] = {}
    for r in topk_ioc_edges(rows, k, endpoint_pair_dedupe=endpoint_pair_dedupe):
        st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
        if st:
            counts[st] = counts.get(st, 0) + 1
    minc = max(1, int(pred_min_count))
    return {s for s, c in counts.items() if int(c) >= minc}


def build_chain_segments(
    top_rows: Sequence[Dict[str, Any]],
    *,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    use_predicted: bool = False,
    pred_min_prob: float = 0.0,
) -> List[Dict[str, Any]]:
    labeled: List[Tuple[int, str, Dict[str, Any]]] = []
    for r in top_rows:
        if use_predicted:
            st = stage_for_edge_predicted(r, min_prob=float(pred_min_prob))
        else:
            if int(r.get("is_ioc", 0)) != 1:
                continue
            st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
        if not st:
            continue
        labeled.append((int(r["t"]), st, dict(r)))
    labeled.sort(key=lambda x: x[0])
    if not labeled:
        return []
    segments: List[Dict[str, Any]] = []
    cur_stage = labeled[0][1]
    t_min = labeled[0][0]
    t_max = labeled[0][0]
    n_edges = 1
    for t, st, _ in labeled[1:]:
        if st == cur_stage:
            t_max = t
            n_edges += 1
        else:
            segments.append({"stage": cur_stage, "t_min": t_min, "t_max": t_max, "n_edges": n_edges})
            cur_stage, t_min, t_max, n_edges = st, t, t, 1
    segments.append({"stage": cur_stage, "t_min": t_min, "t_max": t_max, "n_edges": n_edges})
    return segments


def eval_topk_modes(
    rows: Sequence[Dict[str, Any]],
    *,
    obs: Set[str],
    gt_order: List[str],
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    topks: Sequence[int],
    use_predicted: bool,
    pred_min_prob: float,
    pred_min_count: int,
    ioc_ranked: bool = False,
    endpoint_pair_dedupe: bool = False,
) -> Dict[str, Any]:
    by_k: Dict[str, Any] = {}
    for k in topks:
        if use_predicted:
            pred = stages_from_topk(
                rows,
                k=int(k),
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                ioc_only=False,
                use_predicted=True,
                pred_min_prob=float(pred_min_prob),
                pred_min_count=int(pred_min_count),
                endpoint_pair_dedupe=endpoint_pair_dedupe,
            )
            top_rows = topk_edges(rows, int(k), endpoint_pair_dedupe=endpoint_pair_dedupe)
            segments = build_chain_segments(
                top_rows,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=True,
                pred_min_prob=float(pred_min_prob),
            )
        elif ioc_ranked:
            pred = stages_from_ioc_topk(
                rows,
                k=int(k),
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                pred_min_count=int(pred_min_count),
                endpoint_pair_dedupe=endpoint_pair_dedupe,
            )
            top_rows = topk_ioc_edges(rows, int(k), endpoint_pair_dedupe=endpoint_pair_dedupe)
            segments = build_chain_segments(
                top_rows,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=False,
            )
        else:
            pred = stages_from_topk(
                rows,
                k=int(k),
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                ioc_only=True,
                use_predicted=False,
                pred_min_count=int(pred_min_count),
                endpoint_pair_dedupe=endpoint_pair_dedupe,
            )
            top_rows = topk_edges(rows, int(k), endpoint_pair_dedupe=endpoint_pair_dedupe)
            segments = build_chain_segments(
                top_rows,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=False,
            )
        pred_order = ordered_stage_sequence(pred)
        inter = pred & obs
        recall = float(len(inter)) / float(max(1, len(obs)))
        precision = float(len(inter)) / float(max(1, len(pred))) if pred else 0.0
        lcs_val = lcs_length(pred_order, gt_order)
        by_k[str(int(k))] = {
            "predicted_stages": sorted(pred),
            "stage_recall": recall,
            "stage_precision": precision,
            "ordered_stage_recall_lcs": float(lcs_val) / float(max(1, len(gt_order))),
            "lcs_length": int(lcs_val),
            "chain_segments": segments,
        }
    return by_k
