"""Top-K edge ranking and stage collection for chain reconstruction."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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


def select_topk_source_quota(
    rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    source_file_min_quota: Mapping[str, int],
    endpoint_pair_dedupe: bool = True,
    exclude_etypes: Optional[Set[int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select top-K edges with per-source_file minimum quotas (eval-only).

    Algorithm:
    - Build a score-desc pool (optionally pair-deduped).
    - Optionally exclude certain etype ids.
    - First, satisfy each source_file's quota using its highest-score edges.
    - Then, fill remaining slots by global score desc.
    """
    kk = max(0, int(k))
    quotas: Dict[str, int] = {}
    for sf, n in dict(source_file_min_quota or {}).items():
        sf2 = str(sf).strip()
        nn = int(n)
        if sf2 and nn > 0:
            quotas[sf2] = nn
    ex = set(int(x) for x in (exclude_etypes or set()))

    if endpoint_pair_dedupe:
        pool, pool_meta = dedupe_rows_by_endpoint_pair(rows)
        pool.sort(key=lambda r: float(r["score"]), reverse=True)
    else:
        pool = sorted([dict(r) for r in rows], key=lambda r: float(r["score"]), reverse=True)
        pool_meta = {"n_input_rows": int(len(rows)), "n_pair_deduped_rows": None, "n_collapsed_rows": None}

    # Apply etype exclusion up front.
    if ex:
        pool2: List[Dict[str, Any]] = []
        for r in pool:
            try:
                if int(r.get("etype")) in ex:
                    continue
            except Exception:
                continue
            pool2.append(r)
        pool = pool2

    selected: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, int, int, int, int]] = set()

    def _key(r: Mapping[str, Any]) -> Tuple[str, int, int, int, int]:
        return (
            str(r.get("scenario")),
            int(float(r.get("t", 0))),
            int(r.get("etype", 0)),
            int(r.get("src", 0)),
            int(r.get("dst", 0)),
        )

    per_source_selected: Dict[str, int] = {sf: 0 for sf in quotas}

    # Pass 1: satisfy per-source quotas.
    for sf, need in quotas.items():
        if len(selected) >= kk:
            break
        got = 0
        for r in pool:
            if len(selected) >= kk or got >= need:
                break
            if str(r.get("source_file") or "").strip() != sf:
                continue
            try:
                k2 = _key(r)
            except Exception:
                continue
            if k2 in seen_keys:
                continue
            selected.append(dict(r))
            seen_keys.add(k2)
            got += 1
        per_source_selected[sf] = got

    # Pass 2: fill remaining by global score.
    for r in pool:
        if len(selected) >= kk:
            break
        try:
            k2 = _key(r)
        except Exception:
            continue
        if k2 in seen_keys:
            continue
        selected.append(dict(r))
        seen_keys.add(k2)

    meta = {
        **pool_meta,
        "k": int(kk),
        "endpoint_pair_dedupe": bool(endpoint_pair_dedupe),
        "exclude_etypes": sorted(ex),
        "source_file_min_quota": {k: int(v) for k, v in sorted(quotas.items())},
        "source_file_selected": {k: int(v) for k, v in sorted(per_source_selected.items())},
        "n_selected": int(len(selected)),
    }
    return selected, meta


def select_topk_group_cap(
    rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    cap_key: str = "etype_dst",
    cap_max: int = 5,
    endpoint_pair_dedupe: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select top-K edges with per-group maximum counts (eval-only).

    This is a softer alternative to strict pair dedupe: it reduces "hub" patterns dominating
    top-K, while still allowing multiple edges per group.

    Supported cap_key:
      - "etype_dst": group by (etype, dst)
      - "etype_src": group by (etype, src)
      - "etype_dst_port": group by (etype, dst, dst_port)
      - "etype_src_port": group by (etype, src, src_port)
    """
    kk = max(0, int(k))
    mm = max(1, int(cap_max))

    if endpoint_pair_dedupe:
        pool, pool_meta = dedupe_rows_by_endpoint_pair(rows)
        pool.sort(key=lambda r: float(r["score"]), reverse=True)
    else:
        pool = sorted([dict(r) for r in rows], key=lambda r: float(r["score"]), reverse=True)
        pool_meta = {"n_input_rows": int(len(rows)), "n_pair_deduped_rows": None, "n_collapsed_rows": None}

    def _group(r: Mapping[str, Any]) -> Tuple[Any, ...]:
        et = int(r.get("etype", 0))
        if cap_key == "etype_dst":
            return (et, int(r.get("dst", 0)))
        if cap_key == "etype_src":
            return (et, int(r.get("src", 0)))
        if cap_key == "etype_dst_port":
            # Prefer explicit port if present, otherwise fall back to dst key.
            try:
                port = int(r.get("dst_port", r.get("port", -1)))
            except Exception:
                port = -1
            return (et, int(r.get("dst", 0)), port)
        if cap_key == "etype_src_port":
            try:
                port = int(r.get("src_port", -1))
            except Exception:
                port = -1
            return (et, int(r.get("src", 0)), port)
        # default: etype_dst
        return (et, int(r.get("dst", 0)))

    selected: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, int, int, int, int]] = set()
    group_counts: Dict[Tuple[Any, ...], int] = {}

    def _key(r: Mapping[str, Any]) -> Tuple[str, int, int, int, int]:
        return (
            str(r.get("scenario")),
            int(float(r.get("t", 0))),
            int(r.get("etype", 0)),
            int(r.get("src", 0)),
            int(r.get("dst", 0)),
        )

    for r in pool:
        if len(selected) >= kk:
            break
        try:
            k2 = _key(r)
        except Exception:
            continue
        if k2 in seen_keys:
            continue
        g = _group(r)
        if int(group_counts.get(g, 0)) >= mm:
            continue
        selected.append(dict(r))
        seen_keys.add(k2)
        group_counts[g] = int(group_counts.get(g, 0)) + 1

    meta = {
        **pool_meta,
        "k": int(kk),
        "endpoint_pair_dedupe": bool(endpoint_pair_dedupe),
        "cap_key": str(cap_key),
        "cap_max": int(mm),
        "n_selected": int(len(selected)),
        "n_groups": int(len(group_counts)),
    }
    return selected, meta


def eval_topk_group_cap(
    rows: Sequence[Dict[str, Any]],
    *,
    obs: Set[str],
    gt_order: List[str],
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    topks: Sequence[int],
    cap_key: str,
    cap_max: int,
    endpoint_pair_dedupe: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out: Dict[str, Any] = {}
    last_meta: Dict[str, Any] = {}
    for k in topks:
        selected, meta = select_topk_group_cap(
            rows,
            k=int(k),
            cap_key=str(cap_key),
            cap_max=int(cap_max),
            endpoint_pair_dedupe=bool(endpoint_pair_dedupe),
        )
        last_meta = meta
        counts: Dict[str, int] = {}
        for r in selected:
            if int(r.get("is_ioc", 0)) != 1:
                continue
            st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
            if st:
                counts[st] = counts.get(st, 0) + 1
        pred = set(counts.keys())
        pred_order = ordered_stage_sequence(pred)
        inter = pred & obs
        recall = float(len(inter)) / float(max(1, len(obs)))
        precision = float(len(inter)) / float(max(1, len(pred))) if pred else 0.0
        lcs_val = lcs_length(pred_order, gt_order)
        out[str(int(k))] = {
            "predicted_stages": sorted(pred),
            "stage_recall": recall,
            "stage_precision": precision,
            "ordered_stage_recall_lcs": float(lcs_val) / float(max(1, len(gt_order))),
            "lcs_length": int(lcs_val),
            "chain_segments": build_chain_segments(
                selected,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=False,
                pred_min_prob=0.0,
            ),
        }
    return out, last_meta


def select_topk_group_cap_adaptive(
    rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    cap_key: str = "etype_src",
    probe_mult: int = 5,
    hot_threshold: int = 20,
    hot_cap_max: int = 10,
    endpoint_pair_dedupe: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Adaptive per-group cap: only cap groups that dominate the high-score head.

    - Probe head size K' = probe_mult * K (clipped to pool size)
    - Mark groups with count >= hot_threshold as hot
    - Apply cap (hot_cap_max) only to hot groups when selecting top-K
    """
    kk = max(0, int(k))
    pm = max(1, int(probe_mult))
    th = max(1, int(hot_threshold))
    cap = max(1, int(hot_cap_max))

    if endpoint_pair_dedupe:
        pool, pool_meta = dedupe_rows_by_endpoint_pair(rows)
        pool.sort(key=lambda r: float(r["score"]), reverse=True)
    else:
        pool = sorted([dict(r) for r in rows], key=lambda r: float(r["score"]), reverse=True)
        pool_meta = {"n_input_rows": int(len(rows)), "n_pair_deduped_rows": None, "n_collapsed_rows": None}

    def _group(r: Mapping[str, Any]) -> Tuple[Any, ...]:
        et = int(r.get("etype", 0))
        if cap_key == "etype_dst":
            return (et, int(r.get("dst", 0)))
        if cap_key == "etype_src":
            return (et, int(r.get("src", 0)))
        if cap_key == "etype_dst_port":
            try:
                port = int(r.get("dst_port", r.get("port", -1)))
            except Exception:
                port = -1
            return (et, int(r.get("dst", 0)), port)
        if cap_key == "etype_src_port":
            try:
                port = int(r.get("src_port", -1))
            except Exception:
                port = -1
            return (et, int(r.get("src", 0)), port)
        return (et, int(r.get("src", 0)))

    probe_n = min(len(pool), kk * pm) if kk > 0 else 0
    group_probe_counts: Dict[Tuple[Any, ...], int] = {}
    for r in pool[:probe_n]:
        g = _group(r)
        group_probe_counts[g] = int(group_probe_counts.get(g, 0)) + 1
    hot_groups = {g for g, c in group_probe_counts.items() if int(c) >= th}

    selected: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, int, int, int, int]] = set()
    hot_used: Dict[Tuple[Any, ...], int] = {}

    def _key(r: Mapping[str, Any]) -> Tuple[str, int, int, int, int]:
        return (
            str(r.get("scenario")),
            int(float(r.get("t", 0))),
            int(r.get("etype", 0)),
            int(r.get("src", 0)),
            int(r.get("dst", 0)),
        )

    for r in pool:
        if len(selected) >= kk:
            break
        try:
            k2 = _key(r)
        except Exception:
            continue
        if k2 in seen_keys:
            continue
        g = _group(r)
        if g in hot_groups and int(hot_used.get(g, 0)) >= cap:
            continue
        selected.append(dict(r))
        seen_keys.add(k2)
        if g in hot_groups:
            hot_used[g] = int(hot_used.get(g, 0)) + 1

    meta = {
        **pool_meta,
        "k": int(kk),
        "endpoint_pair_dedupe": bool(endpoint_pair_dedupe),
        "cap_key": str(cap_key),
        "probe_mult": int(pm),
        "probe_n": int(probe_n),
        "hot_threshold": int(th),
        "hot_cap_max": int(cap),
        "n_hot_groups": int(len(hot_groups)),
        "n_selected": int(len(selected)),
    }
    return selected, meta


def eval_topk_group_cap_adaptive(
    rows: Sequence[Dict[str, Any]],
    *,
    obs: Set[str],
    gt_order: List[str],
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    topks: Sequence[int],
    cap_key: str,
    probe_mult: int,
    hot_threshold: int,
    hot_cap_max: int,
    endpoint_pair_dedupe: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out: Dict[str, Any] = {}
    last_meta: Dict[str, Any] = {}
    for k in topks:
        selected, meta = select_topk_group_cap_adaptive(
            rows,
            k=int(k),
            cap_key=str(cap_key),
            probe_mult=int(probe_mult),
            hot_threshold=int(hot_threshold),
            hot_cap_max=int(hot_cap_max),
            endpoint_pair_dedupe=bool(endpoint_pair_dedupe),
        )
        last_meta = meta
        counts: Dict[str, int] = {}
        for r in selected:
            if int(r.get("is_ioc", 0)) != 1:
                continue
            st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
            if st:
                counts[st] = counts.get(st, 0) + 1
        pred = set(counts.keys())
        pred_order = ordered_stage_sequence(pred)
        inter = pred & obs
        recall = float(len(inter)) / float(max(1, len(obs)))
        precision = float(len(inter)) / float(max(1, len(pred))) if pred else 0.0
        lcs_val = lcs_length(pred_order, gt_order)
        out[str(int(k))] = {
            "predicted_stages": sorted(pred),
            "stage_recall": recall,
            "stage_precision": precision,
            "ordered_stage_recall_lcs": float(lcs_val) / float(max(1, len(gt_order))),
            "lcs_length": int(lcs_val),
            "chain_segments": build_chain_segments(
                selected,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=False,
                pred_min_prob=0.0,
            ),
        }
    return out, last_meta


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


def eval_topk_source_quota(
    rows: Sequence[Dict[str, Any]],
    *,
    obs: Set[str],
    gt_order: List[str],
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    topks: Sequence[int],
    source_file_min_quota: Mapping[str, int],
    exclude_etypes: Optional[Set[int]] = None,
    endpoint_pair_dedupe: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out: Dict[str, Any] = {}
    last_meta: Dict[str, Any] = {}
    for k in topks:
        selected, meta = select_topk_source_quota(
            rows,
            k=int(k),
            source_file_min_quota=source_file_min_quota,
            endpoint_pair_dedupe=bool(endpoint_pair_dedupe),
            exclude_etypes=exclude_etypes,
        )
        last_meta = meta

        counts: Dict[str, int] = {}
        for r in selected:
            if int(r.get("is_ioc", 0)) != 1:
                continue
            st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
            if st:
                counts[st] = counts.get(st, 0) + 1
        pred = set(counts.keys())
        pred_order = ordered_stage_sequence(pred)
        inter = pred & obs
        recall = float(len(inter)) / float(max(1, len(obs)))
        precision = float(len(inter)) / float(max(1, len(pred))) if pred else 0.0
        lcs_val = lcs_length(pred_order, gt_order)
        out[str(int(k))] = {
            "predicted_stages": sorted(pred),
            "stage_recall": recall,
            "stage_precision": precision,
            "ordered_stage_recall_lcs": float(lcs_val) / float(max(1, len(gt_order))),
            "lcs_length": int(lcs_val),
            "chain_segments": build_chain_segments(
                selected,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=False,
                pred_min_prob=0.0,
            ),
        }
    return out, last_meta
