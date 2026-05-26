"""Partial attack-chain reconstruction and stage-level evaluation from tail scores."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from graph.alert_eval import ScoredEvent, dedupe_events, rows_to_events


DEFAULT_STAGE_ORDER: Tuple[str, ...] = (
    "resource_development",
    "initial_access_delivery",
    "execution",
    "persistence_privilege_defense_evasion",
    "discovery_collection",
    "command_and_control",
    "exfiltration_impact",
)

STAGE_LABELS: Tuple[str, ...] = ("none",) + DEFAULT_STAGE_ORDER
NUM_STAGE_CLASSES: int = len(STAGE_LABELS)  # 8: index 0 = none
STAGE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(STAGE_LABELS)}
IDX_TO_STAGE: Dict[int, str] = {i: s for i, s in enumerate(STAGE_LABELS)}


def ioc_type_to_stage_idx(
    ioc_type: str,
    ioc_type_to_stage: Mapping[str, str],
) -> int:
    """Return stage class index (0 = none) for a given IOC type string."""
    if not ioc_type:
        return 0
    stage = ioc_type_to_stage.get(ioc_type, "")
    return STAGE_TO_IDX.get(stage, 0)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ioc_type_to_stage(repo_root: Path, rel: str = "config/ioc_type_to_stage.json") -> Dict[str, str]:
    obj = load_json((repo_root / rel).resolve())
    return dict((obj.get("ioc_type_to_stage") or {}))


def build_line_to_ioc_type(ioc_gt: Mapping[str, Any], scenario: str) -> Dict[Tuple[str, int], str]:
    """Map (source_file, line) -> primary IOC type from ground-truth JSON."""
    sc = ioc_gt.get(scenario) or {}
    files = (sc.get("files") if isinstance(sc, dict) else None) or {}
    out: Dict[Tuple[str, int], str] = {}
    for fname, fobj in files.items():
        for ioc in (fobj.get("iocs") if isinstance(fobj, dict) else None) or []:
            if not isinstance(ioc, dict):
                continue
            typ = str(ioc.get("type") or "").strip()
            if not typ:
                continue
            for ln in (ioc.get("lines") or []):
                try:
                    out[(str(fname), int(ln))] = typ
                except Exception:
                    continue
    return out


def stage_for_edge(
    row: Mapping[str, Any],
    *,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
) -> str:
    """Resolve coarse stage for one scored edge (empty if unknown)."""
    it = str(row.get("ioc_type") or "").strip()
    if not it:
        sf = str(row.get("source_file") or "").strip()
        try:
            ridx = int(row.get("row_idx", -1))
        except Exception:
            ridx = -1
        if sf and ridx >= 0:
            it = line_to_type.get((sf, ridx), "") or line_to_type.get((sf, ridx + 1), "")
    if not it:
        return ""
    return str(ioc_type_to_stage.get(it, ""))


def topk_edges(rows: Iterable[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    evs = dedupe_events(rows_to_events(rows))
    evs.sort(key=lambda e: e.score, reverse=True)
    kk = max(0, int(k))
    top = evs[:kk]
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
    # restore metadata from original rows by key
    by_key = {
        (str(r["scenario"]), int(r["t"]), int(r["etype"]), int(r["src"]), int(r["dst"])): r
        for r in rows
    }
    for i, r in enumerate(out):
        k2 = (r["scenario"], r["t"], r["etype"], r["src"], r["dst"])
        src = by_key.get(k2)
        if src:
            out[i] = dict(src)
    return out


def stage_for_edge_predicted(row: Mapping[str, Any]) -> str:
    """Return model-predicted stage from the pred_stage CSV column (empty if absent/none)."""
    ps = str(row.get("pred_stage") or "").strip()
    if ps and ps != "none" and ps in STAGE_TO_IDX:
        return ps
    return ""


def stages_from_topk(
    rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    ioc_only: bool = True,
    use_predicted: bool = False,
) -> Set[str]:
    """Collect predicted stages from top-K edges.

    When *use_predicted* is True, use the model's ``pred_stage`` column
    for **all** edges (not just IOC).  Otherwise fall back to rule-based
    lookup via ``stage_for_edge``.
    """
    pred: Set[str] = set()
    for r in topk_edges(rows, k):
        if use_predicted:
            st = stage_for_edge_predicted(r)
        else:
            if ioc_only and int(r.get("is_ioc", 0)) != 1:
                continue
            st = stage_for_edge(r, ioc_type_to_stage=ioc_type_to_stage, line_to_type=line_to_type)
        if st:
            pred.add(st)
    return pred


def ordered_stage_sequence(stages: Set[str], order: Sequence[str] = DEFAULT_STAGE_ORDER) -> List[str]:
    '''
    organize recognized stages in logical order
    '''
    return [s for s in order if s in stages]


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    ''' calculate longest common subsequence for recall computation
    
    '''
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return int(dp[n][m])


def build_chain_segments(
    top_rows: Sequence[Dict[str, Any]],
    *,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    use_predicted: bool = False,
) -> List[Dict[str, Any]]:
    """Time-ordered stage segments from edges in top-K."""
    labeled: List[Tuple[int, str, Dict[str, Any]]] = []
    for r in top_rows:
        if use_predicted:
            st = stage_for_edge_predicted(r)
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


def _eval_one_mode(
    rows: Sequence[Dict[str, Any]],
    *,
    obs: Set[str],
    gt_order: List[str],
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
    topks: Sequence[int],
    use_predicted: bool,
) -> Dict[str, Any]:
    by_k: Dict[str, Any] = {}
    for k in topks:
        pred = stages_from_topk(
            rows,
            k=int(k),
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            ioc_only=not use_predicted,
            use_predicted=use_predicted,
        )
        pred_order = ordered_stage_sequence(pred)
        inter = pred & obs
        recall = float(len(inter)) / float(max(1, len(obs)))
        precision = float(len(inter)) / float(max(1, len(pred))) if pred else 0.0
        lcs_val = lcs_length(pred_order, gt_order)
        ordered_recall = float(lcs_val) / float(max(1, len(gt_order)))
        top_rows = topk_edges(rows, int(k))
        by_k[str(int(k))] = {
            "predicted_stages": sorted(pred),
            "stage_recall": recall,
            "stage_precision": precision,
            "ordered_stage_recall_lcs": ordered_recall,
            "lcs_length": int(lcs_val),
            "chain_segments": build_chain_segments(
                top_rows,
                ioc_type_to_stage=ioc_type_to_stage,
                line_to_type=line_to_type,
                use_predicted=use_predicted,
            ),
        }
    return by_k


def _has_pred_stage(rows: Sequence[Dict[str, Any]]) -> bool:
    """Check if any row has a non-empty pred_stage value."""
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
) -> Dict[str, Any]:
    obs = set((stages_gt.get("observable") or {}).get("stages") or [])
    sem = set((stages_gt.get("semantic") or {}).get("stages") or [])
    gt_order = ordered_stage_sequence(obs)

    metrics: Dict[str, Any] = {
        "observable_stages": sorted(obs),
        "semantic_stages": sorted(sem),
        "unobserved_semantic": sorted(sem - obs),
        "by_k": _eval_one_mode(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            use_predicted=False,
        ),
    }

    if _has_pred_stage(rows):
        metrics["by_k_predicted"] = _eval_one_mode(
            rows,
            obs=obs,
            gt_order=gt_order,
            ioc_type_to_stage=ioc_type_to_stage,
            line_to_type=line_to_type,
            topks=topks,
            use_predicted=True,
        )

    return metrics
