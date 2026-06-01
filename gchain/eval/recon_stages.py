"""Stage ontology, IOC/GT labeling, and reconstruction score metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from graphcore.edge_meta import pick_primary_ioc_type

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
NUM_STAGE_CLASSES: int = len(STAGE_LABELS)
STAGE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(STAGE_LABELS)}
IDX_TO_STAGE: Dict[int, str] = {i: s for i, s in enumerate(STAGE_LABELS)}


def ioc_type_to_stage_idx(ioc_type: str, ioc_type_to_stage: Mapping[str, str]) -> int:
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
                    key = (str(fname), int(ln))
                except Exception:
                    continue
                prev = out.get(key)
                out[key] = pick_primary_ioc_type([prev, typ]) if prev else typ
    return out


def line_ioc_type_from_row(row: Mapping[str, Any], line_to_type: Mapping[Tuple[str, int], str]) -> str:
    sf = str(row.get("source_file") or "").strip()
    try:
        ridx = int(row.get("row_idx", -1))
    except Exception:
        ridx = -1
    if sf and ridx >= 0:
        return line_to_type.get((sf, ridx), "") or line_to_type.get((sf, ridx + 1), "")
    return ""


def stage_for_edge(
    row: Mapping[str, Any],
    *,
    ioc_type_to_stage: Mapping[str, str],
    line_to_type: Mapping[Tuple[str, int], str],
) -> str:
    it = line_ioc_type_from_row(row, line_to_type)
    if not it:
        it = str(row.get("ioc_type") or "").strip()
    if not it:
        return ""
    return str(ioc_type_to_stage.get(it, ""))


def stage_for_edge_predicted(row: Mapping[str, Any], *, min_prob: float = 0.0) -> str:
    ps = str(row.get("pred_stage") or "").strip()
    if float(min_prob) > 0.0:
        try:
            p = float(row.get("pred_stage_prob") or 0.0)
        except Exception:
            p = 0.0
        if p < float(min_prob):
            return ""
    if ps and ps != "none" and ps in STAGE_TO_IDX:
        return ps
    return ""


def ordered_stage_sequence(stages: Set[str], order: Sequence[str] = DEFAULT_STAGE_ORDER) -> List[str]:
    return [s for s in order if s in stages]


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
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


def recon_scores(pred: Set[str], obs: Set[str], gt_order: List[str]) -> Dict[str, Any]:
    pred_order = ordered_stage_sequence(pred)
    inter = pred & obs
    recall = float(len(inter)) / float(max(1, len(obs)))
    precision = float(len(inter)) / float(max(1, len(pred))) if pred else 0.0
    lcs_val = lcs_length(pred_order, gt_order)
    ordered_recall = float(lcs_val) / float(max(1, len(gt_order)))
    return {
        "predicted_stages": sorted(pred),
        "stage_recall": recall,
        "stage_precision": precision,
        "ordered_stage_recall_lcs": ordered_recall,
        "lcs_length": int(lcs_val),
    }
