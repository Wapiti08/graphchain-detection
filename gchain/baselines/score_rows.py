"""Build scored tail rows compatible with detection + reconstruction eval."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from gchain.train.streams import Stream


def tail_score_rows(
    st: Stream,
    *,
    scenario: str,
    tail_start: int,
    scores: Sequence[float],
) -> List[Dict[str, Any]]:
    """Map per-tail-edge scores to eval rows (same schema as train_loop tail CSV)."""
    n_tail = int(st.src.numel()) - int(tail_start)
    if n_tail <= 0:
        return []
    if len(scores) != n_tail:
        raise ValueError(f"expected {n_tail} tail scores, got {len(scores)}")

    rows: List[Dict[str, Any]] = []
    for off, k in enumerate(range(int(tail_start), int(st.src.numel()))):
        y_ioc = 0
        if st.y_ioc is not None:
            y_ioc = int(st.y_ioc[k].item())
        ridx = -1
        if st.row_idx is not None:
            ridx = int(st.row_idx[k].item())
        sf = ""
        if st.source_file is not None:
            sf = str(st.source_file[k])
        ityp = ""
        if st.ioc_type is not None:
            ityp = str(st.ioc_type[k])
        rows.append(
            {
                "scenario": scenario,
                "t": int(st.t[k].item()),
                "etype": int(st.etype[k].item()),
                "src": int(st.src[k].item()),
                "dst": int(st.dst[k].item()),
                "score": float(scores[off]),
                "is_ioc": y_ioc,
                "source_file": sf,
                "row_idx": ridx,
                "ioc_type": ityp,
                "pred_stage": "",
                "pred_stage_prob": "0.0000",
            }
        )
    return rows
