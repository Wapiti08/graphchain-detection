"""Resolve which edge labels feed optional ranking / stage aux losses."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from gchain.train.streams import Stream


def rank_supervision_tensor(st: "Stream", mode: str) -> "Optional[torch.Tensor]":
    """
    mode:
      - off / none: no tensor
      - ioc: y_ioc (GT line or value match)
      - ioc_line: y_ioc_line (GT line only)
      - rule: y_rule (any weak-rule hit)
      - rule_high: y_rule_high (high-confidence rules only; recommended for lambda_ioc_rank)
    """
    m = str(mode or "ioc").strip().lower()
    if m in {"off", "none", ""}:
        return None
    if m == "ioc":
        return st.y_ioc
    if m == "ioc_line":
        return st.y_ioc_line
    if m == "rule":
        return st.y_rule
    if m == "rule_high":
        return st.y_rule_high
    raise ValueError(f"Unknown --rank-supervision mode: {mode!r}")
