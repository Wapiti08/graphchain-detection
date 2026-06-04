'''
 # @ Create Time: 2026-04-01 10:45:25
 # @ Modified time: 2026-04-01 10:59:18
 # @ Description:

define the unified format for both data, covering the unified entity reference and event components to
construct (temporal) heterogenous graph

 '''
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, Path.cwd().parent.as_posix())


from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from config.ontology import EdgeType, NodeType


@dataclass(frozen=True)
class EntityRef:
    """
    Canonical reference to an entity node in the unified ontology.

    `key` should be a stable identifier string within a dataset/run
    (e.g., package name, pid, ip:port, file path, syscall name).
    """

    type: NodeType
    key: str


@dataclass(frozen=True)
class Event:
    """
    Canonical event that can be turned into a (temporal) heterogeneous graph edge.

    - `ts` is optional when a source has no wall-clock time.
      Graph builders can use `order` as a stable pseudo-time within a sequence.
    """

    edge_type: EdgeType
    src: EntityRef
    dst: EntityRef
    ts: Optional[float] = None
    order: int = 0
    edge_attrs: Dict[str, Any] = field(default_factory=dict)
    src_attrs: Dict[str, Any] = field(default_factory=dict)
    dst_attrs: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_type": self.edge_type.value,
            "src_type": self.src.type.value,
            "src_key": self.src.key,
            "dst_type": self.dst.type.value,
            "dst_key": self.dst.key,
            "ts": self.ts,
            "order": self.order,
            "edge_attrs": dict(self.edge_attrs),
            "src_attrs": dict(self.src_attrs),
            "dst_attrs": dict(self.dst_attrs),
            "raw": dict(self.raw),
        }


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int,)):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        # some CSVs contain floats as strings
        return int(float(s))
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def parse_csv_list(s: Any) -> list[str]:
    """
    Parse a comma-separated list field from CSV into a list of stripped strings.
    """
    if s is None:
        return []
    txt = str(s).strip()
    if txt == "":
        return []
    return [p.strip() for p in txt.split(",") if p.strip()]

