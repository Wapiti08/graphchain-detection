from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from config.ontology import EdgeType
from parsers.events import Event
from parsers.qut.processed import parse_filetop_row, parse_opensnoop_row, parse_syscall_row


ORDER_BASE: Dict[EdgeType, int] = {
    EdgeType.LOAD: 0,
    EdgeType.EXEC: 1000,
    EdgeType.INVOKE: 2000,
    EdgeType.READ: 3000,
    EdgeType.WRITE: 3000,
}


def _stable_key(ev: Event) -> Tuple:
    """
    Deterministic within-type ordering key to ensure reproducible pseudo-time.
    """
    if ev.edge_type == EdgeType.LOAD:
        return (ev.src.key, ev.dst.key)
    if ev.edge_type == EdgeType.EXEC:
        return (ev.dst.key,)
    if ev.edge_type == EdgeType.INVOKE:
        return (ev.dst.key,)  # syscall name
    if ev.edge_type in (EdgeType.READ, EdgeType.WRITE):
        return (ev.dst.key, ev.edge_type.value)
    return (ev.edge_type.value, ev.src.key, ev.dst.key)


def apply_segmented_order(events: Sequence[Event]) -> List[Event]:
    """
    Apply ontology-driven coarse temporal binning.

    - LOAD:   order starts at 0
    - EXEC:   order starts at 1000
    - INVOKE: order starts at 2000
    - READ/WRITE: order starts at 3000
    """
    # Group and stable sort
    groups: Dict[EdgeType, List[Event]] = {}
    for ev in events:
        groups.setdefault(ev.edge_type, []).append(ev)

    out: List[Event] = []
    for et in sorted(groups.keys(), key=lambda x: ORDER_BASE.get(x, 9000)):
        base = ORDER_BASE.get(et, 9000)
        batch = sorted(groups[et], key=_stable_key)
        for j, ev in enumerate(batch):
            out.append(replace(ev, order=base + j))
    return out


def dedup_load(events: Sequence[Event]) -> List[Event]:
    """
    Keep only one PKG->PROC LOAD edge per (src_key, dst_key).
    """
    seen = set()
    out: List[Event] = []
    for ev in events:
        if ev.edge_type == EdgeType.LOAD:
            k = (ev.src.key, ev.dst.key)
            if k in seen:
                continue
            seen.add(k)
        out.append(ev)
    return out


def parse_qut_joined_package(
    pkg_name: str,
    *,
    df_syscall: pd.DataFrame,
    df_opensnoop: pd.DataFrame,
    df_filetop: pd.DataFrame,
) -> List[Event]:
    """
    Join three processed QUT sources for one package into one comprehensive event list.
    """
    row_s = df_syscall[df_syscall.Package_Name == pkg_name].iloc[0]
    row_o = df_opensnoop[df_opensnoop.Package_Name == pkg_name].iloc[0]
    row_f = df_filetop[df_filetop.Package_Name == pkg_name].iloc[0]

    events: List[Event] = []
    events += parse_syscall_row(row_s)
    events += parse_opensnoop_row(row_o)
    events += parse_filetop_row(row_f)

    events = dedup_load(events)
    events = apply_segmented_order(events)
    return events


def load_qut_processed_dfs(repo_root: str | Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three processed QUT CSVs into DataFrames.
    """
    from config.qut_sources import QUT_SOURCES

    root = repo_root
    ps = pd.read_csv(str((root / QUT_SOURCES["syscall_traces"].rel_path)))
    po = pd.read_csv(str((root / QUT_SOURCES["opensnoop_traces"].rel_path)))
    pf = pd.read_csv(str((root / QUT_SOURCES["filetop_traces"].rel_path)))
    return ps, po, pf

