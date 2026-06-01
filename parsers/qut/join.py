from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from config.ontology import EdgeType
from config.qut_sources import QUT_SOURCE_KEYS
from parsers.events import Event
from parsers.qut.processed import (
    parse_filetop_row,
    parse_install_row,
    parse_opensnoop_row,
    parse_pattern_row,
    parse_syscall_row,
    parse_tcp_row,
)

_ROW_PARSERS = {
    "install_traces": parse_install_row,
    "syscall_traces": parse_syscall_row,
    "opensnoop_traces": parse_opensnoop_row,
    "filetop_traces": parse_filetop_row,
    "tcp_traces": parse_tcp_row,
    "pattern_traces": parse_pattern_row,
}

ORDER_BASE: Dict[EdgeType, int] = {
    EdgeType.LOAD: 0,
    EdgeType.EXEC: 1000,
    EdgeType.INVOKE: 2000,
    EdgeType.CONNECT: 2500,
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
        return (ev.dst.key,)  # syscall / pattern name
    if ev.edge_type == EdgeType.CONNECT:
        return (ev.dst.key,)
    if ev.edge_type in (EdgeType.READ, EdgeType.WRITE):
        return (ev.dst.key, ev.edge_type.value)
    return (ev.edge_type.value, ev.src.key, ev.dst.key)


def apply_segmented_order(events: Sequence[Event]) -> List[Event]:
    """
    Apply ontology-driven coarse temporal binning.

    - LOAD:     order starts at 0
    - EXEC:     order starts at 1000
    - INVOKE:   order starts at 2000
    - CONNECT:  order starts at 2500
    - READ/WRITE: order starts at 3000
    """
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
    """Keep only one PKG->PROC LOAD edge per (src_key, dst_key)."""
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


def _row_for_package(df: pd.DataFrame, pkg_name: str, *, package_col: str = "Package_Name") -> Mapping[str, Any]:
    hits = df[df[package_col] == pkg_name]
    if hits.empty:
        raise KeyError(f"Package {pkg_name!r} not found in {package_col}")
    return hits.iloc[0]


def parse_qut_joined_package(
    pkg_name: str,
    *,
    dfs: Optional[Dict[str, pd.DataFrame]] = None,
    df_install: Optional[pd.DataFrame] = None,
    df_syscall: Optional[pd.DataFrame] = None,
    df_opensnoop: Optional[pd.DataFrame] = None,
    df_filetop: Optional[pd.DataFrame] = None,
    df_tcp: Optional[pd.DataFrame] = None,
    df_pattern: Optional[pd.DataFrame] = None,
) -> List[Event]:
    """
    Join all six processed QUT trace tables for one package into one event list.

    Pass either ``dfs`` (keys = QUT_SOURCE_KEYS) or individual DataFrames (legacy).
    """
    if dfs is None:
        dfs = {}
        if df_install is not None:
            dfs["install_traces"] = df_install
        if df_syscall is not None:
            dfs["syscall_traces"] = df_syscall
        if df_opensnoop is not None:
            dfs["opensnoop_traces"] = df_opensnoop
        if df_filetop is not None:
            dfs["filetop_traces"] = df_filetop
        if df_tcp is not None:
            dfs["tcp_traces"] = df_tcp
        if df_pattern is not None:
            dfs["pattern_traces"] = df_pattern

    events: List[Event] = []
    for key in QUT_SOURCE_KEYS:
        if key not in dfs:
            raise KeyError(f"Missing dataframe for {key!r} in joined parse")
        row = _row_for_package(dfs[key], pkg_name)
        events.extend(_ROW_PARSERS[key](row))

    events = dedup_load(events)
    events = apply_segmented_order(events)
    return events


def load_qut_processed_dfs(
    repo_root: str | Any,
    *,
    limit_per_file: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all six processed QUT CSVs into a dict keyed by QUT_SOURCE_KEYS."""
    from config.qut_sources import QUT_SOURCES

    root = repo_root
    out: Dict[str, pd.DataFrame] = {}
    for key in QUT_SOURCE_KEYS:
        spec = QUT_SOURCES[key]
        df = pd.read_csv(str((root / spec.rel_path)))
        if limit_per_file is not None:
            df = df.head(int(limit_per_file))
        out[key] = df
    return out


def list_qut_package_names(
    repo_root: str | Any,
    *,
    limit_per_file: Optional[int] = None,
    require_all_tables: bool = True,
) -> List[str]:
    """
    Return package names present in the processed QUT CSVs.

    If ``require_all_tables`` is True (default), only names appearing in every
    trace table are returned (safe for ``parse_qut_joined_package``).
    """
    from config.qut_sources import QUT_SOURCES

    dfs = load_qut_processed_dfs(repo_root, limit_per_file=limit_per_file)
    names: Optional[set[str]] = None
    for key in QUT_SOURCE_KEYS:
        col = QUT_SOURCES[key].package_col
        df = dfs[key]
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r} in {QUT_SOURCES[key].rel_path}")
        s = {str(x).strip() for x in df[col].tolist() if str(x).strip()}
        names = s if names is None else (names & s)
    assert names is not None
    return sorted(names)
