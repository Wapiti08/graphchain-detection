'''
 # @ Create Time: 2026-04-01 10:46:03
 # @ Modified time: 2026-04-01 11:05:37
 # @ Description:
 '''

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, Path.cwd().parent.parent.as_posix())


from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from config.ontology import EdgeType, NodeType, canonical_edge_attrs, fill_defaults
from parsers.events import EntityRef, Event, parse_csv_list, safe_int


def _pkg_entity(pkg_name: str) -> EntityRef:
    return EntityRef(NodeType.PKG, pkg_name)


def _install_proc_entity(pkg_name: str) -> EntityRef:
    # A stable "installation process" placeholder for package-level aggregates.
    return EntityRef(NodeType.PROC, f"{pkg_name}::install")


def parse_syscall_row(row: Mapping[str, Any]) -> List[Event]:
    """
    Parse one row from QUT SysCall Traces (processed aggregate) into Events.

    Produces:
    - LOAD: PKG -> PROC (entry_point="install")
    - INVOKE: PROC -> SYSCALL for each syscall in Unique_System_Calls_List
      (adds superset attr `count_total` / `count_unique` for downstream feature use)
    """
    pkg = str(row.get("Package_Name", "")).strip()
    if not pkg:
        return []

    pkg_ent = _pkg_entity(pkg)
    proc_ent = _install_proc_entity(pkg)

    events: List[Event] = []
    order = 0

    # PKG -> PROC
    load_attrs = fill_defaults(
        canonical_edge_attrs(EdgeType.LOAD),
        {"entry_point": "install"},
    )
    events.append(
        Event(
            edge_type=EdgeType.LOAD,
            src=pkg_ent,
            dst=proc_ent,
            order=order,
            edge_attrs=load_attrs,
            raw=dict(row),
        )
    )
    order += 1

    total_syscalls = safe_int(row.get("Total_System_Calls"))
    unique_syscalls = safe_int(row.get("Unique_System_Calls"))

    syscall_list = parse_csv_list(row.get("Unique_System_Calls_List"))
    for sc in syscall_list:
        dst = EntityRef(NodeType.SYSCALL, sc)
        invoke_attrs = fill_defaults(
            canonical_edge_attrs(EdgeType.INVOKE),
            {
                "args": "",
                "return_val": 0,
                # superset attrs (not in ontology but allowed in parsing stage)
                "count_total": total_syscalls,
                "count_unique": unique_syscalls,
            },
        )
        events.append(
            Event(
                edge_type=EdgeType.INVOKE,
                src=proc_ent,
                dst=dst,
                order=order,
                edge_attrs=invoke_attrs,
                raw={"package": pkg, "syscall": sc},
            )
        )
        order += 1

    return events


def parse_opensnoop_row(row: Mapping[str, Any]) -> List[Event]:
    """
    Parse one row from QUT Opensnoop Traces (processed aggregate) into Events.

    Since we don't have concrete file paths, we create coarse FILE "bucket" nodes
    representing directory categories used by the dataset features.

    Produces:
    - LOAD: PKG -> PROC
    - WRITE: PROC -> FILE(bucket) with bytes approximated by counts if available
    """
    pkg = str(row.get("Package_Name", "")).strip()
    if not pkg:
        return []

    pkg_ent = _pkg_entity(pkg)
    proc_ent = _install_proc_entity(pkg)

    events: List[Event] = []
    order = 0

    events.append(
        Event(
            edge_type=EdgeType.LOAD,
            src=pkg_ent,
            dst=proc_ent,
            order=order,
            edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.LOAD), {"entry_point": "install"}),
            raw=dict(row),
        )
    )
    order += 1

    # Directory buckets (counts of "installation" paths)
    buckets = {
        "ROOT_DIR": safe_int(row.get("Root_DIR_Installation")),
        "TMP_DIR": safe_int(row.get("Temporary_DIR_Installation")),
        "HOME_DIR": safe_int(row.get("Home_DIR_Installation")),
        "ETC_DIR": safe_int(row.get("Etc_DIR_Installation")),
        "OTHER_DIR": safe_int(row.get("Other_DIR_Installation")),
    }

    for bucket, cnt in buckets.items():
        if cnt <= 0:
            continue
        file_ent = EntityRef(NodeType.FILE, f"{pkg}::bucket::{bucket}")
        # We store the count into the canonical `bytes` slot as a proxy signal.
        # Downstream, you can rename/encode it properly during feature projection.
        write_attrs = fill_defaults(
            canonical_edge_attrs(EdgeType.WRITE),
            {"bytes": cnt, "count_paths": cnt},  # superset `count_paths`
        )
        events.append(
            Event(
                edge_type=EdgeType.WRITE,
                src=proc_ent,
                dst=file_ent,
                order=order,
                edge_attrs=write_attrs,
                dst_attrs={
                    # map directory category into FILE attrs (superset + canonical)
                    "file_type": "DIR_BUCKET",
                    "path_sensitivity": 1 if bucket in {"ROOT_DIR", "ETC_DIR"} else 0,
                    "bucket": bucket,
                },
                raw={"package": pkg, "bucket": bucket, "count": cnt},
            )
        )
        order += 1

    return events


def parse_filetop_row(row: Mapping[str, Any]) -> List[Event]:
    """
    Parse one row from QUT Filetop Traces (processed aggregate) into Events.

    Produces:
    - LOAD: PKG -> PROC
    - READ/WRITE: PROC -> FILE(bucket=ALL_FILES) with bytes from transfer columns
    - EXEC: PROC -> PROC for the top read/write processes lists (as subprocesses)
    """
    pkg = str(row.get("Package_Name", "")).strip()
    if not pkg:
        return []

    pkg_ent = _pkg_entity(pkg)
    proc_ent = _install_proc_entity(pkg)

    events: List[Event] = []
    order = 0

    events.append(
        Event(
            edge_type=EdgeType.LOAD,
            src=pkg_ent,
            dst=proc_ent,
            order=order,
            edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.LOAD), {"entry_point": "install"}),
            raw=dict(row),
        )
    )
    order += 1

    file_ent = EntityRef(NodeType.FILE, f"{pkg}::bucket::ALL_FILES")

    read_bytes = safe_int(row.get("Total_Read_Data_Transfer"))
    if read_bytes > 0:
        events.append(
            Event(
                edge_type=EdgeType.READ,
                src=proc_ent,
                dst=file_ent,
                order=order,
                edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.READ), {"bytes": read_bytes}),
                raw={"package": pkg, "read_bytes": read_bytes},
            )
        )
        order += 1

    write_bytes = safe_int(row.get("Total_Write_Data_Transfer"))
    if write_bytes > 0:
        events.append(
            Event(
                edge_type=EdgeType.WRITE,
                src=proc_ent,
                dst=file_ent,
                order=order,
                edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.WRITE), {"bytes": write_bytes}),
                raw={"package": pkg, "write_bytes": write_bytes},
            )
        )
        order += 1

    # Process lists -> EXEC edges to subprocess nodes (no cmdline, but keep the name)
    for col in ("Read_Processes", "Write_Processes", "File_Access_Processes"):
        for pname in parse_csv_list(row.get(col)):
            subproc = EntityRef(NodeType.PROC, f"{pkg}::proc::{pname}")
            events.append(
                Event(
                    edge_type=EdgeType.EXEC,
                    src=proc_ent,
                    dst=subproc,
                    order=order,
                    edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.EXEC), {"cmdline": ""}),
                    raw={"package": pkg, "proc": pname, "source_col": col},
                )
            )
            order += 1

    return events


def parse_qut_processed_csv(path: str, kind: str) -> List[Event]:
    """
    Convenience loader: read a processed QUT CSV and parse all rows.
    """
    df = pd.read_csv(path)
    parsers = {
        "syscall_traces": parse_syscall_row,
        "opensnoop_traces": parse_opensnoop_row,
        "filetop_traces": parse_filetop_row,
    }
    if kind not in parsers:
        raise ValueError(f"Unknown QUT processed kind: {kind}")

    out: List[Event] = []
    for _, row in df.iterrows():
        out.extend(parsers[kind](row))
    return out

