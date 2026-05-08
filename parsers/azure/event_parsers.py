from __future__ import annotations

from typing import List

import pandas as pd

from config.ontology import EdgeType, NodeType, canonical_edge_attrs, fill_defaults
from parsers.events import EntityRef, Event, safe_int
from parsers.extractors import extract_ips_and_paths, extract_text_signals, suspicious_cmd_flags
from parsers.normalizers import parse_ts_to_unix_seconds
from utils.synthchain_helpers import file_key, host_prefix, net_key, proc_key


def parse_azure_conn_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("TimeGenerated"))
        host = row.get("Computer") or row.get("Machine")
        proc_id = row.get("Process") or None
        proc_name = row.get("ProcessName") or None
        prefix = host_prefix(scenario_id, str(host) if host is not None else None, row.get("SourceIp"))

        src = EntityRef(
            NodeType.PROC,
            proc_key(
                prefix,
                str(proc_id) if proc_id is not None else None,
                str(proc_name) if proc_name is not None else None,
            ),
        )

        dst_ip = row.get("DestinationIp")
        dst_port = row.get("DestinationPort")
        dst = EntityRef(NodeType.NET, net_key(str(dst_ip) if dst_ip is not None else None, dst_port))

        edge_attrs = fill_defaults(
            canonical_edge_attrs(EdgeType.CONNECT),
            {
                "bytes_sent": safe_int(row.get("BytesSent")),
                "bytes_recv": safe_int(row.get("BytesReceived")),
                "direction": str(row.get("Direction") or "").strip().lower(),
                # superset attrs
                "protocol": str(row.get("Protocol") or "").strip().lower(),
            },
        )

        events.append(
            Event(
                edge_type=EdgeType.CONNECT,
                src=src,
                dst=dst,
                ts=ts,
                order=i,
                edge_attrs=edge_attrs,
                dst_attrs={"port": safe_int(dst_port)},
                raw={"log": "azure_conn", "scenario": scenario_id, "source_file": source_file, "row_idx": row_idx},
            )
        )
    return events


def parse_azure_process_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    """
    VMProcess table: treat as a host->process EXEC edge (process "spawn/exists" signal).
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("TimeGenerated"))
        host = row.get("Computer") or row.get("Machine")
        prefix = host_prefix(scenario_id, str(host) if host is not None else None)

        parent = EntityRef(NodeType.PROC, f"{prefix}::proc::HOST")
        proc_id = row.get("Process") or None
        proc_name = row.get("ExecutableName") or row.get("DisplayName") or row.get("ProcessName")
        child = EntityRef(
            NodeType.PROC,
            proc_key(
                prefix,
                str(proc_id) if proc_id is not None else None,
                str(proc_name) if proc_name is not None else None,
            ),
        )

        cmd = str(row.get("CommandLine") or "")
        edge_attrs = fill_defaults(
            canonical_edge_attrs(EdgeType.EXEC),
            {"cmdline": cmd, **suspicious_cmd_flags(cmd), **extract_text_signals(cmd)},
        )

        events.append(
            Event(
                edge_type=EdgeType.EXEC,
                src=parent,
                dst=child,
                ts=ts,
                order=i,
                edge_attrs=edge_attrs,
                raw={"log": "azure_process", "scenario": scenario_id, "source_file": source_file, "row_idx": row_idx},
            )
        )
    return events


def parse_azure_events_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    """
    Azure event logs are mostly free text. Best-effort extraction:
    - IPs -> CONNECT (PROC(host) -> NET(ip))
    - file paths -> WRITE (PROC(host) -> FILE(path))
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("TimeGenerated"))
        host = row.get("Computer")
        prefix = host_prefix(scenario_id, str(host) if host is not None else None)
        proc = EntityRef(NodeType.PROC, f"{prefix}::proc::HOST")

        msg = str(row.get("RenderedDescription") or row.get("Message") or "")
        ips, paths = extract_ips_and_paths(msg)
        txt_signals = extract_text_signals(msg)
        cmd_flags = suspicious_cmd_flags(msg)

        for ip in ips[:5]:
            dst = EntityRef(NodeType.NET, net_key(ip))
            events.append(
                Event(
                    edge_type=EdgeType.CONNECT,
                    src=proc,
                    dst=dst,
                    ts=ts,
                    order=i,
                    edge_attrs=fill_defaults(
                        canonical_edge_attrs(EdgeType.CONNECT),
                        {"bytes_sent": 0, "bytes_recv": 0, "direction": "unknown", "evidence": msg[:500], **txt_signals, **cmd_flags},
                    ),
                    raw={"log": "azure_events", "scenario": scenario_id, "source_file": source_file, "row_idx": row_idx},
                )
            )

        for p in paths[:5]:
            dst = EntityRef(NodeType.FILE, file_key(p))
            events.append(
                Event(
                    edge_type=EdgeType.WRITE,
                    src=proc,
                    dst=dst,
                    ts=ts,
                    order=i,
                    edge_attrs=fill_defaults(
                        canonical_edge_attrs(EdgeType.WRITE),
                        {"bytes": 0, "evidence": msg[:500], **txt_signals, **cmd_flags},
                    ),
                    raw={"log": "azure_events", "scenario": scenario_id, "source_file": source_file, "row_idx": row_idx},
                )
            )

    return events


def parse_azure_syslog_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    """
    Syslog is semi-structured: ProcessName + SyslogMessage.
    We model it as EXEC(host_proc -> proc) with cmdline=syslog message.
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("EventTime [UTC]"))
        host = row.get("Computer") or row.get("HostName")
        host_ip = row.get("HostIP")
        prefix = host_prefix(
            scenario_id,
            str(host) if host is not None else None,
            str(host_ip) if host_ip is not None else None,
        )
        parent = EntityRef(NodeType.PROC, f"{prefix}::proc::HOST")

        proc_name = row.get("ProcessName")
        child = EntityRef(NodeType.PROC, proc_key(prefix, None, str(proc_name) if proc_name is not None else None))
        msg = str(row.get("SyslogMessage") or "")
        txt_signals = extract_text_signals(msg)
        cmd_flags = suspicious_cmd_flags(msg)

        events.append(
            Event(
                edge_type=EdgeType.EXEC,
                src=parent,
                dst=child,
                ts=ts,
                order=i,
                edge_attrs=fill_defaults(
                    canonical_edge_attrs(EdgeType.EXEC),
                    {"cmdline": msg, **txt_signals, **cmd_flags},
                ),
                raw={"log": "azure_syslog", "scenario": scenario_id, "source_file": source_file, "row_idx": row_idx},
            )
        )
    return events

