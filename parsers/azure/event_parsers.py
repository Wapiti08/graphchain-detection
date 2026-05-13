from __future__ import annotations

import html
import re
from typing import Dict, List

import pandas as pd

from config.ontology import EdgeType, NodeType, canonical_edge_attrs, fill_defaults
from parsers.events import EntityRef, Event, safe_int
from parsers.extractors import extract_ips_and_paths, extract_text_signals, suspicious_cmd_flags
from parsers.normalizers import parse_ts_to_unix_seconds
from utils.synthchain_helpers import file_key, host_prefix, net_key, proc_key


def _eventdata_name_values(raw: object) -> Dict[str, str]:
    """Parse `<Data Name="...">...</Data>` pairs from Azure / Sysmon EventData XML."""
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    out: Dict[str, str] = {}
    for m in re.finditer(r'<Data\s+Name="([^"]+)"[^>]*>([^<]*)</Data>', s, flags=re.IGNORECASE):
        key = m.group(1).strip()
        val = html.unescape(m.group(2)).strip()
        if key:
            out[key] = val
    return out


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
    Azure / Sysmon-style operational logs.

    Extraction (in order):
    1) Structured fields from `EventData` XML when present: Image, ParentImage, CommandLine,
       ProcessId, ParentProcessId, Source/Destination IP (+ ports / hostname) -> EXEC / CONNECT.
    2) Executable path `Image` -> WRITE to FILE (anchors rows that have no free-text IP/path).
    3) Fallback: IPs and paths from RenderedDescription / Message (legacy behavior).
    """
    raw_stub = {"log": "azure_events", "scenario": scenario_id, "source_file": source_file}

    def _row_raw(idx: int) -> dict:
        return {**raw_stub, "row_idx": int(idx)}

    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("TimeGenerated"))
        host = row.get("Computer")
        prefix = host_prefix(scenario_id, str(host) if host is not None else None)
        host_proc = EntityRef(NodeType.PROC, f"{prefix}::proc::HOST")

        fields = _eventdata_name_values(row.get("EventData"))
        msg = str(row.get("RenderedDescription") or row.get("Message") or "")

        img = (fields.get("Image") or fields.get("TargetImage") or "").strip()
        pimg = (fields.get("ParentImage") or "").strip()
        pid = str(fields.get("ProcessId") or fields.get("ProcessID") or "").strip() or None
        ppid = str(fields.get("ParentProcessId") or fields.get("ParentProcessID") or "").strip() or None
        cmdline = (fields.get("CommandLine") or fields.get("Commandline") or "").strip()

        evidence = (cmdline or msg)[:500]
        txt = f"{cmdline} {msg}".strip()
        txt_signals = extract_text_signals(txt)
        cmd_flags = suspicious_cmd_flags(cmdline or msg)

        child_proc_e: EntityRef | None = None
        if img:
            ck = proc_key(prefix, pid, img)
            child_proc_e = EntityRef(NodeType.PROC, ck)
            if pimg and pimg.lower() != img.lower():
                pk = proc_key(prefix, ppid, pimg)
                parent_e = EntityRef(NodeType.PROC, pk)
                events.append(
                    Event(
                        edge_type=EdgeType.EXEC,
                        src=parent_e,
                        dst=child_proc_e,
                        ts=ts,
                        order=i,
                        edge_attrs=fill_defaults(
                            canonical_edge_attrs(EdgeType.EXEC),
                            {"cmdline": cmdline or evidence, **txt_signals, **cmd_flags},
                        ),
                        raw=_row_raw(row_idx),
                    )
                )
            else:
                events.append(
                    Event(
                        edge_type=EdgeType.EXEC,
                        src=host_proc,
                        dst=child_proc_e,
                        ts=ts,
                        order=i,
                        edge_attrs=fill_defaults(
                            canonical_edge_attrs(EdgeType.EXEC),
                            {"cmdline": cmdline or evidence, **txt_signals, **cmd_flags},
                        ),
                        raw=_row_raw(row_idx),
                    )
                )

            fk = file_key(img)
            if fk and fk != "UNKNOWN_FILE":
                events.append(
                    Event(
                        edge_type=EdgeType.WRITE,
                        src=child_proc_e,
                        dst=EntityRef(NodeType.FILE, fk),
                        ts=ts,
                        order=i,
                        edge_attrs=fill_defaults(
                            canonical_edge_attrs(EdgeType.WRITE),
                            {"bytes": 0, "evidence": evidence, **txt_signals, **cmd_flags},
                        ),
                        raw=_row_raw(row_idx),
                    )
                )

        proc_for_net = child_proc_e if child_proc_e is not None else host_proc

        dip = (fields.get("DestinationIp") or "").strip()
        dport = fields.get("DestinationPort") or fields.get("DestinationPortName")
        dhost = (fields.get("DestinationHostname") or "").strip() or None

        if dip:
            dst_net = EntityRef(NodeType.NET, net_key(dip, dport, dhost))
            events.append(
                Event(
                    edge_type=EdgeType.CONNECT,
                    src=proc_for_net,
                    dst=dst_net,
                    ts=ts,
                    order=i,
                    edge_attrs=fill_defaults(
                        canonical_edge_attrs(EdgeType.CONNECT),
                        {
                            "bytes_sent": 0,
                            "bytes_recv": 0,
                            "direction": "outbound" if str(fields.get("Initiated") or "").lower() == "true" else "unknown",
                            "evidence": evidence,
                            "protocol": str(fields.get("Protocol") or fields.get("DestinationPortName") or "").strip().lower(),
                            **txt_signals,
                            **cmd_flags,
                        },
                    ),
                    raw=_row_raw(row_idx),
                )
            )

        ips, paths = extract_ips_and_paths(msg)
        src_for_msg = proc_for_net if child_proc_e is not None else host_proc
        for ip in ips[:5]:
            dst = EntityRef(NodeType.NET, net_key(ip))
            events.append(
                Event(
                    edge_type=EdgeType.CONNECT,
                    src=src_for_msg,
                    dst=dst,
                    ts=ts,
                    order=i,
                    edge_attrs=fill_defaults(
                        canonical_edge_attrs(EdgeType.CONNECT),
                        {"bytes_sent": 0, "bytes_recv": 0, "direction": "unknown", "evidence": msg[:500], **txt_signals, **cmd_flags},
                    ),
                    raw=_row_raw(row_idx),
                )
            )

        for p in paths[:5]:
            dst = EntityRef(NodeType.FILE, file_key(p))
            events.append(
                Event(
                    edge_type=EdgeType.WRITE,
                    src=src_for_msg,
                    dst=dst,
                    ts=ts,
                    order=i,
                    edge_attrs=fill_defaults(
                        canonical_edge_attrs(EdgeType.WRITE),
                        {"bytes": 0, "evidence": msg[:500], **txt_signals, **cmd_flags},
                    ),
                    raw=_row_raw(row_idx),
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

