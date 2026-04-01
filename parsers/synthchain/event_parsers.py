'''
 # @ Create Time: 2026-03-31 15:31:45
 # @ Modified time: 2026-04-01 11:13:46
 # @ Description: define the parsing logic for diverse log/csv to extract desired entities in order
 to construct the (temporal) heterogenous graph
 '''
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, Path.cwd().parent.parent.as_posix())

import json
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import pandas as pd

from config.ontology import EdgeType, NodeType, canonical_edge_attrs, fill_defaults
from config.synthchain_sources import SYNTHCHAIN_IOC_CONFIG
from parsers.events import EntityRef, Event, parse_csv_list, safe_float, safe_int
from utils.parse_helpers import extract_ips_and_paths, parse_ts_to_unix_seconds
from utils.synthchain_helpers import (
    file_key,
    host_prefix,
    host_proc_placeholder,
    net_key,
    proc_key,
)


def parse_azure_conn_df(df: pd.DataFrame, scenario_id: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("TimeGenerated"))
        host = row.get("Computer") or row.get("Machine")
        proc_id = row.get("Process") or None
        proc_name = row.get("ProcessName") or None
        prefix = host_prefix(scenario_id, str(host) if host is not None else None, row.get("SourceIp"))

        src = EntityRef(NodeType.PROC, proc_key(prefix, str(proc_id) if proc_id is not None else None, str(proc_name) if proc_name is not None else None))

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

        dst_attrs = {
            "port": safe_int(dst_port),
        }

        events.append(
            Event(
                edge_type=EdgeType.CONNECT,
                src=src,
                dst=dst,
                ts=ts,
                order=i,
                edge_attrs=edge_attrs,
                dst_attrs=dst_attrs,
                raw={"log": "azure_conn", "scenario": scenario_id},
            )
        )
    return events


def parse_azure_process_df(df: pd.DataFrame, scenario_id: str) -> List[Event]:
    """
    VMProcess table: treat as a host->process EXEC edge (process "spawn/exists" signal).
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("TimeGenerated"))
        host = row.get("Computer") or row.get("Machine")
        prefix = host_prefix(scenario_id, str(host) if host is not None else None)

        parent = EntityRef(NodeType.PROC, f"{prefix}::proc::HOST")
        proc_id = row.get("Process") or None
        proc_name = row.get("ExecutableName") or row.get("DisplayName") or row.get("ProcessName")
        child = EntityRef(NodeType.PROC, proc_key(prefix, str(proc_id) if proc_id is not None else None, str(proc_name) if proc_name is not None else None))

        cmdline = row.get("CommandLine") or ""
        edge_attrs = fill_defaults(canonical_edge_attrs(EdgeType.EXEC), {"cmdline": str(cmdline)})
        events.append(
            Event(
                edge_type=EdgeType.EXEC,
                src=parent,
                dst=child,
                ts=ts,
                order=i,
                edge_attrs=edge_attrs,
                raw={"log": "azure_process", "scenario": scenario_id},
            )
        )
    return events


def parse_azure_events_df(df: pd.DataFrame, scenario_id: str) -> List[Event]:
    """
    Azure event logs are mostly free text. We do best-effort extraction:
    - if IPs exist -> CONNECT (PROC(host) -> NET(ip))
    - if file paths exist -> WRITE (PROC(host) -> FILE(path))
    - always keep the raw message/description for downstream text encoding
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("TimeGenerated"))
        host = row.get("Computer")
        prefix = host_prefix(scenario_id, str(host) if host is not None else None)
        proc = EntityRef(NodeType.PROC, f"{prefix}::proc::HOST")

        msg = str(row.get("RenderedDescription") or row.get("Message") or "")
        ips, paths = extract_ips_and_paths(msg)

        # IPs -> CONNECT
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
                        {"bytes_sent": 0, "bytes_recv": 0, "direction": "unknown", "evidence": msg[:500]},
                    ),
                    raw={"log": "azure_events", "scenario": scenario_id},
                )
            )

        # Paths -> WRITE
        for p in paths[:5]:
            dst = EntityRef(NodeType.FILE, file_key(p))
            events.append(
                Event(
                    edge_type=EdgeType.WRITE,
                    src=proc,
                    dst=dst,
                    ts=ts,
                    order=i,
                    edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.WRITE), {"bytes": 0, "evidence": msg[:500]}),
                    raw={"log": "azure_events", "scenario": scenario_id},
                )
            )
    return events


def parse_azure_syslog_df(df: pd.DataFrame, scenario_id: str) -> List[Event]:
    """
    Syslog is semi-structured: ProcessName + SyslogMessage.
    We model it as EXEC(host_proc -> proc) with cmdline=syslog message.
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        ts = parse_ts_to_unix_seconds(row.get("TimeGenerated [UTC]") or row.get("EventTime [UTC]"))
        host = row.get("Computer") or row.get("HostName")
        host_ip = row.get("HostIP")
        prefix = host_prefix(scenario_id, str(host) if host is not None else None, str(host_ip) if host_ip is not None else None)
        parent = EntityRef(NodeType.PROC, f"{prefix}::proc::HOST")

        proc_name = row.get("ProcessName")
        child = EntityRef(NodeType.PROC, proc_key(prefix, None, str(proc_name) if proc_name is not None else None))
        msg = str(row.get("SyslogMessage") or "")

        events.append(
            Event(
                edge_type=EdgeType.EXEC,
                src=parent,
                dst=child,
                ts=ts,
                order=i,
                edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.EXEC), {"cmdline": msg}),
                raw={"log": "azure_syslog", "scenario": scenario_id},
            )
        )
    return events


def parse_zeek_conn_df(df: pd.DataFrame, scenario_id: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        ts = parse_ts_to_unix_seconds(row.get("ts"))
        orig_h = str(row.get("id.orig_h") or "").strip()
        resp_h = str(row.get("id.resp_h") or "").strip()
        resp_p = row.get("id.resp_p")

        src = host_proc_placeholder(scenario_id, host=None, ip=orig_h or None)
        dst = EntityRef(NodeType.NET, net_key(resp_h or None, resp_p))

        edge_attrs = fill_defaults(
            canonical_edge_attrs(EdgeType.CONNECT),
            {
                "bytes_sent": safe_int(row.get("orig_bytes")),
                "bytes_recv": safe_int(row.get("resp_bytes")),
                "direction": "out",
                "proto": str(row.get("proto") or ""),
                "service": str(row.get("service") or ""),
                "duration": safe_float(row.get("duration")),
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
                dst_attrs={"port": safe_int(resp_p)},
                raw={"log": "zeek_conn", "scenario": scenario_id, "uid": row.get("uid")},
            )
        )
    return events


def parse_zeek_dns_df(df: pd.DataFrame, scenario_id: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        ts = parse_ts_to_unix_seconds(row.get("ts"))
        orig_h = str(row.get("id.orig_h") or "").strip()
        query = str(row.get("query") or "").strip()
        answers = str(row.get("answers") or "").strip()

        src = host_proc_placeholder(scenario_id, host=None, ip=orig_h or None)
        dom = EntityRef(NodeType.NET, net_key(ip=None, domain=query))

        # DNS_QUERY: PROC -> NET(domain)
        events.append(
            Event(
                edge_type=EdgeType.DNS_QUERY,
                src=src,
                dst=dom,
                ts=ts,
                order=i,
                edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.DNS_QUERY), {"query_domain": query}),
                raw={"log": "zeek_dns", "scenario": scenario_id, "uid": row.get("uid")},
            )
        )

        # RESOLVE: NET(domain) -> NET(ip)
        if answers:
            for ans in parse_csv_list(answers):
                ip_ent = EntityRef(NodeType.NET, net_key(ans))
                events.append(
                    Event(
                        edge_type=EdgeType.RESOLVE,
                        src=dom,
                        dst=ip_ent,
                        ts=ts,
                        order=i,
                        edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.RESOLVE), {"resolved_ip": ans}),
                        raw={"log": "zeek_dns", "scenario": scenario_id, "uid": row.get("uid")},
                    )
                )

    return events


def parse_zeek_http_df(df: pd.DataFrame, scenario_id: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        ts = parse_ts_to_unix_seconds(row.get("ts"))
        orig_h = str(row.get("id.orig_h") or "").strip()
        resp_h = str(row.get("id.resp_h") or "").strip()
        resp_p = row.get("id.resp_p")
        status = safe_int(row.get("status_code"))

        src = host_proc_placeholder(scenario_id, host=None, ip=orig_h or None)
        dst = EntityRef(NodeType.NET, net_key(resp_h or None, resp_p, domain=str(row.get("host") or "").strip() or None))

        # CONNECT edge carries coarse HTTP info as superset attrs
        events.append(
            Event(
                edge_type=EdgeType.CONNECT,
                src=src,
                dst=dst,
                ts=ts,
                order=i,
                edge_attrs=fill_defaults(
                    canonical_edge_attrs(EdgeType.CONNECT),
                    {
                        "bytes_sent": safe_int(row.get("request_body_len")),
                        "bytes_recv": safe_int(row.get("response_body_len")),
                        "direction": "out",
                        "method": str(row.get("method") or ""),
                        "uri": str(row.get("uri") or ""),
                        "status_code": status,
                    },
                ),
                dst_attrs={"port": safe_int(resp_p)},
                raw={"log": "zeek_http", "scenario": scenario_id, "uid": row.get("uid")},
            )
        )

        # Optional: treat 3xx as REDIRECT NET -> NET (without location info)
        if 300 <= status < 400:
            events.append(
                Event(
                    edge_type=EdgeType.REDIRECT,
                    src=dst,
                    dst=dst,
                    ts=ts,
                    order=i,
                    edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.REDIRECT), {"http_status": status}),
                    raw={"log": "zeek_http", "scenario": scenario_id, "uid": row.get("uid")},
                )
            )
    return events


def parse_eve_json_lines(path: Path, scenario_id: str, limit: Optional[int] = None) -> List[Event]:
    """
    Parse Suricata eve.json (jsonlines).
    We handle `http`, `fileinfo`, and `alert` events conservatively.
    """
    events: List[Event] = []
    order = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            etype = obj.get("event_type")
            src_ip = obj.get("src_ip")
            dst_ip = obj.get("dest_ip")
            src_port = obj.get("src_port")
            dst_port = obj.get("dest_port")

            ts = parse_ts_to_unix_seconds(obj.get("timestamp"))
            src = host_proc_placeholder(scenario_id, host=None, ip=str(src_ip) if src_ip is not None else None)
            dst = EntityRef(NodeType.NET, net_key(str(dst_ip) if dst_ip is not None else None, dst_port))

            if etype == "http":
                http = obj.get("http", {}) or {}
                status = safe_int(http.get("status"))
                events.append(
                    Event(
                        edge_type=EdgeType.CONNECT,
                        src=src,
                        dst=dst,
                        ts=ts,
                        order=order,
                        edge_attrs=fill_defaults(
                            canonical_edge_attrs(EdgeType.CONNECT),
                            {
                                "bytes_sent": 0,
                                "bytes_recv": safe_int(http.get("length")),
                                "direction": "out",
                                "method": str(http.get("http_method") or ""),
                                "uri": str(http.get("url") or ""),
                                "status_code": status,
                                "user_agent": str(http.get("http_user_agent") or ""),
                            },
                        ),
                        dst_attrs={"port": safe_int(dst_port)},
                        raw={"log": "eve", "scenario": scenario_id, "flow_id": obj.get("flow_id")},
                    )
                )
                if 300 <= status < 400:
                    events.append(
                        Event(
                            edge_type=EdgeType.REDIRECT,
                            src=dst,
                            dst=dst,
                            ts=ts,
                            order=order,
                            edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.REDIRECT), {"http_status": status}),
                            raw={"log": "eve", "scenario": scenario_id, "flow_id": obj.get("flow_id")},
                        )
                    )
            elif etype == "fileinfo":
                finfo = obj.get("fileinfo", {}) or {}
                fname = str(finfo.get("filename") or "")
                size = safe_int(finfo.get("size"))
                file_ent = EntityRef(NodeType.FILE, file_key(fname))
                events.append(
                    Event(
                        edge_type=EdgeType.WRITE,
                        src=src,
                        dst=file_ent,
                        ts=ts,
                        order=order,
                        edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.WRITE), {"bytes": size}),
                        raw={"log": "eve", "scenario": scenario_id, "flow_id": obj.get("flow_id")},
                    )
                )
            elif etype == "alert":
                alert = obj.get("alert", {}) or {}
                sig = str(alert.get("signature") or "")
                cred = EntityRef(NodeType.CRED, f"{scenario_id}::alert::{sig}" if sig else f"{scenario_id}::alert::UNKNOWN")
                # model alert signature as PROC -> CRED "evidence" via EXEC (keeps cmdline text slot)
                events.append(
                    Event(
                        edge_type=EdgeType.EXEC,
                        src=src,
                        dst=EntityRef(NodeType.PROC, f"{scenario_id}::proc::ALERT"),
                        ts=ts,
                        order=order,
                        edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.EXEC), {"cmdline": sig}),
                        raw={"log": "eve", "scenario": scenario_id, "flow_id": obj.get("flow_id")},
                    )
                )

            order += 1
            if limit is not None and order >= limit:
                break
    return events


def load_synthchain_events(
    scenario_id: str,
    project_root: str | Path,
    only_ioc_logs: bool = True,
    limit_per_file: Optional[int] = None,
) -> List[Event]:
    """
    Load and parse SynthChain logs for a scenario based on SYNTHCHAIN_IOC_CONFIG.

    - `only_ioc_logs=True` will ignore files marked as has_ioc=False (noise control).
    - `limit_per_file` is useful for quick experiments.
    """
    if scenario_id not in SYNTHCHAIN_IOC_CONFIG:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    cfg = SYNTHCHAIN_IOC_CONFIG[scenario_id]
    base = Path(project_root) / cfg["root"]
    out: List[Event] = []

    for log_name, spec in cfg["logs"].items():
        if only_ioc_logs and not spec.get("has_ioc", False):
            continue

        path = base / spec["filename"]
        if not path.exists():
            # allow config to be ahead of local data; skip missing files
            continue

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            if limit_per_file is not None:
                df = df.head(limit_per_file)

            if log_name == "azure_conn":
                out.extend(parse_azure_conn_df(df, scenario_id))
            elif log_name == "azure_process":
                out.extend(parse_azure_process_df(df, scenario_id))
            elif log_name == "azure_events":
                out.extend(parse_azure_events_df(df, scenario_id))
            elif log_name == "azure_syslog":
                out.extend(parse_azure_syslog_df(df, scenario_id))
            elif log_name == "zeek_conn":
                out.extend(parse_zeek_conn_df(df, scenario_id))
            elif log_name == "zeek_dns":
                out.extend(parse_zeek_dns_df(df, scenario_id))
            elif log_name == "zeek_http":
                out.extend(parse_zeek_http_df(df, scenario_id))
            else:
                # unknown csv type, ignore for now
                continue

        elif path.suffix.lower() == ".json" and path.name == "eve.json":
            out.extend(parse_eve_json_lines(path, scenario_id, limit=limit_per_file))

    return out


