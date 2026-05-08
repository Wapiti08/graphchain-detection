from __future__ import annotations

from typing import List

import pandas as pd

from config.ontology import EdgeType, NodeType, canonical_edge_attrs, fill_defaults
from parsers.events import EntityRef, Event, parse_csv_list, safe_float, safe_int
from parsers.normalizers import parse_ts_to_unix_seconds
from utils.synthchain_helpers import file_key, host_proc_placeholder, net_key


def parse_zeek_conn_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
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
                raw={
                    "log": "zeek_conn",
                    "scenario": scenario_id,
                    "uid": row.get("uid"),
                    "source_file": source_file,
                    "row_idx": row_idx,
                },
            )
        )
    return events


def parse_zeek_dns_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
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
                raw={
                    "log": "zeek_dns",
                    "scenario": scenario_id,
                    "uid": row.get("uid"),
                    "source_file": source_file,
                    "row_idx": row_idx,
                },
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
                        raw={
                            "log": "zeek_dns",
                            "scenario": scenario_id,
                            "uid": row.get("uid"),
                            "source_file": source_file,
                            "row_idx": row_idx,
                        },
                    )
                )

    return events


def parse_zeek_http_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("ts"))
        orig_h = str(row.get("id.orig_h") or "").strip()
        resp_h = str(row.get("id.resp_h") or "").strip()
        resp_p = row.get("id.resp_p")
        status = safe_int(row.get("status_code"))

        src = host_proc_placeholder(scenario_id, host=None, ip=orig_h or None)
        dst = EntityRef(
            NodeType.NET,
            net_key(
                resp_h or None,
                resp_p,
                domain=str(row.get("host") or "").strip() or None,
            ),
        )

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
                raw={
                    "log": "zeek_http",
                    "scenario": scenario_id,
                    "uid": row.get("uid"),
                    "source_file": source_file,
                    "row_idx": row_idx,
                },
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
                    raw={
                        "log": "zeek_http",
                        "scenario": scenario_id,
                        "uid": row.get("uid"),
                        "source_file": source_file,
                        "row_idx": row_idx,
                    },
                )
            )
    return events


def parse_zeek_files_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    """
    Zeek files.log exported as CSV/parquet.

    We model this as PROC(host placeholder) -> FILE with a WRITE edge, carrying file size.
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("ts"))

        orig_h = str(row.get("id.orig_h") or "").strip()
        src = host_proc_placeholder(scenario_id, host=None, ip=orig_h or None)

        fname = str(row.get("filename") or "")
        mime = str(row.get("mime_type") or "")
        sha256 = str(row.get("sha256") or "")
        md5 = str(row.get("md5") or "")
        total_bytes = safe_int(row.get("total_bytes"))

        # prefer filename; fall back to fuid for stable key
        fuid = str(row.get("fuid") or "").strip()
        file_id = file_key(fname) if fname.strip() else f"{scenario_id}::file::{fuid or 'UNKNOWN'}"
        dst = EntityRef(NodeType.FILE, file_id)

        edge_attrs = fill_defaults(
            canonical_edge_attrs(EdgeType.WRITE),
            {
                "bytes": total_bytes,
                # superset attrs
                "mime_type": mime,
                "sha256": sha256.lower() if sha256 else "",
                "md5": md5.lower() if md5 else "",
            },
        )

        dst_attrs = {
            "file_type": mime,
        }

        events.append(
            Event(
                edge_type=EdgeType.WRITE,
                src=src,
                dst=dst,
                ts=ts,
                order=i,
                edge_attrs=edge_attrs,
                dst_attrs=dst_attrs,
                raw={
                    "log": "zeek_files",
                    "scenario": scenario_id,
                    "uid": row.get("uid"),
                    "fuid": row.get("fuid"),
                    "source_file": source_file,
                    "row_idx": row_idx,
                },
            )
        )
    return events


def parse_zeek_ssl_df(df: pd.DataFrame, scenario_id: str, *, source_file: str) -> List[Event]:
    """
    Zeek ssl.log exported as CSV/parquet.

    We model as CONNECT (PROC(host placeholder) -> NET(dst_ip:dst_port)) and attach TLS attrs.
    """
    events: List[Event] = []
    for i, row in df.iterrows():
        row_idx = int(row.get("_row_idx", i))
        ts = parse_ts_to_unix_seconds(row.get("ts"))
        orig_h = str(row.get("id.orig_h") or "").strip()
        resp_h = str(row.get("id.resp_h") or "").strip()
        resp_p = row.get("id.resp_p")

        src = host_proc_placeholder(scenario_id, host=None, ip=orig_h or None)
        dst = EntityRef(
            NodeType.NET,
            net_key(resp_h or None, resp_p, domain=str(row.get("server_name") or "").strip() or None),
        )

        version = str(row.get("version") or "")
        cipher = str(row.get("cipher") or "")
        server_name = str(row.get("server_name") or "")
        validation = str(row.get("validation_status") or "")

        edge_attrs = fill_defaults(
            canonical_edge_attrs(EdgeType.CONNECT),
            {
                "bytes_sent": 0,
                "bytes_recv": 0,
                "direction": "out",
                # superset TLS attrs
                "tls_version": version,
                "tls_cipher": cipher,
                "sni": server_name,
                "validation_status": validation,
            },
        )

        dst_attrs = {"port": safe_int(resp_p)}

        events.append(
            Event(
                edge_type=EdgeType.CONNECT,
                src=src,
                dst=dst,
                ts=ts,
                order=i,
                edge_attrs=edge_attrs,
                dst_attrs=dst_attrs,
                raw={
                    "log": "zeek_ssl",
                    "scenario": scenario_id,
                    "uid": row.get("uid"),
                    "source_file": source_file,
                    "row_idx": row_idx,
                },
            )
        )
    return events

