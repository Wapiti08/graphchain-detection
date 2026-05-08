from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from config.ontology import EdgeType, NodeType, canonical_edge_attrs, fill_defaults
from parsers.events import EntityRef, Event, safe_int
from parsers.normalizers import parse_ts_to_unix_seconds
from utils.synthchain_helpers import file_key, host_proc_placeholder, net_key


def parse_eve_json_lines(
    path: Path,
    scenario_id: str,
    *,
    source_file: str,
    limit: Optional[int] = None,
) -> List[Event]:
    """
    Parse Suricata eve.json (jsonlines).
    Handles http, fileinfo, alert conservatively.
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
                        raw={
                            "log": "eve",
                            "scenario": scenario_id,
                            "flow_id": obj.get("flow_id"),
                            "source_file": source_file,
                            "row_idx": order,
                        },
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
                            raw={
                                "log": "eve",
                                "scenario": scenario_id,
                                "flow_id": obj.get("flow_id"),
                                "source_file": source_file,
                                "row_idx": order,
                            },
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
                        raw={
                            "log": "eve",
                            "scenario": scenario_id,
                            "flow_id": obj.get("flow_id"),
                            "source_file": source_file,
                            "row_idx": order,
                        },
                    )
                )
            elif etype == "alert":
                alert = obj.get("alert", {}) or {}
                sig = str(alert.get("signature") or "")
                # Keep signature in cmdline slot for now (feature encoder can embed it)
                events.append(
                    Event(
                        edge_type=EdgeType.EXEC,
                        src=src,
                        dst=EntityRef(NodeType.PROC, f"{scenario_id}::proc::ALERT"),
                        ts=ts,
                        order=order,
                        edge_attrs=fill_defaults(canonical_edge_attrs(EdgeType.EXEC), {"cmdline": sig}),
                        raw={
                            "log": "eve",
                            "scenario": scenario_id,
                            "flow_id": obj.get("flow_id"),
                            "source_file": source_file,
                            "row_idx": order,
                        },
                    )
                )

            order += 1
            if limit is not None and order >= limit:
                break

    return events

