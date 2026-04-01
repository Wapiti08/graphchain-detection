from __future__ import annotations

from typing import Any, Optional

from config.ontology import NodeType
from parsers.events import EntityRef, safe_int


def host_prefix(scenario_id: str, host: Optional[str], ip: Optional[str] = None) -> str:
    h = (host or "").strip()
    if h:
        return f"{scenario_id}::{h}"
    if ip:
        return f"{scenario_id}::ip::{ip}"
    return f"{scenario_id}::host::UNKNOWN"


def proc_key(prefix: str, proc_id: Optional[str], proc_name: Optional[str]) -> str:
    pid = (proc_id or "").strip()
    pname = (proc_name or "").strip()
    if pid:
        return f"{prefix}::proc::{pid}"
    if pname:
        return f"{prefix}::procname::{pname}"
    return f"{prefix}::proc::UNKNOWN"


def net_key(ip: Optional[str], port: Optional[Any] = None, domain: Optional[str] = None) -> str:
    ip_s = (ip or "").strip()
    dom_s = (domain or "").strip()
    if dom_s and ip_s:
        return f"{dom_s}|{ip_s}:{safe_int(port)}"
    if dom_s:
        return dom_s
    if ip_s:
        p = safe_int(port)
        return f"{ip_s}:{p}" if p else ip_s
    return "UNKNOWN_NET"


def file_key(path: Optional[str]) -> str:
    p = (path or "").strip()
    return p if p else "UNKNOWN_FILE"


def host_proc_placeholder(scenario_id: str, host: Optional[str], ip: Optional[str] = None) -> EntityRef:
    """
    Network sensors (zeek/suricata) usually don't provide process identity.
    We use a stable host-level PROC placeholder as the event source.
    """
    return EntityRef(NodeType.PROC, f"{host_prefix(scenario_id, host, ip)}::proc::HOST")

