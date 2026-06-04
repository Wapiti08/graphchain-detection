"""Helpers to preserve parser/raw fields on graph edges for TGN export and reconstruction."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from parsers.events import Event

# When multiple IOC types hit one edge/line, prefer stage-specific types over generic C2 indicators.
IOC_TYPE_RESOLUTION_PRIORITY: Tuple[str, ...] = (
    "data_exfiltration",
    "execution_chain",
    "execution",
    "process_injection",
    "package_name",
    "malicious_binary",
    "file_transfer",
    "file_artifact",
    "persistence",
    "defense_evasion",
    "crypto_indicator",
    "cloud_recon",
    "c2_agent",
    "suspicious_port",
    "dns_indicator",
    "attack_ip",
    "monitoring",
)


def pick_primary_ioc_type(candidates: Iterable[str]) -> str:
    """Choose one IOC type for stage labeling / export (not alphabetical order)."""
    cset = {str(c).strip() for c in candidates if c and str(c).strip()}
    if not cset:
        return ""
    for pref in IOC_TYPE_RESOLUTION_PRIORITY:
        if pref in cset:
            return pref
    return sorted(cset)[0]


def edge_attrs_for_export(ev: Event) -> Dict[str, Any]:
    """Copy edge attrs and attach stable keys used by TGN export (_row_idx, _source_file, _ioc_type)."""
    ea: Dict[str, Any] = dict(ev.edge_attrs)
    sf = ev.raw.get("source_file")
    if sf:
        ea["_source_file"] = str(sf)
    ri = ev.raw.get("row_idx")
    if isinstance(ri, int):
        ea["_row_idx"] = int(ri)
    types = ea.get("ioc_types")
    if isinstance(types, list) and types:
        ea["_ioc_type"] = pick_primary_ioc_type(types)
    elif isinstance(types, str) and types:
        ea["_ioc_type"] = str(types)
    rule_types = ea.get("rule_ioc_types")
    if isinstance(rule_types, list) and rule_types:
        ea["_rule_ioc_type"] = pick_primary_ioc_type(rule_types)
    elif isinstance(ea.get("_rule_ioc_type"), str) and str(ea.get("_rule_ioc_type")).strip():
        ea["_rule_ioc_type"] = str(ea["_rule_ioc_type"]).strip()
    return ea


def primary_ioc_type_from_attrs(attrs: Dict[str, Any]) -> str:
    types = attrs.get("ioc_types")
    if isinstance(types, list) and types:
        return pick_primary_ioc_type(types)
    t = attrs.get("_ioc_type") or attrs.get("ioc_type")
    if isinstance(t, list) and t:
        return pick_primary_ioc_type(t)
    if isinstance(t, str) and t.strip():
        return t.strip()
    return ""


def primary_rule_ioc_type_from_attrs(attrs: Dict[str, Any]) -> str:
    types = attrs.get("rule_ioc_types")
    if isinstance(types, list) and types:
        return pick_primary_ioc_type(types)
    t = attrs.get("_rule_ioc_type")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return ""
