"""Helpers to preserve parser/raw fields on graph edges for TGN export and reconstruction."""
from __future__ import annotations

from typing import Any, Dict, List

from parsers.events import Event


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
        ea["_ioc_type"] = str(types[0])
    elif isinstance(types, str) and types:
        ea["_ioc_type"] = str(types)
    return ea


def primary_ioc_type_from_attrs(attrs: Dict[str, Any]) -> str:
    t = attrs.get("_ioc_type") or attrs.get("ioc_type")
    if isinstance(t, list) and t:
        return str(t[0])
    if isinstance(t, str):
        return t.strip()
    types = attrs.get("ioc_types")
    if isinstance(types, list) and types:
        return str(types[0])
    return ""
