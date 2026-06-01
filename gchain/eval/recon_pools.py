"""Eval-only candidate pools (row filters before top-K ranking)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple


def ioc_log_source_files_for_scenario(scenario: str) -> Set[str]:
    from config.synthchain_sources import SYNTHCHAIN_IOC_CONFIG

    sc = SYNTHCHAIN_IOC_CONFIG.get(str(scenario)) or {}
    out: Set[str] = set()
    for spec in (sc.get("logs") or {}).values():
        if not isinstance(spec, dict) or not bool(spec.get("has_ioc", False)):
            continue
        fn = str(spec.get("filename") or "").strip()
        if fn:
            out.add(Path(fn).name)
    return out


def filter_rows_to_ioc_log_sources(
    rows: Sequence[Mapping[str, Any]],
    allowed_source_files: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    allowed = {str(x).strip() for x in allowed_source_files if str(x).strip()}
    kept: List[Dict[str, Any]] = []
    excluded_files: Set[str] = set()
    for r in rows:
        sf = str(r.get("source_file") or "").strip()
        if sf in allowed:
            kept.append(dict(r))
        elif sf:
            excluded_files.add(sf)
    return kept, {
        "allowed_source_files": sorted(allowed),
        "n_input_rows": int(len(rows)),
        "n_candidate_rows": int(len(kept)),
        "n_excluded_rows": int(len(rows) - len(kept)),
        "excluded_source_files": sorted(excluded_files),
    }
