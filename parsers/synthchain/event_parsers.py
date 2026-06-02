from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import pandas as pd

from config.synthchain_sources import SYNTHCHAIN_IOC_CONFIG
from parsers.events import Event
from parsers.azure import (
    parse_azure_conn_df,
    parse_azure_events_df,
    parse_azure_process_df,
    parse_azure_syslog_df,
)
from parsers.extractors import extract_ips_and_paths
from parsers.normalizers import IOCIndex, load_ioc_ground_truth
from parsers.suricata import parse_eve_json_lines
from parsers.zeek import (
    parse_zeek_conn_df,
    parse_zeek_dns_df,
    parse_zeek_files_df,
    parse_zeek_http_df,
    parse_zeek_ssl_df,
)


def load_synthchain_events(
    scenario_id: str,
    project_root: str | Path,
    only_ioc_logs: bool = True,
    limit_per_file: Optional[int] = None,
    ioc_ground_truth_path: str | Path | None = "data/SynthChain/iocs/ioc_ground_truth.json",
    *,
    verbose: bool = False,
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

    debug = bool(verbose) or str(os.environ.get("SYNTHCHAIN_DEBUG_LOAD") or "").strip() in {"1", "true", "yes", "y"}
    if debug:
        print(
            f"[load_synthchain_events] scenario={scenario_id} base={base} "
            f"only_ioc_logs={only_ioc_logs} limit_per_file={limit_per_file}"
        )

    for log_name, spec in cfg["logs"].items():
        if only_ioc_logs and not spec.get("has_ioc", False):
            continue

        path = base / spec["filename"]
        if not path.exists():
            # allow config to be ahead of local data; skip missing files
            continue

        # ground-truth filename key (json uses csv/json filenames)
        gt_source_file = Path(spec["filename"]).name

        if path.suffix.lower() == ".csv":
            # prefer parquet if present next to csv (line indices are often built from parquet)
            parquet_path = path.with_suffix(".parquet")
            try:
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                else:
                    df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path)

            # preserve original row index for line-level IOC matching
            df = df.reset_index(drop=False).rename(columns={"index": "_row_idx"})
            if limit_per_file is not None:
                df = df.head(limit_per_file)

            if debug:
                print(f"[load_synthchain_events] parse csv log={log_name} file={gt_source_file} rows={len(df)}")

            if log_name == "azure_conn":
                evs = parse_azure_conn_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "azure_process":
                evs = parse_azure_process_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "azure_events":
                evs = parse_azure_events_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "azure_syslog":
                evs = parse_azure_syslog_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "zeek_conn":
                evs = parse_zeek_conn_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "zeek_dns":
                evs = parse_zeek_dns_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "zeek_http":
                evs = parse_zeek_http_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "zeek_files":
                evs = parse_zeek_files_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            elif log_name == "zeek_ssl":
                evs = parse_zeek_ssl_df(df, scenario_id, source_file=gt_source_file)
                out.extend(evs)
            else:
                # unknown csv type, ignore for now
                continue

            if debug:
                print(
                    f"[load_synthchain_events] -> events_added={len(evs)} (csv log={log_name}) "
                    f"total_so_far={len(out)}"
                )

        elif path.suffix.lower() == ".json" and path.name == "eve.json":
            if debug:
                print(f"[load_synthchain_events] parse eve file={gt_source_file} limit={limit_per_file}")
            evs = parse_eve_json_lines(path, scenario_id, source_file=gt_source_file, limit=limit_per_file)
            out.extend(evs)
            if debug:
                print(f"[load_synthchain_events] -> events_added={len(evs)} (eve) total_so_far={len(out)}")

    # IOC annotation (optional)
    if ioc_ground_truth_path is not None:
        gt_path = Path(project_root) / ioc_ground_truth_path
        if gt_path.exists():
            idx_by_scenario = load_ioc_ground_truth(gt_path)
            idx = idx_by_scenario.get(scenario_id)
            if idx is not None:
                if debug:
                    print(f"[load_synthchain_events] annotate_events_with_iocs input_events={len(out)}")
                out = annotate_events_with_iocs(out, idx)
                if debug:
                    print(f"[load_synthchain_events] annotate_events_with_iocs output_events={len(out)}")

    if debug:
        print(f"[load_synthchain_events] total_events={len(out)}")

    return out


def _event_ioc_tokens(ev: Event) -> List[str]:
    """
    Extract a small set of candidate strings to match against IOC values.
    We avoid scanning the whole message to keep it cheap.
    """
    tokens: List[str] = []

    def add(s: Any) -> None:
        if s is None:
            return
        t = str(s).strip().lower()
        if not t:
            return
        tokens.append(t)

    add(ev.src.key)
    add(ev.dst.key)

    # split compound keys (ip:port, domain|ip:port, etc.)
    for base in list(tokens):
        for sep in ("|", ":", "/", "\\"):
            if sep in base:
                for part in base.split(sep):
                    add(part)

    # Also match common text fields if present.
    add(ev.edge_attrs.get("cmdline"))
    add(ev.edge_attrs.get("evidence"))

    # From evidence/cmdline, extract IPs and file paths (high-value keys).
    txt = " ".join([str(ev.edge_attrs.get("cmdline") or ""), str(ev.edge_attrs.get("evidence") or "")])
    if txt.strip():
        ips, paths = extract_ips_and_paths(txt)
        for ip in ips[:10]:
            add(ip)
        for p in paths[:10]:
            add(p)

    # de-dup preserve order
    seen: set[str] = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def annotate_events_with_iocs(events: List[Event], idx: IOCIndex, *, max_values: int = 5) -> List[Event]:
    """
    Add IOC match metadata into edge_attrs:
    - is_ioc: bool
    - ioc_types: list[str]
    - ioc_values: list[str] (truncated)
    """
    out: List[Event] = []
    values = idx.values
    types_by_value = idx.types_by_value
    hits_by_file_line = idx.hits_by_file_line
    debug = str(os.environ.get("SYNTHCHAIN_DEBUG_LOAD") or "").strip() in {"1", "true", "yes", "y"}
    n_in = len(events)
    n_proc = 0

    for ev in events:
        n_proc += 1
        # 1) line-level match (preferred when available)
        src_file = str(ev.raw.get("source_file") or "")
        row_idx = ev.raw.get("row_idx")
        line_hit = False
        if src_file and isinstance(row_idx, int):
            file_map = hits_by_file_line.get(src_file)
            if file_map:
                # ground truth may be 1-based; try both
                for ln in (row_idx, row_idx + 1):
                    if ln in file_map:
                        pairs = file_map[ln]
                        hits = [v for (_, v) in pairs][:max_values]
                        hit_types = sorted({t for (t, _) in pairs if t})
                        ea = dict(ev.edge_attrs)
                        ea["is_ioc"] = True
                        ea["is_ioc_line"] = True
                        ea["ioc_values"] = hits
                        ea["ioc_types"] = hit_types
                        out.append(replace(ev, edge_attrs=ea))
                        line_hit = True
                        break
                else:
                    pass  # no line hit; fallback to value match
                if line_hit:
                    # This event already got line-level IOC annotation.
                    # Do not run the value-level fallback, to avoid mixing sources.
                    continue

        # 2) value-level match (fallback)
        hits: List[str] = []
        hit_types: set[str] = set()
        for tok in _event_ioc_tokens(ev):
            if tok in values:
                hits.append(tok)
                hit_types |= set(types_by_value.get(tok, set()))
                if len(hits) >= max_values:
                    break

        if hits:
            ea = dict(ev.edge_attrs)
            ea["is_ioc"] = True
            ea["is_ioc_line"] = False
            ea["ioc_values"] = hits
            ea["ioc_types"] = sorted(hit_types) if hit_types else []
            out.append(replace(ev, edge_attrs=ea))
        else:
            ea = dict(ev.edge_attrs)
            if "is_ioc" not in ea:
                ea["is_ioc"] = False
            if "is_ioc_line" not in ea:
                ea["is_ioc_line"] = False
            out.append(replace(ev, edge_attrs=ea))

    if debug:
        print(f"[annotate_events_with_iocs] processed={n_proc} in={n_in} out={len(out)}")
        if n_proc != n_in or len(out) != n_in:
            raise RuntimeError(
                f"annotate_events_with_iocs length mismatch: processed={n_proc} in={n_in} out={len(out)}"
            )
    return out


