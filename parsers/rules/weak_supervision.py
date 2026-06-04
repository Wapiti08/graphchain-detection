"""Scenario-agnostic weak supervision from IOC-taxonomy rules (no GT line/value IOCs)."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config.ontology import EdgeType
from graphcore.edge_meta import pick_primary_ioc_type
from parsers.events import Event


def _repo_rules_path(project_root: Path) -> Path:
    return project_root / "config" / "weak_supervision_rules.json"


@lru_cache(maxsize=4)
def load_weak_supervision_rules(project_root: str) -> Dict[str, Any]:
    path = _repo_rules_path(Path(project_root))
    if not path.is_file():
        raise FileNotFoundError(f"Missing weak supervision config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _compile_tier2(patterns: Sequence[Mapping[str, Any]]) -> List[Tuple[str, re.Pattern, str, bool]]:
    out: List[Tuple[str, re.Pattern, str, bool]] = []
    for p in patterns:
        rid = str(p.get("id") or "").strip()
        pat = str(p.get("pattern") or "").strip()
        conf = str(p.get("confidence") or "medium").strip().lower()
        if not rid or not pat:
            continue
        rank_ok = p.get("rank_eligible", True) is not False
        out.append((rid, re.compile(pat), conf, bool(rank_ok)))
    return out


def _rank_policy(rules: Mapping[str, Any]) -> Dict[str, Any]:
    pol = rules.get("rank_policy")
    return dict(pol) if isinstance(pol, dict) else {}


def _exclude_rule_types(rules: Mapping[str, Any]) -> set[str]:
    pol = _rank_policy(rules)
    excluded = set(pol.get("exclude_rule_types") or [])
    for entry in rules.get("tier2_benign_adjacent_disabled") or []:
        if isinstance(entry, dict) and entry.get("id"):
            excluded.add(str(entry["id"]))
    return excluded


def _finalize_rule_confidence(
    hits: List[str],
    *,
    rules: Mapping[str, Any],
    ioc_log_source: bool,
    raw_conf_rank: int,
) -> str:
    """Map raw hit strength -> exported confidence with rank_policy gates."""
    pol = _rank_policy(rules)
    excluded = _exclude_rule_types(rules)
    active = [h for h in hits if h not in excluded]
    if not active:
        return ""

    strong = {str(x) for x in (pol.get("strong_rule_types_single_hit") or [])}
    min_types = int(pol.get("min_rule_types_for_high") or 2)
    need_ioc_src = bool(pol.get("require_ioc_log_source_for_high", False))

    qualifies_high = False
    if any(h in strong for h in active):
        qualifies_high = True
    elif len(active) >= max(1, min_types):
        qualifies_high = True

    if need_ioc_src and not ioc_log_source:
        qualifies_high = False

    if qualifies_high and raw_conf_rank >= 2:
        return "high"
    if raw_conf_rank >= 1 or active:
        return "medium"
    return ""


def _event_text(ev: Event) -> str:
    ea = ev.edge_attrs
    return " ".join(
        [
            str(ea.get("cmdline") or ""),
            str(ea.get("evidence") or ""),
        ]
    ).strip()


def _port_from_net_key(key: str) -> Optional[int]:
    k = (key or "").strip().lower()
    if not k:
        return None
    # ip:port, domain|ip:port
    if "|" in k:
        k = k.split("|")[-1]
    if ":" in k:
        tail = k.rsplit(":", 1)[-1]
        try:
            p = int(tail)
            if 0 < p < 65536:
                return p
        except ValueError:
            return None
    return None


def infer_rule_hits_for_event(
    ev: Event,
    rules: Mapping[str, Any],
    *,
    ioc_log_source: bool = False,
) -> Tuple[List[str], str]:
    """
    Return (rule_type_ids, confidence) where confidence is 'high' | 'medium' | ''.
    Does not read ioc_ground_truth.json or is_ioc flags.
    """
    tier1 = rules.get("tier1") or {}
    rule_type_to_ioc = dict(rules.get("rule_type_to_ioc_type") or {})
    hits: List[str] = []
    conf_rank = 0  # 0 none, 1 medium, 2 high

    def bump(level: int) -> None:
        nonlocal conf_rank
        conf_rank = max(conf_rank, level)

    text = _event_text(ev)
    text_l = text.lower()
    ea = ev.edge_attrs
    src_file = str(ev.raw.get("source_file") or "").lower()

    # --- Tier 1: Suricata / IDS ---
    for suf in tier1.get("suricata_source_suffixes") or []:
        if src_file.endswith(str(suf).lower()):
            hits.append("suricata_alert")
            bump(2)
            break
    if "suricata_alert" not in hits:
        for mk in tier1.get("suricata_cmdline_markers") or []:
            if str(mk).lower() in text_l:
                hits.append("suricata_alert")
                bump(2)
                break

    # --- Tier 1: INJECT ---
    et_name = ev.edge_type.value if isinstance(ev.edge_type, EdgeType) else str(ev.edge_type)
    if et_name in {str(x) for x in (tier1.get("inject_edge_types") or [])}:
        hits.append("inject_edge")
        bump(2)

    # --- Tier 1: suspicious port on CONNECT-like edges ---
    if et_name in {str(x) for x in (tier1.get("connect_edge_types") or [])}:
        suspicious = {int(p) for p in (tier1.get("suspicious_ports") or [])}
        for key in (ev.dst.key, ev.src.key):
            port = _port_from_net_key(str(key))
            if port is not None and port in suspicious:
                hits.append("suspicious_port")
                bump(1)
                break

    excluded = _exclude_rule_types(rules)
    pol = _rank_policy(rules)
    download_single_max = str(
        pol.get("lolbin_download_single_flag_max_confidence") or "medium"
    ).strip().lower()

    # --- Tier 1: LOLBin / cmd flags already on edge_attrs ---
    lolbin_map = dict(tier1.get("lolbin_flags") or {})
    needs_pair = {str(x) for x in (tier1.get("lolbin_requires_second_flag_for_high") or [])}
    active_flags = [fk for fk in lolbin_map if bool(ea.get(fk))]
    if active_flags:
        for fk in active_flags:
            rt = str(lolbin_map[fk])
            if rt in excluded:
                continue
            if rt not in hits:
                hits.append(rt)
        download_only = active_flags and all(
            fk in needs_pair for fk in active_flags
        )
        if any(fk not in needs_pair for fk in active_flags):
            bump(2)
        elif len(active_flags) >= 2:
            bump(2)
        elif download_only and download_single_max != "high":
            bump(1)
        else:
            bump(1)

    # --- Tier 2: cmdline regex (rank-eligible only; benign install patterns disabled in config) ---
    tier2_compiled = _compile_tier2(rules.get("tier2_cmdline_patterns") or [])
    if text:
        for rid, rx, conf, rank_ok in tier2_compiled:
            if rid in excluded or not rank_ok:
                continue
            if rx.search(text):
                if rid not in hits:
                    hits.append(rid)
                bump(2 if conf == "high" else 1)

    # de-dup preserve order
    seen: set[str] = set()
    ordered: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            ordered.append(h)

    if not ordered:
        return [], ""

    active = [h for h in ordered if h not in excluded]
    if not active:
        return [], ""

    conf = _finalize_rule_confidence(
        ordered,
        rules=rules,
        ioc_log_source=ioc_log_source,
        raw_conf_rank=conf_rank,
    )
    return ordered, conf


def rule_types_to_ioc_types(
    rule_types: Sequence[str],
    rules: Mapping[str, Any],
) -> List[str]:
    m = rules.get("rule_type_to_ioc_type") or {}
    out: List[str] = []
    for rt in rule_types:
        it = str(m.get(rt) or "").strip()
        if it:
            out.append(it)
    return out


def annotate_events_with_weak_rules(
    events: List[Event],
    rules: Mapping[str, Any],
    *,
    ioc_log_sources: Optional[Mapping[str, bool]] = None,
) -> List[Event]:
    """
    Attach weak-rule metadata (parallel to GT IOC annotation):
    - is_rule_hit, is_rule_hit_high
    - rule_types, rule_ioc_types, rule_confidence
    - _rule_ioc_type (primary, for stage weak labels)
    """
    out: List[Event] = []
    for ev in events:
        src = str(ev.raw.get("source_file") or "")
        ioc_src = bool((ioc_log_sources or {}).get(src, False))
        rule_types, conf = infer_rule_hits_for_event(ev, rules, ioc_log_source=ioc_src)
        if not rule_types:
            out.append(ev)
            continue
        ioc_types = rule_types_to_ioc_types(rule_types, rules)
        ea = dict(ev.edge_attrs)
        ea["is_rule_hit"] = True
        ea["is_rule_hit_high"] = conf == "high"
        ea["rule_types"] = list(rule_types)
        ea["rule_ioc_types"] = list(ioc_types)
        ea["rule_confidence"] = conf
        if ioc_types:
            ea["_rule_ioc_type"] = pick_primary_ioc_type(ioc_types)
        out.append(replace(ev, edge_attrs=ea))
    return out


def ioc_log_source_map_for_scenario(scenario_id: str, project_root: Path) -> Dict[str, bool]:
    """Map ground-truth source filename -> log is IOC-bearing (has_ioc in synthchain_sources)."""
    from config.synthchain_sources import SYNTHCHAIN_IOC_CONFIG

    cfg = SYNTHCHAIN_IOC_CONFIG.get(scenario_id)
    if not cfg:
        return {}
    out: Dict[str, bool] = {}
    for spec in (cfg.get("logs") or {}).values():
        fn = Path(str(spec.get("filename") or "")).name
        if fn:
            out[fn] = bool(spec.get("has_ioc", False))
    return out
