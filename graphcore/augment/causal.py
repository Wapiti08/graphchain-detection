from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from config.ontology import EdgeType, NodeType, canonical_edge_attrs, fill_defaults
from parsers.events import EntityRef, Event


CausalMode = Literal["off", "level0", "level1"]


def _t(ev: Event) -> float:
    return float(ev.ts) if ev.ts is not None else float(ev.order)


def _dt(a: Event, b: Event) -> float:
    return _t(b) - _t(a)


def _same_proc(ev: Event) -> str:
    # Treat the "actor" as the PROC source when available.
    return ev.src.key if ev.src.type == NodeType.PROC else ""


def _proc_cause_edge(
    scenario_hint: str,
    src_proc_key: str,
    dst_proc_key: str,
    *,
    ts: Optional[float],
    order: int,
    delta_t: float,
    rule: str,
    confidence: float = 1.0,
) -> Event:
    # Use PROC->PROC CAUSE edge
    src = EntityRef(NodeType.PROC, src_proc_key)
    dst = EntityRef(NodeType.PROC, dst_proc_key)
    attrs = fill_defaults(
        canonical_edge_attrs(EdgeType.CAUSE),
        {"delta_t": delta_t, "cause_rule": rule, "confidence": confidence},
    )
    return Event(
        edge_type=EdgeType.CAUSE,
        src=src,
        dst=dst,
        ts=ts,
        order=order,
        edge_attrs=attrs,
        raw={"log": "__causal__", "scenario": scenario_hint, "rule": rule},
    )


def augment_events_with_causal(
    events: List[Event],
    *,
    mode: CausalMode = "level0",
    window: float = 50.0,
    max_out_per_proc: int = 1000,
) -> List[Event]:
    """
    Deterministic causal augmentation for temporal graph learning.

    - mode="level0": connect adjacent events of the same PROC (sequence dependency)
    - mode="level1": level0 + lightweight bridge rules (shared NET/FILE dst within window)

    Time handling:
    - uses Event.ts if present
    - otherwise uses Event.order (pseudo-time) when ts is absent

    `window` is interpreted in the same unit as t(ev):
    - seconds if ts is present
    - steps if ts is None
    """
    if mode == "off" or not events:
        return events

    # stable, deterministic ordering first
    base = sorted(events, key=lambda e: (_t(e), e.order, e.edge_type.value, e.src.key, e.dst.key))

    scenario_hint = str(base[0].raw.get("scenario") or "")

    added: List[Event] = []
    next_order = (max((e.order for e in base), default=0) + 1) if base else 0

    # --- Level 0: per-proc adjacent edges ---
    by_proc: Dict[str, List[Event]] = {}
    for ev in base:
        p = _same_proc(ev)
        if p:
            by_proc.setdefault(p, []).append(ev)

    for proc_key, seq in by_proc.items():
        # already sorted by global key; ensure stable
        seq = sorted(seq, key=lambda e: (_t(e), e.order, e.edge_type.value, e.dst.key))
        out_ct = 0
        for a, b in zip(seq, seq[1:]):
            delta = _dt(a, b)
            if delta < 0:
                continue
            if delta > window:
                continue

            added.append(
                _proc_cause_edge(
                    scenario_hint,
                    proc_key,
                    proc_key,
                    ts=b.ts,
                    order=next_order,
                    delta_t=delta,
                    rule="next_event_same_proc",
                    confidence=1.0,
                )
            )
            next_order += 1
            out_ct += 1
            if out_ct >= max_out_per_proc:
                break

    if mode == "level0":
        return base + added

    # --- Level 1: shared object bridge rules (still PROC->PROC cause edges) ---
    # Map dst entity -> last event (per proc) within window
    last_by_proc_dst: Dict[Tuple[str, str], Event] = {}
    for ev in base:
        proc_key = _same_proc(ev)
        if not proc_key:
            continue

        # only consider edges that touch NET/FILE as "objects"
        if ev.dst.type not in (NodeType.NET, NodeType.FILE):
            continue

        k = (proc_key, f"{ev.dst.type.value}:{ev.dst.key}")
        prev = last_by_proc_dst.get(k)
        if prev is not None:
            delta = _dt(prev, ev)
            if 0 <= delta <= window:
                rule = "shared_object_bridge"
                added.append(
                    _proc_cause_edge(
                        scenario_hint,
                        proc_key,
                        proc_key,
                        ts=ev.ts,
                        order=next_order,
                        delta_t=delta,
                        rule=rule,
                        confidence=0.8,
                    )
                )
                next_order += 1
        last_by_proc_dst[k] = ev

    return base + added

