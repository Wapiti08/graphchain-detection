"""
Shared tail-score evaluation: dedupe, high-score selection, time windows,
connected-component clustering (same rules as scripts/aggregate_alerts.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ScoredEvent:
    scenario: str
    t: int
    etype: int
    src: int
    dst: int
    score: float
    is_ioc: int


def rows_to_events(rows: Iterable[Dict[str, Any]]) -> List[ScoredEvent]:
    out: List[ScoredEvent] = []
    for row in rows:
        out.append(
            ScoredEvent(
                scenario=str(row["scenario"]),
                t=int(float(row["t"])),
                etype=int(row["etype"]),
                src=int(row["src"]),
                dst=int(row["dst"]),
                score=float(row["score"]),
                is_ioc=int(row.get("is_ioc", "0") or 0),
            )
        )
    return out


def dedupe_events(evs: List[ScoredEvent]) -> List[ScoredEvent]:
    merged: Dict[Tuple[str, int, int, int, int], ScoredEvent] = {}
    for e in evs:
        k = (e.scenario, e.t, e.etype, e.src, e.dst)
        old = merged.get(k)
        if old is None:
            merged[k] = e
            continue
        merged[k] = ScoredEvent(
            scenario=e.scenario,
            t=e.t,
            etype=e.etype,
            src=e.src,
            dst=e.dst,
            score=max(old.score, e.score),
            is_ioc=1 if (old.is_ioc or e.is_ioc) else 0,
        )
    return list(merged.values())


def quantile_cutoff(values: List[float], q: float) -> float:
    if not values:
        return float("inf")
    q = max(0.0, min(1.0, float(q)))
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return float(xs[idx])


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent.get(x, x)
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rka = self.rank.get(ra, 0)
        rkb = self.rank.get(rb, 0)
        if rka < rkb:
            ra, rb = rb, ra
            rka, rkb = rkb, rka
        self.parent[rb] = ra
        if rka == rkb:
            self.rank[ra] = rka + 1


def time_windows(events: List[ScoredEvent], window: int) -> List[List[ScoredEvent]]:
    if not events:
        return []
    evs = sorted(events, key=lambda e: (e.t, -e.score))
    out: List[List[ScoredEvent]] = []
    cur: List[ScoredEvent] = []
    start_t = evs[0].t
    for e in evs:
        if cur and (e.t - start_t) > window:
            out.append(cur)
            cur = []
            start_t = e.t
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def cluster_connected(events: List[ScoredEvent]) -> List[List[ScoredEvent]]:
    uf = UnionFind()
    for e in events:
        uf.union(e.src, e.dst)
    comps: Dict[int, List[ScoredEvent]] = {}
    for e in events:
        root = uf.find(e.src)
        comps.setdefault(root, []).append(e)
    return list(comps.values())


def select_high_score(evs: List[ScoredEvent], *, topk_events: int, score_quantile: float) -> List[ScoredEvent]:
    evs_sorted = sorted(evs, key=lambda e: e.score, reverse=True)
    if int(topk_events) > 0:
        return evs_sorted[: int(topk_events)]
    cutoff = quantile_cutoff([e.score for e in evs_sorted], float(score_quantile))
    return [e for e in evs_sorted if e.score >= cutoff]


def count_alerts(
    by_scenario: Dict[str, List[ScoredEvent]],
    *,
    window: int,
    score_quantile: float,
    min_events: int,
    topk_events: int,
    dedupe: bool,
) -> Tuple[int, int, int, int]:
    """
    Returns (num_alerts, num_tail_after_dedupe, num_flagged_in_selection, num_ioc_in_flagged).
    """
    num_alerts = 0
    num_tail = 0
    num_flagged = 0
    num_ioc_flagged = 0
    for _sc, evs in sorted(by_scenario.items()):
        if not evs:
            continue
        if dedupe:
            evs = dedupe_events(evs)
        num_tail += len(evs)
        kept = select_high_score(evs, topk_events=topk_events, score_quantile=score_quantile)
        num_flagged += len(kept)
        num_ioc_flagged += int(sum(e.is_ioc for e in kept))
        for win in time_windows(kept, int(window)):
            for cl in cluster_connected(win):
                if len(cl) >= int(min_events):
                    num_alerts += 1
    return num_alerts, num_tail, num_flagged, num_ioc_flagged


def precision_at_k_all(rows: List[Dict[str, Any]], ks: Sequence[int]) -> Dict[int, float]:
    """Global tail: sort all rows by score desc; precision@K = mean(is_ioc in top K)."""
    if not rows:
        return {int(k): float("nan") for k in ks}
    evs = rows_to_events(rows)
    evs = sorted(evs, key=lambda e: e.score, reverse=True)
    out: Dict[int, float] = {}
    for k in ks:
        kk = min(int(k), len(evs))
        if kk == 0:
            out[int(k)] = float("nan")
        else:
            out[int(k)] = float(sum(e.is_ioc for e in evs[:kk]) / float(kk))
    return out


def tail_alert_metrics(
    rows: List[Dict[str, Any]],
    *,
    topks: Sequence[int],
    alert_window: int,
    alert_quantile: float,
    alert_min_events: int,
    alert_topk_events: int,
    dedupe: bool,
) -> Dict[str, float]:
    """Metrics aligned with aggregate_alerts defaults / knobs."""
    p_at = precision_at_k_all(rows, topks)
    by_sc: Dict[str, List[ScoredEvent]] = {}
    for e in rows_to_events(rows):
        by_sc.setdefault(e.scenario, []).append(e)
    n_alerts, n_tail, n_flagged, n_ioc_f = count_alerts(
        by_sc,
        window=int(alert_window),
        score_quantile=float(alert_quantile),
        min_events=int(alert_min_events),
        topk_events=int(alert_topk_events),
        dedupe=bool(dedupe),
    )
    prec_flagged = float(n_ioc_f) / float(max(1, n_flagged))
    return {
        **{f"p_at_{k}": float(p_at.get(int(k), float("nan"))) for k in topks},
        "num_tail_deduped": float(n_tail),
        "num_flagged": float(n_flagged),
        "num_alerts": float(n_alerts),
        "flagged_rate": float(n_flagged) / float(max(1, n_tail)),
        "alerts_per_tail_event": float(n_alerts) / float(max(1, n_tail)),
        "precision_in_flagged": prec_flagged,
    }
