"""Frequency / rarity baseline: rare train-prefix patterns score higher on the tail."""
from __future__ import annotations

from collections import Counter
from math import log
from typing import Dict, List, Sequence, Tuple

from gchain.train.streams import Stream


def _edge_key(st: Stream, idx: int) -> Tuple[int, int, int]:
    return (
        int(st.etype[idx].item()),
        int(st.src[idx].item()),
        int(st.dst[idx].item()),
    )


def freq_rarity_scores(
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
) -> List[float]:
    """Score tail edges by negative log frequency of (etype, src, dst) in the train prefix."""
    counts: Counter[Tuple[int, int, int]] = Counter()
    for i in range(int(train_end)):
        counts[_edge_key(st, i)] += 1

    n_train = max(1, int(train_end))
    out: List[float] = []
    for i in range(int(tail_start), int(st.src.numel())):
        key = _edge_key(st, i)
        c = int(counts.get(key, 0))
        # Unseen or rare patterns get higher scores (TGN uses -log p; higher = more anomalous).
        freq = (float(c) + 1.0) / float(n_train + 1)
        out.append(-log(freq))
    return out


def score_fn_for_latency(st: Stream, train_end: int, tail_start: int):
    """Return a callable for latency benchmarking (recomputes on each call)."""

    def _fn() -> List[float]:
        return freq_rarity_scores(st, train_end=train_end, tail_start=tail_start)

    return _fn
