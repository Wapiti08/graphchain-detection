"""Scoring latency helpers for benchmark tables."""
from __future__ import annotations

import statistics
import time
from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")


def ms_per_1k_edges(
    score_fn: Callable[[], Sequence[T]],
    *,
    n_tail_edges: int,
    warmup: int = 1,
    repeats: int = 3,
) -> float:
    """Median wall time to score ``n_tail_edges`` edges, scaled to ms per 1k edges."""
    if n_tail_edges <= 0:
        return float("nan")
    for _ in range(max(0, int(warmup))):
        score_fn()
    samples: List[float] = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        score_fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        samples.append(elapsed_ms * 1000.0 / float(n_tail_edges))
    return float(statistics.median(samples))
