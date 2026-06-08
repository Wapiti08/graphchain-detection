"""Random-score baseline (lower bound)."""
from __future__ import annotations

import random
from typing import List

from gchain.train.streams import Stream


def random_scores(
    st: Stream,
    *,
    tail_start: int,
    seed: int = 0,
) -> List[float]:
    rng = random.Random(int(seed))
    n_tail = int(st.src.numel()) - int(tail_start)
    return [rng.random() for _ in range(n_tail)]


def score_fn_for_latency(st: Stream, tail_start: int, seed: int = 0):
    def _fn() -> List[float]:
        return random_scores(st, tail_start=tail_start, seed=seed)

    return _fn
