"""Path-LOF style baseline (minimal stub; extend with graph path features)."""
from __future__ import annotations

from typing import List

from gchain.baselines.freq_rarity import freq_rarity_scores
from gchain.train.streams import Stream


def path_lof_scores(
    st: Stream,
    *,
    train_end: int,
    tail_start: int,
) -> List[float]:
    """Placeholder: delegate to freq-rarity until path-LOF graph features are wired."""
    return freq_rarity_scores(st, train_end=train_end, tail_start=tail_start)


def score_fn_for_latency(st: Stream, train_end: int, tail_start: int):
    def _fn() -> List[float]:
        return path_lof_scores(st, train_end=train_end, tail_start=tail_start)

    return _fn
