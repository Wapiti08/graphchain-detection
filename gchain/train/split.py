from __future__ import annotations

import math


def time_split_idx(num_events: int, train_frac: float) -> int:
    """Index splitting [0:split) train prefix vs [split:E) eval tail."""
    if num_events <= 1:
        return 0
    k = int(math.floor(float(train_frac) * float(num_events)))
    return max(1, min(num_events - 1, k))
