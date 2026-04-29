from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def parse_ts_to_unix_seconds(x: Any) -> Optional[float]:
    """
    Parse various timestamp formats into unix seconds (float).

    - Handles strings like "22/04/2025, 13:03:00.020" and ISO-8601.
    - Returns None if parsing fails.
    """
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "":
            return None

        # Most SynthChain sources are either:
        # - Azure: "22/04/2025, 13:03:00.020" (dayfirst=True)
        # - Zeek:  "2025-12-14 16:00:00.035702944" (dayfirst=False)
        # - EVE:   "2025-12-14T06:16:15.290604-0500" (dayfirst=False)
        if s[0].isdigit() and "-" in s[:10]:
            ts = pd.to_datetime(s, utc=True, errors="coerce", dayfirst=False)
        else:
            ts = pd.to_datetime(s, utc=True, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            return None
        return float(ts.timestamp())
    except Exception:
        return None

