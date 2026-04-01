from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import pandas as pd


IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s\"']+")
NIX_PATH_RE = re.compile(r"(?:^|[\s\"'])(/[^ \t\n\r\"']+)")


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
        # Try ISO-ish first to avoid pandas warnings, then fall back to dayfirst.
        if s[0].isdigit() and "-" in s[:10]:
            ts = pd.to_datetime(s, utc=True, errors="coerce", dayfirst=False)
        else:
            ts = pd.to_datetime(s, utc=True, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            return None
        return float(ts.timestamp())
    except Exception:
        return None


def unique_preserve_order(xs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_ips_and_paths(text: str) -> Tuple[List[str], List[str]]:
    ips = IP_RE.findall(text)
    paths = WIN_PATH_RE.findall(text)
    paths += [m.group(1) for m in NIX_PATH_RE.finditer(text)]
    return unique_preserve_order(ips), unique_preserve_order([p.strip() for p in paths if p.strip()])

