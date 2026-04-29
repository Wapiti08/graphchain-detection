from __future__ import annotations

import re
from typing import List, Tuple


IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s\"']+")
NIX_PATH_RE = re.compile(r"(?:^|[\s\"'])(/[^ \t\n\r\"']+)")


def _unique_preserve_order(xs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_ips_and_paths(text: str) -> Tuple[List[str], List[str]]:
    """
    Best-effort extraction for semi-structured logs (Azure events/syslog).
    """
    ips = IP_RE.findall(text)
    paths = WIN_PATH_RE.findall(text)
    paths += [m.group(1) for m in NIX_PATH_RE.finditer(text)]
    return _unique_preserve_order(ips), _unique_preserve_order([p.strip() for p in paths if p.strip()])

