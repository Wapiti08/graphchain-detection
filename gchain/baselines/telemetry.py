"""Scheme A telemetry filters for SynthChain single-source baselines."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from gchain.train.streams import Stream

TelemetryKind = Literal["full", "audit", "zeek", "eve"]


class Telemetry(str, Enum):
    FULL = "full"
    AUDIT = "audit"
    ZEEK = "zeek"
    EVE = "eve"


# Basenames / patterns aligned with config/synthchain_sources.py log filenames.
_AUDIT_NAMES: Set[str] = {
    "azure_events.csv",
    "azure_conn.csv",
    "azure_process.csv",
    "azure_syslog.csv",
}
_ZEEK_PREFIXES: Tuple[str, ...] = ("zeek_",)
_EVE_NAMES: Set[str] = {"eve.json"}


def source_file_basename(source_file: str) -> str:
    s = str(source_file or "").strip()
    if not s:
        return ""
    return Path(s).name


def classify_source_file(source_file: str) -> Optional[Telemetry]:
    """Return telemetry family for a row's source_file, or None if unknown."""
    name = source_file_basename(source_file).lower()
    if not name:
        return None
    if name in _EVE_NAMES:
        return Telemetry.EVE
    if name in {n.lower() for n in _AUDIT_NAMES}:
        return Telemetry.AUDIT
    if any(name.startswith(p) for p in _ZEEK_PREFIXES):
        return Telemetry.ZEEK
    return None


def edge_mask_for_telemetry(
    source_files: Optional[Sequence[str]],
    telemetry: TelemetryKind,
) -> List[bool]:
    """Per-edge mask; ``full`` keeps all edges."""
    if telemetry == "full" or source_files is None:
        n = len(source_files) if source_files is not None else 0
        return [True] * n
    target = Telemetry(telemetry)
    out: List[bool] = []
    for sf in source_files:
        fam = classify_source_file(str(sf))
        out.append(fam == target)
    return out


def scenario_supports_telemetry(scenario: str, telemetry: TelemetryKind) -> bool:
    """Whether ``scenario`` has any edges from the requested telemetry family."""
    if telemetry == "full":
        return True
    from config.synthchain_sources import SYNTHCHAIN_IOC_CONFIG

    sc = SYNTHCHAIN_IOC_CONFIG.get(str(scenario)) or {}
    for spec in (sc.get("logs") or {}).values():
        if not isinstance(spec, dict):
            continue
        fn = source_file_basename(str(spec.get("filename") or "")).lower()
        if not fn:
            continue
        if telemetry == "audit" and fn in {n.lower() for n in _AUDIT_NAMES}:
            return True
        if telemetry == "zeek" and any(fn.startswith(p) for p in _ZEEK_PREFIXES):
            return True
        if telemetry == "eve" and fn in {n.lower() for n in _EVE_NAMES}:
            return True
    return False


def filter_stream_indices(
    st: "Stream",
    telemetry: TelemetryKind,
) -> List[int]:
    """Return edge indices kept for ``telemetry``."""
    n = int(st.src.numel())
    if telemetry == "full":
        return list(range(n))
    mask = edge_mask_for_telemetry(st.source_file, telemetry)
    if len(mask) != n:
        return list(range(n))
    return [i for i, m in enumerate(mask) if m]


def subset_stream(st: "Stream", indices: Sequence[int]) -> "Stream":
    """Slice stream tensors to ``indices`` (order preserved)."""
    import torch

    if not indices:
        z = torch.zeros(0, dtype=torch.long)
        zm = torch.zeros((0, st.msg.size(-1)), dtype=st.msg.dtype)
        return type(st)(
            src=z,
            dst=z,
            t=z,
            msg=zm,
            etype=z,
            y_ioc=None if st.y_ioc is None else z,
            y_ioc_line=None if st.y_ioc_line is None else z,
            y_rule=None if st.y_rule is None else z,
            y_rule_high=None if st.y_rule_high is None else z,
            row_idx=None if st.row_idx is None else z,
            source_file=None,
            ioc_type=None,
            rule_ioc_type=None,
        )
    idx = torch.tensor(list(indices), dtype=torch.long)
    sf = st.source_file
    it = st.ioc_type
    rit = st.rule_ioc_type
    return type(st)(
        src=st.src[idx],
        dst=st.dst[idx],
        t=st.t[idx],
        msg=st.msg[idx],
        etype=st.etype[idx],
        y_ioc=(st.y_ioc[idx] if st.y_ioc is not None else None),
        y_ioc_line=(st.y_ioc_line[idx] if st.y_ioc_line is not None else None),
        y_rule=(st.y_rule[idx] if st.y_rule is not None else None),
        y_rule_high=(st.y_rule_high[idx] if st.y_rule_high is not None else None),
        row_idx=(st.row_idx[idx] if st.row_idx is not None else None),
        source_file=(tuple(sf[i] for i in indices) if sf is not None else None),
        ioc_type=(tuple(it[i] for i in indices) if it is not None else None),
        rule_ioc_type=(tuple(rit[i] for i in indices) if rit is not None else None),
    )
