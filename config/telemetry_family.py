"""Telemetry-family filters for SynthChain log loading."""
from __future__ import annotations

from typing import Literal

TelemetryFamily = Literal["full", "audit", "zeek", "eve"]


def normalize_telemetry_family(value: str | None) -> TelemetryFamily:
    family = str(value or "full").strip().lower()
    aliases = {
        "all": "full",
        "azure": "audit",
        "ids": "eve",
        "suricata": "eve",
    }
    family = aliases.get(family, family)
    if family not in {"full", "audit", "zeek", "eve"}:
        raise ValueError(
            f"Unknown telemetry_family={value!r}; expected full, audit, zeek, or eve."
        )
    return family  # type: ignore[return-value]


def log_name_in_family(log_name: str, family: TelemetryFamily) -> bool:
    name = str(log_name or "").strip().lower()
    if family == "full":
        return True
    if family == "audit":
        return name.startswith("azure_")
    if family == "zeek":
        return name.startswith("zeek_")
    if family == "eve":
        return name in {"eve", "suricata"} or name.startswith("eve_") or name.startswith("suricata_")
    return False
