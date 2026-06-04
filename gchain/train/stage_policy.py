"""Stage supervision policy: IOC types excluded from stage CE (deployment/eval aligned)."""
from __future__ import annotations

from typing import Mapping

# Network indicators → command_and_control in ioc_type_to_stage.json; excluded from stage head training.
NETWORK_IOC_TYPES = frozenset(
    {
        "attack_ip",
        "suspicious_port",
        "dns_indicator",
        "c2_agent",
    }
)


def stage_for_ioc_type(ioc_type: str, ioc_type_to_stage: Mapping[str, str]) -> str:
    it = str(ioc_type or "").strip()
    if not it:
        return ""
    return str(ioc_type_to_stage.get(it, "") or "").strip()


def is_ioc_type_stage_eligible(ioc_type: str, *, project_root: str = "") -> bool:
    """Process/cmd-centric IOC types may receive stage CE; network types may not."""
    _ = project_root
    it = str(ioc_type or "").strip()
    if not it:
        return False
    return it not in NETWORK_IOC_TYPES
