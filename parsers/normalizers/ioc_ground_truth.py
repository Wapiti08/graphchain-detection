from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Set, Tuple


@dataclass(frozen=True)
class IOCIndex:
    """
    Fast lookup structure for IOC value -> types, per scenario.
    """

    values: Set[str]
    types_by_value: Mapping[str, Set[str]]


def load_ioc_ground_truth(path: str | Path) -> Mapping[str, IOCIndex]:
    """
    Load SynthChain IOC ground truth json into a per-scenario index.

    Expected structure:
    { "sc1": { "files": { "azure_events.csv": { "iocs": [ {"type": "...", "value": "..."} ]}}}}
    """
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))

    out: Dict[str, IOCIndex] = {}
    for scenario_id, scenario in obj.items():
        values: Set[str] = set()
        types_by_value: Dict[str, Set[str]] = {}

        files = (scenario or {}).get("files", {}) or {}
        for _, f in files.items():
            for ioc in (f or {}).get("iocs", []) or []:
                v = str(ioc.get("value", "")).strip()
                t = str(ioc.get("type", "")).strip()
                if not v:
                    continue
                v_norm = v.lower()
                values.add(v_norm)
                if v_norm not in types_by_value:
                    types_by_value[v_norm] = set()
                if t:
                    types_by_value[v_norm].add(t)

        out[str(scenario_id)] = IOCIndex(values=values, types_by_value=types_by_value)

    return out

