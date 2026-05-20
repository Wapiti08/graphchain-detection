#!/usr/bin/env python3
"""
Build reproducible SynthChain stage ground-truth files from:

- ATT&CK Navigator layers in each scenario sanidata/ (payload_ttp*.json / payload_ttps*.json / attack_navigator_*.json)
- IOC ground truth (data/SynthChain/iocs/ioc_ground_truth.json)

Outputs one JSON per scenario with:
- semantic stages: derived from Navigator 'tactic' via config/attack_stage_map.json
- observable stages: derived from IOC 'type' via config/ioc_type_to_stage.json (line-level evidence-backed)

Example:
  python3 scripts/build_synthchain_stage_gt.py
  python3 scripts/build_synthchain_stage_gt.py --out-dir artifacts/stage_gt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Technique:
    technique_id: str
    tactic: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_tactic(x: object) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    s = s.lower().replace("_", "-").replace(" ", "-")
    return s


def _extract_techniques_from_layer(obj: Any) -> List[Technique]:
    """
    Accepts ATT&CK Navigator layer JSON format:
      { "techniques": [ {"techniqueID": "...", "tactic": "..."} , ... ] }
    """
    if not isinstance(obj, dict):
        return []
    techs = obj.get("techniques")
    if not isinstance(techs, list):
        return []
    out: List[Technique] = []
    for t in techs:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("techniqueID") or "").strip()
        tac = _norm_tactic(t.get("tactic"))
        if not tid:
            continue
        out.append(Technique(technique_id=tid, tactic=tac))
    return out


def _iter_scenario_layer_paths(sanidata_dir: Path) -> Iterable[Path]:
    # allow multiple variants across scenarios
    patterns = [
        "payload_ttp_*.json",
        "payload_ttps_*.json",
        "attack_navigator_*.json",
    ]
    for pat in patterns:
        for p in sorted(sanidata_dir.glob(pat)):
            if p.is_file():
                yield p


def _collect_semantic_stages_for_scenario(
    *, layer_paths: Sequence[Path], tactic_to_stage: Mapping[str, str]
) -> Tuple[Set[str], List[Dict[str, str]]]:
    stages: Set[str] = set()
    techniques: List[Dict[str, str]] = []
    for p in layer_paths:
        obj = _load_json(p)
        techs = _extract_techniques_from_layer(obj)
        for te in techs:
            if te.tactic:
                st = tactic_to_stage.get(te.tactic)
                if st:
                    stages.add(st)
            techniques.append(
                {
                    "technique_id": te.technique_id,
                    "tactic": te.tactic,
                    "source": p.name,
                }
            )
    return stages, techniques


def _collect_observable_stages_from_iocs(
    *, ioc_gt: Mapping[str, Any], scenario_id: str, ioc_type_to_stage: Mapping[str, str]
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Observable stages are evidence-backed by IOC annotations (line-level or value-level).
    We map IOC 'type' -> stage, and count IOC entries per stage.
    """
    sc = ioc_gt.get(scenario_id) or {}
    files = (sc.get("files") if isinstance(sc, dict) else None) or {}

    stages: Set[str] = set()
    counts: Dict[str, int] = {}
    for _, fobj in (files.items() if isinstance(files, dict) else []):
        for ioc in (fobj.get("iocs") if isinstance(fobj, dict) else None) or []:
            if not isinstance(ioc, dict):
                continue
            typ = str(ioc.get("type") or "").strip()
            st = ioc_type_to_stage.get(typ)
            if not st:
                continue
            stages.add(st)
            counts[st] = counts.get(st, 0) + 1
    return stages, counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=str, default="", help="Repo root (defaults to script parent).")
    p.add_argument(
        "--scenarios",
        type=str,
        default="sc1,sc2,sc3,sc4,sc5,sc6,sc7",
        help="Comma-separated scenario ids.",
    )
    p.add_argument(
        "--sanidata-root",
        type=str,
        default="data/SynthChain/sanidata",
        help="Root directory containing sc*/ folders with payload_ttp(s) layers.",
    )
    p.add_argument(
        "--ioc-gt",
        type=str,
        default="data/SynthChain/iocs/ioc_ground_truth.json",
        help="IOC ground truth JSON path (relative to repo).",
    )
    p.add_argument(
        "--attack-stage-map",
        type=str,
        default="config/attack_stage_map.json",
        help="Tactic->stage mapping JSON (relative to repo).",
    )
    p.add_argument(
        "--ioc-type-to-stage",
        type=str,
        default="config/ioc_type_to_stage.json",
        help="IOC type->stage mapping JSON (relative to repo).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/stage_gt",
        help="Output directory (relative to repo root).",
    )
    args = p.parse_args()

    repo = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    sanidata_root = (repo / args.sanidata_root).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_map = _load_json((repo / args.attack_stage_map).resolve())
    ioc_map = _load_json((repo / args.ioc_type_to_stage).resolve())
    tactic_to_stage = dict((stage_map.get("tactic_to_stage") or {}))
    ioc_type_to_stage = dict((ioc_map.get("ioc_type_to_stage") or {}))

    ioc_gt = _load_json((repo / args.ioc_gt).resolve())
    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]

    summary: List[Dict[str, Any]] = []

    for sc in scenarios:
        sc_dir = sanidata_root / sc
        layer_paths = list(_iter_scenario_layer_paths(sc_dir)) if sc_dir.is_dir() else []
        semantic_stages, techniques = _collect_semantic_stages_for_scenario(
            layer_paths=layer_paths, tactic_to_stage=tactic_to_stage
        )
        observable_stages, obs_counts = _collect_observable_stages_from_iocs(
            ioc_gt=ioc_gt, scenario_id=sc, ioc_type_to_stage=ioc_type_to_stage
        )

        out = {
            "scenario": sc,
            "semantic": {
                "stages": sorted(semantic_stages),
                "techniques": techniques,
                "layer_files": [p.name for p in layer_paths],
            },
            "observable": {
                "stages": sorted(observable_stages),
                "ioc_counts_by_stage": {k: int(v) for k, v in sorted(obs_counts.items())},
            },
        }
        (out_dir / f"{sc}.stages_gt.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        summary.append(
            {
                "scenario": sc,
                "semantic_stage_count": int(len(semantic_stages)),
                "observable_stage_count": int(len(observable_stages)),
                "layer_files": [p.name for p in layer_paths],
            }
        )

    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} scenarios -> {out_dir}")


if __name__ == "__main__":
    main()

