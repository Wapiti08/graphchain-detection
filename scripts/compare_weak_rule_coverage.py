#!/usr/bin/env python3
"""Compare weak-rule annotation counts: baseline rules vs an updated rules file."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from parsers.synthchain import load_synthchain_events
from parsers.rules.weak_supervision import (
    annotate_events_with_weak_rules,
    ioc_log_source_map_for_scenario,
    load_weak_supervision_rules,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rule_id_configured(rules: Dict[str, Any], rule_id: str) -> bool:
    if rule_id in (rules.get("rule_type_to_ioc_type") or {}):
        return True
    for pat in rules.get("tier2_cmdline_patterns") or []:
        if isinstance(pat, dict) and str(pat.get("id") or "") == rule_id:
            return True
    return False


def _rule_stats(events: List[Any], *, new_rule_id: str) -> Dict[str, int]:
    n = len(events)
    hit = sum(1 for e in events if e.edge_attrs.get("is_rule_hit"))
    high = sum(1 for e in events if e.edge_attrs.get("is_rule_hit_high"))
    new_hits = sum(
        1
        for e in events
        if new_rule_id in (e.edge_attrs.get("rule_types") or [])
    )
    return {
        "num_events": n,
        "rule_hit": hit,
        "rule_high": high,
        "new_rule_hits": new_hits,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare weak-rule coverage across SynthChain scenarios.")
    p.add_argument(
        "--baseline-rules",
        type=str,
        default="config/weak_supervision_rules.json",
    )
    p.add_argument(
        "--updated-rules",
        type=str,
        default="config/weak_supervision_rules_update_ablation.json",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="sc1,sc2,sc3,sc4,sc5,sc6,sc7",
    )
    p.add_argument(
        "--new-rule-id",
        type=str,
        default="tier2_staging_path_download",
        help="Rule id to count as the analyst-added update.",
    )
    p.add_argument("--only-ioc-logs", action="store_true", default=True)
    p.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/rules_update_ablation/rule_coverage_compare.csv",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="artifacts/rules_update_ablation/rule_coverage_compare.json",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    baseline_rules_path = (repo / args.baseline_rules).resolve()
    updated_rules_path = (repo / args.updated_rules).resolve()
    updated_rules = _load_json(updated_rules_path)
    new_rule_configured = _rule_id_configured(updated_rules, args.new_rule_id)

    print(
        "Rules config: "
        f"baseline={baseline_rules_path} "
        f"updated={updated_rules_path} "
        f"updated_version={updated_rules.get('version', '')} "
        f"new_rule_id={args.new_rule_id} "
        f"configured={new_rule_configured}"
    )
    if not new_rule_configured:
        print(
            f"Warning: {args.new_rule_id!r} is not defined in the updated rules file; "
            "coverage deltas for this rule will be zero."
        )

    rows: List[Dict[str, Any]] = []
    load_weak_supervision_rules.cache_clear()
    baseline_rules = load_weak_supervision_rules(str(repo), args.baseline_rules)
    updated_rules_loaded = load_weak_supervision_rules(str(repo), args.updated_rules)

    for sc in scenarios:
        ev_raw = load_synthchain_events(
            sc,
            project_root=repo,
            only_ioc_logs=bool(args.only_ioc_logs),
            annotate_weak_rules=False,
        )
        src_map = ioc_log_source_map_for_scenario(sc, repo)
        ev_base = annotate_events_with_weak_rules(
            ev_raw,
            baseline_rules,
            ioc_log_sources=src_map,
        )
        ev_upd = annotate_events_with_weak_rules(
            ev_raw,
            updated_rules_loaded,
            ioc_log_sources=src_map,
        )
        sb = _rule_stats(ev_base, new_rule_id=args.new_rule_id)
        su = _rule_stats(ev_upd, new_rule_id=args.new_rule_id)
        rows.append(
            {
                "scenario": sc,
                "baseline_rule_hit": sb["rule_hit"],
                "updated_rule_hit": su["rule_hit"],
                "delta_rule_hit": su["rule_hit"] - sb["rule_hit"],
                "baseline_rule_high": sb["rule_high"],
                "updated_rule_high": su["rule_high"],
                "delta_rule_high": su["rule_high"] - sb["rule_high"],
                "updated_new_rule_hits": su["new_rule_hits"],
                "new_rule_id": args.new_rule_id,
                "baseline_events": sb["num_events"],
                "updated_events": su["num_events"],
            }
        )

    out_csv = (repo / args.out_csv).resolve()
    out_json = (repo / args.out_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    for r in rows:
        print(
            f"  {r['scenario']}: rule_hit +{r['delta_rule_hit']} "
            f"rule_high +{r['delta_rule_high']} "
            f"new_rule={r['updated_new_rule_hits']}"
        )
    total_delta_hit = sum(int(r["delta_rule_hit"]) for r in rows)
    total_delta_high = sum(int(r["delta_rule_high"]) for r in rows)
    total_new_rule = sum(int(r["updated_new_rule_hits"]) for r in rows)
    print(
        "Summary: "
        f"delta_rule_hit={total_delta_hit} "
        f"delta_rule_high={total_delta_high} "
        f"{args.new_rule_id}={total_new_rule}"
    )
    if total_delta_hit == 0 and total_delta_high == 0 and total_new_rule == 0:
        print(
            "Note: updated rules produced no additional hits on the selected logs; "
            "treat this run as a pipeline/regression prototype unless the rule is revised."
        )


if __name__ == "__main__":
    main()
