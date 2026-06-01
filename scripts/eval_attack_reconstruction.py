#!/usr/bin/env python3
"""Evaluate partial attack-chain reconstruction from TGN tail scores + stage GT."""
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

from gchain.eval.attack_reconstruct import (
    build_line_to_ioc_type,
    evaluate_reconstruction,
    ioc_log_source_files_for_scenario,
    load_ioc_type_to_stage,
    load_json,
)


def _read_score_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores-csv", type=str, required=True, help="best_eval_tail_scores.csv from training.")
    p.add_argument("--stage-gt", type=str, required=True, help="artifacts/stage_gt/<sc>.stages_gt.json")
    p.add_argument(
        "--ioc-gt",
        type=str,
        default="data/SynthChain/iocs/ioc_ground_truth.json",
        help="IOC ground truth for line-level type fallback.",
    )
    p.add_argument("--topks", type=str, default="10,50,100,500", help="Comma-separated K values.")
    p.add_argument(
        "--pred-min-prob",
        type=float,
        default=0.0,
        help="For by_k_pred_stage / by_alert_pred_stage: require pred_stage_prob >= this (0 = off).",
    )
    p.add_argument(
        "--pred-min-count",
        type=int,
        default=1,
        help="Min edges (per top-K or per alert) voting the same stage to include it.",
    )
    p.add_argument(
        "--include-ioc-pool-upper-bound",
        action="store_true",
        help="Emit by_k_ioc_pool_upper_bound (oracle IOC pool top-K; diagnostic ceiling only).",
    )
    p.add_argument(
        "--ioc-log-sources",
        action="store_true",
        help="Also evaluate by_k_ioc_log_sources (top-K pool = edges from has_ioc=True logs only).",
    )
    p.add_argument(
        "--no-ioc-log-sources",
        action="store_true",
        help="Skip by_k_ioc_log_sources even when scenario is known (default: emit for SynthChain scenarios).",
    )
    p.add_argument(
        "--no-pair-dedupe-recon",
        action="store_true",
        help="Skip by_k_pair_dedupe (max score per scenario,etype,src,dst then top-K).",
    )
    p.add_argument(
        "--no-alert-reconstruction",
        action="store_true",
        help="Skip by_alert_rule / by_alert_pred_stage metrics.",
    )
    p.add_argument("--alert-window", type=int, default=3600, help="Alert time window (same unit as t).")
    p.add_argument(
        "--alert-quantile",
        type=float,
        default=0.99,
        help="Per-scenario score quantile before clustering (ignored if --alert-topk-events > 0).",
    )
    p.add_argument("--alert-min-events", type=int, default=3, help="Min events per connected alert cluster.")
    p.add_argument(
        "--alert-topk-events",
        type=int,
        default=0,
        help="If >0, keep only top-K events by score per scenario before clustering.",
    )
    p.add_argument(
        "--no-alert-dedupe",
        action="store_true",
        help="Do not dedupe (scenario,t,etype,src,dst) before alert selection.",
    )
    p.add_argument("--out", type=str, default="", help="Output JSON (default: beside scores as reconstruction_metrics.json).")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    scores_csv = Path(args.scores_csv)
    if not scores_csv.is_absolute():
        scores_csv = (repo / scores_csv).resolve()
    stage_gt_path = Path(args.stage_gt)
    if not stage_gt_path.is_absolute():
        stage_gt_path = (repo / stage_gt_path).resolve()
    ioc_gt_path = (repo / args.ioc_gt).resolve()

    rows = _read_score_rows(scores_csv)
    stages_gt = load_json(stage_gt_path)
    scenario = str(stages_gt.get("scenario") or "")
    if not scenario and rows:
        scenario = str(rows[0].get("scenario") or "")

    ioc_gt = load_json(ioc_gt_path)
    ioc_type_to_stage = load_ioc_type_to_stage(repo)
    line_to_type = build_line_to_ioc_type(ioc_gt, scenario) if scenario else {}

    topks = [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]

    ioc_log_files = None
    if bool(args.ioc_log_sources) or not bool(args.no_ioc_log_sources):
        if scenario:
            allowed = ioc_log_source_files_for_scenario(scenario)
            if allowed:
                ioc_log_files = allowed

    metrics = evaluate_reconstruction(
        rows=rows,
        stages_gt=stages_gt,
        ioc_type_to_stage=ioc_type_to_stage,
        line_to_type=line_to_type,
        topks=topks,
        pred_min_prob=float(args.pred_min_prob),
        pred_min_count=int(args.pred_min_count),
        include_ioc_pool_upper_bound=bool(args.include_ioc_pool_upper_bound),
        ioc_log_source_files=ioc_log_files,
        run_alert_reconstruction=not bool(args.no_alert_reconstruction),
        alert_window=int(args.alert_window),
        alert_quantile=float(args.alert_quantile),
        alert_min_events=int(args.alert_min_events),
        alert_topk_events=int(args.alert_topk_events),
        alert_dedupe=not bool(args.no_alert_dedupe),
    )
    metrics["scenario"] = scenario
    metrics["scores_csv"] = str(scores_csv)
    metrics["stage_gt"] = str(stage_gt_path)
    metrics["n_score_rows"] = len(rows)

    if bool(args.no_pair_dedupe_recon):
        metrics.pop("by_k_pair_dedupe", None)
        metrics.pop("candidate_pool_pair_dedupe", None)

    out_path = Path(args.out) if args.out else scores_csv.parent / "reconstruction_metrics.json"
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
