#!/usr/bin/env python3
"""Ingest precomputed tail scores (FuseChain / static GNN) into benchmark JSON."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gchain.baselines.evaluate import evaluate_from_score_rows, write_result
from gchain.baselines.telemetry import TelemetryKind


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _filter_telemetry(rows: List[Dict[str, Any]], telemetry: TelemetryKind) -> List[Dict[str, Any]]:
    if telemetry == "full":
        return rows
    from gchain.baselines.telemetry import Telemetry, classify_source_file

    target = Telemetry(telemetry)
    out: List[Dict[str, Any]] = []
    for r in rows:
        fam = classify_source_file(str(r.get("source_file") or ""))
        if fam == target:
            out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, required=True, help="Label for main table row, e.g. fusechain.")
    p.add_argument("--scores-csv", type=str, required=True)
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--telemetry", type=str, default="full", choices=("full", "audit", "zeek", "eve"))
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/benchmarks/per_method",
    )
    p.add_argument(
        "--latency-ms-per-1k",
        type=float,
        default=float("nan"),
        help="Optional measured latency; omit for NaN in table.",
    )
    p.add_argument("--stage-gt-dir", type=str, default="artifacts/stage_gt")
    p.add_argument("--ioc-gt", type=str, default="data/SynthChain/iocs/ioc_ground_truth.json")
    args = p.parse_args()

    csv_path = Path(args.scores_csv)
    if not csv_path.is_absolute():
        csv_path = (_REPO / csv_path).resolve()
    rows = _read_rows(csv_path)
    rows = [r for r in rows if str(r.get("scenario") or "").strip() == args.scenario]
    telemetry: TelemetryKind = args.telemetry  # type: ignore[assignment]
    rows = _filter_telemetry(rows, telemetry)

    result = evaluate_from_score_rows(
        method=args.method,
        scenario=args.scenario,
        rows=rows,
        repo_root=_REPO,
        telemetry=telemetry,
        stage_gt_dir=args.stage_gt_dir,
        ioc_gt=args.ioc_gt,
        latency_ms_per_1k=(
            None if str(args.latency_ms_per_1k) == "nan" else float(args.latency_ms_per_1k)
        ),
    )
    out_path = (_REPO / args.out_dir / args.method / args.telemetry / f"{args.scenario}.json").resolve()
    write_result(out_path, result)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
