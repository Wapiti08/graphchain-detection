#!/usr/bin/env python3
"""Benchmark FuseChain tail inference latency and patch benchmark JSON / ingest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gchain.baselines.evaluate import evaluate_from_score_rows, write_result
from gchain.train.streams import load_stream_from_tgn_pt
from gchain.train.tgn_infer import benchmark_tail_latency_ms_per_1k, resolve_checkpoint

DEFAULT_SCENARIOS = ("sc1", "sc2", "sc3", "sc4", "sc5", "sc6", "sc7")


def _read_score_rows(path: Path) -> List[Dict[str, Any]]:
    import csv

    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _resolve_scores_csv(run_dir: Path) -> Optional[Path]:
    for name in ("best_eval_all_scores.csv", "best_eval_tail_scores.csv", "eval_all_scores.csv"):
        p = run_dir / name
        if p.is_file():
            return p
    return None


def _patch_json_latency(path: Path, latency: float) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["latency_ms_per_1k"] = float(latency)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="FuseChain tail inference latency (ms per 1k edges).")
    p.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    p.add_argument(
        "--runs-dir",
        type=str,
        default="artifacts/tgn_runs",
        help="Root containing per_scenario_scX_rule_stage/ runs.",
    )
    p.add_argument(
        "--run-suffix",
        type=str,
        default="rule_stage",
        help="Run folder suffix: per_scenario_{sc}_{suffix}",
    )
    p.add_argument("--graphs-dir", type=str, default="artifacts/graphs")
    p.add_argument("--device", type=str, default="", help="Override checkpoint device (cpu/cuda/mps).")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--bench-dir",
        type=str,
        default="artifacts/benchmarks/per_method/fusechain/full",
        help="Patch latency into existing sc*.json when present.",
    )
    p.add_argument(
        "--ingest-missing",
        action="store_true",
        help="If benchmark JSON missing, ingest detection metrics from run scores CSV.",
    )
    p.add_argument("--out-json", type=str, default="artifacts/benchmarks/fusechain_latency.json")
    args = p.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    runs_root = (_REPO / args.runs_dir).resolve()
    graphs_dir = (_REPO / args.graphs_dir).resolve()
    bench_dir = (_REPO / args.bench_dir).resolve()
    device = str(args.device).strip() or None

    results: Dict[str, Any] = {}
    for sc in scenarios:
        run_dir = runs_root / f"per_scenario_{sc}_{args.run_suffix}"
        ckpt = resolve_checkpoint(run_dir)
        tgn_pt = graphs_dir / f"synthchain_{sc}.tgn.pt"
        if ckpt is None:
            print(f"skip {sc}: no checkpoint under {run_dir}", file=sys.stderr)
            continue
        if not tgn_pt.is_file():
            print(f"skip {sc}: missing {tgn_pt}", file=sys.stderr)
            continue

        st = load_stream_from_tgn_pt(tgn_pt)
        lat, n_tail = benchmark_tail_latency_ms_per_1k(
            st,
            ckpt_path=ckpt,
            device=device,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        results[sc] = {
            "latency_ms_per_1k": lat,
            "n_tail": n_tail,
            "checkpoint": str(ckpt),
            "tgn_pt": str(tgn_pt),
        }
        print(f"{sc}: latency_ms_per_1k={lat:.3f} (n_tail={n_tail})")

        bench_json = bench_dir / f"{sc}.json"
        if bench_json.is_file():
            _patch_json_latency(bench_json, lat)
            print(f"  patched {bench_json}")
        elif args.ingest_missing:
            scores_csv = _resolve_scores_csv(run_dir)
            if scores_csv is None:
                print(f"  skip ingest {sc}: no scores csv in {run_dir}", file=sys.stderr)
                continue
            rows = [r for r in _read_score_rows(scores_csv) if str(r.get("scenario") or "").strip() == sc]
            payload = evaluate_from_score_rows(
                method="fusechain",
                scenario=sc,
                rows=rows,
                repo_root=_REPO,
                telemetry="full",
                latency_ms_per_1k=lat,
            )
            write_result(bench_json, payload)
            print(f"  wrote {bench_json}")

    out_path = (_REPO / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
