#!/usr/bin/env python3
"""Run one baseline method across SynthChain scenarios; write per-scenario JSON."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gchain.baselines.evaluate import evaluate_scenario, write_result
from gchain.baselines.static_gnn import StaticGNNConfig
from gchain.baselines.telemetry import TelemetryKind

DEFAULT_SCENARIOS = ("sc1", "sc2", "sc3", "sc4", "sc5", "sc6", "sc7")


def _parse_scenarios(raw: str) -> tuple[str, ...]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    return tuple(parts) if parts else DEFAULT_SCENARIOS


def main() -> None:
    p = argparse.ArgumentParser(description="SynthChain baseline benchmark runner.")
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=("freq_rarity", "path_lof", "random", "graphsage", "rgcn"),
        help="Scoring method; graphsage/rgcn train static GNN on the train prefix.",
    )
    p.add_argument("--epochs", type=int, default=25, help="Static GNN training epochs.")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--telemetry",
        type=str,
        default="full",
        choices=("full", "audit", "zeek", "eve"),
        help="Scheme A telemetry filter (full = all sources).",
    )
    p.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    p.add_argument(
        "--graphs-dir",
        type=str,
        default="artifacts/graphs",
        help="Directory with synthchain_scX.tgn.pt exports.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/benchmarks/per_method",
        help="Output root; writes <out>/<method>/<telemetry>/<sc>.json",
    )
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage-gt-dir", type=str, default="artifacts/stage_gt")
    p.add_argument("--ioc-gt", type=str, default="data/SynthChain/iocs/ioc_ground_truth.json")
    args = p.parse_args()

    scenarios = _parse_scenarios(args.scenarios)
    graphs_dir = (_REPO / args.graphs_dir).resolve()
    out_root = (_REPO / args.out_dir / args.method / args.telemetry).resolve()
    telemetry: TelemetryKind = args.telemetry  # type: ignore[assignment]

    for sc in scenarios:
        tgn_pt = graphs_dir / f"synthchain_{sc}.tgn.pt"
        if not tgn_pt.is_file():
            print(f"skip {sc}: missing {tgn_pt}", file=sys.stderr)
            continue
        full_pt = graphs_dir / f"synthchain_{sc}.full.pt"
        static_cfg = None
        if args.method in ("graphsage", "rgcn"):
            static_cfg = StaticGNNConfig(
                variant=args.method,  # type: ignore[arg-type]
                epochs=int(args.epochs),
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                lr=float(args.lr),
                device=str(args.device),
                seed=int(args.seed),
                full_pt=full_pt if full_pt.is_file() else None,
            )
        result = evaluate_scenario(
            method=args.method,
            scenario=sc,
            tgn_pt=tgn_pt,
            repo_root=_REPO,
            telemetry=telemetry,
            train_frac=float(args.train_frac),
            stage_gt_dir=args.stage_gt_dir,
            ioc_gt=args.ioc_gt,
            seed=int(args.seed),
            static_gnn_config=static_cfg,
            full_pt=full_pt if full_pt.is_file() else None,
        )
        out_path = out_root / f"{sc}.json"
        write_result(out_path, result)
        print(f"wrote {out_path} status={result.get('status')}")


if __name__ == "__main__":
    main()
