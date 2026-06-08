#!/usr/bin/env python3
"""Train static GraphSAGE / RGCN on SynthChain; export tail scores + benchmark JSON."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gchain.baselines.evaluate import evaluate_scenario, write_result
from gchain.baselines.static_gnn import StaticGNNConfig, train_and_infer_tail
from gchain.baselines.score_rows import tail_score_rows
from gchain.baselines.telemetry import TelemetryKind, filter_stream_indices, subset_stream
from gchain.train.eval_io import write_eval_rows_csv
from gchain.train.split import time_split_idx
from gchain.train.streams import load_stream_from_tgn_pt

DEFAULT_SCENARIOS = ("sc1", "sc2", "sc3", "sc4", "sc5", "sc6", "sc7")


def _parse_scenarios(raw: str) -> tuple[str, ...]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    return tuple(parts) if parts else DEFAULT_SCENARIOS


def main() -> None:
    p = argparse.ArgumentParser(description="Train static GNN baselines (GraphSAGE / RGCN).")
    p.add_argument("--variant", type=str, default="graphsage", choices=("graphsage", "rgcn"))
    p.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    p.add_argument("--telemetry", type=str, default="full", choices=("full", "audit", "zeek", "eve"))
    p.add_argument("--graphs-dir", type=str, default="artifacts/graphs")
    p.add_argument("--out-dir", type=str, default="artifacts/benchmarks/static_gnn_runs")
    p.add_argument("--bench-dir", type=str, default="artifacts/benchmarks/per_method")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export-csv", action="store_true", help="Write eval_tail_scores.csv per scenario.")
    p.add_argument("--write-bench-json", action="store_true", help="Also write per_method benchmark JSON.")
    p.add_argument("--stage-gt-dir", type=str, default="artifacts/stage_gt")
    p.add_argument("--ioc-gt", type=str, default="data/SynthChain/iocs/ioc_ground_truth.json")
    args = p.parse_args()

    scenarios = _parse_scenarios(args.scenarios)
    graphs_dir = (_REPO / args.graphs_dir).resolve()
    run_dir = (_REPO / args.out_dir / args.variant).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry: TelemetryKind = args.telemetry  # type: ignore[assignment]

    for sc in scenarios:
        tgn_pt = graphs_dir / f"synthchain_{sc}.tgn.pt"
        full_pt = graphs_dir / f"synthchain_{sc}.full.pt"
        if not tgn_pt.is_file():
            print(f"skip {sc}: missing {tgn_pt}", file=sys.stderr)
            continue

        cfg = StaticGNNConfig(
            variant=args.variant,  # type: ignore[arg-type]
            epochs=int(args.epochs),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            lr=float(args.lr),
            device=str(args.device),
            seed=int(args.seed),
            full_pt=full_pt if full_pt.is_file() else None,
        )

        st_full = load_stream_from_tgn_pt(tgn_pt)
        keep = filter_stream_indices(st_full, telemetry)
        st = subset_stream(st_full, keep)
        n = int(st.src.numel())
        if n < 2:
            print(f"skip {sc}: too few edges after telemetry filter", file=sys.stderr)
            continue
        split = time_split_idx(n, float(args.train_frac))

        scores, model, cfg = train_and_infer_tail(
            st,
            train_end=split,
            tail_start=split,
            variant=args.variant,
            config=cfg,
        )
        rows = tail_score_rows(st, scenario=sc, tail_start=split, scores=scores)

        if args.export_csv:
            csv_path = run_dir / sc / "eval_tail_scores.csv"
            write_eval_rows_csv(csv_path, rows)
            print(f"wrote {csv_path}")

        ckpt = run_dir / sc / "static_gnn.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        try:
            import torch

            torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt)
            print(f"wrote {ckpt}")
        except Exception as exc:  # pragma: no cover
            print(f"warning: could not save checkpoint for {sc}: {exc}", file=sys.stderr)

        if args.write_bench_json:
            result = evaluate_scenario(
                method=args.variant,
                scenario=sc,
                tgn_pt=tgn_pt,
                repo_root=_REPO,
                telemetry=telemetry,
                train_frac=float(args.train_frac),
                stage_gt_dir=args.stage_gt_dir,
                ioc_gt=args.ioc_gt,
                seed=int(args.seed),
                static_gnn_config=cfg,
                full_pt=full_pt if full_pt.is_file() else None,
            )
            bench_path = (
                _REPO / args.bench_dir / args.variant / telemetry / f"{sc}.json"
            ).resolve()
            write_result(bench_path, result)
            print(f"wrote {bench_path}")


if __name__ == "__main__":
    main()
