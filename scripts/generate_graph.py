from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate temporal heterogeneous graphs.")
    p.add_argument(
        "--dataset",
        choices=["synthchain", "qut"],
        required=True,
        help="Dataset to parse and build graph from.",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="sc1",
        help="Scenario id for synthchain (e.g., sc1..sc7).",
    )
    p.add_argument(
        "--qut-kind",
        type=str,
        default="syscall_traces",
        choices=["syscall_traces", "opensnoop_traces", "filetop_traces", "all"],
        help="Which QUT processed CSV to parse.",
    )
    p.add_argument(
        "--package-name",
        type=str,
        default="",
        help="QUT only: build graph for one package (required for --qut-kind all).",
    )
    p.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Max rows/lines per file (useful for quick runs).",
    )
    p.add_argument(
        "--only-ioc-logs",
        action="store_true",
        help="Synthchain only: only parse logs marked has_ioc=True.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="artifacts/graphs",
        help="Output directory for saved graphs.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="",
        help="Output filename stem override (no extension).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    # Allow running as a script without installing the package.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "synthchain":
        from parsers.synthchain import load_synthchain_events

        events = load_synthchain_events(
            args.scenario,
            project_root=repo_root,
            only_ioc_logs=bool(args.only_ioc_logs),
            limit_per_file=args.limit_per_file,
        )
        stem = args.name or f"synthchain_{args.scenario}"

    else:  # qut
        import pandas as pd

        from config.qut_sources import QUT_SOURCES
        if args.qut_kind == "all":
            if not args.package_name:
                raise SystemExit("--package-name is required when --qut-kind all")
            from parsers.qut.join import parse_qut_joined_package

            df_s = pd.read_csv(repo_root / QUT_SOURCES["syscall_traces"].rel_path)
            df_o = pd.read_csv(repo_root / QUT_SOURCES["opensnoop_traces"].rel_path)
            df_f = pd.read_csv(repo_root / QUT_SOURCES["filetop_traces"].rel_path)
            events = parse_qut_joined_package(
                args.package_name,
                df_syscall=df_s,
                df_opensnoop=df_o,
                df_filetop=df_f,
            )
            stem = args.name or f"qut_joined_{args.package_name}"
        else:
            from parsers.qut.processed import (
                parse_filetop_row,
                parse_opensnoop_row,
                parse_syscall_row,
            )

            spec = QUT_SOURCES[args.qut_kind]
            path = repo_root / spec.rel_path
            df = pd.read_csv(path)
            if args.limit_per_file is not None:
                df = df.head(args.limit_per_file)

            row_parser = {
                "syscall_traces": parse_syscall_row,
                "opensnoop_traces": parse_opensnoop_row,
                "filetop_traces": parse_filetop_row,
            }[args.qut_kind]

            events = []
            for _, row in df.iterrows():
                events.extend(row_parser(row))

            stem = args.name or f"qut_{args.qut_kind}"

    from graph import build_hetero_graph

    data, stats = build_hetero_graph(events)

    # Save graph
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing torch. Use the python env where torch+pyg are installed."
        ) from e

    stats_dict = {
        "num_events": stats.num_events,
        "num_nodes_by_type": {k.value: v for k, v in stats.num_nodes_by_type.items()},
        "num_edges_by_type": {k.value: v for k, v in stats.num_edges_by_type.items()},
    }

    # (1) Safe payload: tensors + primitives only (works with default weights_only=True)
    out_path = out_dir / f"{stem}.pt"
    torch.save({"data_dict": data.to_dict(), "stats": stats_dict}, out_path)

    # (2) Full object payload: convenient, but requires torch.load(..., weights_only=False)
    out_path_full = out_dir / f"{stem}.full.pt"
    torch.save({"data": data, "stats": stats_dict}, out_path_full)

    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_full}")
    print(f"Events: {stats.num_events}")
    print("Nodes:", {k.value: v for k, v in stats.num_nodes_by_type.items() if v})
    print("Edges:", {k.value: v for k, v in stats.num_edges_by_type.items() if v})


if __name__ == "__main__":
    main()

