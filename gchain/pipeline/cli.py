from __future__ import annotations

import argparse
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate temporal heterogeneous graphs and optional TGN streams.")
    p.add_argument(
        "--dataset",
        choices=["synthchain"],
        default="synthchain",
        help="Dataset to parse and build graph from (SynthChain only).",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="sc1",
        help="Scenario id (e.g., sc1..sc7).",
    )
    p.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Export synthchain_sc1..sc7 (or override with comma-separated --scenario).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --all-scenarios, skip if target .tgn.pt already exists.",
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
        help="Only parse logs marked has_ioc=True.",
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
    p.add_argument(
        "--causal",
        type=str,
        default="off",
        choices=["off", "level0", "level1"],
        help="Add deterministic causal dependency edges.",
    )
    p.add_argument(
        "--causal-window",
        type=float,
        default=50.0,
        help="Causal window in seconds (if ts) or steps (if no ts).",
    )
    p.add_argument(
        "--export-tgn",
        action="store_true",
        help="Also export a flattened TGN-style event stream (.tgn.pt).",
    )
    return p.parse_args(argv)
