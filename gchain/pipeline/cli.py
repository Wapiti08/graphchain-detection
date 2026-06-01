from __future__ import annotations

import argparse
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate temporal heterogeneous graphs and optional TGN streams.")
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
    from config.qut_sources import QUT_KIND_CHOICES

    p.add_argument(
        "--qut-kind",
        type=str,
        default="all",
        choices=sorted(QUT_KIND_CHOICES),
        help=(
            "Which QUT processed CSV to parse, or 'all' to join six trace families "
            "(install, syscall, opensnoop, filetop, tcp, pattern) for --package-name."
        ),
    )
    p.add_argument(
        "--package-name",
        type=str,
        default="",
        help="QUT: one package when using --qut-kind all (omit when --all-packages).",
    )
    p.add_argument(
        "--all-packages",
        action="store_true",
        help="QUT only: export qut_joined_<pkg> for every Package_Name (requires --qut-kind all).",
    )
    p.add_argument(
        "--all-scenarios",
        action="store_true",
        help="SynthChain only: export synthchain_sc1..sc7 (or override with --scenarios).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --all-packages / --all-scenarios, skip if target .tgn.pt already exists.",
    )
    p.add_argument(
        "--max-packages",
        type=int,
        default=None,
        help="QUT batch only: process at most this many packages (smoke tests).",
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
