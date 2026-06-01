from __future__ import annotations

import argparse
from typing import List, Optional

from gchain.train.args_common import add_training_args


def _resolve_input_mode(args: argparse.Namespace) -> str:
    """Infer synthchain vs tgnpt from explicit flag or --tgn-pt."""
    mode = str(getattr(args, "input_mode", "") or "").strip().lower()
    tgn_pt = str(getattr(args, "tgn_pt", "") or "").strip()
    if tgn_pt and mode not in ("synthchain", "tgnpt"):
        return "tgnpt"
    if mode in ("synthchain", "tgnpt"):
        return mode
    return "synthchain"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train/validate a TGN with time-split SSL (+ optional weak IOC/stage supervision). "
            "Use --input-mode synthchain for SynthChain scenarios, or --input-mode tgnpt with "
            "--tgn-pt for any exported .tgn.pt stream (e.g. QUT-DV25)."
        )
    )

    p.add_argument(
        "--input-mode",
        type=str,
        default="",
        choices=["", "synthchain", "tgnpt"],
        help="Data source: synthchain (scenario *.tgn.pt) or tgnpt (single --tgn-pt). "
        "If omitted and --tgn-pt is set, defaults to tgnpt.",
    )

    sc = p.add_argument_group("SynthChain input (--input-mode synthchain)")
    sc.add_argument(
        "--scenarios",
        type=str,
        default="sc1,sc2,sc3,sc4,sc5,sc6,sc7",
        help="Comma-separated scenario ids.",
    )
    sc.add_argument("--holdout", type=str, default="", help="LOSO holdout scenario (e.g. sc3).")
    sc.add_argument("--graphs-dir", type=str, default="artifacts/graphs")
    sc.add_argument(
        "--auto-generate",
        action="store_true",
        help="Generate missing synthchain_<sc>.tgn.pt via gchain.pipeline.generate_graph.",
    )

    tg = p.add_argument_group("Pre-exported stream (--input-mode tgnpt)")
    tg.add_argument(
        "--tgn-pt",
        type=str,
        default="",
        help="Path to a single .tgn.pt (QUT joined package, custom export, etc.).",
    )
    tg.add_argument(
        "--name",
        type=str,
        default="",
        help="Run/scenario label in outputs (defaults to .tgn.pt stem).",
    )

    add_training_args(p)

    args = p.parse_args(argv)
    if bool(getattr(args, "hard_neg", False)):
        args.neg_sampling = "pool"

    args.input_mode = _resolve_input_mode(args)

    if args.input_mode == "tgnpt":
        if not str(args.tgn_pt).strip():
            raise SystemExit("--input-mode tgnpt requires --tgn-pt.")
        if not str(args.out).strip():
            stem = str(args.name).strip()
            if not stem:
                from pathlib import Path

                stem = Path(args.tgn_pt).stem.replace(".tgn", "")
            args.out = f"artifacts/tgn_runs/{stem}"
    else:
        if not str(args.out).strip():
            args.out = "artifacts/tgn_runs/synthchain_multi"

    return args
