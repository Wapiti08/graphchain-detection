from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from gchain.pipeline.cli import parse_args
from gchain.pipeline.generate import generate_graph, generate_synthchain_all_scenarios


def _parse_csv_list(s: str) -> List[str]:
    out: List[str] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def cli_main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        if bool(args.all_scenarios):
            sc_arg = str(args.scenario)
            scenarios = _parse_csv_list(sc_arg) if "," in sc_arg else [f"sc{i}" for i in range(1, 8)]
            generate_synthchain_all_scenarios(
                repo_root=repo_root,
                scenarios=scenarios,
                out=args.out,
                only_ioc_logs=bool(args.only_ioc_logs),
                limit_per_file=args.limit_per_file,
                causal=args.causal,
                causal_window=float(args.causal_window),
                export_tgn=bool(args.export_tgn),
                skip_existing=bool(args.skip_existing),
                weak_rules_relpath=str(args.weak_rules),
                verbose=True,
            )
            return

        generate_graph(
            repo_root=repo_root,
            scenario=args.scenario,
            out=args.out,
            limit_per_file=args.limit_per_file,
            only_ioc_logs=bool(args.only_ioc_logs),
            name=args.name,
            causal=args.causal,
            causal_window=float(args.causal_window),
            export_tgn=bool(args.export_tgn),
            weak_rules_relpath=str(args.weak_rules),
            verbose=True,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
