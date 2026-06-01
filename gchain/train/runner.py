from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from gchain.train.cli import parse_args
from gchain.train.streams import Stream, load_stream_from_tgn_pt
from gchain.train.train_loop import train


def cli_main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    if args.input_mode == "tgnpt":
        tgn_path = (repo_root / args.tgn_pt).resolve()
        if not tgn_path.exists():
            raise SystemExit(
                f"Missing `{tgn_path}`. Generate via gchain.pipeline, e.g.\n"
                f"  python -m gchain.pipeline --dataset qut --qut-kind all --package-name <PKG> --export-tgn"
            )

        name = str(args.name).strip() or tgn_path.stem.replace(".tgn", "")
        args.name = name
        args.scenarios = name

        streams: Dict[str, Stream] = {name: load_stream_from_tgn_pt(tgn_path)}
        train(
            args,
            repo_root=repo_root,
            streams_override=streams,
            eval_protocol="single_tgnpt",
            run_meta={"input_mode": "tgnpt", "name": name, "tgn_pt": str(Path(args.tgn_pt))},
        )
        return

    train(args, repo_root=repo_root)
