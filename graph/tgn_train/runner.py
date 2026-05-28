from __future__ import annotations

from typing import List, Optional

from graph.tgn_train.cli import parse_args
from graph.tgn_train.train_loop import train


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Backward-compatible CLI entrypoint."""
    args = parse_args(argv)
    train(args)

