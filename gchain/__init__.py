"""
GraphChain detection stack: pipeline (build artifacts), train (TGN), eval (metrics).
Heterogeneous graph primitives live in ``graphcore``.
"""

from gchain.pipeline import GenerateGraphResult, generate_graph, cli_main as pipeline_cli_main
from gchain.train.runner import cli_main as train_cli_main

__all__ = [
    "GenerateGraphResult",
    "generate_graph",
    "pipeline_cli_main",
    "train_cli_main",
]
