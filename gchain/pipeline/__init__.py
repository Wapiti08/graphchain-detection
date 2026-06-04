"""Graph construction pipeline: parse events → hetero graph → optional TGN export."""

from gchain.pipeline.generate import (
    GenerateGraphResult,
    generate_graph,
    generate_synthchain_all_scenarios,
)
from gchain.pipeline.runner import cli_main

__all__ = [
    "GenerateGraphResult",
    "generate_graph",
    "generate_synthchain_all_scenarios",
    "cli_main",
]
