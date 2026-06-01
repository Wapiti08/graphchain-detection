"""Heterogeneous graph construction and TGN stream export (dataset-agnostic core)."""

from graphcore.augment import augment_events_with_causal
from graphcore.builder import BuildStats, build_hetero_graph
from graphcore.tgn_input import TGNEventStream, hetero_to_tgn_event_stream

__all__ = [
    "BuildStats",
    "TGNEventStream",
    "augment_events_with_causal",
    "build_hetero_graph",
    "hetero_to_tgn_event_stream",
]
