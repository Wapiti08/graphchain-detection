"""Partial attack-chain reconstruction (re-exports from recon_* modules)."""
from __future__ import annotations

from gchain.eval.recon_alerts import evaluate_alert_reconstruction
from gchain.eval.recon_eval import evaluate_reconstruction
from gchain.eval.recon_pools import (
    filter_rows_to_ioc_log_sources,
    ioc_log_source_files_for_scenario,
)
from gchain.eval.recon_stages import (
    DEFAULT_STAGE_ORDER,
    IDX_TO_STAGE,
    NUM_STAGE_CLASSES,
    STAGE_LABELS,
    STAGE_TO_IDX,
    build_line_to_ioc_type,
    ioc_type_to_stage_idx,
    load_ioc_type_to_stage,
    load_json,
    lcs_length,
    ordered_stage_sequence,
    recon_scores,
    stage_for_edge,
    stage_for_edge_predicted,
)
from gchain.eval.recon_topk import (
    build_chain_segments,
    dedupe_rows_by_endpoint_pair,
    eval_topk_modes,
    stages_from_ioc_topk,
    stages_from_topk,
    topk_edges,
    topk_ioc_edges,
)

# Backward-compatible aliases for older names.
_eval_one_mode = eval_topk_modes
_recon_scores = recon_scores

__all__ = [
    "DEFAULT_STAGE_ORDER",
    "IDX_TO_STAGE",
    "NUM_STAGE_CLASSES",
    "STAGE_LABELS",
    "STAGE_TO_IDX",
    "build_chain_segments",
    "build_line_to_ioc_type",
    "dedupe_rows_by_endpoint_pair",
    "evaluate_alert_reconstruction",
    "evaluate_reconstruction",
    "filter_rows_to_ioc_log_sources",
    "ioc_log_source_files_for_scenario",
    "ioc_type_to_stage_idx",
    "load_ioc_type_to_stage",
    "load_json",
    "lcs_length",
    "ordered_stage_sequence",
    "stage_for_edge",
    "stage_for_edge_predicted",
    "stages_from_ioc_topk",
    "stages_from_topk",
    "topk_edges",
    "topk_ioc_edges",
]
