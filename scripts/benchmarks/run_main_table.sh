#!/usr/bin/env bash
# Run SynthChain main-table baselines (detection + reconstruction + latency).
# Requires: artifacts/graphs/synthchain_sc{1..7}.tgn.pt and artifacts/stage_gt/*.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GRAPHS_DIR="${GRAPHS_DIR:-artifacts/graphs}"
OUT_DIR="${OUT_DIR:-artifacts/benchmarks/per_method}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
SCENARIOS="${SCENARIOS:-sc1,sc2,sc3,sc4,sc5,sc6,sc7}"
PYTHON="${PYTHON:-python3}"

run_method() {
  local method="$1"
  local telemetry="$2"
  echo "==> ${method} / ${telemetry}"
  "${PYTHON}" scripts/benchmarks/run_baseline.py \
    --method "${method}" \
    --telemetry "${telemetry}" \
    --scenarios "${SCENARIOS}" \
    --graphs-dir "${GRAPHS_DIR}" \
    --out-dir "${OUT_DIR}" \
    --train-frac "${TRAIN_FRAC}"
}

# Single-source (Scheme A)
run_method freq_rarity audit
run_method freq_rarity zeek
run_method freq_rarity eve

# Full multi-source baselines implemented in-repo
run_method freq_rarity full
run_method path_lof full

# Static GNN (GraphSAGE / RGCN on train-prefix graph; uses .full.pt node attrs when present)
STATIC_EPOCHS="${STATIC_EPOCHS:-25}"
STATIC_DEVICE="${STATIC_DEVICE:-cpu}"
for variant in graphsage rgcn; do
  echo "==> static ${variant} / full"
  "${PYTHON}" scripts/benchmarks/run_baseline.py \
    --method "${variant}" \
    --telemetry full \
    --scenarios "${SCENARIOS}" \
    --graphs-dir "${GRAPHS_DIR}" \
    --out-dir "${OUT_DIR}" \
    --train-frac "${TRAIN_FRAC}" \
    --epochs "${STATIC_EPOCHS}" \
    --device "${STATIC_DEVICE}"
done

# FuseChain: ingest training scores when available
# Example FuseChain per scenario:
#   python3 scripts/benchmarks/ingest_scores.py \
#     --method fusechain --scenario sc1 \
#     --scores-csv artifacts/tgn_runs/sc1_rule_stage/best_eval_tail_scores.csv

echo "==> merge main table"
"${PYTHON}" scripts/benchmarks/merge_main_table.py \
  --bench-dir "${OUT_DIR}" \
  --out-csv artifacts/benchmarks/main_table.csv \
  --out-latex artifacts/benchmarks/main_table.tex \
  --scenarios "${SCENARIOS}"

echo "done: artifacts/benchmarks/main_table.csv"
