#!/usr/bin/env bash
# Per-scenario protocol aligned with sc1/sc2:
#   - Pure SSL link prediction (no --lambda-stage / --lambda-ioc-rank unless EXTRA_TRAIN_ARGS set)
#   - Export all edges at best epoch (--save-scores-split all)
#   - Reconstruction on best_eval_all_scores.csv (primary metric: by_k)
#
# Usage:
#   bash scripts/run_per_scenario_ssl_recon.sh sc3
#   EPOCHS=20 bash scripts/run_per_scenario_ssl_recon.sh sc3
#   REGEN_GRAPH=1 bash scripts/run_per_scenario_ssl_recon.sh sc2
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

SC="${1:?scenario id, e.g. sc2 or sc3}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
GRAPHS_DIR="${GRAPHS_DIR:-artifacts/graphs}"
OUT="${OUT:-artifacts/tgn_runs/per_scenario_${SC}_all_scores}"
STAGE_GT="${STAGE_GT:-artifacts/stage_gt/${SC}.stages_gt.json}"
TOPKS="${TOPKS:-10,50,100,500}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
REGEN_GRAPH="${REGEN_GRAPH:-0}"

if [[ "${REGEN_GRAPH}" == "1" ]]; then
  echo "========== generate graph ${SC}"
  "${PY}" scripts/generate_graph.py --dataset synthchain --scenario "${SC}" --export-tgn
fi

echo "========== train ${SC} -> ${OUT}"
"${PY}" scripts/train_tgn_synthchain.py \
  --scenarios "${SC}" \
  --graphs-dir "${GRAPHS_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --auto-generate \
  --warmup \
  --eval-ioc \
  --aux-supervision train_only \
  --save-scores \
  --save-scores-split all \
  --select-metric auprc \
  --out "${OUT}" \
  ${EXTRA_TRAIN_ARGS}

SCORES="${OUT}/best_eval_all_scores.csv"
if [[ ! -f "${SCORES}" ]]; then
  SCORES="${OUT}/eval_all_scores.csv"
fi
if [[ ! -f "${SCORES}" ]]; then
  echo "run_per_scenario_ssl_recon: missing scores under ${OUT}" >&2
  exit 1
fi
if [[ ! -f "${STAGE_GT}" ]]; then
  echo "run_per_scenario_ssl_recon: missing ${STAGE_GT}; run scripts/build_synthchain_stage_gt.py" >&2
  exit 1
fi

echo "========== reconstruction ${SC} (${SCORES})"
"${PY}" scripts/eval_attack_reconstruction.py \
  --scores-csv "${SCORES}" \
  --stage-gt "${STAGE_GT}" \
  --topks "${TOPKS}" \
  --pred-min-prob 0.5 \
  --pred-min-count 2 \
  --out "${OUT}/reconstruction_metrics.json"

echo "Done. Summary: ${OUT}/run_summary.json"
echo "         Recon:  ${OUT}/reconstruction_metrics.json"
