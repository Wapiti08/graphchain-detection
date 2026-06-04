#!/usr/bin/env bash
# Per-scenario reconstruction from a joint (multi-scenario) training run.
# Uses best_eval_tail_scores.csv + --scenario-filter (default eval split).
#
# Usage:
#   bash scripts/run_joint_recon_by_scenario.sh artifacts/tgn_runs/joint_all_stage_gt
#   SCENARIOS=sc5,sc6 bash scripts/run_joint_recon_by_scenario.sh artifacts/tgn_runs/joint_all_stage_gt
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

RUN_DIR="${1:?run dir (e.g. artifacts/tgn_runs/joint_all_stage_gt)}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "run_joint_recon_by_scenario: not a directory: ${RUN_DIR}" >&2
  exit 1
fi

resolve_scores() {
  local dir="$1"
  local f
  for f in \
    "${dir}/best_eval_tail_scores.csv" \
    "${dir}/best_eval_tail_scores (1).csv" \
    "${dir}/eval_tail_scores.csv"
  do
    if [[ -f "${f}" ]]; then
      echo "${f}"
      return 0
    fi
  done
  return 1
}

SCORES="$(resolve_scores "${RUN_DIR}")" || true
if [[ -z "${SCORES}" ]]; then
  echo "run_joint_recon_by_scenario: no tail scores under ${RUN_DIR}" >&2
  echo "  expected: best_eval_tail_scores.csv (from --eval-ioc training)" >&2
  exit 1
fi
echo "Using scores: ${SCORES}"

STAGE_GT_DIR="${STAGE_GT_DIR:-artifacts/stage_gt}"
TOPKS="${TOPKS:-10,50,100,500}"
SCENARIOS="${SCENARIOS:-sc1,sc2,sc3,sc4,sc5,sc6,sc7}"

ADAPTIVE_KEY="${ADAPTIVE_KEY:-etype_src}"
ADAPTIVE_PROBE_MULT="${ADAPTIVE_PROBE_MULT:-5}"
ADAPTIVE_HOT_TH="${ADAPTIVE_HOT_THRESHOLD:-20}"
ADAPTIVE_HOT_CAP="${ADAPTIVE_HOT_CAP:-10}"

IFS=',' read -r -a SC_LIST <<< "${SCENARIOS}"

RECON_ARGS=(
  --scores-csv "${SCORES}"
  --topks "${TOPKS}"
  --group-cap-adaptive-key "${ADAPTIVE_KEY}"
  --group-cap-adaptive-probe-mult "${ADAPTIVE_PROBE_MULT}"
  --group-cap-adaptive-hot-threshold "${ADAPTIVE_HOT_TH}"
  --group-cap-adaptive-hot-cap "${ADAPTIVE_HOT_CAP}"
)

for SC in "${SC_LIST[@]}"; do
  SC="$(echo "${SC}" | xargs)"
  [[ -n "${SC}" ]] || continue
  STAGE_GT="${STAGE_GT_DIR}/${SC}.stages_gt.json"
  if [[ ! -f "${STAGE_GT}" ]]; then
    echo "run_joint_recon_by_scenario: missing ${STAGE_GT}" >&2
    exit 1
  fi
  OUT="${RUN_DIR}/recon_${SC}.json"
  echo "========== ${SC} -> ${OUT}"
  "${PY}" scripts/eval_attack_reconstruction.py \
    "${RECON_ARGS[@]}" \
    --scenario-filter "${SC}" \
    --stage-gt "${STAGE_GT}" \
    --out "${OUT}"
done

echo "Done. Outputs: ${RUN_DIR}/recon_sc*.json"
