#!/usr/bin/env bash
# Deployment-aligned per-scenario protocol:
#   - SSL link detection (primary representation)
#   - Stage head trained on rule_ioc_type (y_rule_high), NOT GT IOC lines/values
#   - No IOC rank loss by default (--lambda-ioc-rank 0)
#   - all-scores export + adaptive cap recon
#   - Report by_k_pred_stage (and by_k_group_cap_adaptive) in summary CSV
#
# Prerequisite: graphs with weak rules v2 (rule_ioc_type, y_rule_high in *.tgn.pt)
#
# Usage:
#   REGEN_GRAPH=1 EPOCHS=20 DEVICE=cuda bash scripts/run_per_scenario_rule_stage.sh all
#   bash scripts/run_per_scenario_rule_stage.sh sc4
#   RECON_ONLY=1 bash scripts/run_per_scenario_rule_stage.sh sc4
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

SC="${1:?scenario id (sc1..sc7) or all}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
GRAPHS_DIR="${GRAPHS_DIR:-artifacts/graphs}"
TOPKS="${TOPKS:-10,50,100,500}"
STAGE_GT_DIR="${STAGE_GT_DIR:-artifacts/stage_gt}"

LAMBDA_STAGE="${LAMBDA_STAGE:-0.5}"
LAMBDA_STAGE_NONE="${LAMBDA_STAGE_NONE:-0.1}"
STAGE_NONE_RATIO="${STAGE_NONE_SAMPLE_RATIO:-0.2}"
STAGE_SUPERVISION="${STAGE_SUPERVISION:-rule_high}"
STAGE_TAG="${STAGE_TAG:-rule_stage}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

REGEN_GRAPH="${REGEN_GRAPH:-0}"
RECON_ONLY="${RECON_ONLY:-0}"
RUN_SUMMARIZE="${RUN_SUMMARIZE:-1}"

ADAPTIVE_KEY="${ADAPTIVE_KEY:-etype_src}"
ADAPTIVE_PROBE_MULT="${ADAPTIVE_PROBE_MULT:-5}"
ADAPTIVE_HOT_TH="${ADAPTIVE_HOT_THRESHOLD:-20}"
ADAPTIVE_HOT_CAP="${ADAPTIVE_HOT_CAP:-10}"

RECON_ARGS=(
  --topks "${TOPKS}"
  --group-cap-adaptive-key "${ADAPTIVE_KEY}"
  --group-cap-adaptive-probe-mult "${ADAPTIVE_PROBE_MULT}"
  --group-cap-adaptive-hot-threshold "${ADAPTIVE_HOT_TH}"
  --group-cap-adaptive-hot-cap "${ADAPTIVE_HOT_CAP}"
)

TRAIN_STAGE_ARGS=(
  --aux-supervision train_only
  --lambda-stage "${LAMBDA_STAGE}"
  --lambda-stage-none "${LAMBDA_STAGE_NONE}"
  --stage-none-sample-ratio "${STAGE_NONE_RATIO}"
  --stage-supervision "${STAGE_SUPERVISION}"
  --lambda-ioc-rank 0
  --rank-supervision off
)

if [[ "${SC}" == "all" ]]; then
  SCENARIOS=(sc1 sc2 sc3 sc4 sc5 sc6 sc7)
else
  SCENARIOS=("${SC}")
fi

resolve_scores() {
  local out="$1"
  if [[ -f "${out}/best_eval_all_scores.csv" ]]; then
    echo "${out}/best_eval_all_scores.csv"
  elif [[ -f "${out}/eval_all_scores.csv" ]]; then
    echo "${out}/eval_all_scores.csv"
  else
    echo ""
  fi
}

run_recon() {
  local sc="$1"
  local out="$2"
  local stage_gt="${STAGE_GT_DIR}/${sc}.stages_gt.json"
  local scores
  scores="$(resolve_scores "${out}")"
  if [[ -z "${scores}" ]]; then
    echo "run_per_scenario_rule_stage: missing scores under ${out}" >&2
    return 1
  fi
  if [[ ! -f "${stage_gt}" ]]; then
    echo "run_per_scenario_rule_stage: missing ${stage_gt}" >&2
    return 1
  fi
  echo "========== reconstruction ${sc} -> ${out}/reconstruction_metrics.json"
  "${PY}" scripts/eval_attack_reconstruction.py \
    --scores-csv "${scores}" \
    --stage-gt "${stage_gt}" \
    "${RECON_ARGS[@]}" \
    --out "${out}/reconstruction_metrics.json"
}

run_stage_one() {
  local sc="$1"
  local out="artifacts/tgn_runs/per_scenario_${sc}_${STAGE_TAG}"
  if [[ "${#SCENARIOS[@]}" -eq 1 && -n "${OUT:-}" ]]; then
    out="${OUT}"
  fi

  if [[ "${REGEN_GRAPH}" == "1" ]]; then
    echo "========== generate graph ${sc} (rule_ioc_type in stream)"
    "${PY}" scripts/generate_graph.py --dataset synthchain --scenario "${sc}" --export-tgn
  fi

  if [[ "${RECON_ONLY}" != "1" ]]; then
    echo "========== train ${sc} (${STAGE_TAG}, stage=${STAGE_SUPERVISION}) -> ${out}"
    "${PY}" scripts/train_tgn_synthchain.py \
      --scenarios "${sc}" \
      --graphs-dir "${GRAPHS_DIR}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --auto-generate \
      --warmup \
      --eval-ioc \
      --save-scores \
      --save-scores-split all \
      --select-metric auprc \
      --out "${out}" \
      "${TRAIN_STAGE_ARGS[@]}" \
      ${EXTRA_TRAIN_ARGS}
  else
    echo "========== RECON_ONLY: skip train ${sc} (${out})"
  fi

  run_recon "${sc}" "${out}"
  echo "  run_summary: ${out}/run_summary.json"
  echo "  primary deploy metric: by_k_pred_stage + by_k_group_cap_adaptive in reconstruction_metrics.json"
}

for sc in "${SCENARIOS[@]}"; do
  run_stage_one "${sc}"
done

if [[ "${RUN_SUMMARIZE}" == "1" && "${#SCENARIOS[@]}" -ge 1 ]]; then
  echo "========== summarize rule_stage runs"
  "${PY}" scripts/summarize_synthchain_runs.py \
    --runs-dir artifacts/tgn_runs \
    --pattern "per_scenario_sc*_${STAGE_TAG}" \
    --topks "${TOPKS}" \
    --out "artifacts/tgn_runs/synthchain_summary_${STAGE_TAG}.csv"
  echo "  summary: artifacts/tgn_runs/synthchain_summary_${STAGE_TAG}.csv"
fi

echo "Done."
