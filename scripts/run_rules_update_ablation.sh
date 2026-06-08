#!/usr/bin/env bash
# Rules-update ablation: baseline vs analyst-added weak rule (staging-path download).
#
# Phases:
#   1) Coverage compare (no GPU): rule_hit / rule_high deltas per scenario
#   2) Re-export .tgn.pt into separate graph dirs
#   3) Per-scenario deployable SSL train on a small scenario subset (default sc1,sc3,sc5)
#   4) Merge train metrics into regression CSV
#
# Usage:
#   STATS_ONLY=1 bash scripts/run_rules_update_ablation.sh
#   EPOCHS=5 DEVICE=cpu SCENARIOS=sc1,sc3,sc5 bash scripts/run_rules_update_ablation.sh
#   TRAIN_ONLY=1 bash scripts/run_rules_update_ablation.sh   # skip export if graphs exist
#   LAMBDA_STAGE=0.5 STAGE_SUPERVISION=gt_ioc bash scripts/run_rules_update_ablation.sh  # lab ablation
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
else
  PY=""
  for cand in "${ROOT}/.venv/bin/python" /opt/anaconda3/bin/python python3; do
    if [[ -x "${cand}" ]] && "${cand}" -c "import pandas" 2>/dev/null; then
      PY="${cand}"
      break
    fi
  done
  PY="${PY:-python3}"
fi

BASELINE_RULES="${BASELINE_RULES:-config/weak_supervision_rules.json}"
UPDATED_RULES="${UPDATED_RULES:-config/weak_supervision_rules_update_ablation.json}"
NEW_RULE_ID="${NEW_RULE_ID:-tier2_setup_py_direct_exec}"
SCENARIOS="${SCENARIOS:-sc1,sc3,sc5}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
LAMBDA_STAGE="${LAMBDA_STAGE:-0}"
LAMBDA_STAGE_NONE="${LAMBDA_STAGE_NONE:-0}"
STAGE_SUPERVISION="${STAGE_SUPERVISION:-rule_high}"
LAMBDA_IOC_RANK="${LAMBDA_IOC_RANK:-0}"
RANK_SUPERVISION="${RANK_SUPERVISION:-off}"
GRAPHS_BASELINE="${GRAPHS_BASELINE:-artifacts/graphs_rules_baseline}"
GRAPHS_UPDATED="${GRAPHS_UPDATED:-artifacts/graphs_rules_updated}"
OUT_BASELINE="${OUT_BASELINE:-artifacts/tgn_runs/rules_update_baseline}"
OUT_UPDATED="${OUT_UPDATED:-artifacts/tgn_runs/rules_update_v3}"
ABLATION_DIR="${ABLATION_DIR:-artifacts/rules_update_ablation}"

STATS_ONLY="${STATS_ONLY:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EXPORT_ONLY="${EXPORT_ONLY:-0}"

mkdir -p "${ABLATION_DIR}"

echo "========== config"
echo "python=${PY}"
echo "scenarios=${SCENARIOS} epochs=${EPOCHS} batch=${BATCH} device=${DEVICE}"
echo "stage: lambda=${LAMBDA_STAGE} none_lambda=${LAMBDA_STAGE_NONE} supervision=${STAGE_SUPERVISION}"
echo "rank:  lambda=${LAMBDA_IOC_RANK} supervision=${RANK_SUPERVISION}"
echo "rules: baseline=${BASELINE_RULES} updated=${UPDATED_RULES}"
echo "new_rule_id=${NEW_RULE_ID}"

echo "========== phase 1: weak-rule coverage (baseline vs updated)"
"${PY}" scripts/compare_weak_rule_coverage.py \
  --baseline-rules "${BASELINE_RULES}" \
  --updated-rules "${UPDATED_RULES}" \
  --new-rule-id "${NEW_RULE_ID}" \
  --scenarios "sc1,sc2,sc3,sc4,sc5,sc6,sc7" \
  --out-csv "${ABLATION_DIR}/rule_coverage_compare.csv" \
  --out-json "${ABLATION_DIR}/rule_coverage_compare.json"

if [[ "${STATS_ONLY}" == "1" ]]; then
  echo "STATS_ONLY=1: done after coverage compare."
  exit 0
fi

IFS=',' read -r -a SC_ARR <<< "${SCENARIOS}"

if [[ "${TRAIN_ONLY}" != "1" ]]; then
  echo "========== phase 2a: export graphs (baseline rules) -> ${GRAPHS_BASELINE}"
  for sc in "${SC_ARR[@]}"; do
    sc="$(echo "${sc}" | xargs)"
    [[ -n "${sc}" ]] || continue
    "${PY}" -m gchain.pipeline --dataset synthchain --scenario "${sc}" \
      --only-ioc-logs --export-tgn --out "${GRAPHS_BASELINE}" \
      --weak-rules "${BASELINE_RULES}"
  done

  echo "========== phase 2b: export graphs (updated rules) -> ${GRAPHS_UPDATED}"
  for sc in "${SC_ARR[@]}"; do
    sc="$(echo "${sc}" | xargs)"
    [[ -n "${sc}" ]] || continue
    "${PY}" -m gchain.pipeline --dataset synthchain --scenario "${sc}" \
      --only-ioc-logs --export-tgn --out "${GRAPHS_UPDATED}" \
      --weak-rules "${UPDATED_RULES}"
  done
fi

if [[ "${EXPORT_ONLY}" == "1" ]]; then
  echo "EXPORT_ONLY=1: done after graph export."
  exit 0
fi

COMMON_TRAIN=(
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --auto-generate
  --warmup
  --eval-ioc
  --save-scores
  --select-metric auprc
  --aux-supervision train_only
  --lambda-stage "${LAMBDA_STAGE}"
  --lambda-stage-none "${LAMBDA_STAGE_NONE}"
  --stage-supervision "${STAGE_SUPERVISION}"
  --lambda-ioc-rank "${LAMBDA_IOC_RANK}"
  --rank-supervision "${RANK_SUPERVISION}"
)

train_one() {
  local graphs_dir="$1"
  local out_root="$2"
  local tag="$3"
  local sc="$4"
  local out="${out_root}/per_scenario_${sc}_${tag}"
  echo "========== train ${sc} (${tag}) -> ${out}"
  "${PY}" scripts/train_tgn_synthchain.py \
    --scenarios "${sc}" \
    --graphs-dir "${graphs_dir}" \
    --out "${out}" \
    "${COMMON_TRAIN[@]}"
}

echo "========== phase 3a: train baseline graphs"
for sc in "${SC_ARR[@]}"; do
  sc="$(echo "${sc}" | xargs)"
  [[ -n "${sc}" ]] || continue
  train_one "${GRAPHS_BASELINE}" "${OUT_BASELINE}" "rule_stage" "${sc}"
done

echo "========== phase 3b: train updated graphs"
for sc in "${SC_ARR[@]}"; do
  sc="$(echo "${sc}" | xargs)"
  [[ -n "${sc}" ]] || continue
  train_one "${GRAPHS_UPDATED}" "${OUT_UPDATED}" "rule_stage_v3" "${sc}"
done

echo "========== phase 4: merge train regression table"
"${PY}" scripts/merge_rules_update_ablation.py \
  --baseline-root "${OUT_BASELINE}" \
  --updated-root "${OUT_UPDATED}" \
  --scenarios "${SCENARIOS}" \
  --out-csv "${ABLATION_DIR}/train_regression_compare.csv"

echo "Done."
echo "  coverage: ${ABLATION_DIR}/rule_coverage_compare.csv"
echo "  train:    ${ABLATION_DIR}/train_regression_compare.csv"
