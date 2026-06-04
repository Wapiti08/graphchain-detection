#!/usr/bin/env bash
# Per-scenario weak supervision (IOC-taxonomy rules, no GT line/value IOCs):
#   - Regenerate graph (y_rule / y_rule_high in *.tgn.pt) when REGEN_GRAPH=1
#   - SSL + --lambda-ioc-rank with --rank-supervision rule_high (default)
#   - all-scores export + reconstruction with adaptive group-cap (primary reporting)
#
# Compare to per_scenario_sc*_all_scores (pure SSL). Optional fair baseline recon refresh.
#
# Usage:
#   bash scripts/run_per_scenario_weak_rule.sh sc4
#   EPOCHS=20 DEVICE=cuda bash scripts/run_per_scenario_weak_rule.sh all
#   REGEN_GRAPH=1 bash scripts/run_per_scenario_weak_rule.sh sc4
#   RANK_SUPERVISION=rule LAMBDA_IOC_RANK=0.2 bash scripts/run_per_scenario_weak_rule.sh sc4
#   REFRESH_BASELINE_RECON=1 bash scripts/run_per_scenario_weak_rule.sh sc4   # adaptive recon on _all_scores only
#   RECON_ONLY=1 bash scripts/run_per_scenario_weak_rule.sh sc4                # skip train if scores exist
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

LAMBDA_IOC_RANK="${LAMBDA_IOC_RANK:-0.05}"
IOC_RANK_MARGIN="${IOC_RANK_MARGIN:-0.5}"
# rule = any rank-eligible hit (medium+high); rule_high = gated (IOC log + strong/multi-signal)
RANK_SUPERVISION="${RANK_SUPERVISION:-rule_high}"
RULE_TAG="${RULE_TAG:-${RANK_SUPERVISION}}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

REGEN_GRAPH="${REGEN_GRAPH:-0}"
RECON_ONLY="${RECON_ONLY:-0}"
REFRESH_BASELINE_RECON="${REFRESH_BASELINE_RECON:-0}"
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

TRAIN_WEAK_ARGS=(
  --aux-supervision train_only
  --lambda-ioc-rank "${LAMBDA_IOC_RANK}"
  --ioc-rank-margin "${IOC_RANK_MARGIN}"
  --rank-supervision "${RANK_SUPERVISION}"
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
    echo "run_per_scenario_weak_rule: missing scores under ${out}" >&2
    return 1
  fi
  if [[ ! -f "${stage_gt}" ]]; then
    echo "run_per_scenario_weak_rule: missing ${stage_gt}" >&2
    return 1
  fi
  echo "========== reconstruction ${sc} -> ${out}/reconstruction_metrics.json"
  "${PY}" scripts/eval_attack_reconstruction.py \
    --scores-csv "${scores}" \
    --stage-gt "${stage_gt}" \
    "${RECON_ARGS[@]}" \
    --out "${out}/reconstruction_metrics.json"
}

run_weak_one() {
  local sc="$1"
  local out="artifacts/tgn_runs/per_scenario_${sc}_${RULE_TAG}"
  if [[ "${#SCENARIOS[@]}" -eq 1 && -n "${OUT:-}" ]]; then
    out="${OUT}"
  fi

  if [[ "${REGEN_GRAPH}" == "1" ]]; then
    echo "========== generate graph ${sc} (weak rules in stream)"
    "${PY}" scripts/generate_graph.py --dataset synthchain --scenario "${sc}" --export-tgn
  fi

  if [[ "${RECON_ONLY}" != "1" ]]; then
    echo "========== train ${sc} (${RULE_TAG}) -> ${out}"
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
      "${TRAIN_WEAK_ARGS[@]}" \
      ${EXTRA_TRAIN_ARGS}
  else
    echo "========== RECON_ONLY: skip train ${sc} (${out})"
  fi

  run_recon "${sc}" "${out}"
  echo "  run_summary: ${out}/run_summary.json"
}

refresh_baseline_recon() {
  local sc="$1"
  local out="artifacts/tgn_runs/per_scenario_${sc}_all_scores"
  if [[ ! -d "${out}" ]]; then
    echo "========== skip baseline recon ${sc}: missing ${out}"
    return 0
  fi
  echo "========== refresh SSL baseline recon (adaptive cap) ${sc}"
  run_recon "${sc}" "${out}"
}

for sc in "${SCENARIOS[@]}"; do
  if [[ "${REFRESH_BASELINE_RECON}" == "1" ]]; then
    refresh_baseline_recon "${sc}"
  fi
  run_weak_one "${sc}"
done

if [[ "${RUN_SUMMARIZE}" == "1" && "${#SCENARIOS[@]}" -ge 1 ]]; then
  echo "========== summarize weak-rule runs"
  "${PY}" scripts/summarize_synthchain_runs.py \
    --runs-dir artifacts/tgn_runs \
    --pattern "per_scenario_sc*_${RULE_TAG}" \
    --topks "${TOPKS}" \
    --out "artifacts/tgn_runs/synthchain_summary_${RULE_TAG}.csv"
  echo "  summary: artifacts/tgn_runs/synthchain_summary_${RULE_TAG}.csv"
  echo "  compare: artifacts/tgn_runs/per_scenario_sc*_all_scores"
fi

echo "Done."
