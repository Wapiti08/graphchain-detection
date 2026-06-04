#!/usr/bin/env bash
# Pure SSL baseline ablations (7 scenarios):
#   (A) --neg-sampling window
#   (B) window + --train-only-benign
# Compare to per_scenario_sc*_all_scores (default random neg, no train-only-benign).
#
# Usage:
#   bash scripts/run_per_scenario_ssl_baseline.sh all
#   EPOCHS=20 DEVICE=cuda bash scripts/run_per_scenario_ssl_baseline.sh sc4
#   BASELINE_MODE=window_only EPOCHS=20 bash scripts/run_per_scenario_ssl_baseline.sh all
#   BASELINE_MODE=window_benign_only bash scripts/run_per_scenario_ssl_baseline.sh sc3
#
# BASELINE_MODE: both (default) | window_only | window_benign_only
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
BASELINE_MODE="${BASELINE_MODE:-both}"
NEG_WINDOW_SECONDS="${NEG_WINDOW_SECONDS:-3600}"
NEG_WINDOW_MAX_CANDS="${NEG_WINDOW_MAX_CANDS:-4096}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

WINDOW_ARGS=(
  --neg-sampling window
  --neg-window-seconds "${NEG_WINDOW_SECONDS}"
  --neg-window-max-cands "${NEG_WINDOW_MAX_CANDS}"
)

if [[ "${SC}" == "all" ]]; then
  SCENARIOS=(sc1 sc2 sc3 sc4 sc5 sc6 sc7)
else
  SCENARIOS=("${SC}")
fi

run_one() {
  local sc="$1"
  local tag="$2"
  local benign_flag="$3"
  local out="artifacts/tgn_runs/per_scenario_${sc}_ssl_${tag}"
  local stage_gt="artifacts/stage_gt/${sc}.stages_gt.json"
  local train_args=("${WINDOW_ARGS[@]}")
  if [[ "${benign_flag}" == "1" ]]; then
    train_args+=(--train-only-benign)
  fi

  echo "========== train ${sc} (${tag}) -> ${out}"
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
    --aux-supervision train_only \
    --save-scores \
    --save-scores-split all \
    --select-metric auprc \
    --out "${out}" \
    "${train_args[@]}" \
    ${EXTRA_TRAIN_ARGS}

  local scores="${out}/best_eval_all_scores.csv"
  if [[ ! -f "${scores}" ]]; then
    scores="${out}/eval_all_scores.csv"
  fi

  echo "========== reconstruction ${sc} (${tag})"
  "${PY}" scripts/eval_attack_reconstruction.py \
    --scores-csv "${scores}" \
    --stage-gt "${stage_gt}" \
    --topks "${TOPKS}" \
    --group-cap-adaptive-key etype_src \
    --group-cap-adaptive-probe-mult 5 \
    --group-cap-adaptive-hot-threshold 20 \
    --group-cap-adaptive-hot-cap 10 \
    --out "${out}/reconstruction_metrics.json"
}

for sc in "${SCENARIOS[@]}"; do
  if [[ "${BASELINE_MODE}" == "both" || "${BASELINE_MODE}" == "window_only" ]]; then
    run_one "${sc}" "window" 0
  fi
  if [[ "${BASELINE_MODE}" == "both" || "${BASELINE_MODE}" == "window_benign_only" ]]; then
    run_one "${sc}" "window_benign" 1
  fi
done

echo "========== summarize"
if [[ "${BASELINE_MODE}" == "both" || "${BASELINE_MODE}" == "window_only" ]]; then
  "${PY}" scripts/summarize_synthchain_runs.py \
    --runs-dir artifacts/tgn_runs \
    --pattern "per_scenario_sc*_ssl_window" \
    --topks "${TOPKS}" \
    --out artifacts/tgn_runs/synthchain_summary_ssl_window.csv
fi
if [[ "${BASELINE_MODE}" == "both" || "${BASELINE_MODE}" == "window_benign_only" ]]; then
  "${PY}" scripts/summarize_synthchain_runs.py \
    --runs-dir artifacts/tgn_runs \
    --pattern "per_scenario_sc*_ssl_window_benign" \
    --topks "${TOPKS}" \
    --out artifacts/tgn_runs/synthchain_summary_ssl_window_benign.csv
fi

echo "Done."
echo "  window:       artifacts/tgn_runs/synthchain_summary_ssl_window.csv"
echo "  window_benign: artifacts/tgn_runs/synthchain_summary_ssl_window_benign.csv"
echo "Compare to:    artifacts/tgn_runs/per_scenario_sc*_all_scores (original SSL baseline)"
