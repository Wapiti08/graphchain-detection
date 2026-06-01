#!/usr/bin/env bash
# Per-scenario evaluation (primary protocol): train and test on the SAME scenario only.
# Self-supervised link prediction on the scenario prefix; evaluate IOC ranking on its tail.
# Auxiliary IOC/stage losses (if enabled) use only that scenario's labels.
#
# Usage (from repo root):
#   bash scripts/run_per_scenario_eval.sh
#   EPOCHS=20 DEVICE=cuda EXTRA_TRAIN_ARGS='--early-stop-patience 5 --early-stop-min-delta 0.001' bash scripts/run_per_scenario_eval.sh
#   # Pure SSL (no IOC rank / stage head):
#   EXTRA_TRAIN_ARGS='--aux-supervision off' bash scripts/run_per_scenario_eval.sh
#   # With weak stage supervision (shared ioc_type->stage ontology):
#   EXTRA_TRAIN_ARGS='--lambda-stage 0.5 --lambda-ioc-rank 0.1' bash scripts/run_per_scenario_eval.sh
#   # Also run attack reconstruction (uses best_eval_all_scores.csv when present):
#   RUN_RECON=1 SAVE_SCORES_SPLIT=all bash scripts/run_per_scenario_eval.sh
#   # Or one scenario, same protocol as sc1/sc2:
#   bash scripts/run_per_scenario_ssl_recon.sh sc3
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
BATCH="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cpu}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SELECT_METRIC="${SELECT_METRIC:-p_at_100}"
RUN_RECON="${RUN_RECON:-0}"
STAGE_GT_DIR="${STAGE_GT_DIR:-artifacts/stage_gt}"

if ! "${PY}" -c "import torch, torch_geometric" 2>/dev/null; then
  echo "run_per_scenario_eval: need torch + torch_geometric" >&2
  exit 1
fi

if [[ ! -d "${ROOT}/data/SynthChain" ]]; then
  echo "run_per_scenario_eval: missing data/SynthChain" >&2
  exit 1
fi

for SC in sc1 sc2 sc3 sc4 sc5 sc6 sc7; do
  OUT="artifacts/tgn_runs/per_scenario_${SC}"
  echo "========== per-scenario ${SC} -> ${OUT}"
  "${PY}" scripts/train_tgn_synthchain.py \
    --scenarios "${SC}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --auto-generate \
    --warmup \
    --eval-ioc \
    --aux-supervision train_only \
    --select-metric "${SELECT_METRIC}" \
    --save-scores \
    --save-scores-split "${SAVE_SCORES_SPLIT:-tail}" \
    --out "${OUT}" \
    ${EXTRA_TRAIN_ARGS}

  if [[ "${RUN_RECON}" == "1" ]]; then
    GT="${STAGE_GT_DIR}/${SC}.stages_gt.json"
    SCORES="${OUT}/best_eval_all_scores.csv"
    if [[ ! -f "${SCORES}" ]]; then
      SCORES="${OUT}/eval_all_scores.csv"
    fi
    if [[ ! -f "${SCORES}" ]]; then
      SCORES="${OUT}/best_eval_tail_scores.csv"
    fi
    if [[ -f "${SCORES}" && -f "${GT}" ]]; then
      echo "========== reconstruction ${SC} (${SCORES})"
      "${PY}" scripts/eval_attack_reconstruction.py \
        --scores-csv "${SCORES}" \
        --stage-gt "${GT}" \
        --topks "${RECON_TOPKS:-10,50,100,500}" \
        --pred-min-prob "${RECON_PRED_MIN_PROB:-0.5}" \
        --pred-min-count "${RECON_PRED_MIN_COUNT:-2}" \
        --out "${OUT}/reconstruction_metrics.json"
    else
      echo "run_per_scenario_eval: skip recon ${SC} (missing scores or ${GT})" >&2
    fi
  fi
done

echo "========== merge detection summaries"
"${PY}" scripts/merge_loso_summaries.py \
  --runs-root artifacts/tgn_runs \
  --pattern "per_scenario_*" \
  --require-protocol per_scenario \
  --out-csv artifacts/tgn_runs/per_scenario_summary.csv

if [[ "${RUN_RECON}" == "1" ]]; then
  echo "========== merge reconstruction summaries"
  "${PY}" scripts/merge_attack_reconstruction_summaries.py \
    --runs-root artifacts/tgn_runs \
    --pattern "per_scenario_*" \
    --out-csv artifacts/tgn_runs/per_scenario_reconstruction_summary.csv
fi

echo "run_per_scenario_eval: done."
echo "  detection: artifacts/tgn_runs/per_scenario_summary.csv"
