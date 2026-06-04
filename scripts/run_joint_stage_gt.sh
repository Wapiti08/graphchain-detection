#!/usr/bin/env bash
# Joint GT-IOC stage training: SSL pretrain, then frozen backbone + low lambda_stage.
#
# Phase 1: joint SSL (lambda_stage=0) -> ${OUT}_ssl/best_ckpt_joint.pt
# Phase 2: load ckpt, --freeze-ssl-backbone, train stage head only
#
# Usage:
#   EPOCHS_SSL=15 EPOCHS_STAGE=5 DEVICE=cuda bash scripts/run_joint_stage_gt.sh
#   RECON_ONLY=1 OUT=artifacts/tgn_runs/joint_stage_frozen bash scripts/run_joint_stage_gt.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

OUT="${OUT:-artifacts/tgn_runs/joint_stage_frozen}"
SSL_OUT="${SSL_OUT:-${OUT}_ssl}"
EPOCHS_SSL="${EPOCHS_SSL:-15}"
EPOCHS_STAGE="${EPOCHS_STAGE:-5}"
BATCH="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
GRAPHS_DIR="${GRAPHS_DIR:-artifacts/graphs}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
SELECT_METRIC="${SELECT_METRIC:-auprc}"
RECON_ONLY="${RECON_ONLY:-0}"
RUN_RECON="${RUN_RECON:-1}"
SKIP_SSL="${SKIP_SSL:-0}"

LAMBDA_STAGE="${LAMBDA_STAGE:-0.2}"
LAMBDA_STAGE_NONE="${LAMBDA_STAGE_NONE:-0.05}"
STAGE_NONE_RATIO="${STAGE_NONE_SAMPLE_RATIO:-0.2}"
INIT_CKPT="${INIT_CKPT:-}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

COMMON_ARGS=(
  --scenarios sc1,sc2,sc3,sc4,sc5,sc6,sc7
  --batch-size "${BATCH}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --train-frac "${TRAIN_FRAC}"
  --graphs-dir "${GRAPHS_DIR}"
  --auto-generate
  --warmup
  --eval-ioc
  --save-scores
  --select-metric "${SELECT_METRIC}"
  --aux-supervision train_only
  --stage-supervision gt_ioc
  --lambda-ioc-rank 0
  --rank-supervision off
)

resolve_ckpt() {
  local dir="$1"
  for f in "${dir}/best_ckpt_joint.pt" "${dir}/best_ckpt.pt" "${dir}/ckpt_joint.pt"; do
    if [[ -f "${f}" ]]; then
      echo "${f}"
      return 0
    fi
  done
  return 1
}

if [[ "${RECON_ONLY}" != "1" && "${SKIP_SSL}" != "1" ]]; then
  echo "========== phase 1: joint SSL (no stage) -> ${SSL_OUT}"
  "${PY}" scripts/train_tgn_synthchain.py \
    "${COMMON_ARGS[@]}" \
    --epochs "${EPOCHS_SSL}" \
    --lambda-stage 0 \
    --lambda-stage-none 0 \
    --out "${SSL_OUT}" \
    ${EXTRA_TRAIN_ARGS}
fi

if [[ "${RECON_ONLY}" != "1" ]]; then
  if [[ -z "${INIT_CKPT}" ]]; then
    INIT_CKPT="$(resolve_ckpt "${SSL_OUT}")" || {
      echo "run_joint_stage_gt: no checkpoint under ${SSL_OUT}; run phase 1 or set INIT_CKPT" >&2
      exit 1
    }
  fi
  echo "========== phase 2: frozen SSL + stage (lambda=${LAMBDA_STAGE}) -> ${OUT}"
  echo "  init_ckpt=${INIT_CKPT}"
  "${PY}" scripts/train_tgn_synthchain.py \
    "${COMMON_ARGS[@]}" \
    --epochs "${EPOCHS_STAGE}" \
    --lambda-stage "${LAMBDA_STAGE}" \
    --lambda-stage-none "${LAMBDA_STAGE_NONE}" \
    --stage-none-sample-ratio "${STAGE_NONE_RATIO}" \
    --init-ckpt "${INIT_CKPT}" \
    --freeze-ssl-backbone \
    --out "${OUT}" \
    ${EXTRA_TRAIN_ARGS}
fi

if [[ "${RUN_RECON}" == "1" ]]; then
  echo "========== per-scenario recon (tail scores)"
  bash scripts/run_joint_recon_by_scenario.sh "${OUT}"
fi

echo "Done."
echo "  ssl ckpt:  ${SSL_OUT}/best_ckpt_joint.pt"
echo "  stage run: ${OUT}/run_summary.json"
