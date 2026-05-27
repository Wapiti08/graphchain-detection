#!/usr/bin/env bash
# Leave-one-scenario-out (LOSO) stress test on SynthChain sc1..sc7.
# Each fold: SSL on 6 scenarios' prefixes; test on the held-out scenario tail.
# Auxiliary IOC/stage losses apply only to train scenarios (never the holdout).
# Primary evaluation for the paper: bash scripts/run_per_scenario_eval.sh
#
# Usage (from repo root):
#   bash scripts/run_loso_synthchain.sh
#   # Recommended (paper-style LOSO + early stopping on val p@100):
#   EPOCHS=20 SEED=42 DEVICE=cuda \
#     EXTRA_TRAIN_ARGS='--early-stop-patience 5 --early-stop-min-delta 0.001' \
#     bash scripts/run_loso_synthchain.sh
#   # With stage classifier (learned attack reconstruction):
#   EXTRA_TRAIN_ARGS='--early-stop-patience 5 --early-stop-min-delta 0.001 --lambda-stage 0.5' \
#     bash scripts/run_loso_synthchain.sh
#   # Use a different K for checkpoint selection:
#   SELECT_METRIC=p_at_500 bash scripts/run_loso_synthchain.sh
#   # Smoke / CI:
#   EPOCHS=2 DEVICE=cpu bash scripts/run_loso_synthchain.sh
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

if ! "${PY}" -c "import torch, torch_geometric" 2>/dev/null; then
  echo "run_loso_synthchain: need torch + torch_geometric" >&2
  exit 1
fi

if [[ ! -d "${ROOT}/data/SynthChain" ]]; then
  echo "run_loso_synthchain: missing data/SynthChain" >&2
  exit 1
fi

for H in sc1 sc2 sc3 sc4 sc5 sc6 sc7; do
  OUT="artifacts/tgn_runs/loso_holdout_${H}"
  echo "========== LOSO holdout=${H} -> ${OUT}"
  "${PY}" scripts/train_tgn_synthchain.py \
    --holdout "${H}" \
    --scenarios sc1,sc2,sc3,sc4,sc5,sc6,sc7 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --auto-generate \
    --warmup \
    --eval-ioc \
    --aux-supervision train_only \
    --select-metric "${SELECT_METRIC}" \
    --out "${OUT}" \
    ${EXTRA_TRAIN_ARGS}
done

echo "========== merge summaries"
"${PY}" scripts/merge_loso_summaries.py \
  --runs-root artifacts/tgn_runs \
  --pattern "loso_holdout_*" \
  --out-csv artifacts/tgn_runs/loso_summary.csv

echo "run_loso_synthchain: done. See artifacts/tgn_runs/loso_summary.csv"
