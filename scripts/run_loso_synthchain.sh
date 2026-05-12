#!/usr/bin/env bash
# Leave-one-scenario-out (LOSO) on SynthChain sc1..sc7.
# Each fold: train on 6 scenarios, validate on the held-out scenario (see train_tgn_synthchain.py).
#
# Usage (from repo root):
#   bash scripts/run_loso_synthchain.sh
#   EPOCHS=10 SEED=0 bash scripts/run_loso_synthchain.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
BATCH="${BATCH_SIZE:-256}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

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
    --device cpu \
    --auto-generate \
    --warmup \
    --eval-ioc \
    --select-metric auprc \
    --out "${OUT}" \
    ${EXTRA_TRAIN_ARGS}
done

echo "========== merge summaries"
"${PY}" scripts/merge_loso_summaries.py \
  --runs-root artifacts/tgn_runs \
  --pattern "loso_holdout_*" \
  --out-csv artifacts/tgn_runs/loso_summary.csv

echo "run_loso_synthchain: done. See artifacts/tgn_runs/loso_summary.csv"
