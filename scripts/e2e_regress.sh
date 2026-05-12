#!/usr/bin/env bash
# End-to-end smoke regression: SynthChain sc1 graph + TGN + short train + scores + alerts.
# Run from repo root:  bash scripts/e2e_regress.sh
# Requires: Python with torch + torch_geometric, data under data/SynthChain/ for sc1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

if ! "${PY}" -m pytest --version >/dev/null 2>&1; then
  echo "e2e_regress: pytest not installed (pip install pytest)." >&2
  exit 1
fi

if [[ ! -d "${ROOT}/data/SynthChain" ]]; then
  echo "e2e_regress: missing data/SynthChain — cannot build sc1 graph." >&2
  exit 1
fi

echo "==> generate sc1 graph + TGN export (limited rows, fast)"
"${PY}" scripts/generate_graph.py \
  --dataset synthchain \
  --scenario sc1 \
  --export-tgn \
  --limit-per-file 400

echo "==> pytest"
QUT_CSV="${ROOT}/data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv"
if [[ -f "${QUT_CSV}" ]]; then
  "${PY}" -m pytest tests/ -q --tb=short
else
  echo "WARN: QUT CSV not found; running SynthChain + TGN export tests only."
  "${PY}" -m pytest tests/test_synthchain_parser.py tests/test_tgn_export.py -q --tb=short
fi

echo "==> train_tgn_sc1 (1 epoch)"
"${PY}" scripts/train_tgn_sc1.py --epochs 1 --batch-size 256 --out artifacts/tgn_runs/e2e_smoke_sc1

echo "==> train_tgn_synthchain (sc1 only, tail scores for aggregate)"
"${PY}" scripts/train_tgn_synthchain.py \
  --scenarios sc1 \
  --epochs 1 \
  --batch-size 256 \
  --device cpu \
  --eval-ioc \
  --save-scores-each-epoch \
  --out artifacts/tgn_runs/e2e_smoke_multi

SCORES="${ROOT}/artifacts/tgn_runs/e2e_smoke_multi/eval_tail_scores_epoch001.csv"
if [[ ! -f "${SCORES}" ]]; then
  echo "e2e_regress: expected scores CSV missing: ${SCORES}" >&2
  exit 1
fi

echo "==> aggregate_alerts (smoke output)"
"${PY}" scripts/aggregate_alerts.py \
  --scores-csv artifacts/tgn_runs/e2e_smoke_multi/eval_tail_scores_epoch001.csv \
  --out-dir artifacts/alerts_e2e_smoke \
  --min-events 1 \
  --topk-events 120

echo "e2e_regress: OK"
