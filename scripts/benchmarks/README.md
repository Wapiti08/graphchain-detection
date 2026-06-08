# SynthChain quantitative benchmarks

Main-table protocol for comparisons (all on SynthChain sc1–sc7).

## Layout

| Path | Role |
|------|------|
| `gchain/baselines/` | Telemetry filters, freq-rarity / path-LOF scorers, unified eval |
| `gchain/eval/latency.py` | Median ms per 1k tail edges |
| `scripts/benchmarks/run_baseline.py` | One method × telemetry × scenarios → JSON |
| `scripts/benchmarks/train_static_gnn.py` | Train GraphSAGE / RGCN + export CSV / benchmark JSON |
| `scripts/benchmarks/bench_fusechain_latency.py` | FuseChain tail inference latency (warmup + score) |
| `scripts/benchmarks/ingest_scores.py` | FuseChain (or external static GNN) from tail score CSV |
| `scripts/benchmarks/merge_main_table.py` | JSON → `main_table.csv` (+ optional LaTeX) |
| `artifacts/benchmarks/` | Outputs (gitignored) |

## Prerequisites

```bash
# Graph exports + stage GT
bash scripts/e2e_regress.sh   # or your pipeline export
python3 scripts/build_synthchain_stage_gt.py --out-dir artifacts/stage_gt
```

## Run

```bash
bash scripts/benchmarks/run_main_table.sh
```

Or a single baseline:

```bash
python3 scripts/benchmarks/run_baseline.py --method freq_rarity --telemetry full
```

Static GNN (GraphSAGE / RGCN):

```bash
# End-to-end benchmark JSON (trains on 70% prefix, scores tail)
python3 scripts/benchmarks/run_baseline.py --method graphsage --telemetry full --epochs 25 --device cpu
python3 scripts/benchmarks/run_baseline.py --method rgcn --telemetry full

# Train + export tail CSV + optional checkpoint
python3 scripts/benchmarks/train_static_gnn.py \
  --variant graphsage --export-csv --write-bench-json --device cpu
```

Protocol: build a **static** multi-relational graph from the train prefix of `synthchain_scX.tgn.pt`; node features combine degree/relation histograms and (when `synthchain_scX.full.pt` exists) hetero node attrs. Train link prediction with negative sampling; tail anomaly score = `-log p(edge)`.

Ingest FuseChain scores after training:

```bash
python3 scripts/benchmarks/ingest_scores.py \
  --method fusechain --scenario sc1 --telemetry full \
  --scores-csv artifacts/tgn_runs/per_scenario_sc1_rule_stage/best_eval_all_scores.csv
```

FuseChain latency (prefix warmup + tail scoring; excludes offline training):

```bash
python3 scripts/benchmarks/bench_fusechain_latency.py \
  --scenarios sc1,sc2,sc3,sc4,sc5,sc6,sc7 \
  --device cuda \
  --ingest-missing

python3 scripts/benchmarks/merge_main_table.py
```

## Main table columns

Macro-average over applicable scenarios (EVE N/A on sc2/sc5/sc6):

- **AUROC**, **AUPRC**, **P@500** — tail detection (`gchain/train/metrics.py`, `precision_at_k_all`)
- **StageRec@500**, **LCS@500** — `evaluate_reconstruction` → `by_k["500"]`
- **Lat (ms/1k)** — scoring only (excludes TGN training)

## Scheme A telemetry

| Row | `--telemetry` | Sources |
|-----|---------------|---------|
| Audit-only | `audit` | `azure_events`, `azure_conn`, `azure_process`, `azure_syslog` |
| Network-sensor-only | `zeek` | `zeek_*.csv` |
| Alert-only | `eve` | `eve.json` |

sc2/sc5/sc6 are audit-only scenarios (composite `azure_events` only in ground truth).
