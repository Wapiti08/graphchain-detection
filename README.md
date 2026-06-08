# FuseChain-Detection
detection of ongoing supply chain vulnerabilities with temporal graph neural networks

## Python packages

| Package | Role |
|---------|------|
| `parsers/` | Raw logs / CSV → canonical `Event` |
| `config/` | Ontology, dataset paths |
| `graphcore/` | Hetero graph build + TGN stream export (`build_hetero_graph`, `hetero_to_tgn_event_stream`) |
| `gchain/` | Pipeline (`gchain.pipeline`), TGN training (`gchain.train`), detection metrics (`gchain.eval`) |

CLI: `python -m gchain.pipeline`, `python -m gchain.train` (or `scripts/train_tgn.py`).

## Data Processing (Feature Extraction)

### Unified Entity Ontology

The canonical graph schema is based on HetHunt's heterogeneous runtime graph and extended with the IoC entities observed in the SynthChain scenarios. The goal is to keep the graph explainable while avoiding one-off node types for every raw log field.

#### Node Types

- `PKG` --- package identity and dependency unit.
    - attrs: `ecosystem`, `name`, `version`, `registry`, `is_direct_dependency`

- `PROC` --- process/runtime entity, including install scripts, subprocesses, interpreters, LOLBins, build jobs, and container entrypoints.
    - attrs: `proc_name`, `pid`, `ppid`, `cmdline`, `user`, `working_dir`, `image_path`, `is_lolbin`, `parent_depth`

- `CMD` --- command string or shell action invoked by a process.
    - attrs: `command`, `interpreter`, `args`, `shell`, `encoded`, `is_obfuscated`

- `FILE` --- file-system entity that can be read, written, deleted, packed, or staged.
    - attrs: `path`, `file_type`, `mime_type`, `size`, `hash_md5`, `hash_sha1`, `hash_sha256`, `path_sensitivity`, `is_archive`, `is_model_artifact`

- `NET` --- network endpoint, including IPs, domains, URLs, and service ports.
    - attrs: `ip`, `domain`, `url`, `port`, `protocol`, `service`, `is_known_registry`, `tls_valid`, `tls_version`, `cipher`, `server_name`, `sni_matches_cert`, `validation_status`

- `HOST` --- machine, VM, container, or attacker/victim environment.
    - attrs: `hostname`, `host_ip`, `os`, `role`, `cloud_provider`, `container_id`

- `USER` --- local, cloud, package-registry, or service account identity.
    - attrs: `username`, `domain`, `account_type`, `privilege_level`

- `CRED` --- token, password, API key, cloud secret, or credential-like material.
    - attrs: `cred_type`, `provider`, `scope`, `is_secret_leak`

- `ARTIFACT` --- supply-chain artifact such as wheel/tarball/npm package, build zip, executable, model file, or container image.
    - attrs: `artifact_type`, `name`, `version`, `hash_md5`, `hash_sha256`, `source`, `signed`, `signature_valid`

- `SYSCALL` --- system call or low-level behavior type.
    - attrs: `name`, `category`

- `ALERT` --- detection or IDS signal such as Suricata alert signatures and MITRE/TTP labels.
    - attrs: `signature`, `category`, `severity`, `mitre_technique`, `confidence`

#### Edge Types

- `DEPEND`: `PKG -> PKG`
    - attrs: `version_constraint`, `dependency_type`

- `LOAD`: `PKG -> PROC`
    - attrs: `entry_point`, `phase`

- `EXEC`: `PROC -> PROC`, `PROC -> CMD`, `HOST -> PROC`
    - attrs: `cmdline`, `args`, `exit_code`, `phase`

- `INVOKE`: `PROC -> SYSCALL`
    - attrs: `args`, `return_val`

- `READ`: `PROC -> FILE`
    - attrs: `bytes`, `operation`, `evidence`

- `WRITE`: `PROC -> FILE`
    - attrs: `bytes`, `operation`, `evidence`

- `DELETE`: `PROC -> FILE`
    - attrs: `operation`, `evidence`

- `CONNECT`: `PROC -> NET`, `HOST -> NET`
    - attrs: `bytes_sent`, `bytes_recv`, `direction`, `duration`, `protocol`, `service`

- `DNS_QUERY`: `PROC -> NET`
    - attrs: `query_domain`, `query_type`, `rcode`, `answers`, `ttls`

- `RESOLVE`: `NET -> NET`
    - attrs: `resolved_ip`

- `REDIRECT`: `NET -> NET`
    - attrs: `http_status`, `location`

- `RELAY`: `NET -> NET`
    - attrs: `delta_t`, `hop_count`

- `EXFILTRATE`: `PROC -> NET`, `FILE -> NET`
    - attrs: `bytes_sent`, `channel`, `archive_type`, `evidence`

- `INJECT`: `PROC -> PROC`, `PROC -> ARTIFACT`
    - attrs: `injection_type`, `target`, `phase`

- `AUTHENTICATE`: `USER -> HOST`, `USER -> NET`, `PROC -> NET`
    - attrs: `auth_method`, `success`, `credential_type`

- `USES_CRED`: `PROC -> CRED`, `USER -> CRED`
    - attrs: `usage`, `provider`, `scope`

- `HOSTS`: `HOST -> NET`, `HOST -> PROC`
    - attrs: `port`, `service`, `listen_state`

- `GENERATES`: `PROC -> ARTIFACT`, `PKG -> ARTIFACT`
    - attrs: `phase`, `tool`, `hash_sha256`

- `TRIGGERS`: `ARTIFACT -> PROC`, `CMD -> PROC`, `ALERT -> PROC`
    - attrs: `trigger_type`, `condition`, `evidence`

#### Common Event Attributes

- Temporal attrs: `ts`, `order`, `phase`, `duration`
- Provenance attrs: `scenario_id`, `log_source`, `raw_type`, `event_id`, `flow_id`, `uid`
- IoC attrs: `ioc_label`, `confidence`, `threat_type`, `mitre_technique`
- HTTP attrs: `method`, `uri`, `status_code`, `status_msg`, `user_agent`, `resp_bytes`
- TLS attrs: `server_name`, `cipher`, `tls_version`, `sni_matches_cert`, `validation_status`
- DNS attrs: `query_domain`, `query_type`, `answers`, `rcode`


## Deployment-aligned protocol (link + rule stage + recon)

Training uses **no GT IOC line/value labels**. Stage supervision comes from
`config/weak_supervision_rules.json` → `rule_ioc_type` on `y_rule_high` edges only.
Evaluation still uses held-out stage GT (`artifacts/stage_gt/*.json`) for metrics.

| Phase | What runs | Deployable? |
|-------|-----------|-------------|
| Train prefix | SSL link loss + optional stage CE (`--stage-supervision rule_high`) | Yes |
| Train (avoid as primary) | `--stage-supervision gt_ioc`, `--rank-supervision ioc_line` | No (oracle) |
| Inference | Anomaly score on all edges + `pred_stage` in scores CSV | Yes |
| Reporting | `by_k_pred_stage`, `by_k_group_cap_adaptive`, optional `by_alert_pred_stage` | Post-hoc / analyst view |
| Lab metric only | `by_k` (stages from GT IOC edges in top-K) | Eval upper-bound style |

Regenerate graphs (weak rules v2 in stream):

```bash
python -m gchain.pipeline --dataset synthchain --all-scenarios --only-ioc-logs --export-tgn
```

Run deployment-aligned per-scenario training + adaptive recon:

```bash
EPOCHS=20 DEVICE=cuda bash scripts/run_per_scenario_rule_stage.sh all
# summary: artifacts/tgn_runs/synthchain_summary_rule_stage.csv
```

Manual train flags (equivalent):

```bash
python scripts/train_tgn_synthchain.py --scenarios sc4 \
  --lambda-stage 0.5 --lambda-stage-none 0.1 --stage-supervision rule_high \
  --lambda-ioc-rank 0 --rank-supervision off \
  --save-scores --save-scores-split all --select-metric auprc \
  --out artifacts/tgn_runs/per_scenario_sc4_rule_stage
```

Primary SSL + adaptive recon (research baseline) remains `bash scripts/run_per_scenario_ssl_recon.sh`.
Weak-rule **ranking** ablation: `bash scripts/run_per_scenario_weak_rule.sh`.

### Code map (what was added/changed)

| Area | Path | Change |
|------|------|--------|
| Rules | `config/weak_supervision_rules.json` | v2 rank_policy; benign install cmd excluded from rank |
| Annotate | `parsers/rules/weak_supervision.py` | `infer_rule_hits_for_event`, `annotate_events_with_weak_rules` |
| Parser hook | `parsers/synthchain/event_parsers.py` | `annotate_weak_rules=True` on load |
| TGN export | `graphcore/tgn_input.py`, `graphcore/edge_meta.py` | `y_rule`, `y_rule_high`, `rule_ioc_type` in `.tgn.pt` |
| Pipeline | `gchain/pipeline/generate.py` | save `rule_ioc_type` in blob |
| Train CLI | `gchain/train/args_common.py` | `--stage-supervision`, `--rank-supervision` |
| Stage labels | `gchain/train/modeling.py` | `build_stage_labels(..., stage_supervision=rule_high)` |
| Streams | `gchain/train/streams.py` | load `rule_ioc_type` from `.tgn.pt` |
| Train loop | `gchain/train/train_loop.py` | stage CE + `lambda_stage_none` for deployable masks |
| Rank aux | `gchain/train/supervision.py` | `rank_supervision_tensor()` |
| Script (rank ablation) | `scripts/run_per_scenario_weak_rule.sh` | rule rank + adaptive recon |
| Script (deploy) | `scripts/run_per_scenario_rule_stage.sh` | **link + rule stage**, no rank loss |
| Tests | `tests/test_weak_supervision_rules.py` | rule v2 policy tests |

### Rules-update ablation (extensibility demo)

Simulates analyst-added weak rule `tier2_staging_path_download` (curl/wget staging under `/dev/shm` or `/tmp`), then compares coverage and per-scenario train metrics:

```bash
# fast: rule_hit / rule_high deltas on sc1..sc7 (no GPU)
STATS_ONLY=1 bash scripts/run_rules_update_ablation.sh

# full: re-export graphs + train sc1,sc3,sc5 (default 5 epochs, cpu)
bash scripts/run_rules_update_ablation.sh

# outputs under artifacts/rules_update_ablation/
#   rule_coverage_compare.csv
#   train_regression_compare.csv
```

Pipeline accepts alternate rules: `--weak-rules config/weak_supervision_rules_update_ablation.json`.

## Quick Running
```
python -m gchain.pipeline --dataset synthchain --scenario sc1

# (recommended) export flattened TGN event stream
python -m gchain.pipeline --dataset synthchain --scenario sc1 --export-tgn

# SynthChain: batch-export sc1..sc7 .tgn.pt
python -m gchain.pipeline --dataset synthchain --all-scenarios --export-tgn

# train/validate on sc1 with strict time split (past -> future)
python scripts/train_tgn_sc1.py --epochs 5 --batch-size 256

# PRIMARY: per-scenario eval (SSL on prefix, test tail; same scenario train+test)
# Optional weak aux: EXTRA_TRAIN_ARGS='--lambda-stage 0.5 --lambda-ioc-rank 0.1'
# RUN_RECON=1 also writes per_scenario_reconstruction_summary.csv
bash scripts/run_per_scenario_eval.sh

# single scenario via train_tgn_synthchain (equivalent to one fold above)
python scripts/train_tgn_synthchain.py --scenarios sc1 --epochs 5 --batch-size 256 --auto-generate --eval-ioc --warmup

# joint training across all scenarios (ablation; not the primary protocol)
python scripts/train_tgn_synthchain.py --epochs 5 --batch-size 256 --auto-generate

# LOSO stress test: train 6 scenarios, test holdout (cross-attack-family transfer lower bound)
bash scripts/run_loso_synthchain.sh

# pure self-supervised (no IOC rank / stage CE on train scenarios):
# EXTRA_TRAIN_ARGS='--aux-supervision off' bash scripts/run_per_scenario_eval.sh

# Optional: shallow IOC supervision (margin on anomaly score vs non-IOC in same batch)
# python scripts/train_tgn_synthchain.py --holdout sc3 --eval-ioc --warmup --lambda-ioc-rank 0.1 --ioc-rank-margin 0.5 ...

# aggregate_alerts shares gchain.eval.alert_eval with train metrics (pf / ar / fr / p@K in epoch logs)
python scripts/aggregate_alerts.py --scores-csv artifacts/tgn_runs/synthchain_multi/best_eval_tail_scores.csv --out-dir artifacts/alerts

# full pipeline smoke: pytest + sc1 graph/TGN + 1-epoch train + tail scores + aggregate
# needs data/SynthChain
bash scripts/e2e_regress.sh

# generate ground truth stage mapping data
python3 scripts/build_synthchain_stage_gt.py --out-dir artifacts/stage_gt

# partial attack-chain reconstruction (needs tail scores + stage GT; regen graphs with --export-tgn for row_idx/ioc_type)
python scripts/eval_attack_reconstruction.py \
  --scores-csv artifacts/tgn_runs/per_scenario_sc5/best_eval_tail_scores.csv \
  --stage-gt artifacts/stage_gt/sc5.stages_gt.json
python scripts/merge_attack_reconstruction_summaries.py \
  --pattern "per_scenario_*" \
  --out-csv artifacts/tgn_runs/per_scenario_reconstruction_summary.csv

# quantitative main table (detection + reconstruction + latency; see scripts/benchmarks/README.md)
bash scripts/benchmarks/run_main_table.sh

```