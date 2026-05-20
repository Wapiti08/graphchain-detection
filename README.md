# GraphChain-Detection
detection of ongoing supply chain vulnerabilities with temporal graph neural networks

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


## Quick Running
```
python scripts/generate_graph.py --dataset synthchain --scenario sc1

# (recommended) export flattened TGN event stream
python scripts/generate_graph.py --dataset synthchain --scenario sc1 --export-tgn

# train/validate on sc1 with strict time split (past -> future)
python scripts/train_tgn_sc1.py --epochs 5 --batch-size 256

# joint training across all scenarios (per-scenario time split)
python scripts/train_tgn_synthchain.py --epochs 5 --batch-size 256 --auto-generate

# strict cross-scenario generalization: hold out one scenario (example: test on sc3)
python scripts/train_tgn_synthchain.py --holdout sc3 --epochs 5 --batch-size 256 --auto-generate

# LOSO (all folds sc1..sc7); writes artifacts/tgn_runs/loso_holdout_*/run_summary.json and loso_summary.csv
# Optional: EPOCHS=10 SEED=1 bash scripts/run_loso_synthchain.sh
bash scripts/run_loso_synthchain.sh

# Optional: shallow IOC supervision (margin on anomaly score vs non-IOC in same batch)
# python scripts/train_tgn_synthchain.py --holdout sc3 --eval-ioc --warmup --lambda-ioc-rank 0.1 --ioc-rank-margin 0.5 ...

# aggregate_alerts shares graph/alert_eval.py with train metrics (pf / ar / fr / p@K in epoch logs)
python scripts/aggregate_alerts.py --scores-csv artifacts/tgn_runs/synthchain_multi/best_eval_tail_scores.csv --out-dir artifacts/alerts

# full pipeline smoke: pytest + sc1 graph/TGN + 1-epoch train + tail scores + aggregate
# needs data/SynthChain; if QUT CSV is absent, only SynthChain-related tests run
bash scripts/e2e_regress.sh

# generate ground truth stage mapping data
python3 scripts/build_synthchain_stage_gt.py --out-dir artifacts/stage_gt

# partial attack-chain reconstruction (needs tail scores + stage GT; regen graphs with --export-tgn for row_idx/ioc_type)
python scripts/eval_attack_reconstruction.py \
  --scores-csv artifacts/tgn_runs/loso_holdout_sc5/best_eval_tail_scores.csv \
  --stage-gt artifacts/stage_gt/sc5.stages_gt.json
python scripts/merge_attack_reconstruction_summaries.py

```