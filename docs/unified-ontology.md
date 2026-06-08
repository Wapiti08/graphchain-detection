# Unified Node and Edge Ontology

This document specifies node and edge types with their definitions, key attributes, and explicit exploitation indicators. The canonical implementation lives in [`config/ontology.py`](../config/ontology.py); parsers map raw telemetry into this schema.

## Node Types

| Node Type | Key Attributes | Semantics | Exploitation Indicators |
|-----------|----------------|-----------|-------------------------|
| `PKG` | `package_name`, `version`, `registry`, `dependency_role` | Software package or dependency entity. | Dependency confusion, malicious package installation, compromised package lineage. |
| `PROC` | `proc_name`, `pid`, `ppid`, `user`, `image_path`, `is_lolbin`, `parent_depth` | Runtime process or execution context. | LOLBin abuse, suspicious parent-child chains, install-script execution, process spawning. |
| `CMD` | `command`, `interpreter`, `args`, `encoded`, `shell` | Command-line or shell command extracted from process, syslog, or alert text. | PowerShell, bash, npm install hooks, encoded commands, curl/wget/certutil download commands. |
| `FILE` | `path`, `file_type`, `mime_type`, `hash_md5`, `hash_sha256`, `size`, `path_sensitivity` | File, directory, archive, script, payload, or filesystem artifact. | Payload drop, persistence path write, secret file access, suspicious archive/model artifact. |
| `NET` | `ip`, `domain`, `port`, `protocol`, `service`, `tls_version`, `cipher`, `server_name`, `validation_status`, `is_known_registry` | Network endpoint, DNS name, URL host, remote service, or resolved address. | C2 endpoint, exfiltration destination, unknown registry, suspicious port, self-signed TLS, no-SNI connection. |
| `HOST` | `host_id`, `hostname`, `container_id`, `role`, `os` | Runtime host, VM, container, sandbox, victim, or attacker-controlled service host. | Cross-container movement, exposed service, attacker/victim role separation. |
| `USER` | `user_name`, `uid`, `account`, `tenant`, `privilege` | Local user, cloud account, service account, or identity context. | Credential abuse, privilege escalation, suspicious service-account activity. |
| `CRED` | `cred_type`, `provider`, `scope`, `is_secret_leak` | Credential, token, secret, key, password, or cloud access material. | Token leakage, Azure key exposure, package credential theft, secret exfiltration. |
| `ARTIFACT` | `artifact_type`, `name`, `hash_md5`, `hash_sha256`, `uri`, `is_model_artifact` | Supply-chain artifact such as archive, wheel, model, payload, or package asset. | Malicious ML model, staged payload, poisoned archive, artifact substitution. |
| `ALERT` | `alert_signature`, `category`, `severity`, `signature_id`, `source` | Detection alert or IDS/security signal, e.g., Suricata EVE alert. | IDS signature hit, suspicious network behavior, policy violation, exploit indicator. |
| `SYSCALL` | `syscall_name`, `category` | System-call behavior type node. | Evasion, privilege-relevant calls, filesystem/network behavior patterns. |

## Edge Types

| Edge Type | Src → Dst | Attributes | Semantics | Exploitation Coverage |
|-----------|-----------|------------|-----------|----------------------|
| `DEPEND` | `PKG` → `PKG` | `version_constraint`, `dependency_type`, `registry` | Package dependency relation. | Dependency confusion, malicious transitive dependency, package substitution. |
| `LOAD` | `PKG` → `PROC` | `entry_point`, `install_phase`, `script_type` | Package invokes runtime execution. | Malicious install script, post-install trigger, runtime payload activation. |
| `EXEC` | `PROC` → `PROC`, `PROC` → `CMD` | `cmdline`, `interpreter`, `args`, `is_lolbin`, `parent_depth` | Process execution, subprocess creation, or command invocation. | Command hijacking, shell execution, LOLBin chain, encoded payload execution. |
| `INVOKE` | `PROC` → `SYSCALL` | `args`, `return_val`, `syscall_category` | System-call invocation. | Evasion, privilege escalation, suspicious filesystem/network syscall behavior. |
| `READ` | `PROC` → `FILE` | `bytes`, `operation`, `evidence` | File read. | Secret collection, credential access, reconnaissance, model/artifact inspection. |
| `WRITE` | `PROC` → `FILE` | `bytes`, `operation`, `evidence` | File write, create, or modification. | Payload drop, persistence, artifact generation, staged script creation. |
| `DELETE` | `PROC` → `FILE` | `operation`, `evidence`, `path_sensitivity` | File deletion or cleanup. | Trace removal, payload cleanup, anti-forensics. |
| `CONNECT` | `PROC` → `NET` | `bytes_sent`, `bytes_recv`, `duration`, `direction`, `protocol`, `service`, `method`, `uri`, `status_code`, `user_agent` | Network connection or HTTP transaction. | C2 communication, payload retrieval, exfiltration, suspicious registry access. |
| `DNS_QUERY` | `PROC` → `NET` | `query_domain`, `query_type`, `rcode`, `answers`, `ttls` | DNS query initiated by a process or host context. | DNS tunneling, reconnaissance, C2 lookup, fast-flux preparation. |
| `RESOLVE` | `NET` → `NET` | `resolved_ip`, `query_domain`, `ttl`, `rcode` | DNS name resolution. | DNS-based C2, fast-flux, domain-to-IP infrastructure linkage. |
| `REDIRECT` | `NET` → `NET` | `http_status`, `location`, `method`, `uri` | HTTP redirect relation. | URL obfuscation, staging redirect, redirect-based payload delivery. |
| `RELAY` | `NET` → `NET` | `Δt`, `protocol`, `bytes` | Proxy, relay, or multi-hop transfer relation. | Staging infrastructure, proxy chain, traffic relay. |
| `EXFILTRATE` | `PROC` → `NET` | `bytes_sent`, `method`, `uri`, `content_type`, `evidence` | Outbound transfer interpreted as exfiltration-like behavior. | Credential upload, compressed data transfer, FTP/HTTP exfiltration, callback with sensitive content. |
| `USES_CRED` | `PROC` → `CRED`, `USER` → `CRED` | `cred_type`, `provider`, `scope`, `evidence` | Credential use, access, or exposure. | Token handoff, Azure key leakage, password/secret abuse. |
| `ALERT_ON` | `ALERT` → `PROC`, `ALERT` → `NET`, `ALERT` → `FILE` | `alert_signature`, `category`, `severity`, `signature_id` | Security alert associated with an entity. | Suricata alert context, suspicious endpoint or payload flagged by IDS. |
| `INJECT` | `PROC` → `PROC` | `injection_type`, `target_proc`, `evidence` | Code injection into another process. | Process hollowing, DLL injection, runtime tampering. |
| `CAUSE` | `PROC` → `PROC` | `Δt̂`, `cause_rule`, `confidence`, `is_derived` | Derived dependency edge, not directly observed telemetry. | Temporal attack-chain continuity, same-process sequence, shared-object bridge. |
