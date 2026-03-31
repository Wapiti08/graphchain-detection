# configs/synthchain_ioc_config.py

SYNTHCHAIN_IOC_CONFIG = {
    # SC1: Steganography PyPI attack (colorsapi) — Windows
    # Ground truth: azure_events (465), azure_conn (141), azure_process (42)
    # Total IOCs: 648
    "sc1": {
        "root": "data/SynthChain/sc1",
        "logs": {
            "azure_events": {
                "filename": "windows/azure_events.csv",
                "has_ioc": True,
                "log_type": "AE",
                "entities": ["src_ip", "dst_ip", "user", "process", "file_path", "cmdline"],
                "timestamp_col": "TimeGenerated",
            },
            "azure_conn": {
                "filename": "windows/azure_conn.csv",
                "has_ioc": True,
                "log_type": "AC",
                "entities": ["src_ip", "dst_ip", "src_port", "dst_port", "bytes_sent", "bytes_recv"],
                "timestamp_col": "TimeGenerated",
            },
            "azure_process": {
                "filename": "windows/azure_process.csv",
                "has_ioc": True,
                "log_type": "AP",
                "entities": ["user", "process", "parent_process", "cmdline"],
                "timestamp_col": "TimeGenerated",
            },
        },
    },

    # SC2: LOLBin chain PyPI attack (pystallerer) — Windows
    # Ground truth: all IOCs in azure_events only
    # Total IOCs: ~141 (51 pkg + 3 attack_ip + 10 exfil + 656 inject + auxiliaries)
    "sc2": {
        "root": "data/SynthChain/sc2",
        "logs": {
            "azure_events": {
                "filename": "azure_events.csv",
                "has_ioc": True,
                "log_type": "AE",
                "entities": ["src_ip", "dst_ip", "user", "process", "parent_process", "file_path", "cmdline", "injection_target"],
                "timestamp_col": "TimeGenerated",
            },
        },
    },

    # SC3: NPM FTP exfiltration (olymptrade) — Linux
    # Ground truth: eve.json (213), azure_syslog (14), zeek_conn (96), zeek_dns (6), zeek_http (4)
    # Total IOCs: 534
    "sc3": {
        "root": "data/SynthChain/sc3",
        "logs": {
            "eve": {
                "filename": "eve.json",
                "has_ioc": True,
                "log_type": "EVE",
                "entities": ["src_ip", "dst_ip", "src_port", "dst_port", "alert_signature"],
                "timestamp_col": "timestamp",
            },
            "azure_syslog": {
                "filename": "azure_syslog.csv",
                "has_ioc": True,
                "log_type": "AS",
                "entities": ["process", "cmdline", "user"],
                "timestamp_col": "TimeGenerated",
            },
            "zeek_conn": {
                "filename": "zeek_conn.csv",
                "has_ioc": True,
                "log_type": "ZC",
                "entities": ["src_ip", "dst_ip", "src_port", "dst_port", "bytes_sent", "bytes_recv", "duration"],
                "timestamp_col": "ts",
            },
            "zeek_dns": {
                "filename": "zeek_dns.csv",
                "has_ioc": True,
                "log_type": "ZD",
                "entities": ["src_ip", "query_domain", "resolved_ip", "query_type"],
                "timestamp_col": "ts",
            },
            "zeek_http": {
                "filename": "zeek_http.csv",
                "has_ioc": True,
                "log_type": "ZH",
                "entities": ["src_ip", "dst_ip", "method", "uri", "status_code", "resp_bytes"],
                "timestamp_col": "ts",
            },
        },
    },

    # SC4: NPM two-stage token handoff (audit-ejs / audit-vue) — Linux
    # Ground truth: eve.json (203), azure_syslog (26), zeek_conn (6), zeek_dns (4), zeek_ssl (3)
    # Total IOCs: 242
    "sc4": {
        "root": "data/SynthChain/sc4",
        "logs": {
            "eve": {
                "filename": "eve.json",
                "has_ioc": True,
                "log_type": "EVE",
                "entities": ["src_ip", "dst_ip", "src_port", "dst_port", "alert_signature"],
                "timestamp_col": "timestamp",
            },
            "azure_syslog": {
                "filename": "azure_syslog.csv",
                "has_ioc": True,
                "log_type": "AS",
                "entities": ["process", "cmdline", "user"],
                "timestamp_col": "TimeGenerated",
            },
            "zeek_conn": {
                "filename": "zeek_conn.csv",
                "has_ioc": True,
                "log_type": "ZC",
                "entities": ["src_ip", "dst_ip", "src_port", "dst_port", "bytes_sent", "bytes_recv"],
                "timestamp_col": "ts",
            },
            "zeek_dns": {
                "filename": "zeek_dns.csv",
                "has_ioc": True,
                "log_type": "ZD",
                "entities": ["src_ip", "query_domain", "resolved_ip"],
                "timestamp_col": "ts",
            },
            "zeek_ssl": {
                "filename": "zeek_ssl.csv",
                "has_ioc": True,
                "log_type": "ZS",
                "entities": ["src_ip", "dst_ip", "server_name", "cipher", "version"],
                "timestamp_col": "ts",
            },
        },
    },

    # SC5: Trojanized installer 3CX-style (X_TRADER / ampglobalusa5setup) — Windows
    # Ground truth: all IOCs in azure_events
    # Total IOCs: 503 (21 download + 117 install_chain_1st + 402 install_chain_2nd + 326 c:\tt\ + 71 IMDS)
    "sc5": {
        "root": "data/SynthChain/sc5",
        "logs": {
            "azure_events": {
                "filename": "azure_events.csv",
                "has_ioc": True,
                "log_type": "AE",
                "entities": ["src_ip", "dst_ip", "process", "parent_process", "file_path", "cmdline", "hash_md5", "hash_sha256"],
                "timestamp_col": "TimeGenerated",
            },
        },
    },

    # SC6: Cloud-based supply chain exploitation — Windows/WinLinux hybrid
    # Ground truth: all IOCs in azure_events
    # Total IOCs: 392
    "sc6": {
        "root": "data/SynthChain/sc6",
        "logs": {
            "azure_events": {
                "filename": "victim/azure_events.csv",
                "has_ioc": True,
                "log_type": "AE",
                "entities": ["src_ip", "dst_ip", "process", "parent_process", "file_path", "cmdline", "hash_md5", "hash_sha256"],
                "timestamp_col": "TimeGenerated",
            },
        },
    },

    # SC7: Trojaned neural network model (Docker) — Linux
    # Ground truth: eve.json (201), azure_syslog (4), zeek_conn (155), zeek_dns (4), zeek_http (77), zeek_files (77)
    # Total IOCs: 510
    "sc7": {
        "root": "data/SynthChain/sc7",
        "logs": {
            "eve": {
                "filename": "eve.json",
                "has_ioc": True,
                "log_type": "EVE",
                "entities": ["src_ip", "dst_ip", "src_port", "dst_port", "alert_signature"],
                "timestamp_col": "timestamp",
            },
            "zeek_conn": {
                "filename": "zeek_conn.csv",
                "has_ioc": True,
                "log_type": "ZC",
                "entities": ["src_ip", "dst_ip", "src_port", "dst_port", "bytes_sent", "bytes_recv"],
                "timestamp_col": "ts",
            },
            "zeek_http": {
                "filename": "zeek_http.csv",
                "has_ioc": True,
                "log_type": "ZH",
                "entities": ["src_ip", "dst_ip", "method", "uri", "status_code", "resp_bytes"],
                "timestamp_col": "ts",
            },
            "zeek_files": {
                "filename": "zeek_files.csv",
                "has_ioc": True,
                "log_type": "ZF",
                "entities": ["src_ip", "dst_ip", "filename", "mime_type", "total_bytes"],
                "timestamp_col": "ts",
            },
        },
    },
}