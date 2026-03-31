# GraphChain-Detection
detection of ongoing supply chain vulnerabilities with temporal graph neural networks

## Data Processing (Feature Extraction)

- Unified Entity Ontology:

    - Unified Node Types (6) with attributes:

        - PROC --- process: install script, subprocess, python interpreter
            - attr1: is_lolbin (Living Off the Land Binaries) - bitsadmin, powershell

            - attr2: parent_depth - Docker → Python → payload chain depth (benign is within 2-3, deeper means more suspicious)

        - FILE --- file system entity: file(path) be read/created/wrote/delete
            - attr1: path_sensitivity - Writing to the Startup folder = Persistence

            - attr2: file_type - .png hidden payload (steganography)

        - NET --- network endpoint: IP:port or domain names
            - attr1: port - Sc4: 2121/50000 = FTP; Sc6: 8081 = exfil

            - attr2: is_known_registry - Sc4/5: npm registry vs unknown IP

            - attr3: tls_valid - Sc5: self-signed cert, no SNI

        - SYSCALL --- behavior type node

        - PKG --- package: current packages with their dependencies

        - CRED --- token/secret/crediential
            - attr1: cred_type - Sc5: token; Sc6: Azure key; Sc2: password


    - Unified Edge Types with attributes (12):

        EXEC (cmdline):     PROC -> PROC :   cover initial execution, subprocess creation

        READ (bytes):     PROC -> FILE:    information collection

        WRITE (bytes):    PROC -> FILE:    persistence

        CONNECT (bytes_sent, bytes_recv, direction):  PROC -> NET:     network connection

        INVOKE (args, return_val):   PROC -> SYSCALL: system call

        DEPEND (version_constraint):   PKG -> PKG:      dependency

        LOAD (entry_point):     PKG -> PROC:     package invoke execution

        REDIRECT (http_status): NET -> NET:      http redirected

        RESOLVE (resolved_ip):  NET -> NET:      DNS resolve

        RELAY (\delta t):    NET -> NET:      Proxy/multi-hop transfer

        INJECT (injection_type):   PROC -> PROC:    inject code in process

        DNS_QUERY (query_domain): PROC -> NET:     who raises dns query

    - 