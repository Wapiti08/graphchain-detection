from __future__ import annotations

import re
from typing import Dict, List


# --- URL / domain ---
URL_RE = re.compile(r"\bhttps?://[^\s\"'<>]+", re.IGNORECASE)
DOMAIN_RE = re.compile(
    r"\b(?:(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)\.)+(?:[a-z]{2,63})\b",
    re.IGNORECASE,
)

# --- hashes / keys (best-effort) ---
MD5_RE = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)
SHA1_RE = re.compile(r"\b[a-f0-9]{40}\b", re.IGNORECASE)
SHA256_RE = re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE)

# Azure-style secret-ish (very rough; keep as boolean/count only)
AZURE_CONNSTR_RE = re.compile(r"\b(DefaultEndpointsProtocol|AccountKey|SharedAccessKey)\b", re.IGNORECASE)
JWT_RE = re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b")
BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{80,}={0,2}\b")


SUSPICIOUS_CMD_PATTERNS: Dict[str, re.Pattern] = {
    "has_powershell": re.compile(r"\bpowershell(\.exe)?\b", re.IGNORECASE),
    "has_bitsadmin": re.compile(r"\bbitsadmin(\.exe)?\b", re.IGNORECASE),
    "has_certutil": re.compile(r"\bcertutil(\.exe)?\b", re.IGNORECASE),
    "has_curl": re.compile(r"\bcurl\b", re.IGNORECASE),
    "has_wget": re.compile(r"\bwget\b", re.IGNORECASE),
    "has_base64_flag": re.compile(r"(?:-enc|-encodedcommand)\b", re.IGNORECASE),
    "has_invoke_webrequest": re.compile(r"\binvoke-webrequest\b", re.IGNORECASE),
    "has_invoke_expression": re.compile(r"\binvoke-expression\b|\biex\b", re.IGNORECASE),
}


KNOWN_REGISTRY_DOMAINS = {
    # PyPI
    "pypi.org",
    "files.pythonhosted.org",
    # npm
    "npmjs.org",
    "registry.npmjs.org",
    # GitHub (common in supply-chain)
    "github.com",
    "raw.githubusercontent.com",
}


def _unique_preserve_order(xs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _domain_to_tld(d: str) -> str:
    parts = d.lower().strip(".").split(".")
    if len(parts) < 2:
        return ""
    return parts[-1]


def extract_text_signals(text: str) -> Dict[str, object]:
    """
    Extract lightweight, low-noise signals from unstructured text.

    Returns only low-cardinality features (counts/buckets/booleans) to avoid
    attribute explosion from raw URLs/domains/hashes.
    """
    t = text or ""

    urls = _unique_preserve_order(URL_RE.findall(t))
    domains = _unique_preserve_order(DOMAIN_RE.findall(t))
    md5s = _unique_preserve_order(MD5_RE.findall(t))
    sha1s = _unique_preserve_order(SHA1_RE.findall(t))
    sha256s = _unique_preserve_order(SHA256_RE.findall(t))

    has_jwt = bool(JWT_RE.search(t))
    has_base64_blob = bool(BASE64_RE.search(t))
    has_azure_secret_hint = bool(AZURE_CONNSTR_RE.search(t))

    # Domain buckets (keep tiny, stable set)
    tld_counts: Dict[str, int] = {}
    has_known_registry = False
    for d in domains:
        if d.lower() in KNOWN_REGISTRY_DOMAINS:
            has_known_registry = True
        tld = _domain_to_tld(d)
        if tld:
            tld_counts[tld] = tld_counts.get(tld, 0) + 1

    # keep only top-3 tlds by count to stay low-cardinality
    top_tlds = sorted(tld_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    top_tld_1 = top_tlds[0][0] if len(top_tlds) > 0 else ""
    top_tld_2 = top_tlds[1][0] if len(top_tlds) > 1 else ""
    top_tld_3 = top_tlds[2][0] if len(top_tlds) > 2 else ""

    return {
        "n_urls": len(urls),
        "n_domains": len(domains),
        "n_md5": len(md5s),
        "n_sha1": len(sha1s),
        "n_sha256": len(sha256s),
        "has_known_registry_domain": has_known_registry,
        "top_tld_1": top_tld_1,
        "top_tld_2": top_tld_2,
        "top_tld_3": top_tld_3,
        "has_jwt": has_jwt,
        "has_base64_blob": has_base64_blob,
        "has_azure_secret_hint": has_azure_secret_hint,
    }


def suspicious_cmd_flags(text: str) -> Dict[str, bool]:
    t = text or ""
    return {k: bool(rx.search(t)) for k, rx in SUSPICIOUS_CMD_PATTERNS.items()}

