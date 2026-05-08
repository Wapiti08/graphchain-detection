"""
Unified entity ontology for GraphChain-Detection.

This file is intentionally *config-only*:
- defines canonical node/edge types
- defines desired attributes (name + dtype + default)

Parsers should map raw logs into this ontology.
Graph builders should consume events expressed in this ontology.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple


AttrDType = Literal["text", "cat", "num", "bool"]


@dataclass(frozen=True)
class AttrSpec:
    name: str
    dtype: AttrDType
    default: Any = None
    description: str = ""


class NodeType(str, Enum):
    PROC = "PROC"
    FILE = "FILE"
    NET = "NET"
    SYSCALL = "SYSCALL"
    PKG = "PKG"
    CRED = "CRED"


class EdgeType(str, Enum):
    EXEC = "EXEC"  # PROC -> PROC
    READ = "READ"  # PROC -> FILE
    WRITE = "WRITE"  # PROC -> FILE
    CONNECT = "CONNECT"  # PROC -> NET
    INVOKE = "INVOKE"  # PROC -> SYSCALL
    DEPEND = "DEPEND"  # PKG -> PKG
    LOAD = "LOAD"  # PKG -> PROC
    CAUSE = "CAUSE"  # PROC -> PROC (causal/temporal dependency)
    REDIRECT = "REDIRECT"  # NET -> NET
    RESOLVE = "RESOLVE"  # NET -> NET
    RELAY = "RELAY"  # NET -> NET
    INJECT = "INJECT"  # PROC -> PROC
    DNS_QUERY = "DNS_QUERY"  # PROC -> NET


EdgeTriplet = Tuple[NodeType, EdgeType, NodeType]


# ---- Node attribute specs (from README.md) ----
NODE_ATTRS: Mapping[NodeType, Tuple[AttrSpec, ...]] = {
    NodeType.PROC: (
        AttrSpec(
            name="is_lolbin",
            dtype="bool",
            default=False,
            description="Living Off the Land binary (e.g., bitsadmin, powershell).",
        ),
        AttrSpec(
            name="parent_depth",
            dtype="num",
            default=0,
            description="Execution chain depth (e.g., Docker → Python → payload).",
        ),
    ),
    NodeType.FILE: (
        AttrSpec(
            name="path_sensitivity",
            dtype="num",
            default=0,
            description="Higher for sensitive paths (e.g., Startup folder persistence).",
        ),
        AttrSpec(
            name="file_type",
            dtype="cat",
            default="",
            description="File extension/type (e.g., .png steganography).",
        ),
    ),
    NodeType.NET: (
        AttrSpec(
            name="port",
            dtype="num",
            default=0,
            description="Network port (e.g., 2121/50000 FTP, 8081 exfil).",
        ),
        AttrSpec(
            name="is_known_registry",
            dtype="bool",
            default=False,
            description="Known package registry vs unknown endpoint.",
        ),
        AttrSpec(
            name="tls_valid",
            dtype="bool",
            default=True,
            description="TLS validity (self-signed/no SNI often suspicious).",
        ),
    ),
    NodeType.SYSCALL: tuple(),
    NodeType.PKG: tuple(),
    NodeType.CRED: (
        AttrSpec(
            name="cred_type",
            dtype="cat",
            default="",
            description="Credential type (token/Azure key/password/etc.).",
        ),
    ),
}


# ---- Edge attribute specs (from README.md) ----
EDGE_ATTRS: Mapping[EdgeType, Tuple[AttrSpec, ...]] = {
    EdgeType.EXEC: (
        AttrSpec(name="cmdline", dtype="text", default="", description="Command line."),
    ),
    EdgeType.READ: (
        AttrSpec(name="bytes", dtype="num", default=0, description="Bytes read."),
    ),
    EdgeType.WRITE: (
        AttrSpec(name="bytes", dtype="num", default=0, description="Bytes written."),
    ),
    EdgeType.CONNECT: (
        AttrSpec(name="bytes_sent", dtype="num", default=0, description="Bytes sent."),
        AttrSpec(name="bytes_recv", dtype="num", default=0, description="Bytes received."),
        AttrSpec(
            name="direction",
            dtype="cat",
            default="",
            description="Connection direction (in/out/unknown).",
        ),
    ),
    EdgeType.INVOKE: (
        AttrSpec(name="args", dtype="text", default="", description="Syscall args."),
        AttrSpec(
            name="return_val",
            dtype="num",
            default=0,
            description="Syscall return value.",
        ),
    ),
    EdgeType.DEPEND: (
        AttrSpec(
            name="version_constraint",
            dtype="text",
            default="",
            description="Dependency constraint (e.g., >=1.2,<2.0).",
        ),
    ),
    EdgeType.LOAD: (
        AttrSpec(
            name="entry_point",
            dtype="text",
            default="",
            description="Package entry point or loaded module.",
        ),
    ),
    EdgeType.CAUSE: (
        AttrSpec(
            name="delta_t",
            dtype="num",
            default=0,
            description="Time/step difference between cause and effect.",
        ),
        AttrSpec(
            name="cause_rule",
            dtype="cat",
            default="",
            description="Deterministic rule id that created this causal edge.",
        ),
        AttrSpec(
            name="confidence",
            dtype="num",
            default=1.0,
            description="Rule confidence (1.0 for hard rules).",
        ),
    ),
    EdgeType.REDIRECT: (
        AttrSpec(
            name="http_status",
            dtype="num",
            default=0,
            description="HTTP status code for redirect.",
        ),
    ),
    EdgeType.RESOLVE: (
        AttrSpec(
            name="resolved_ip",
            dtype="text",
            default="",
            description="Resolved IP for a DNS lookup.",
        ),
    ),
    EdgeType.RELAY: (
        AttrSpec(
            name="delta_t",
            dtype="num",
            default=0,
            description="Time delta between hops.",
        ),
    ),
    EdgeType.INJECT: (
        AttrSpec(
            name="injection_type",
            dtype="cat",
            default="",
            description="Injection type (e.g., DLL, shellcode).",
        ),
    ),
    EdgeType.DNS_QUERY: (
        AttrSpec(
            name="query_domain",
            dtype="text",
            default="",
            description="Domain being queried.",
        ),
    ),
}


# ---- Edge schema (source/rel/target) ----
EDGE_SCHEMA: Mapping[EdgeType, EdgeTriplet] = {
    EdgeType.EXEC: (NodeType.PROC, EdgeType.EXEC, NodeType.PROC),
    EdgeType.READ: (NodeType.PROC, EdgeType.READ, NodeType.FILE),
    EdgeType.WRITE: (NodeType.PROC, EdgeType.WRITE, NodeType.FILE),
    EdgeType.CONNECT: (NodeType.PROC, EdgeType.CONNECT, NodeType.NET),
    EdgeType.INVOKE: (NodeType.PROC, EdgeType.INVOKE, NodeType.SYSCALL),
    EdgeType.DEPEND: (NodeType.PKG, EdgeType.DEPEND, NodeType.PKG),
    EdgeType.LOAD: (NodeType.PKG, EdgeType.LOAD, NodeType.PROC),
    EdgeType.CAUSE: (NodeType.PROC, EdgeType.CAUSE, NodeType.PROC),
    EdgeType.REDIRECT: (NodeType.NET, EdgeType.REDIRECT, NodeType.NET),
    EdgeType.RESOLVE: (NodeType.NET, EdgeType.RESOLVE, NodeType.NET),
    EdgeType.RELAY: (NodeType.NET, EdgeType.RELAY, NodeType.NET),
    EdgeType.INJECT: (NodeType.PROC, EdgeType.INJECT, NodeType.PROC),
    EdgeType.DNS_QUERY: (NodeType.PROC, EdgeType.DNS_QUERY, NodeType.NET),
}


def canonical_node_attrs(node_type: NodeType) -> Dict[str, AttrSpec]:
    return {a.name: a for a in NODE_ATTRS.get(node_type, tuple())}


def canonical_edge_attrs(edge_type: EdgeType) -> Dict[str, AttrSpec]:
    return {a.name: a for a in EDGE_ATTRS.get(edge_type, tuple())}


def fill_defaults(
    specs: Mapping[str, AttrSpec],
    values: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Fill missing canonical attributes using defaults.
    Extra keys in `values` are preserved (useful during "superset" extraction).
    """
    out: Dict[str, Any] = dict(values or {})
    for k, spec in specs.items():
        if k not in out or out[k] is None:
            out[k] = spec.default
    return out


def edge_triplet(edge_type: EdgeType) -> EdgeTriplet:
    return EDGE_SCHEMA[edge_type]

