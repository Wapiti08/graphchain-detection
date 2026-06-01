from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from graphcore.edge_meta import primary_ioc_type_from_attrs

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class TGNEventStream:
    src: "torch.Tensor"   # [E] int64
    dst: "torch.Tensor"   # [E] int64
    t: "torch.Tensor"     # [E] float32
    msg: "torch.Tensor"   # [E, D] float32
    etype: "torch.Tensor" # [E] int64
    y_ioc: Optional["torch.Tensor"] = None  # [E] int64 (0/1)
    y_ioc_line: Optional["torch.Tensor"] = None  # [E] int64 (0/1)
    row_idx: Optional["torch.Tensor"] = None  # [E] int64 (-1 if unknown)
    source_file: Optional[Tuple[str, ...]] = None  # len E
    ioc_type: Optional[Tuple[str, ...]] = None  # len E (primary IOC type, "" if none)
    meta: Optional[Dict[str, Any]] = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _hash_bucket(s: Any, num_buckets: int) -> int:
    ''' process multiple relation values to fixed-length vector

    args:
        num_buckets: define the length of vector
    '''
    if num_buckets <= 1:
        return 0
    if s is None:
        return 0
    return (hash(str(s)) % num_buckets + num_buckets) % num_buckets


def _vectorize_edge_attrs(
    attrs: Dict[str, Any],
    *,
    cat_hash_buckets: int = 8,
) -> List[float]:
    ''' process diverse value (numeric and categorical) into fixed-length vectors

    '''
    # numeric
    bytes_sent = _safe_float(attrs.get("bytes_sent"))
    bytes_recv = _safe_float(attrs.get("bytes_recv"))
    bytes_rw = _safe_float(attrs.get("bytes"))
    delta_t = _safe_float(attrs.get("delta_t"))

    # direction (CONNECT)
    dir_raw = str(attrs.get("direction") or "").lower().strip()
    dir_map = {"in": 0, "inbound": 0, "out": 1, "outbound": 1}
    dir_id = dir_map.get(dir_raw, 2)
    dir_oh = [0.0, 0.0, 0.0]
    dir_oh[dir_id] = 1.0

    # booleans
    bool_keys = [
        "has_powershell",
        "has_bitsadmin",
        "has_certutil",
        "has_curl",
        "has_wget",
        "has_base64_flag",
        "has_invoke_webrequest",
        "has_invoke_expression",
        "has_jwt",
        "has_base64_blob",
        "has_azure_secret_hint",
        "has_known_registry_domain",
    ]
    bool_feats = [1.0 if bool(attrs.get(k)) else 0.0 for k in bool_keys]

    # hashed categorical sketch
    cat_keys = ["method", "status_code", "proto", "service", "tls_version", "tls_cipher", "sni", "cause_rule"]
    cat_vec = [0.0] * cat_hash_buckets
    for k in cat_keys:
        v = attrs.get(k)
        if v is None or str(v).strip() == "":
            continue
        cat_vec[_hash_bucket(f"{k}={v}", cat_hash_buckets)] += 1.0

    return [bytes_sent, bytes_recv, bytes_rw, delta_t, *dir_oh, *bool_feats, *cat_vec]


def hetero_to_tgn_event_stream(
    data: "HeteroData",
    *,
    cat_hash_buckets: int = 8,
    include_meta: bool = False,
) -> TGNEventStream:
    """
    Flatten HeteroData into a single TGN-style interaction stream.
    """
    import torch

    # global id offsets
    # Deterministic global-id layout: sort node types to keep offsets stable
    # across runs and across different graph construction paths.
    node_offsets: Dict[str, int] = {}
    total = 0
    for nt in sorted(data.node_types):
        node_offsets[nt] = total
        total += int(data[nt].num_nodes)
    num_nodes_by_type: Dict[str, int] = {nt: int(data[nt].num_nodes) for nt in sorted(data.node_types)}

    # Deterministic edge-type ids: sort edge types to keep etype stable across scenarios.
    edge_types_sorted = sorted(data.edge_types)
    etype_to_id: Dict[Tuple[str, str, str], int] = {et: i for i, et in enumerate(edge_types_sorted)}

    src_all: List[int] = []
    dst_all: List[int] = []
    t_all: List[float] = []
    etype_all: List[int] = []
    msg_all: List[List[float]] = []
    y_ioc_all: List[int] = []
    y_ioc_line_all: List[int] = []
    row_idx_all: List[int] = []
    source_file_all: List[str] = []
    ioc_type_all: List[str] = []
    meta: Dict[str, Any] = {"edge_type": []} if include_meta else {}
    if include_meta:
        # With `node_offsets` + `num_nodes_by_type`, you can deterministically decode:
        # global_id -> (node_type, local_id).
        meta["node_offsets"] = dict(node_offsets)
        meta["num_nodes_by_type"] = dict(num_nodes_by_type)
        meta["edge_type_to_id"] = {str(k): int(v) for k, v in etype_to_id.items()}

    for et in edge_types_sorted:
        store = data[et]
        edge_index = store.edge_index
        edge_time = store.edge_time
        raw_list = getattr(store, "edge_attrs_raw", None)
        if raw_list is None:
            raw_list = [{} for _ in range(edge_index.size(1))]

        s_nt, rel, d_nt = et
        s_off = node_offsets[s_nt]
        d_off = node_offsets[d_nt]
        eid = etype_to_id[et]

        for j in range(edge_index.size(1)):
            raw_attrs = raw_list[j] or {}
            src_all.append(int(edge_index[0, j]) + s_off)
            dst_all.append(int(edge_index[1, j]) + d_off)
            t_all.append(float(edge_time[j]))
            etype_all.append(eid)
            msg_all.append(_vectorize_edge_attrs(raw_attrs, cat_hash_buckets=cat_hash_buckets))
            y_ioc_all.append(1 if bool(raw_attrs.get("is_ioc")) else 0)
            y_ioc_line_all.append(1 if bool(raw_attrs.get("is_ioc_line")) else 0)
            ri = raw_attrs.get("_row_idx")
            row_idx_all.append(int(ri) if isinstance(ri, int) else -1)
            source_file_all.append(str(raw_attrs.get("_source_file") or ""))
            ioc_type_all.append(primary_ioc_type_from_attrs(raw_attrs))
            if include_meta:
                meta["edge_type"].append(et)

    order = sorted(range(len(t_all)), key=lambda i: (t_all[i], etype_all[i], src_all[i], dst_all[i]))
    src = torch.tensor([src_all[i] for i in order], dtype=torch.long)
    dst = torch.tensor([dst_all[i] for i in order], dtype=torch.long)
    t = torch.tensor([t_all[i] for i in order], dtype=torch.float)
    etype = torch.tensor([etype_all[i] for i in order], dtype=torch.long)
    msg = torch.tensor([msg_all[i] for i in order], dtype=torch.float)
    y_ioc = torch.tensor([y_ioc_all[i] for i in order], dtype=torch.long)
    y_ioc_line = torch.tensor([y_ioc_line_all[i] for i in order], dtype=torch.long)
    row_idx = torch.tensor([row_idx_all[i] for i in order], dtype=torch.long)

    return TGNEventStream(
        src=src,
        dst=dst,
        t=t,
        msg=msg,
        etype=etype,
        y_ioc=y_ioc,
        y_ioc_line=y_ioc_line,
        row_idx=row_idx,
        source_file=tuple(source_file_all[i] for i in order),
        ioc_type=tuple(ioc_type_all[i] for i in order),
        meta=(meta if include_meta else None),
    )

