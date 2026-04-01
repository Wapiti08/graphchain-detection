from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from typing import TYPE_CHECKING

from config.ontology import EdgeType, NodeType, edge_triplet
from parsers.events import Event

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import HeteroData


@dataclass
class BuildStats:
    num_events: int
    num_nodes_by_type: Dict[NodeType, int]
    num_edges_by_type: Dict[EdgeType, int]


def _event_time(ev: Event) -> float:
    # Prefer real timestamp, fall back to pseudo-time.
    if ev.ts is not None:
        return float(ev.ts)
    return float(ev.order)


def build_hetero_graph(
    events: List[Event],
    *,
    include_raw_attrs: bool = True,
) -> Tuple["HeteroData", BuildStats]:
    """
    Build a PyG HeteroData graph from canonical Events.

    What this builder guarantees:
    - Deduplicates nodes per NodeType using EntityRef.key
    - Creates per-edge-type edge_index tensors
    - Adds edge_time (float tensor) and edge_order (long tensor) per edge type

    Note:
    - Feature encoding is intentionally deferred. We attach raw attribute dicts
      (edge_attrs/src_attrs/dst_attrs) as python lists if `include_raw_attrs=True`.
      You can later project/encode them into tensors using `models/encoders.py`.
    """
    try:
        import torch  # type: ignore
        from torch_geometric.data import HeteroData  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Graph construction requires `torch` and `torch-geometric`.\n"
            "Install deps (example): `pip install -r requirements.txt`.\n"
            "Note: torch-geometric may need platform-specific wheels."
        ) from e

    data = HeteroData()

    # --- node id maps ---
    node_id: Dict[NodeType, Dict[str, int]] = {nt: {} for nt in NodeType}
    node_keys: Dict[NodeType, List[str]] = {nt: [] for nt in NodeType}

    def get_node_index(nt: NodeType, key: str) -> int:
        m = node_id[nt]
        if key in m:
            return m[key]
        idx = len(node_keys[nt])
        m[key] = idx
        node_keys[nt].append(key)
        return idx

    # --- edge accumulation ---
    edge_src: Dict[EdgeType, List[int]] = {et: [] for et in EdgeType}
    edge_dst: Dict[EdgeType, List[int]] = {et: [] for et in EdgeType}
    edge_time: Dict[EdgeType, List[float]] = {et: [] for et in EdgeType}
    edge_order: Dict[EdgeType, List[int]] = {et: [] for et in EdgeType}

    edge_attrs_raw: Dict[EdgeType, List[Dict[str, Any]]] = {et: [] for et in EdgeType}
    src_attrs_raw: Dict[NodeType, List[Dict[str, Any]]] = {nt: [] for nt in NodeType}
    dst_attrs_raw: Dict[NodeType, List[Dict[str, Any]]] = {nt: [] for nt in NodeType}

    # Track latest attrs for nodes (best-effort merge).
    node_attrs_latest: Dict[NodeType, Dict[int, Dict[str, Any]]] = {
        nt: {} for nt in NodeType
    }

    for i, ev in enumerate(events):
        et = ev.edge_type
        s_nt, _, d_nt = edge_triplet(et)

        # Trust ontology triplet; still ensure node types match.
        # If parsers ever emit mismatched types, we adapt to actual types.
        s_type = ev.src.type if isinstance(ev.src.type, NodeType) else s_nt
        d_type = ev.dst.type if isinstance(ev.dst.type, NodeType) else d_nt

        s_idx = get_node_index(s_type, ev.src.key)
        d_idx = get_node_index(d_type, ev.dst.key)

        edge_src[et].append(s_idx)
        edge_dst[et].append(d_idx)
        edge_time[et].append(_event_time(ev))
        edge_order[et].append(int(ev.order))

        if include_raw_attrs:
            edge_attrs_raw[et].append(dict(ev.edge_attrs))

        if ev.src_attrs:
            node_attrs_latest[s_type][s_idx] = {
                **node_attrs_latest[s_type].get(s_idx, {}),
                **dict(ev.src_attrs),
            }
        if ev.dst_attrs:
            node_attrs_latest[d_type][d_idx] = {
                **node_attrs_latest[d_type].get(d_idx, {}),
                **dict(ev.dst_attrs),
            }

    # --- materialize nodes ---
    for nt in NodeType:
        keys = node_keys[nt]
        if not keys:
            continue

        data[nt.value].num_nodes = len(keys)
        data[nt.value].node_key = keys  # stable identifier per node

        if include_raw_attrs:
            # Fill with {} for nodes without attrs.
            raw_list: List[Dict[str, Any]] = []
            for idx in range(len(keys)):
                raw_list.append(node_attrs_latest[nt].get(idx, {}))
            data[nt.value].node_attrs_raw = raw_list

    # --- materialize edges ---
    num_edges_by_type: Dict[EdgeType, int] = {}
    for et in EdgeType:
        s_nt, _, d_nt = edge_triplet(et)
        srcs = edge_src[et]
        dsts = edge_dst[et]
        if not srcs:
            continue

        # Use the actual ontology-defined triplet for edge store name.
        store = data[s_nt.value, et.value, d_nt.value]

        store.edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        store.edge_time = torch.tensor(edge_time[et], dtype=torch.float)
        store.edge_order = torch.tensor(edge_order[et], dtype=torch.long)

        if include_raw_attrs:
            store.edge_attrs_raw = edge_attrs_raw[et]

        num_edges_by_type[et] = len(srcs)

    stats = BuildStats(
        num_events=len(events),
        num_nodes_by_type={nt: len(node_keys[nt]) for nt in NodeType},
        num_edges_by_type=num_edges_by_type,
    )
    return data, stats

