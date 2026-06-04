from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(frozen=True)
class Stream:
    src: "torch.Tensor"  # [E] int64
    dst: "torch.Tensor"  # [E] int64
    t: "torch.Tensor"  # [E] int64 (for TGNMemory last_update)
    msg: "torch.Tensor"  # [E, D] float32
    etype: "torch.Tensor"  # [E] int64
    y_ioc: Optional["torch.Tensor"] = None  # [E] int64 (0/1)
    y_ioc_line: Optional["torch.Tensor"] = None  # [E] int64 (0/1)
    y_rule: Optional["torch.Tensor"] = None  # [E] int64 weak-rule hit
    y_rule_high: Optional["torch.Tensor"] = None  # [E] int64 high-confidence rule
    row_idx: Optional["torch.Tensor"] = None  # [E] int64 (-1 if unknown)
    source_file: Optional[Tuple[str, ...]] = None
    ioc_type: Optional[Tuple[str, ...]] = None
    rule_ioc_type: Optional[Tuple[str, ...]] = None


def load_stream_from_tgn_pt(path: Path) -> Stream:
    import torch

    blob = torch.load(path, weights_only=True)
    sf = blob.get("source_file")
    it = blob.get("ioc_type")
    rit = blob.get("rule_ioc_type")
    return Stream(
        src=blob["src"].long(),
        dst=blob["dst"].long(),
        t=blob["t"].long(),
        msg=blob["msg"].float(),
        etype=blob["etype"].long(),
        y_ioc=(blob.get("y_ioc").long() if blob.get("y_ioc") is not None else None),
        y_ioc_line=(blob.get("y_ioc_line").long() if blob.get("y_ioc_line") is not None else None),
        y_rule=(blob.get("y_rule").long() if blob.get("y_rule") is not None else None),
        y_rule_high=(blob.get("y_rule_high").long() if blob.get("y_rule_high") is not None else None),
        row_idx=(blob.get("row_idx").long() if blob.get("row_idx") is not None else None),
        source_file=(tuple(str(x) for x in sf) if sf is not None else None),
        ioc_type=(tuple(str(x) for x in it) if it is not None else None),
        rule_ioc_type=(tuple(str(x) for x in rit) if rit is not None else None),
    )


def num_nodes_in_stream(st: Stream) -> int:
    import torch

    if st.src.numel() == 0:
        return 0
    return int(torch.max(torch.stack([st.src.max(), st.dst.max()])).item()) + 1


def offset_stream_nodes(st: Stream, base: int) -> Stream:
    if base == 0:
        return st
    return Stream(
        src=st.src + int(base),
        dst=st.dst + int(base),
        t=st.t,
        msg=st.msg,
        etype=st.etype,
        y_ioc=st.y_ioc,
        y_ioc_line=st.y_ioc_line,
        y_rule=st.y_rule,
        y_rule_high=st.y_rule_high,
        row_idx=st.row_idx,
        source_file=st.source_file,
        ioc_type=st.ioc_type,
        rule_ioc_type=st.rule_ioc_type,
    )


def ensure_scenario_stream(
    *,
    repo_root: Path,
    graphs_dir: Path,
    scenario: str,
    auto_generate: bool,
) -> Tuple[Path, Stream]:
    tgn_path = graphs_dir / f"synthchain_{scenario}.tgn.pt"
    if tgn_path.exists():
        return tgn_path, load_stream_from_tgn_pt(tgn_path)

    if not auto_generate:
        raise SystemExit(
            f"Missing `{tgn_path}`. Generate it via:\n"
            f"  python -m gchain.pipeline --dataset synthchain --scenario {scenario} --export-tgn\n"
        )

    from gchain.pipeline import generate_graph

    result = generate_graph(
        repo_root=repo_root,
        scenario=scenario,
        out=str(graphs_dir.relative_to(repo_root)),
        export_tgn=True,
        verbose=False,
    )
    if result.tgn_pt is None or not result.tgn_pt.exists():
        raise SystemExit(f"Auto-generation finished but `{tgn_path}` was not created.")
    return tgn_path, load_stream_from_tgn_pt(tgn_path)


def regenerate_scenario_stream(*, repo_root: Path, graphs_dir: Path, scenario: str) -> None:
    from gchain.pipeline import generate_graph

    generate_graph(
        repo_root=repo_root,
        scenario=scenario,
        out=str(graphs_dir.relative_to(repo_root)),
        export_tgn=True,
        verbose=False,
    )

