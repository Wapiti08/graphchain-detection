#!/usr/bin/env python3
"""Export telemetry-filtered .tgn.pt views for single-source baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gchain.baselines.telemetry import filter_stream_indices, subset_stream
from gchain.train.streams import load_stream_from_tgn_pt

DEFAULT_SCENARIOS = ("sc1", "sc2", "sc3", "sc4", "sc5", "sc6", "sc7")


def _export_blob(st, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "src": st.src,
        "dst": st.dst,
        "t": st.t,
        "msg": st.msg,
        "etype": st.etype,
    }
    if st.y_ioc is not None:
        blob["y_ioc"] = st.y_ioc
    if st.y_ioc_line is not None:
        blob["y_ioc_line"] = st.y_ioc_line
    if st.y_rule is not None:
        blob["y_rule"] = st.y_rule
    if st.y_rule_high is not None:
        blob["y_rule_high"] = st.y_rule_high
    if st.row_idx is not None:
        blob["row_idx"] = st.row_idx
    if st.source_file is not None:
        blob["source_file"] = list(st.source_file)
    if st.ioc_type is not None:
        blob["ioc_type"] = list(st.ioc_type)
    if st.rule_ioc_type is not None:
        blob["rule_ioc_type"] = list(st.rule_ioc_type)
    torch.save(blob, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", type=str, required=True, choices=("audit", "zeek", "eve"))
    p.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    p.add_argument("--graphs-dir", type=str, default="artifacts/graphs")
    p.add_argument("--out-dir", type=str, default="artifacts/graphs/benchmark")
    args = p.parse_args()

    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    graphs_dir = (_REPO / args.graphs_dir).resolve()
    out_dir = (_REPO / args.out_dir / args.telemetry).resolve()

    for sc in scenarios:
        src = graphs_dir / f"synthchain_{sc}.tgn.pt"
        if not src.is_file():
            print(f"skip {sc}: missing {src}", file=sys.stderr)
            continue
        st = load_stream_from_tgn_pt(src)
        keep = filter_stream_indices(st, args.telemetry)  # type: ignore[arg-type]
        sub = subset_stream(st, keep)
        dst = out_dir / f"synthchain_{sc}.tgn.pt"
        _export_blob(sub, dst)
        print(f"wrote {dst} ({sub.src.numel()} edges)")


if __name__ == "__main__":
    main()
