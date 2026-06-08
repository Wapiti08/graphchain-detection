#!/usr/bin/env python3
"""Count stage-eligible weak-rule labels per scenario (rule / rule_high)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gchain.eval.attack_reconstruct import ioc_type_to_stage_idx, load_ioc_type_to_stage
from gchain.train.modeling import _stage_eligible_counts_per_scenario
from gchain.train.streams import load_stream_from_tgn_pt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=str, default="sc1,sc2,sc3,sc4,sc5,sc6,sc7")
    p.add_argument("--graphs-dir", type=str, default="artifacts/graphs")
    args = p.parse_args()

    graphs_dir = (_REPO / args.graphs_dir).resolve()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    for sc in scenarios:
        tgn = graphs_dir / f"synthchain_{sc}.tgn.pt"
        if not tgn.is_file():
            print(f"{sc}: missing {tgn}")
            continue
        st = load_stream_from_tgn_pt(tgn)
        streams = {sc: st}
        for mode in ("rule_high", "rule"):
            counts = _stage_eligible_counts_per_scenario(
                streams,
                stage_supervision=mode,
                repo_root=_REPO,
                ioc_type_to_stage_idx=ioc_type_to_stage_idx,
                load_stage_map=load_ioc_type_to_stage,
            )
            print(f"{sc}\t{mode}\tstage_eligible={counts.get(sc, 0)}")


if __name__ == "__main__":
    main()
