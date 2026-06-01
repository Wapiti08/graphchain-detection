from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gchain.eval.alert_eval import (
    ScoredEvent,
    cluster_connected,
    count_alerts,
    dedupe_events,
    select_high_score,
    time_windows,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate high-score TGN events into subgraph alerts.")
    p.add_argument(
        "--scores-csv",
        type=str,
        default="artifacts/tgn_runs/synthchain_multi/best_eval_tail_scores.csv",
        help="Input per-event scores CSV (from train_tgn_synthchain.py).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/alerts",
        help="Output directory for alerts.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=3600,
        help="Time window size (same unit as t in scores CSV).",
    )
    p.add_argument(
        "--topk-events",
        type=int,
        default=0,
        help="If >0, keep only top-K events by score (per scenario).",
    )
    p.add_argument(
        "--score-quantile",
        type=float,
        default=0.99,
        help="Keep events with score >= quantile (per scenario). Ignored if --topk-events > 0.",
    )
    p.add_argument(
        "--min-events",
        type=int,
        default=3,
        help="Drop alerts with fewer than this many events.",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=200,
        help="Cap stored evidence events per alert.",
    )
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate (scenario,t,etype,src,dst) rows; default is merge and keep max score.",
    )
    return p.parse_args()


def _read_scores(path: Path) -> List[ScoredEvent]:
    out: List[ScoredEvent] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                ScoredEvent(
                    scenario=str(row["scenario"]),
                    t=int(float(row["t"])),
                    etype=int(row["etype"]),
                    src=int(row["src"]),
                    dst=int(row["dst"]),
                    score=float(row["score"]),
                    is_ioc=int(row.get("is_ioc", "0") or 0),
                )
            )
    return out


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    in_path = (repo_root / args.scores_csv).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _read_scores(in_path)
    by_scenario: Dict[str, List[ScoredEvent]] = {}
    for e in events:
        by_scenario.setdefault(e.scenario, []).append(e)

    alerts_path = out_dir / "alerts.jsonl"
    summary_path = out_dir / "alerts.csv"

    alert_rows: List[Dict[str, object]] = []
    alert_id = 0
    with alerts_path.open("w") as jf:
        for sc, evs in sorted(by_scenario.items()):
            if not evs:
                continue

            if not bool(args.no_dedupe):
                evs = dedupe_events(evs)

            kept = select_high_score(evs, topk_events=int(args.topk_events), score_quantile=float(args.score_quantile))

            for win in time_windows(kept, int(args.window)):
                clusters = cluster_connected(win)
                for cl in clusters:
                    cl = sorted(cl, key=lambda e: (e.t, -e.score))
                    if len(cl) < int(args.min_events):
                        continue
                    nodes = sorted({e.src for e in cl} | {e.dst for e in cl})
                    ioc_hits = int(sum(e.is_ioc for e in cl))
                    max_score = float(max(e.score for e in cl))
                    t0, t1 = int(cl[0].t), int(cl[-1].t)
                    evidence = [
                        {
                            "t": int(e.t),
                            "etype": int(e.etype),
                            "src": int(e.src),
                            "dst": int(e.dst),
                            "score": float(e.score),
                            "is_ioc": int(e.is_ioc),
                        }
                        for e in cl[: int(args.max_events)]
                    ]

                    rec = {
                        "alert_id": alert_id,
                        "scenario": sc,
                        "t_start": t0,
                        "t_end": t1,
                        "num_events": int(len(cl)),
                        "num_nodes": int(len(nodes)),
                        "max_score": max_score,
                        "ioc_hits": ioc_hits,
                        "nodes": nodes,
                        "evidence": evidence,
                    }
                    jf.write(json.dumps(rec) + "\n")
                    alert_rows.append(
                        {
                            "alert_id": alert_id,
                            "scenario": sc,
                            "t_start": t0,
                            "t_end": t1,
                            "num_events": int(len(cl)),
                            "num_nodes": int(len(nodes)),
                            "max_score": max_score,
                            "ioc_hits": ioc_hits,
                        }
                    )
                    alert_id += 1

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["alert_id", "scenario", "t_start", "t_end", "num_events", "num_nodes", "max_score", "ioc_hits"],
        )
        w.writeheader()
        for r in alert_rows:
            w.writerow(r)

    n_alerts, n_tail, n_flagged, _ = count_alerts(
        {k: list(v) for k, v in by_scenario.items()},
        window=int(args.window),
        score_quantile=float(args.score_quantile),
        min_events=int(args.min_events),
        topk_events=int(args.topk_events),
        dedupe=not bool(args.no_dedupe),
    )
    print(f"Saved: {alerts_path}")
    print(f"Saved: {summary_path}")
    print(f"Alerts: {len(alert_rows)} (sanity count_alerts={n_alerts}, tail_deduped={n_tail}, flagged={n_flagged})")


if __name__ == "__main__":
    main()
