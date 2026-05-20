#!/usr/bin/env python3
"""Merge per-fold reconstruction_metrics.json from LOSO runs into one CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=str, default="artifacts/tgn_runs")
    p.add_argument("--pattern", type=str, default="loso_holdout_*")
    p.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/tgn_runs/loso_reconstruction_summary.csv",
    )
    p.add_argument("--topks", type=str, default="10,50,100,500")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = (repo / args.runs_root).resolve()
    out_csv = (repo / args.out_csv).resolve()
    topks = [str(int(x.strip())) for x in args.topks.split(",") if x.strip()]

    rows_out: List[Dict[str, object]] = []
    for d in sorted(root.glob(args.pattern)):
        if not d.is_dir():
            continue
        js = d / "reconstruction_metrics.json"
        if not js.is_file():
            continue
        data = json.loads(js.read_text(encoding="utf-8"))
        sc = str(data.get("scenario") or "")
        row: Dict[str, object] = {
            "fold": d.name,
            "scenario": sc,
            "n_score_rows": data.get("n_score_rows"),
            "n_observable_stages": len(data.get("observable_stages") or []),
            "n_semantic_stages": len(data.get("semantic_stages") or []),
            "n_unobserved_semantic": len(data.get("unobserved_semantic") or []),
        }
        by_k = data.get("by_k") or {}
        for k in topks:
            bk = by_k.get(k) or {}
            row[f"stage_recall@{k}"] = bk.get("stage_recall")
            row[f"stage_precision@{k}"] = bk.get("stage_precision")
            row[f"ordered_stage_recall_lcs@{k}"] = bk.get("ordered_stage_recall_lcs")
        rows_out.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        print(f"No reconstruction_metrics.json under {root}/{args.pattern}")
        return

    fieldnames: List[str] = list(rows_out[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {out_csv} ({len(rows_out)} folds)")


if __name__ == "__main__":
    main()
