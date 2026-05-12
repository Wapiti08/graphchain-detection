#!/usr/bin/env python3
"""Merge per-fold run_summary.json from LOSO runs into one CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-root",
        type=str,
        default="artifacts/tgn_runs",
        help="Directory containing loso_holdout_sc*/ folders.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="loso_holdout_*",
        help="Glob under runs-root for fold directories.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/tgn_runs/loso_summary.csv",
        help="Output CSV path (relative to repo root).",
    )
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = (repo / args.runs_root).resolve()
    out_csv = (repo / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    folds = sorted(root.glob(args.pattern))
    rows: list[dict[str, object]] = []
    for d in folds:
        if not d.is_dir():
            continue
        js = d / "run_summary.json"
        if not js.is_file():
            continue
        data = json.loads(js.read_text(encoding="utf-8"))
        holdout = str(data.get("holdout") or "")
        if not holdout:
            # skip non-holdout dirs that matched glob
            continue
        rows.append(
            {
                "holdout": holdout,
                "best_epoch": data.get("best_epoch"),
                "best_metric": data.get("best_metric"),
                "select_metric": data.get("select_metric"),
                "best_auroc": data.get("best_auroc"),
                "best_auprc": data.get("best_auprc"),
                "last_auroc": data.get("last_auroc"),
                "last_auprc": data.get("last_auprc"),
                "last_val_loss": data.get("last_val_loss"),
                "last_val_acc": data.get("last_val_acc"),
                "run_dir": str(d.relative_to(repo)),
            }
        )

    fieldnames = [
        "holdout",
        "best_epoch",
        "best_metric",
        "select_metric",
        "best_auroc",
        "best_auprc",
        "last_auroc",
        "last_auprc",
        "last_val_loss",
        "last_val_acc",
        "run_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} folds -> {out_csv}")


if __name__ == "__main__":
    main()
