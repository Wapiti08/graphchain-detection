#!/usr/bin/env python3
"""Merge per-run run_summary.json from TGN experiments (LOSO, per-scenario, etc.) into one CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _flatten(prefix: str, d: object) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        out[f"{prefix}{k}"] = v
    return out


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
        help="Glob under runs-root for run directories.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/tgn_runs/loso_summary.csv",
        help="Output CSV path (relative to repo root).",
    )
    p.add_argument(
        "--require-protocol",
        type=str,
        default="",
        help="If set, only include runs whose eval_protocol matches (e.g. per_scenario, loso_holdout).",
    )
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = (repo / args.runs_root).resolve()
    out_csv = (repo / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    folds = sorted(root.glob(args.pattern))
    rows: List[Dict[str, object]] = []
    extra_keys: set[str] = set()

    for d in folds:
        if not d.is_dir():
            continue
        js = d / "run_summary.json"
        if not js.is_file():
            continue
        data = json.loads(js.read_text(encoding="utf-8"))
        if args.require_protocol:
            if str(data.get("eval_protocol") or "") != str(args.require_protocol):
                continue
        scenario = str(data.get("scenario") or data.get("holdout") or "").strip()
        if not scenario:
            ts = data.get("test_scenarios") or []
            if isinstance(ts, list) and len(ts) == 1:
                scenario = str(ts[0])
        if not scenario:
            continue

        base: Dict[str, object] = {
            "scenario": scenario,
            "eval_protocol": data.get("eval_protocol"),
            "holdout": data.get("holdout") or "",
            "best_epoch": data.get("best_epoch"),
            "best_metric": data.get("best_metric"),
            "select_metric": data.get("select_metric"),
            "best_auroc": data.get("best_auroc"),
            "best_auprc": data.get("best_auprc"),
            "last_auroc": data.get("last_auroc"),
            "last_auprc": data.get("last_auprc"),
            "last_val_loss": data.get("last_val_loss"),
            "last_val_acc": data.get("last_val_acc"),
            "epochs": data.get("epochs"),
            "epochs_completed": data.get("epochs_completed"),
            "early_stopped": data.get("early_stopped"),
            "early_stop_patience": data.get("early_stop_patience"),
            "early_stop_min_delta": data.get("early_stop_min_delta"),
            "seed": data.get("seed"),
            "run_dir": str(d.relative_to(repo)),
        }
        bt = _flatten("best_tail_", data.get("best_tail_eval"))
        lt = _flatten("last_tail_", data.get("last_tail_eval"))
        extra_keys |= set(bt.keys())
        extra_keys |= set(lt.keys())
        base.update(bt)
        base.update(lt)
        rows.append(base)

    core = [
        "scenario",
        "eval_protocol",
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
        "epochs",
        "epochs_completed",
        "early_stopped",
        "early_stop_patience",
        "early_stop_min_delta",
        "seed",
        "run_dir",
    ]
    fieldnames = core + sorted(extra_keys)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Wrote {len(rows)} folds -> {out_csv}")


if __name__ == "__main__":
    main()
