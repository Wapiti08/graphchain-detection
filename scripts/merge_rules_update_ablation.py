#!/usr/bin/env python3
"""Merge baseline vs updated-rules training runs into a regression comparison CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    js = run_dir / "run_summary.json"
    if not js.is_file():
        return None
    return json.loads(js.read_text(encoding="utf-8"))


def _metric(data: Dict[str, Any], key: str) -> Optional[float]:
    v = data.get(key)
    if v is None:
        tail = data.get("best_tail_eval") or {}
        if isinstance(tail, dict):
            v = tail.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", type=str, required=True)
    p.add_argument("--updated-root", type=str, required=True)
    p.add_argument(
        "--scenarios",
        type=str,
        default="sc1,sc3,sc5",
        help="Comma-separated scenarios; run dirs are {root}/per_scenario_{sc}_*",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/rules_update_ablation/train_regression_compare.csv",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    base_root = (repo / args.baseline_root).resolve()
    upd_root = (repo / args.updated_root).resolve()

    def find_run(root: Path, sc: str) -> Optional[Path]:
        matches = sorted(root.glob(f"per_scenario_{sc}_*"))
        if not matches:
            matches = sorted(root.glob(f"*{sc}*"))
        return matches[-1] if matches else None

    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        bdir = find_run(base_root, sc)
        udir = find_run(upd_root, sc)
        bdata = _load_summary(bdir) if bdir else None
        udata = _load_summary(udir) if udir else None
        row: Dict[str, Any] = {
            "scenario": sc,
            "baseline_run": str(bdir.relative_to(repo)) if bdir else "",
            "updated_run": str(udir.relative_to(repo)) if udir else "",
        }
        tail_keys = ("auroc", "auprc", "p_at_100", "pf")
        for key in ("best_auprc", "best_auroc"):
            bv = _metric(bdata, key) if bdata else None
            uv = _metric(udata, key) if udata else None
            row[f"baseline_{key}"] = bv
            row[f"updated_{key}"] = uv
            row[f"delta_{key}"] = (uv - bv) if bv is not None and uv is not None else ""
        for tk in tail_keys:
            key = f"best_tail_{tk}"
            bv = None
            uv = None
            if bdata:
                tail = bdata.get("best_tail_eval") or {}
                if isinstance(tail, dict):
                    bv = tail.get(tk)
            if udata:
                tail = udata.get("best_tail_eval") or {}
                if isinstance(tail, dict):
                    uv = tail.get(tk)
            try:
                bv_f = float(bv) if bv is not None else None
                uv_f = float(uv) if uv is not None else None
            except (TypeError, ValueError):
                bv_f, uv_f = None, None
            row[f"baseline_{key}"] = bv_f
            row[f"updated_{key}"] = uv_f
            row[f"delta_{key}"] = (uv_f - bv_f) if bv_f is not None and uv_f is not None else ""
        rows.append(row)

    out_csv = (repo / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")
    for r in rows:
        da = r.get("delta_best_auprc", "")
        print(f"  {r['scenario']}: delta_best_auprc={da}")


if __name__ == "__main__":
    main()
