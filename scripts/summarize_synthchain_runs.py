#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _extract_recon_mode(
    recon: Dict[str, Any],
    *,
    mode: str,
    topks: List[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    block = recon.get(mode)
    if not isinstance(block, dict):
        for k in topks:
            out[f"{mode}.recall@{k}"] = ""
        return out

    if block.get("available") is False:
        for k in topks:
            out[f"{mode}.recall@{k}"] = ""
        out[f"{mode}.available"] = False
        return out

    out[f"{mode}.available"] = True
    for k in topks:
        kk = str(k)
        v = ""
        if isinstance(block.get(kk), dict):
            v = block[kk].get("stage_recall", "")
        out[f"{mode}.recall@{k}"] = v
    return out


def _scenario_id_from_dirname(name: str) -> str:
    # expected: per_scenario_sc3_all_scores
    toks = name.split("_")
    for t in toks:
        if t.startswith("sc") and t[2:].isdigit():
            return t
    return ""


def summarize_one(run_dir: Path, *, topks: List[int]) -> Dict[str, Any]:
    sc = _scenario_id_from_dirname(run_dir.name)
    row: Dict[str, Any] = {"scenario": sc or run_dir.name, "run_dir": str(run_dir)}

    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.is_file():
        rs = _load_json(run_summary_path)
        row["best_epoch"] = rs.get("best_epoch", "")
        row["best_auroc"] = rs.get("best_auroc", "")
        row["best_auprc"] = rs.get("best_auprc", "")
        for k in [10, 50, 100, 500]:
            row[f"best.p@{k}"] = _safe_get(rs, ["best_tail_eval", f"p_at_{k}"], "")
    else:
        row["best_epoch"] = ""
        row["best_auroc"] = ""
        row["best_auprc"] = ""
        for k in [10, 50, 100, 500]:
            row[f"best.p@{k}"] = ""

    recon_path = run_dir / "reconstruction_metrics.json"
    if recon_path.is_file():
        recon = _load_json(recon_path)
        row["recon.n_score_rows"] = recon.get("n_score_rows", "")
        obs = recon.get("observable_stages")
        row["recon.observable_stage_count"] = len(obs) if isinstance(obs, list) else ""
        row["recon.unobserved_semantic_count"] = (
            len(recon.get("unobserved_semantic") or []) if isinstance(recon.get("unobserved_semantic"), list) else ""
        )

        row.update(_extract_recon_mode(recon, mode="by_k", topks=topks))
        row.update(_extract_recon_mode(recon, mode="by_k_pair_dedupe", topks=topks))
        row.update(_extract_recon_mode(recon, mode="by_k_source_quota", topks=topks))
        row.update(_extract_recon_mode(recon, mode="by_k_group_cap", topks=topks))
        row.update(_extract_recon_mode(recon, mode="by_k_group_cap_adaptive", topks=topks))
        row.update(_extract_recon_mode(recon, mode="by_k_pred_stage", topks=topks))
        row.update(_extract_recon_mode(recon, mode="by_alert_rule", topks=topks))
        row.update(_extract_recon_mode(recon, mode="by_alert_pred_stage", topks=topks))
    else:
        row["recon.n_score_rows"] = ""
        row["recon.observable_stage_count"] = ""
        row["recon.unobserved_semantic_count"] = ""
        for mode in [
            "by_k",
            "by_k_pair_dedupe",
            "by_k_source_quota",
            "by_k_group_cap",
            "by_k_group_cap_adaptive",
            "by_k_pred_stage",
            "by_alert_rule",
            "by_alert_pred_stage",
        ]:
            for k in topks:
                row[f"{mode}.recall@{k}"] = ""
            row[f"{mode}.available"] = ""

    return row


def _parse_topks(s: str) -> List[int]:
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize per-scenario SynthChain runs (run_summary + reconstruction).")
    ap.add_argument(
        "--runs-dir",
        type=str,
        default="artifacts/tgn_runs",
        help="Directory containing per_scenario_sc*_*/ folders.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="per_scenario_sc*_all_scores",
        help="Glob pattern under --runs-dir to locate runs.",
    )
    ap.add_argument(
        "--topks",
        type=str,
        default="10,50,100,500",
        help="Comma-separated K values for reconstruction stage recall extraction.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path (defaults to stdout).",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    runs_dir = (repo / args.runs_dir).resolve()
    topks = _parse_topks(str(args.topks))

    run_dirs = sorted([p for p in runs_dir.glob(str(args.pattern)) if p.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        raise SystemExit(f"No runs found under {runs_dir} matching {args.pattern!r}")

    rows = [summarize_one(p, topks=topks) for p in run_dirs]
    # stable column order
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    out_path = Path(args.out).resolve() if str(args.out).strip() else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f = out_path.open("w", newline="", encoding="utf-8")
        close = True
    else:
        import sys

        f = sys.stdout
        close = False

    try:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    finally:
        if close:
            f.close()


if __name__ == "__main__":
    main()

