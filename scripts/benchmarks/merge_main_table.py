#!/usr/bin/env python3
"""Merge per-method benchmark JSON into main_table.csv (+ optional LaTeX)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]

DEFAULT_SCENARIOS = ("sc1", "sc2", "sc3", "sc4", "sc5", "sc6", "sc7")

# Rows for the paper main table (method label, method key, telemetry).
MAIN_TABLE_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("Audit-only", "freq_rarity", "audit"),
    ("Network-sensor-only", "freq_rarity", "zeek"),
    ("Alert-only (IDS)", "freq_rarity", "eve"),
    ("Freq-rarity", "freq_rarity", "full"),
    ("Path-LOF", "path_lof", "full"),
    ("GraphSAGE (static)", "graphsage", "full"),
    ("RGCN (static)", "rgcn", "full"),
    ("FuseChain", "fusechain", "full"),
)

COLUMNS = (
    "label",
    "method",
    "telemetry",
    "auroc",
    "auprc",
    "p_at_500",
    "stage_recall_at_500",
    "lcs_at_500",
    "latency_ms_per_1k",
    "n_scenarios",
    "n_na",
)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(vals: Sequence[float]) -> float:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _macro_aggregate(
    per_scenario: Mapping[str, Mapping[str, Any]],
    scenarios: Sequence[str],
) -> Dict[str, Any]:
    det_keys = ("auroc", "auprc", "p_at_500")
    recon_keys = ("stage_recall", "ordered_stage_recall_lcs")
    lat_vals: List[float] = []
    det_acc: Dict[str, List[float]] = {k: [] for k in det_keys}
    recon_acc: Dict[str, List[float]] = {k: [] for k in recon_keys}
    n_na = 0
    for sc in scenarios:
        row = per_scenario.get(sc)
        if row is None or row.get("status") != "ok":
            n_na += 1
            continue
        det = row.get("detection") or {}
        for k in det_keys:
            v = det.get(k if k != "p_at_500" else "p_at_500")
            if v is not None and not math.isnan(float(v)):
                det_acc[k].append(float(v))
        recon = (row.get("reconstruction") or {}).get("at_500") or {}
        for k in recon_keys:
            v = recon.get(k)
            if v is not None and not math.isnan(float(v)):
                recon_acc[k].append(float(v))
        lat = row.get("latency_ms_per_1k")
        if lat is not None and not math.isnan(float(lat)):
            lat_vals.append(float(lat))

    return {
        "auroc": _mean(det_acc["auroc"]),
        "auprc": _mean(det_acc["auprc"]),
        "p_at_500": _mean(det_acc["p_at_500"]),
        "stage_recall_at_500": _mean(recon_acc["stage_recall"]),
        "lcs_at_500": _mean(recon_acc["ordered_stage_recall_lcs"]),
        "latency_ms_per_1k": _mean(lat_vals),
        "n_scenarios": len(scenarios),
        "n_na": n_na,
    }


def collect_results(
    bench_root: Path,
    scenarios: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label, method, telemetry in MAIN_TABLE_ROWS:
        per_sc: Dict[str, Mapping[str, Any]] = {}
        for sc in scenarios:
            path = bench_root / method / telemetry / f"{sc}.json"
            payload = _load_json(path)
            if payload is not None:
                per_sc[sc] = payload
        agg = _macro_aggregate(per_sc, scenarios)
        out.append(
            {
                "label": label,
                "method": method,
                "telemetry": telemetry,
                **agg,
            }
        )
    return out


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "---"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x):
        return "---"
    return f"{x:.{nd}f}"


def write_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"Method & AUROC & AUPRC & P@500 & StageRec@500 & LCS@500 & Lat (ms/1k) \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            " & ".join(
                [
                    str(r.get("label", "")),
                    _fmt(r.get("auroc")),
                    _fmt(r.get("auprc")),
                    _fmt(r.get("p_at_500")),
                    _fmt(r.get("stage_recall_at_500")),
                    _fmt(r.get("lcs_at_500")),
                    _fmt(r.get("latency_ms_per_1k"), nd=1),
                ]
            )
            + r" \\"
        )
    lines.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bench-dir",
        type=str,
        default="artifacts/benchmarks/per_method",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/benchmarks/main_table.csv",
    )
    p.add_argument(
        "--out-latex",
        type=str,
        default="",
        help="Optional path for LaTeX tabular fragment.",
    )
    p.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    args = p.parse_args()

    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    bench_root = (_REPO / args.bench_dir).resolve()
    rows = collect_results(bench_root, scenarios)

    out_csv = (_REPO / args.out_csv).resolve()
    write_csv(out_csv, rows)
    print(f"wrote {out_csv}")

    if args.out_latex:
        out_tex = (_REPO / args.out_latex).resolve()
        write_latex(out_tex, rows)
        print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
