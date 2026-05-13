#!/usr/bin/env python3
"""Report y_ioc / y_ioc_line balance in SynthChain *.tgn.pt streams (full vs time-split prefix/tail).

Uses the same time index split as `scripts/train_tgn_synthchain.py` (--train-frac).
Does not require training; only loads `artifacts/graphs/synthchain_<sc>.tgn.pt` (or --graphs-dir).

Example (repo root):

  python3 scripts/report_tgn_ioc_balance.py
  python3 scripts/report_tgn_ioc_balance.py --scenarios sc5,sc6 --train-frac 0.7 --json
  python3 scripts/report_tgn_ioc_balance.py --check-log-files   # list configured logs present/missing (raw data)
  python3 scripts/report_tgn_ioc_balance.py --verify-pt         # cross-check .tgn.pt length vs synthchain_*.pt stats
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _time_split_idx(num_events: int, train_frac: float) -> int:
    if num_events <= 1:
        return 0
    k = int(math.floor(float(train_frac) * float(num_events)))
    return max(1, min(num_events - 1, k))


def _summarize(y: Any, lo: int, hi: int) -> Dict[str, float]:
    import torch

    if y is None or hi <= lo:
        return {"n": 0.0, "ioc": 0.0, "rate": float("nan")}
    sl = y[lo:hi]
    n = int(sl.numel())
    ioc = int(sl.sum().item())
    return {"n": float(n), "ioc": float(ioc), "rate": float(ioc) / float(max(1, n))}


def _tail_unique_ioc_stats(
    *,
    src: Any,
    dst: Any,
    t: Any,
    etype: Any,
    y: Any,
    split_idx: int,
    hi: int,
) -> Dict[str, float]:
    """Unique (t, etype, src, dst) in [split_idx, hi); how many keys have any IOC-labeled row."""
    import torch

    if y is None or hi <= split_idx:
        return {"unique_edges": 0.0, "unique_with_any_ioc": 0.0, "rate_keys_any_ioc": float("nan")}

    keys: set[Tuple[int, int, int, int]] = set()
    keys_ioc: set[Tuple[int, int, int, int]] = set()
    for i in range(split_idx, hi):
        tv = t[i].item()
        ti = int(tv) if isinstance(tv, (int, float)) else int(tv)
        k2 = (ti, int(etype[i].item()), int(src[i].item()), int(dst[i].item()))
        keys.add(k2)
        if int(y[i].item()) == 1:
            keys_ioc.add(k2)
    u = len(keys)
    ui = len(keys_ioc)
    return {
        "unique_edges": float(u),
        "unique_with_any_ioc": float(ui),
        "rate_keys_any_ioc": float(ui) / float(max(1, u)),
    }


def _verify_pt_vs_tgn(graphs_dir: Path, scenario: str, e_tgn: int, torch: Any) -> Dict[str, Any]:
    """Load synthchain_<sc>.pt from generate_graph; compare stats to TGN stream length."""
    pt_path = graphs_dir / f"synthchain_{scenario}.pt"
    out: Dict[str, Any] = {"pt_path": str(pt_path.name), "pt_exists": pt_path.is_file()}
    if not pt_path.is_file():
        out["match"] = None
        out["note"] = "no .pt (run generate_graph.py --dataset synthchain --scenario ... --export-tgn)"
        return out

    blob = torch.load(pt_path, weights_only=True)
    stats = blob.get("stats")
    if not isinstance(stats, dict):
        out["match"] = None
        out["note"] = "stats missing in .pt"
        return out

    ne = stats.get("num_events")
    ne_i = int(ne) if ne is not None else -1
    ebt = stats.get("num_edges_by_type") or {}
    edge_sum = 0
    if isinstance(ebt, dict):
        for v in ebt.values():
            try:
                edge_sum += int(v)
            except (TypeError, ValueError):
                pass

    out["pt_num_events"] = ne_i
    out["pt_edge_sum"] = int(edge_sum)
    out["match_events"] = ne_i == int(e_tgn)
    out["match_edge_sum"] = edge_sum == int(e_tgn) if edge_sum > 0 else None
    out["match"] = bool(out["match_events"])
    if ne_i >= 0 and edge_sum > 0 and edge_sum != ne_i:
        out["pt_internal_warn"] = f"num_events={ne_i} != sum(edges)={edge_sum}"
    return out


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit("Need torch: pip install torch") from e

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graphs-dir",
        type=str,
        default="artifacts/graphs",
        help="Directory containing synthchain_<scenario>.tgn.pt",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="sc1,sc2,sc3,sc4,sc5,sc6,sc7",
        help="Comma-separated scenario ids.",
    )
    p.add_argument("--train-frac", type=float, default=0.7, help="Same as train_tgn_synthchain.py.")
    p.add_argument("--json", action="store_true", help="Print one JSON object instead of a table.")
    p.add_argument(
        "--check-log-files",
        action="store_true",
        help="For each scenario, print SynthChain config log paths (exist / missing). "
        "Missing files are skipped silently by generate_graph / load_synthchain_events.",
    )
    p.add_argument(
        "--only-ioc-logs",
        action="store_true",
        help="When combined with --check-log-files, mirror generate_graph --only-ioc-logs (default off).",
    )
    p.add_argument(
        "--verify-pt",
        action="store_true",
        help="Cross-check each synthchain_<sc>.tgn.pt length E against synthchain_<sc>.pt stats.num_events.",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    graphs_dir = (repo / args.graphs_dir).resolve()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    if args.check_log_files:
        from config.synthchain_sources import SYNTHCHAIN_IOC_CONFIG

        only_ioc = bool(args.only_ioc_logs)
        for sc in scenarios:
            if sc not in SYNTHCHAIN_IOC_CONFIG:
                print(f"{sc}: unknown scenario", file=sys.stderr)
                continue
            cfg = SYNTHCHAIN_IOC_CONFIG[sc]
            base = repo / cfg["root"]
            print(f"=== {sc} logs under {cfg['root']} (only_ioc_logs={only_ioc})")
            for log_name, spec in cfg["logs"].items():
                if only_ioc and not spec.get("has_ioc", False):
                    continue
                rel = spec["filename"]
                p = base / rel
                st = "ok" if p.is_file() else "MISSING"
                print(f"  [{st}] {log_name}: {p.relative_to(repo)}")
            print()

    rows_out: List[Dict[str, Any]] = []

    for sc in scenarios:
        path = graphs_dir / f"synthchain_{sc}.tgn.pt"
        if not path.is_file():
            row = {"scenario": sc, "path": str(path.relative_to(repo)), "error": "missing_file"}
            rows_out.append(row)
            continue

        blob = torch.load(path, weights_only=True)
        src = blob["src"]
        dst = blob["dst"]
        t = blob["t"]
        etype = blob["etype"]
        y_ioc = blob.get("y_ioc")
        y_line = blob.get("y_ioc_line")
        n = int(src.numel())
        split = _time_split_idx(n, float(args.train_frac))

        full_i = _summarize(y_ioc, 0, n) if y_ioc is not None else None
        pre_i = _summarize(y_ioc, 0, split) if y_ioc is not None else None
        tail_i = _summarize(y_ioc, split, n) if y_ioc is not None else None
        full_l = _summarize(y_line, 0, n) if y_line is not None else None
        pre_l = _summarize(y_line, 0, split) if y_line is not None else None
        tail_l = _summarize(y_line, split, n) if y_line is not None else None

        uniq = None
        if y_ioc is not None:
            uniq = _tail_unique_ioc_stats(
                src=src, dst=dst, t=t, etype=etype, y=y_ioc, split_idx=split, hi=n
            )

        row: Dict[str, Any] = {
            "scenario": sc,
            "path": str(path.relative_to(repo)),
            "events": n,
            "split_idx": split,
            "tail_rows": int(n - split),
            "train_frac": float(args.train_frac),
        }
        if full_i:
            row["y_ioc_all_n"] = int(full_i["n"])
            row["y_ioc_all_pos"] = int(full_i["ioc"])
            row["y_ioc_all_rate"] = full_i["rate"]
        if pre_i:
            row["y_ioc_prefix_n"] = int(pre_i["n"])
            row["y_ioc_prefix_pos"] = int(pre_i["ioc"])
            row["y_ioc_prefix_rate"] = pre_i["rate"]
        if tail_i:
            row["y_ioc_tail_n"] = int(tail_i["n"])
            row["y_ioc_tail_pos"] = int(tail_i["ioc"])
            row["y_ioc_tail_rate"] = tail_i["rate"]
        if full_l:
            row["y_ioc_line_all_rate"] = full_l["rate"]
        if pre_l:
            row["y_ioc_line_prefix_rate"] = pre_l["rate"]
        if tail_l:
            row["y_ioc_line_tail_rate"] = tail_l["rate"]
        if uniq:
            row["tail_unique_edges"] = int(uniq["unique_edges"])
            row["tail_unique_edges_with_ioc"] = int(uniq["unique_with_any_ioc"])
            row["tail_unique_frac_ioc"] = uniq["rate_keys_any_ioc"]

        if args.verify_pt:
            row["pt_verify"] = _verify_pt_vs_tgn(graphs_dir, sc, n, torch)

        rows_out.append(row)

    if args.json:
        print(json.dumps(rows_out, indent=2))
        return

    # Human-readable aligned table
    hdr = (
        f"{'sc':<5} {'E':>6} {'split':>6} {'Tn':>5} | "
        f"{'ioc%all':>8} {'ioc%pre':>8} {'ioc%tail':>8} | "
        f"{'u_tail':>6} {'u_ioc%':>7} | "
        f"{'line%all':>8} {'line%tail':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in rows_out:
        if row.get("error"):
            print(f"{row['scenario']:<5}  {row.get('error','')} {row.get('path','')}")
            continue
        iall = row.get("y_ioc_all_rate", float("nan"))
        ipre = row.get("y_ioc_prefix_rate", float("nan"))
        itail = row.get("y_ioc_tail_rate", float("nan"))
        ut = row.get("tail_unique_edges", 0)
        uf = row.get("tail_unique_frac_ioc", float("nan"))
        tn = row.get("tail_rows", 0)
        la = row.get("y_ioc_line_all_rate")
        lt = row.get("y_ioc_line_tail_rate")
        line_all = "  n/a  " if la is None else f"{100.0 * float(la):>7.2f}%"
        line_tail = "  n/a  " if lt is None else f"{100.0 * float(lt):>7.2f}%"
        print(
            f"{row['scenario']:<5} {row['events']:>6} {row['split_idx']:>6} {tn:>5} | "
            f"{100.0 * iall:>7.2f}% {100.0 * ipre:>7.2f}% {100.0 * itail:>7.2f}% | "
            f"{ut:>6} {100.0 * uf:>6.2f}% | "
            f"{line_all} {line_tail}"
        )
    print()
    print("E = edges in the flattened TGN stream (.tgn.pt), not raw CSV line count.")
    print("Tn = tail row count (E - split). u_tail = unique (int(t),etype,src,dst) in tail; many rows can share one key.")
    print("ioc% = y_ioc (line OR value token hit). line% = y_ioc_line (ground-truth line hit only; stricter).")
    print("If ioc%≈100% but line% is low, value-level IOC matching is over-firing on sc5/sc6-like scenarios.")

    if args.verify_pt:
        print()
        print("=== verify synthchain_<sc>.pt (generate_graph) vs .tgn.pt stream length")
        vh = f"{'sc':<5} {'E_tgn':>7} {'pt_evt':>7} {'Σedge':>7} {'OK':>4}"
        print(vh)
        print("-" * len(vh))
        for row in rows_out:
            if row.get("error"):
                print(f"{row['scenario']:<5}  (no .tgn.pt)")
                continue
            pv = row.get("pt_verify") or {}
            e_t = int(row["events"])
            if not pv.get("pt_exists"):
                print(f"{row['scenario']:<5} {e_t:>7}     —       —    —")
                if pv.get("note"):
                    print(f"      {pv['note']}")
                continue
            pne = int(pv.get("pt_num_events", -1))
            esu = int(pv.get("pt_edge_sum", 0))
            ok = pv.get("match")
            ok_s = "OK" if ok else ("BAD" if ok is False else "?")
            print(f"{row['scenario']:<5} {e_t:>7} {pne:>7} {esu:>7} {ok_s:>4}")
            if pv.get("pt_internal_warn"):
                print(f"      !! {pv['pt_internal_warn']}")
            if pv.get("note") and pv.get("match") is None and pv.get("pt_exists"):
                print(f"      !! {pv['note']}")
        print("pt_evt = stats.num_events in synthchain_<sc>.pt ; Σedge = sum(num_edges_by_type).")


if __name__ == "__main__":
    main()
