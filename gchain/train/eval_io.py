from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

ScoresSplit = Literal["tail", "all"]

EVAL_TAIL_CSV_FIELDS = [
    "scenario",
    "t",
    "etype",
    "src",
    "dst",
    "score",
    "is_ioc",
    "source_file",
    "row_idx",
    "ioc_type",
    "pred_stage",
    "pred_stage_prob",
]


def resolve_scores_csv(run_dir: Path, *, split: ScoresSplit = "tail") -> Optional[Path]:
    """Pick TGN score CSV; use ``tail`` for detection benchmarks (held-out 30% only)."""
    if split == "tail":
        names = (
            "best_eval_tail_scores.csv",
            "eval_tail_scores.csv",
            "best_eval_all_scores.csv",
            "eval_all_scores.csv",
        )
    else:
        names = (
            "best_eval_all_scores.csv",
            "eval_all_scores.csv",
            "best_eval_tail_scores.csv",
            "eval_tail_scores.csv",
        )
    for name in names:
        p = run_dir / name
        if p.is_file():
            return p
    return None


def write_eval_rows_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = DictWriter(f, fieldnames=EVAL_TAIL_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

