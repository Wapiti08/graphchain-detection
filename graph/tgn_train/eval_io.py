from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from typing import Dict, Iterable, List

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


def write_eval_rows_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = DictWriter(f, fieldnames=EVAL_TAIL_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

