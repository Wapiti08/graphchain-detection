from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def roc_auc(y_true: List[int], y_score: List[float]) -> float:
    # Returns AUROC in [0,1]. If undefined (only one class), returns NaN.
    if not y_true:
        return float("nan")
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(len(y_true)), key=lambda i: y_score[i], reverse=True)
    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    auc = 0.0
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return float(auc)


def pr_auc(y_true: List[int], y_score: List[float]) -> float:
    # Returns area under precision-recall curve. If undefined, returns NaN.
    if not y_true:
        return float("nan")
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return float("nan")

    order = sorted(range(len(y_true)), key=lambda i: y_score[i], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(1, (tp + fp))
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def parse_topk(s: str) -> List[int]:
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
        except Exception:
            continue
        if k > 0:
            out.append(k)
    return sorted(set(out))


def resolve_p_at_k(select_metric: str, select_p_at_k: int) -> Optional[int]:
    """Return K when selection uses tail p@K; None for auprc/auroc."""
    sm = str(select_metric).strip().lower()
    if sm == "p_at":
        return max(1, int(select_p_at_k))
    if sm.startswith("p_at_"):
        try:
            return max(1, int(sm[5:]))
        except ValueError:
            return None
    return None


def selection_metric_label(select_metric: str, select_p_at_k: int) -> str:
    pk = resolve_p_at_k(select_metric, select_p_at_k)
    if pk is not None:
        return f"p_at_{pk}"
    return str(select_metric)


def selection_score(
    select_metric: str,
    select_p_at_k: int,
    *,
    cur_auroc: float,
    cur_auprc: float,
    tail_eval: Dict[str, float],
) -> float:
    pk = resolve_p_at_k(select_metric, select_p_at_k)
    if pk is not None:
        return float(tail_eval.get(f"p_at_{pk}", float("nan")))
    sm = str(select_metric).strip().lower()
    if sm == "auprc":
        return float(cur_auprc)
    if sm == "auroc":
        return float(cur_auroc)
    return float("nan")


def topk_ioc_hits(eval_rows: List[Dict[str, object]], ks: Sequence[int]) -> Dict[int, int]:
    rows = sorted(eval_rows, key=lambda r: float(r["score"]), reverse=True)
    out: Dict[int, int] = {}
    for k in ks:
        kk = min(int(k), len(rows))
        out[int(k)] = int(sum(int(rows[i]["is_ioc"]) for i in range(kk)))
    return out

