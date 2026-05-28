from __future__ import annotations

import argparse
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/validate a TGN on SynthChain scenarios (time-split; optional holdout scenario)."
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="sc1,sc2,sc3,sc4,sc5,sc6,sc7",
        help="Comma-separated scenarios to use as the scenario universe.",
    )
    p.add_argument(
        "--holdout",
        type=str,
        default="",
        help="If set (e.g. sc3), train on all other scenarios and test on holdout.",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Within each scenario, train on earliest fraction; validate on the rest.",
    )
    p.add_argument("--memory-dim", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--etype-dim", type=int, default=16)
    p.add_argument(
        "--neg-sampling",
        type=str,
        default="random",
        choices=["random", "pool", "inbatch", "window"],
        help=(
            "Negative sampling strategy for self-supervised link prediction. "
            "random: dst ~ Uniform[0,num_nodes); "
            "pool: sample dst from per-etype pool built on train prefix (similar to --hard-neg); "
            "inbatch: use other dst within the same batch (harder); "
            "window: sample dst from same etype within a time window in the train prefix (hard, distribution-aware)."
        ),
    )
    p.add_argument(
        "--neg-window-seconds",
        type=int,
        default=3600,
        help="Time window (seconds) for --neg-sampling window (uses train prefix pools).",
    )
    p.add_argument(
        "--neg-window-max-cands",
        type=int,
        default=4096,
        help="Cap the number of candidates considered per edge for --neg-sampling window (0 = no cap).",
    )
    p.add_argument(
        "--hard-neg",
        action="store_true",
        help="DEPRECATED: use --neg-sampling pool instead.",
    )
    p.add_argument(
        "--train-only-benign",
        action="store_true",
        help=(
            "During training (prefix only), exclude IOC-labeled edges from loss and memory updates "
            "when y_ioc is available. This makes the self-supervised objective focus on normal behavior."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graphs-dir", type=str, default="artifacts/graphs")
    p.add_argument("--out", type=str, default="artifacts/tgn_runs/synthchain_multi")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Force device. Use cpu if CUDA driver is old.",
    )
    p.add_argument(
        "--auto-generate",
        action="store_true",
        help="If a scenario graph/stream is missing, try generating it via scripts/generate_graph.py.",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="For evaluation, warm up memory on the scenario prefix (no-grad) before scoring the tail.",
    )
    p.add_argument(
        "--eval-ioc",
        action="store_true",
        help="On evaluation tails, compute AUROC/AUPRC using msg[is_ioc] as label.",
    )
    p.add_argument("--save-scores", action="store_true", help="Save per-event scores for evaluation tails to a CSV under --out.")
    p.add_argument(
        "--save-scores-split",
        type=str,
        default="tail",
        choices=["tail", "all"],
        help=(
            "Which part of the scenario to export scores for when --save-scores is set. "
            "tail = only the holdout tail (after --train-frac split, default). "
            "all = export prefix+tail (full timeline), useful for full-chain reconstruction."
        ),
    )
    p.add_argument(
        "--save-scores-each-epoch",
        action="store_true",
        help="Save per-event tail scores for every epoch (eval_tail_scores_epochXXX.csv).",
    )
    p.add_argument(
        "--select-metric",
        type=str,
        default="auprc",
        help=(
            "Metric for best checkpoint / early stopping (requires --eval-ioc): "
            "auprc, auroc, p_at (uses --select-p-at-k), or p_at_<K> e.g. p_at_100."
        ),
    )
    p.add_argument("--select-p-at-k", type=int, default=100, help="K for --select-metric p_at (ignored for p_at_<K> or auprc/auroc).")
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if --select-metric does not improve for this many epochs (0 = disabled). Requires --eval-ioc.",
    )
    p.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum improvement on --select-metric to reset early-stopping patience.")
    p.add_argument("--topk", type=str, default="10,50,100,500", help="Comma-separated K values for top-K IOC hit reporting on eval tail scores.")
    p.add_argument("--eval-alert-window", type=int, default=3600, help="Same as aggregate_alerts --window when reporting alert-rate / precision-in-flagged.")
    p.add_argument("--eval-alert-quantile", type=float, default=0.99)
    p.add_argument("--eval-alert-min-events", type=int, default=3)
    p.add_argument("--eval-alert-topk-events", type=int, default=0, help="0 = use quantile.")
    p.add_argument("--no-eval-alert-dedupe", action="store_true", help="Disable dedupe for alert metrics (matches aggregate_alerts --no-dedupe).")
    p.add_argument("--lambda-ioc-rank", type=float, default=0.0, help="If >0, add margin ranking loss so IOC edges get higher anomaly score than non-IOC in same batch (train prefix only).")
    p.add_argument("--ioc-rank-margin", type=float, default=0.5, help="Margin for IOC ranking loss (anomaly score = -log sigmoid(pos_logit)).")
    p.add_argument("--lambda-stage", type=float, default=0.0, help="If >0, add stage classification CE loss on IOC edges with known stage labels (train prefix).")
    p.add_argument(
        "--lambda-stage-none",
        type=float,
        default=0.0,
        help=(
            "If >0, add an auxiliary CE loss that pushes non-IOC edges to stage 'none' (class 0) "
            "during training prefix. This helps pred_stage avoid hallucinating stages in real deployments."
        ),
    )
    p.add_argument(
        "--stage-none-sample-ratio",
        type=float,
        default=1.0,
        help=(
            "Sampling ratio for non-IOC edges when applying --lambda-stage-none (0..1). "
            "Use <1 to reduce compute / class imbalance."
        ),
    )
    p.add_argument("--stage-hidden-dim", type=int, default=64, help="Hidden dim of the stage classifier MLP head.")
    p.add_argument(
        "--aux-supervision",
        type=str,
        default="train_only",
        choices=["train_only", "off"],
        help=(
            "IOC rank / stage CE auxiliary losses: train_only = only on --train scenarios "
            "(holdout scenario never gets aux gradients; per-scenario run uses that scenario). "
            "off = pure self-supervised link prediction."
        ),
    )

    args = p.parse_args(argv)
    if bool(getattr(args, "hard_neg", False)):
        args.neg_sampling = "pool"
    return args

