from __future__ import annotations

import argparse


def add_training_args(p: argparse.ArgumentParser) -> None:
    """Shared TGN training / evaluation flags (dataset-agnostic)."""
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Train on earliest fraction of each stream; validate on the rest.",
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
            "Negative sampling: random | pool (per-etype train-prefix dst) | "
            "inbatch | window (time-local hard negatives)."
        ),
    )
    p.add_argument("--neg-window-seconds", type=int, default=3600)
    p.add_argument("--neg-window-max-cands", type=int, default=4096)
    p.add_argument("--hard-neg", action="store_true", help="DEPRECATED: use --neg-sampling pool.")
    p.add_argument(
        "--train-only-benign",
        action="store_true",
        help="Train prefix only: skip IOC-labeled edges in loss/memory when y_ioc exists.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="", help="Run output dir (default depends on --input-mode).")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", action="store_true", help="Warm up memory on prefix before tail eval.")
    p.add_argument("--eval-ioc", action="store_true", help="Tail AUROC/AUPRC when y_ioc is in the stream.")
    p.add_argument("--save-scores", action="store_true")
    p.add_argument("--save-scores-split", type=str, default="tail", choices=["tail", "all"])
    p.add_argument("--save-scores-each-epoch", action="store_true")
    p.add_argument("--select-metric", type=str, default="auprc")
    p.add_argument("--select-p-at-k", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--topk", type=str, default="10,50,100,500")
    p.add_argument("--eval-alert-window", type=int, default=3600)
    p.add_argument("--eval-alert-quantile", type=float, default=0.99)
    p.add_argument("--eval-alert-min-events", type=int, default=3)
    p.add_argument("--eval-alert-topk-events", type=int, default=0)
    p.add_argument("--no-eval-alert-dedupe", action="store_true")
    p.add_argument("--lambda-ioc-rank", type=float, default=0.0)
    p.add_argument("--ioc-rank-margin", type=float, default=0.5)
    p.add_argument("--lambda-stage", type=float, default=0.0)
    p.add_argument(
        "--lambda-stage-none",
        type=float,
        default=0.0,
        help="Aux CE pushing non-IOC edges to stage 'none' on train prefix.",
    )
    p.add_argument("--stage-none-sample-ratio", type=float, default=1.0)
    p.add_argument("--stage-hidden-dim", type=int, default=64)
    p.add_argument(
        "--aux-supervision",
        type=str,
        default="train_only",
        choices=["train_only", "off"],
    )
