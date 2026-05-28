from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from graph.alert_eval import tail_alert_metrics
from graph.attack_reconstruct import IDX_TO_STAGE, ioc_type_to_stage_idx, load_ioc_type_to_stage
from graph.tgn_train.eval_io import write_eval_rows_csv
from graph.tgn_train.metrics import (
    parse_topk,
    pr_auc,
    resolve_p_at_k,
    roc_auc,
    selection_metric_label,
    selection_score,
    topk_ioc_hits,
)
from graph.tgn_train.modeling import build_models, build_stage_labels
from graph.tgn_train.neg_sampling import build_neg_pools, build_time_pools, inbatch_neg_dst, sample_window_neg_dst
from graph.tgn_train.streams import Stream, ensure_scenario_stream, load_stream_from_tgn_pt, num_nodes_in_stream, offset_stream_nodes, regenerate_scenario_stream


def _parse_scenarios(s: str) -> List[str]:
    out: List[str] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def _time_split_idx(num_events: int, train_frac: float) -> int:
    if num_events <= 1:
        return 0
    k = int(math.floor(float(train_frac) * float(num_events)))
    return max(1, min(num_events - 1, k))


def train(args: "object", *, repo_root: Optional[Path] = None) -> None:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    if getattr(args, "device", "cpu") == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import torch
    import torch.nn.functional as F

    from graph.attack_reconstruct import NUM_STAGE_CLASSES

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    graphs_dir = (repo_root / args.graphs_dir).resolve()
    graphs_dir.mkdir(parents=True, exist_ok=True)

    scenario_universe = _parse_scenarios(args.scenarios)
    if not scenario_universe:
        raise SystemExit("--scenarios is empty")

    if int(args.early_stop_patience) > 0 and not bool(args.eval_ioc):
        print("warning: --early-stop-patience is ignored without --eval-ioc.", flush=True)

    if args.holdout:
        holdout = args.holdout.strip()
        train_scenarios = [s for s in scenario_universe if s != holdout]
        test_scenarios = [holdout]
    else:
        train_scenarios = list(scenario_universe)
        test_scenarios = list(scenario_universe)
    train_scenarios_set = set(train_scenarios)

    if args.holdout:
        eval_protocol = "loso_holdout"
    elif len(test_scenarios) == 1 and set(train_scenarios) == {test_scenarios[0]}:
        eval_protocol = "per_scenario"
    else:
        eval_protocol = "joint_multi"

    use_aux_supervision = str(args.aux_supervision) != "off"

    streams: Dict[str, Stream] = {}
    for sc in sorted(set(train_scenarios + test_scenarios)):
        _, st = ensure_scenario_stream(
            repo_root=repo_root,
            graphs_dir=graphs_dir,
            scenario=sc,
            auto_generate=bool(args.auto_generate),
        )
        streams[sc] = st

    msg_dims = {sc: int(streams[sc].msg.size(-1)) for sc in streams}
    uniq_dims = sorted(set(msg_dims.values()))
    if len(uniq_dims) > 1:
        if not bool(args.auto_generate):
            raise SystemExit(
                "Inconsistent msg dimensions across scenarios (likely stale *.tgn.pt files). "
                f"Found dims: {msg_dims}. Re-export with --export-tgn for all scenarios, "
                "or rerun with --auto-generate."
            )
        target_dim = min(uniq_dims)
        for sc, d in msg_dims.items():
            if d != target_dim:
                regenerate_scenario_stream(repo_root=repo_root, graphs_dir=graphs_dir, scenario=sc)
                streams[sc] = load_stream_from_tgn_pt(graphs_dir / f"synthchain_{sc}.tgn.pt")
        msg_dims2 = {sc: int(streams[sc].msg.size(-1)) for sc in streams}
        if len(set(msg_dims2.values())) > 1:
            raise SystemExit(f"Still inconsistent msg dimensions after regeneration: {msg_dims2}")

    scenario_base: Dict[str, int] = {}
    base = 0
    for sc in sorted(streams.keys()):
        scenario_base[sc] = base
        base += num_nodes_in_stream(streams[sc])
    for sc in list(streams.keys()):
        streams[sc] = offset_stream_nodes(streams[sc], scenario_base[sc])

    all_src = torch.cat([streams[sc].src for sc in streams], dim=0)
    all_dst = torch.cat([streams[sc].dst for sc in streams], dim=0)
    all_etype = torch.cat([streams[sc].etype for sc in streams], dim=0)
    num_nodes = int(torch.max(torch.stack([all_src.max(), all_dst.max()])).item()) + 1 if all_src.numel() else 0
    num_etypes = int(all_etype.max().item()) + 1 if all_etype.numel() else 1
    raw_msg_dim = int(next(iter(streams.values())).msg.size(-1)) + int(args.etype_dim)

    use_stage = float(args.lambda_stage) > 0.0
    memory, link_pred, etype_emb, stage_pred = build_models(
        num_nodes=num_nodes,
        num_etypes=num_etypes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=int(args.memory_dim),
        time_dim=int(args.time_dim),
        etype_dim=int(args.etype_dim),
        use_stage=use_stage,
        stage_hidden_dim=int(args.stage_hidden_dim),
        num_stage_classes=NUM_STAGE_CLASSES,
        device=device,
    )

    y_stage_per_sc = build_stage_labels(
        streams=streams,
        repo_root=repo_root,
        ioc_type_to_stage_idx=ioc_type_to_stage_idx,
        load_ioc_type_to_stage=load_ioc_type_to_stage,
    )

    all_params = list(memory.parameters()) + list(link_pred.parameters()) + list(etype_emb.parameters())
    if stage_pred is not None:
        all_params += list(stage_pred.parameters())
    opt = torch.optim.Adam(all_params, lr=float(args.lr))
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device).fill_(-1)

    def sample_neg(true_dst: "torch.Tensor", e: "torch.Tensor", pools: Dict[int, "torch.Tensor"]) -> "torch.Tensor":
        if str(args.neg_sampling) == "pool" and int(e.item()) in pools and pools[int(e.item())].numel() > 1:
            pool = pools[int(e.item())].to(true_dst.device)
            j = torch.randint(0, int(pool.numel()), (1,), device=true_dst.device)
            neg = pool[j].view_as(true_dst)
            if int(neg.item()) == int(true_dst.item()):
                neg = pool[(j + 1) % int(pool.numel())].view_as(true_dst)
            return neg
        neg = torch.randint(0, num_nodes, true_dst.size(), device=true_dst.device)
        if true_dst.numel() == 1 and int(neg.item()) == int(true_dst.item()):
            neg = (neg + 1) % num_nodes
        return neg

    def run_one_scenario(
        sc: str,
        *,
        train_mode: bool,
        prefix_only: bool,
        collect_eval: bool = False,
        score_all: bool = False,
    ) -> Tuple[float, float, Optional[List[Dict[str, object]]], Optional[Tuple[float, float]]]:
        st = streams[sc]
        src = st.src.to(device)
        dst = st.dst.to(device)
        t = st.t.to(device)
        msg = st.msg.to(device)
        etype = st.etype.to(device)
        y_ioc = getattr(st, "y_ioc", None)
        if y_ioc is not None:
            y_ioc = y_ioc.to(device)
        row_idx_cpu = getattr(st, "row_idx", None)
        source_file = getattr(st, "source_file", None)
        ioc_type = getattr(st, "ioc_type", None)

        split_idx = _time_split_idx(int(src.numel()), float(args.train_frac))
        if score_all:
            lo, hi = (0, int(src.numel()))
        else:
            lo, hi = (0, split_idx) if prefix_only else (split_idx, int(src.numel()))

        pools = build_neg_pools(dst, etype, split_idx) if str(args.neg_sampling) == "pool" else {}
        time_pools = build_time_pools(dst, t, etype, split_idx) if str(args.neg_sampling) == "window" else {}

        total_loss = 0.0
        correct = 0.0
        count = 0.0
        rows: Optional[List[Dict[str, object]]] = [] if collect_eval else None
        y_true: List[int] = []
        y_score: List[float] = []

        y_stage_sc = y_stage_per_sc.get(sc)

        if train_mode:
            memory.train()
            link_pred.train()
            etype_emb.train()
            if stage_pred is not None:
                stage_pred.train()
        else:
            memory.eval()
            link_pred.eval()
            etype_emb.eval()
            if stage_pred is not None:
                stage_pred.eval()

        memory.reset_state()

        if (not train_mode) and (not prefix_only) and (not score_all) and bool(args.warmup) and split_idx > 0:
            with torch.no_grad():
                for i in range(0, split_idx, int(args.batch_size)):
                    memory.detach()
                    j = min(split_idx, i + int(args.batch_size))
                    s = src[i:j]
                    d = dst[i:j]
                    tt = t[i:j]
                    m = msg[i:j]
                    e = etype[i:j]
                    eemb = etype_emb(e)
                    raw_msg = torch.cat([m, eemb], dim=-1)
                    memory.update_state(s, d, tt, raw_msg.detach())

        for i in range(lo, hi, int(args.batch_size)):
            memory.detach()
            j = min(hi, i + int(args.batch_size))
            s = src[i:j]
            d = dst[i:j]
            tt = t[i:j]
            m = msg[i:j]
            e = etype[i:j]

            if str(args.neg_sampling) == "inbatch":
                neg_d = inbatch_neg_dst(d)
            elif str(args.neg_sampling) == "window":
                neg_d = torch.stack(
                    [
                        sample_window_neg_dst(
                            d[k : k + 1],
                            tt[k : k + 1],
                            e[k : k + 1],
                            time_pools,
                            window_seconds=int(args.neg_window_seconds),
                            max_cands=int(args.neg_window_max_cands),
                        ).view(())
                        for k in range(int(d.numel()))
                    ]
                )
            else:
                neg_d = torch.stack([sample_neg(d[k : k + 1], e[k : k + 1], pools).view(()) for k in range(int(d.numel()))])

            eemb = etype_emb(e)
            raw_msg = torch.cat([m, eemb], dim=-1)

            n_id = torch.unique(torch.cat([s, d, neg_d], dim=0))
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, _ = memory(n_id)

            z_s = z[assoc[s]]
            z_d = z[assoc[d]]
            z_neg = z[assoc[neg_d]]

            pos_inp = torch.cat([z_s, z_d, raw_msg], dim=-1)
            neg_inp = torch.cat([z_s, z_neg, raw_msg], dim=-1)

            pos_logit = link_pred(pos_inp).view(-1)
            neg_logit = link_pred(neg_inp).view(-1)

            benign_mask: Optional["torch.Tensor"] = None
            if bool(args.train_only_benign) and train_mode and prefix_only and (y_ioc is not None):
                benign_mask = (y_ioc[i:j] == 0)

            if benign_mask is not None:
                if int(benign_mask.sum().item()) == 0:
                    loss = None
                else:
                    pos_logit_eff = pos_logit[benign_mask]
                    neg_logit_eff = neg_logit[benign_mask]
                    y = torch.cat([torch.ones_like(pos_logit_eff), torch.zeros_like(neg_logit_eff)], dim=0)
                    logit = torch.cat([pos_logit_eff, neg_logit_eff], dim=0)
                    loss = F.binary_cross_entropy_with_logits(logit, y)
            else:
                y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=0)
                logit = torch.cat([pos_logit, neg_logit], dim=0)
                loss = F.binary_cross_entropy_with_logits(logit, y)

            aux_active = use_aux_supervision and train_mode and (sc in train_scenarios_set)
            if loss is not None and aux_active and float(args.lambda_ioc_rank) > 0.0 and y_ioc is not None:
                yb = y_ioc[i:j].float()
                pos_prob = torch.sigmoid(pos_logit).view(-1)
                score_anom = -torch.log(pos_prob.clamp_min(1e-12))
                mask_i = yb > 0.5
                mask_n = yb < 0.5
                if int(mask_i.sum().item()) > 0 and int(mask_n.sum().item()) > 0:
                    si = score_anom[mask_i]
                    sn = score_anom[mask_n]
                    idx = torch.randint(0, int(sn.numel()), (int(si.numel()),), device=device)
                    pair_sn = sn[idx]
                    margin = float(args.ioc_rank_margin)
                    loss_ioc = F.relu(margin - (si - pair_sn)).mean()
                    loss = loss + float(args.lambda_ioc_rank) * loss_ioc

            if loss is not None and aux_active and use_stage and stage_pred is not None and y_stage_sc is not None:
                stage_labels_batch = y_stage_sc[i:j].to(device)
                has_label = stage_labels_batch > 0
                if int(has_label.sum().item()) > 0:
                    stage_logits = stage_pred(pos_inp)
                    loss_stage = F.cross_entropy(stage_logits[has_label], stage_labels_batch[has_label])
                    loss = loss + float(args.lambda_stage) * loss_stage
                # Regularize stage head on non-IOC edges -> 'none' (class 0)
                # This is important for real-world use where we cannot filter by IOC.
                if float(getattr(args, "lambda_stage_none", 0.0)) > 0.0 and y_ioc is not None:
                    yb = y_ioc[i:j].to(device).view(-1)
                    mask_none = (yb == 0)
                    if int(mask_none.sum().item()) > 0:
                        # Optional downsampling to control imbalance/compute.
                        ratio = float(getattr(args, "stage_none_sample_ratio", 1.0))
                        ratio = max(0.0, min(1.0, ratio))
                        if ratio < 1.0:
                            keep = (torch.rand(mask_none.size(0), device=device) < ratio)
                            mask_none = mask_none & keep
                        if int(mask_none.sum().item()) > 0:
                            stage_logits_all = stage_pred(pos_inp)
                            none_labels = torch.zeros(int(mask_none.sum().item()), dtype=torch.long, device=device)
                            loss_none = F.cross_entropy(stage_logits_all[mask_none], none_labels)
                            loss = loss + float(args.lambda_stage_none) * loss_none

            if train_mode and loss is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                if benign_mask is not None:
                    if int(benign_mask.sum().item()) > 0:
                        prob = torch.sigmoid(logit)
                        pred = (prob > 0.5).float()
                        correct += float((pred == y).sum().item())
                        count += float(y.numel())
                        assert loss is not None
                        total_loss += float(loss.item()) * float(y.numel())
                else:
                    prob = torch.sigmoid(logit)
                    pred = (prob > 0.5).float()
                    correct += float((pred == y).sum().item())
                    count += float(y.numel())
                    total_loss += float(loss.item()) * float(y.numel())

                if collect_eval:
                    pos_prob = torch.sigmoid(pos_logit)
                    pos_score = (-torch.log(pos_prob.clamp_min(1e-12))).detach().cpu()
                    if y_ioc is not None:
                        lbl = y_ioc[i:j].to(torch.int64).detach().cpu()
                    else:
                        lbl = torch.zeros((int(pos_score.numel()),), dtype=torch.int64)
                    ri_sl = row_idx_cpu[i:j] if row_idx_cpu is not None else None
                    sf_sl = source_file[i:j] if source_file is not None else None
                    it_sl = ioc_type[i:j] if ioc_type is not None else None

                    batch_pred_stages: List[str] = []
                    batch_pred_probs: List[float] = []
                    if stage_pred is not None:
                        stage_logits = stage_pred(pos_inp)
                        stage_probs = torch.softmax(stage_logits, dim=-1)
                        stage_cls = torch.argmax(stage_probs, dim=-1).detach().cpu()
                        stage_max_p = stage_probs.max(dim=-1).values.detach().cpu()
                        for k in range(int(pos_score.numel())):
                            cidx = int(stage_cls[k].item())
                            batch_pred_stages.append(IDX_TO_STAGE.get(cidx, "none"))
                            batch_pred_probs.append(float(stage_max_p[k].item()))
                    else:
                        batch_pred_stages = ["" for _ in range(int(pos_score.numel()))]
                        batch_pred_probs = [0.0 for _ in range(int(pos_score.numel()))]

                    for k in range(int(pos_score.numel())):
                        y_true.append(int(lbl[k].item()))
                        y_score.append(float(pos_score[k].item()))
                        assert rows is not None
                        ridx = int(ri_sl[k].item()) if ri_sl is not None else -1
                        sf = str(sf_sl[k]) if sf_sl is not None else ""
                        ityp = str(it_sl[k]) if it_sl is not None else ""
                        rows.append(
                            {
                                "scenario": sc,
                                "t": int(tt[k].item()),
                                "etype": int(e[k].item()),
                                "src": int(s[k].item()),
                                "dst": int(d[k].item()),
                                "score": float(pos_score[k].item()),
                                "is_ioc": int(lbl[k].item()),
                                "source_file": sf,
                                "row_idx": ridx,
                                "ioc_type": ityp,
                                "pred_stage": batch_pred_stages[k],
                                "pred_stage_prob": f"{batch_pred_probs[k]:.4f}",
                            }
                        )

            if benign_mask is not None:
                if int(benign_mask.sum().item()) > 0:
                    memory.update_state(s[benign_mask], d[benign_mask], tt[benign_mask], raw_msg[benign_mask].detach())
            else:
                memory.update_state(s, d, tt, raw_msg.detach())

        metrics = None
        if collect_eval and bool(args.eval_ioc):
            metrics = (roc_auc(y_true, y_score), pr_auc(y_true, y_score))
        return (total_loss / max(1.0, count)), (correct / max(1.0, count)), rows, metrics

    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topk(args.topk)

    select_p_at_k = resolve_p_at_k(args.select_metric, int(args.select_p_at_k))
    if select_p_at_k is not None:
        if not bool(args.eval_ioc):
            raise SystemExit("--select-metric p_at* requires --eval-ioc.")
        if select_p_at_k not in topks:
            topks = sorted(set(topks) | {select_p_at_k})
            print(f"note: added K={select_p_at_k} to --topk for checkpoint selection (now {topks}).", flush=True)
    elif str(args.select_metric).strip().lower() not in ("auprc", "auroc"):
        raise SystemExit(f"Unknown --select-metric {args.select_metric!r}; use auprc, auroc, p_at, or p_at_<K>.")
    metric_label = selection_metric_label(args.select_metric, int(args.select_p_at_k))

    best_metric = float("-inf")
    best_epoch: Optional[int] = None
    best_auroc_at_best: float = float("nan")
    best_auprc_at_best: float = float("nan")
    best_ckpt_path = out_dir / ("best_ckpt_holdout.pt" if args.holdout else "best_ckpt_joint.pt")
    best_scores_path = out_dir / "best_eval_tail_scores.csv"
    best_scores_all_path = out_dir / "best_eval_all_scores.csv"

    last_tr_loss = last_tr_acc = last_va_loss = last_va_acc = float("nan")
    last_auroc = last_auprc = float("nan")
    last_tail_eval: Dict[str, float] = {}
    best_tail_eval: Dict[str, float] = {}

    es_best = float("-inf")
    es_patience = 0
    last_completed_epoch = 0

    for ep in range(1, int(args.epochs) + 1):
        last_completed_epoch = int(ep)
        train_losses: List[float] = []
        train_accs: List[float] = []
        for sc in train_scenarios:
            tl, ta, _, _ = run_one_scenario(sc, train_mode=True, prefix_only=True, collect_eval=False)
            train_losses.append(tl)
            train_accs.append(ta)
        tr_loss = float(sum(train_losses) / max(1, len(train_losses)))
        tr_acc = float(sum(train_accs) / max(1, len(train_accs)))

        with torch.no_grad():
            val_losses: List[float] = []
            val_accs: List[float] = []
            eval_rows: List[Dict[str, object]] = []
            eval_rows_all: List[Dict[str, object]] = []
            eval_aurocs: List[float] = []
            eval_auprcs: List[float] = []
            for sc in test_scenarios:
                vl, va, rows, metrics = run_one_scenario(
                    sc, train_mode=False, prefix_only=False, collect_eval=bool(args.save_scores or args.eval_ioc)
                )
                val_losses.append(vl)
                val_accs.append(va)
                if rows is not None:
                    eval_rows.extend(rows)
                if metrics is not None:
                    auroc_v, auprc_v = metrics
                    if not math.isnan(auroc_v):
                        eval_aurocs.append(float(auroc_v))
                    if not math.isnan(auprc_v):
                        eval_auprcs.append(float(auprc_v))

                if bool(args.save_scores) and str(getattr(args, "save_scores_split", "tail")) == "all":
                    _, _, rows_all, _ = run_one_scenario(
                        sc,
                        train_mode=False,
                        prefix_only=False,
                        collect_eval=True,
                        score_all=True,
                    )
                    if rows_all is not None:
                        eval_rows_all.extend(rows_all)
            va_loss = float(sum(val_losses) / max(1, len(val_losses)))
            va_acc = float(sum(val_accs) / max(1, len(val_accs)))

        extra = ""
        cur_auroc = float("nan")
        cur_auprc = float("nan")
        if bool(args.eval_ioc) and (eval_aurocs or eval_auprcs):
            cur_auroc = float(sum(eval_aurocs) / max(1, len(eval_aurocs))) if eval_aurocs else float("nan")
            cur_auprc = float(sum(eval_auprcs) / max(1, len(eval_auprcs))) if eval_auprcs else float("nan")
            extra = f" | AUROC {cur_auroc:.3f} AUPRC {cur_auprc:.3f}"

        if topks and eval_rows:
            hits = topk_ioc_hits(eval_rows, topks)
            extra += " | " + " ".join([f"top{k}={hits[k]}" for k in topks])

        epoch_tail_eval: Dict[str, float] = {}
        if bool(args.eval_ioc) and eval_rows:
            epoch_tail_eval = tail_alert_metrics(
                eval_rows,
                topks=topks,
                alert_window=int(args.eval_alert_window),
                alert_quantile=float(args.eval_alert_quantile),
                alert_min_events=int(args.eval_alert_min_events),
                alert_topk_events=int(args.eval_alert_topk_events),
                dedupe=not bool(args.no_eval_alert_dedupe),
            )
            last_tail_eval = dict(epoch_tail_eval)
            extra += (
                f" | pf={epoch_tail_eval['precision_in_flagged']:.3f}"
                f" ar={epoch_tail_eval['alerts_per_tail_event']:.4f}"
                f" fr={epoch_tail_eval['flagged_rate']:.3f}"
            )
            for k in topks:
                pk = epoch_tail_eval.get(f"p_at_{int(k)}", float("nan"))
                extra += f" p@{int(k)}={pk:.3f}"
        print(
            f"epoch {ep:03d} | train({len(train_scenarios)}) loss {tr_loss:.4f} acc {tr_acc:.3f} "
            f"| val({len(test_scenarios)}) loss {va_loss:.4f} acc {va_acc:.3f}{extra}"
        )

        last_tr_loss, last_tr_acc = float(tr_loss), float(tr_acc)
        last_va_loss, last_va_acc = float(va_loss), float(va_acc)
        last_auroc, last_auprc = float(cur_auroc), float(cur_auprc)

        if bool(args.save_scores) or bool(args.save_scores_each_epoch):
            csv_path = out_dir / ("eval_tail_scores.csv" if bool(args.save_scores) else f"eval_tail_scores_epoch{ep:03d}.csv")
            write_eval_rows_csv(csv_path, eval_rows)
            if bool(args.save_scores) and str(getattr(args, "save_scores_split", "tail")) == "all":
                write_eval_rows_csv(out_dir / "eval_all_scores.csv", eval_rows_all)

        if bool(args.eval_ioc):
            cur = selection_score(
                args.select_metric,
                int(args.select_p_at_k),
                cur_auroc=cur_auroc,
                cur_auprc=cur_auprc,
                tail_eval=epoch_tail_eval,
            )
            if not math.isnan(cur) and cur > best_metric:
                best_metric = float(cur)
                best_epoch = int(ep)
                best_auroc_at_best = float(cur_auroc)
                best_auprc_at_best = float(cur_auprc)
                ckpt_dict: Dict[str, object] = {
                    "memory": memory.state_dict(),
                    "link_pred": link_pred.state_dict(),
                    "etype_emb": etype_emb.state_dict(),
                    "config": vars(args),
                    "train_scenarios": train_scenarios,
                    "test_scenarios": test_scenarios,
                    "scenario_base": scenario_base,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "metric_name": metric_label,
                }
                if stage_pred is not None:
                    ckpt_dict["stage_pred"] = stage_pred.state_dict()
                torch.save(ckpt_dict, best_ckpt_path)
                if eval_rows:
                    write_eval_rows_csv(best_scores_path, eval_rows)
                if bool(args.save_scores) and str(getattr(args, "save_scores_split", "tail")) == "all":
                    if eval_rows_all:
                        write_eval_rows_csv(best_scores_all_path, eval_rows_all)
                if epoch_tail_eval:
                    best_tail_eval = dict(epoch_tail_eval)

            if int(args.early_stop_patience) > 0:
                cur_es = selection_score(
                    args.select_metric,
                    int(args.select_p_at_k),
                    cur_auroc=cur_auroc,
                    cur_auprc=cur_auprc,
                    tail_eval=epoch_tail_eval,
                )
                if not math.isnan(cur_es):
                    if cur_es > es_best + float(args.early_stop_min_delta):
                        es_best = float(cur_es)
                        es_patience = 0
                    else:
                        es_patience += 1
                        if es_patience >= int(args.early_stop_patience):
                            print(
                                f"early_stop: {metric_label} did not improve by "
                                f">{args.early_stop_min_delta} for {args.early_stop_patience} epochs "
                                f"(best_seen={es_best:.4f}). Stopping at epoch {ep:03d}."
                            )
                            break

    final_ckpt: Dict[str, object] = {
        "memory": memory.state_dict(),
        "link_pred": link_pred.state_dict(),
        "etype_emb": etype_emb.state_dict(),
        "config": vars(args),
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "scenario_base": scenario_base,
    }
    if stage_pred is not None:
        final_ckpt["stage_pred"] = stage_pred.state_dict()
    torch.save(final_ckpt, out_dir / ("ckpt_holdout.pt" if args.holdout else "ckpt_joint.pt"))

    if best_epoch is not None:
        print(f"Best by {metric_label}: epoch {best_epoch} = {best_metric:.4f}")
    if int(args.early_stop_patience) > 0 and last_completed_epoch < int(args.epochs):
        print(f"Completed {last_completed_epoch}/{int(args.epochs)} epochs (early stopping).")

    def _jf(x: object) -> object:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    best_metric_out: Optional[float] = None
    if best_epoch is not None and not math.isinf(float(best_metric)):
        best_metric_out = float(best_metric)

    primary_scenario = str(args.holdout).strip() if args.holdout else (test_scenarios[0] if len(test_scenarios) == 1 else "")
    summary = {
        "eval_protocol": eval_protocol,
        "scenario": primary_scenario,
        "holdout": str(args.holdout) if args.holdout else "",
        "scenarios": list(scenario_universe),
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "aux_supervision": str(args.aux_supervision),
        "lambda_ioc_rank": float(args.lambda_ioc_rank),
        "lambda_stage": float(args.lambda_stage),
        "epochs": int(args.epochs),
        "epochs_completed": int(last_completed_epoch),
        "early_stopped": bool(last_completed_epoch < int(args.epochs)),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "seed": int(args.seed),
        "select_metric": str(args.select_metric),
        "select_metric_label": metric_label,
        "select_p_at_k": int(args.select_p_at_k),
        "best_epoch": best_epoch,
        "best_metric": best_metric_out,
        "best_auroc": _jf(best_auroc_at_best),
        "best_auprc": _jf(best_auprc_at_best),
        "last_train_loss": _jf(last_tr_loss),
        "last_train_acc": _jf(last_tr_acc),
        "last_val_loss": _jf(last_va_loss),
        "last_val_acc": _jf(last_va_acc),
        "last_auroc": _jf(last_auroc),
        "last_auprc": _jf(last_auprc),
        "last_tail_eval": {k: _jf(v) for k, v in last_tail_eval.items()},
        "best_tail_eval": {k: _jf(v) for k, v in best_tail_eval.items()},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {out_dir}")

