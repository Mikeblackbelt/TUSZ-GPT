"""Stage 2 -- supervised fine-tuning of the pretrained GPT trunk with two heads.

    python -m eeg_gpt.training.finetune --master-csv ../master_full.csv --checkpoint-dir ../checkpoints \
        --pretrained pretrain_run/best.pt --out ft_run --context-blocks 8 --type-grouping focal_generalized

Binary head   : --label preictal  (your SPH/SOP rows: preictal=1, background=0, ictal masked)
                --label horizon   (Mahdi 2026: onset within --horizon-s after the window; ictal rows still masked)
Multitype head: seizure type on ictal rows (--type-preictal-weight > 0 adds preictal rows with the
                upcoming seizure's type); classes from --type-grouping (default focal/generalized).
Evaluation    : AUROC, threshold-sweep balanced accuracy, per-session AUROC, firing-power smoothing,
                the log band-power logistic baseline on the same windows, type confusion on ictal rows.
Protocols     : default = TUSZ patient-disjoint splits (--train-split train --eval-split dev).
                --adapt-patient <id>: Mahdi's patient-adaptive setting done honestly -- fine-tune on the
                patient's EARLIER sessions, evaluate on the LATER ones (chronological, no overlap).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from eeg_gpt.data.datasets import LabeledWindows, patient_id, session_number
from eeg_gpt.data.tokenizer import AmplitudeTokenizer
from eeg_gpt.models.gpt import EEGGPT, SeizureModel, count_params
from eeg_gpt.training.metrics import (auroc, bandpower_baseline, bandpower_features, confusion, macro_f1,
                                      per_session_auroc, smoothed_auroc, threshold_sweep)

log = logging.getLogger("finetune")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--master-csv", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--stage", default="proc")
    p.add_argument("--out", required=True)
    p.add_argument("--pretrained", default=None, help="pretrain best.pt; omit for a from-scratch control (the A/B)")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="dev")
    p.add_argument("--adapt-patient", default=None, help="patient id: fine-tune on earlier sessions, eval on later")
    p.add_argument("--adapt-train-frac", type=float, default=0.5)
    p.add_argument("--label", choices=["preictal", "horizon"], default="preictal")
    p.add_argument("--horizon-s", type=float, default=30.0)
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--context-blocks", type=int, default=1)
    p.add_argument("--type-grouping", choices=["none", "focal_generalized", "fnsz_gnsz_other"], default="focal_generalized")
    p.add_argument("--type-weight", type=float, default=1.0)
    p.add_argument("--type-preictal-weight", type=float, default=0.0)
    p.add_argument("--min-ictal-per-batch", type=int, default=4)
    p.add_argument("--binary-weight", type=float, default=1.0)
    p.add_argument("--pos-weight-cap", type=float, default=20.0)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0, help="0 = full pass")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pool", choices=["mean", "last", "meanmax"], default="mean")
    p.add_argument("--smooth-k", type=int, default=15)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-capacity", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--load", default=None, help="fine-tuned checkpoint for --eval-only")
    return p


class BalancedBatchSampler(Sampler):
    """Session-blocked batches (k sessions open at a time, cache-friendly) with
    `min_ictal` slots reserved for ictal rows so the type head sees every batch."""

    def __init__(self, df, batch_size, min_ictal=0, sessions_per_batch=4, seed=0):
        self.groups = [g.index.tolist() for _, g in df.groupby("session_key", sort=False)]
        self.ictal = df.index[df["_cat"] == "ictal"].tolist()
        self.bs, self.k, self.spb, self.seed = batch_size, min(min_ictal, batch_size - 1) if self.ictal else 0, sessions_per_batch, seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        groups = [list(g) for g in self.groups]; rng.shuffle(groups)
        for g in groups:
            rng.shuffle(g)
        ict = list(self.ictal); rng.shuffle(ict); ip = 0
        open_g, nxt, batch, fill = [], 0, [], self.bs - self.k
        while True:
            while len(open_g) < self.spb and nxt < len(groups):
                open_g.append(groups[nxt]); nxt += 1
            if not open_g:
                break
            for g in list(open_g):
                if not g:
                    open_g.remove(g); continue
                batch.append(g.pop())
                if len(batch) == fill:
                    for _ in range(self.k):
                        if ip >= len(ict):
                            ip = 0; rng.shuffle(ict)
                        batch.append(ict[ip]); ip += 1
                    yield batch; batch = []
            open_g = [g for g in open_g if g]

    def __len__(self):
        return sum(len(g) for g in self.groups) // max(1, self.bs - self.k)


def make_datasets(args, tok):
    kw = dict(stage=args.stage, window_s=args.window_s, fs=args.fs, context_blocks=args.context_blocks, horizon_s=args.horizon_s,
              type_grouping=args.type_grouping, tokenizer=tok, cache_capacity=args.cache_capacity)
    if args.adapt_patient:
        allds = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, **kw)
        sess = sorted([s for s in allds.sessions if patient_id(s) == args.adapt_patient], key=session_number)
        if len(sess) < 2:
            raise SystemExit(f"patient {args.adapt_patient}: need >= 2 sessions, found {sess}")
        n_tr = max(1, int(round(len(sess) * args.adapt_train_frac)))
        tr = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, sessions=sess[:n_tr], **kw)
        ev = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, sessions=sess[n_tr:], **kw)
        log.info(f"patient-adaptive: {args.adapt_patient} train sessions {sess[:n_tr]} -> eval sessions {sess[n_tr:]}")
        return tr, ev
    return (LabeledWindows(args.master_csv, args.checkpoint_dir, args.train_split, **kw),
            LabeledWindows(args.master_csv, args.checkpoint_dir, args.eval_split, **kw))


def binary_targets(batch, label):
    return batch["horizon_pos"] if label == "horizon" else batch["occ"]


def make_loader(dataset, *, batch_size=None, batch_sampler=None, num_workers=0, shuffle=False, drop_last=False):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


def run_eval(model, loader, args, device, ds, do_baseline=False, train_loader=None):
    model.eval()
    S, Y, V, SID, TIME, TP, TT, TI, Z = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["tokens"].to(device, non_blocking=torch.cuda.is_available()))
            S.append(torch.sigmoid(out["binary_logit"].float()).cpu()); Y.append(binary_targets(b, args.label)); V.append(b["occ_valid"])
            SID.append(b["session_idx"]); TIME.append(b["row_idx"].float()); TP.append(out["type_logits"].argmax(-1).cpu())
            TT.append(b["type_idx"]); TI.append(b["is_ictal"])
            if do_baseline:
                Z.append(torch.stack([bandpower_features(z, args.fs) for z in b["z_last"]]))
    S, Y, V, SID, TIME = (torch.cat(x).numpy() for x in (S, Y, V, SID, TIME))
    TP, TT, TI = (torch.cat(x).numpy() for x in (TP, TT, TI))
    m = V > 0.5
    res = {"auroc": auroc(S[m], Y[m]), "sweep": threshold_sweep(S[m], Y[m]), "n_valid": int(m.sum()), "n_pos": int(Y[m].sum())}
    res["per_session"] = per_session_auroc(SID[m], S[m], Y[m])
    res["smooth"] = {k: smoothed_auroc(SID[m], TIME[m], S[m], Y[m], k)["auroc_smoothed"] for k in (5, args.smooth_k, 4 * args.smooth_k)}
    ict = TI > 0.5
    n_types = len(ds.type_classes)
    cm = confusion(TP[ict & (TT >= 0)], TT[ict & (TT >= 0)], n_types)
    res["type_cm_ictal"] = cm.tolist(); res["type_macro_f1_ictal"] = macro_f1(cm)
    res["type_pred_hist"] = np.bincount(TP[ict], minlength=n_types).tolist()
    if do_baseline and train_loader is not None:
        Xtr, ytr = [], []
        with torch.no_grad():
            for b in train_loader:
                mm = b["occ_valid"] > 0.5
                if not bool(mm.any()):
                    continue
                Xtr.append(torch.stack([bandpower_features(z, args.fs) for z in b["z_last"][mm]])); ytr.append(binary_targets(b, args.label)[mm])
        bl = bandpower_baseline(torch.cat(Xtr), torch.cat(ytr), torch.cat(Z)[m], torch.from_numpy(Y[m]).float())
        res["baseline"] = {"train_auroc": bl["train_auroc"], "dev_auroc": bl["dev_auroc"]}
    model.train()
    return res


def log_eval(res, tag, ds):
    ps, sw = res["per_session"], res["sweep"]
    log.info(f"{tag} | binary AUROC {res['auroc']:.4f} | sweep BA {sw['balanced_accuracy']:.3f} (sens {sw['sensitivity']:.2f} spec {sw['specificity']:.2f}) "
             f"| n={res['n_valid']} pos={res['n_pos']}")
    log.info(f"{tag} | per-session AUROC: n={ps['n']} median={ps['median']:.3f} frac>0.6={ps['frac_above_0.6']:.2f} | smoothed AUROC " +
             " ".join(f"k={k}:{v:.3f}" for k, v in res["smooth"].items()))
    if "baseline" in res:
        log.info(f"{tag} | band-power baseline: train {res['baseline']['train_auroc']:.3f} dev {res['baseline']['dev_auroc']:.3f}")
    log.info(f"{tag} | TYPE (ictal rows) macro-F1 {res['type_macro_f1_ictal']:.3f} | classes {ds.type_classes} | pred hist {res['type_pred_hist']} | cm {res['type_cm_ictal']}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "finetune.log"))])
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- backbone: from the pretraining checkpoint (architecture + tokenizer settings come with it) ---
    if args.pretrained:
        ck = torch.load(args.pretrained, map_location="cpu")
        cfg = ck["config"]
        tok = AmplitudeTokenizer(cfg["n_levels"], cfg["clip_k"])
        backbone = EEGGPT(cfg["n_levels"], cfg["n_channels"], cfg["d"], cfg["layers"], cfg["heads"], cfg["ffn"], cfg["block_size"],
                          args.dropout, [int(o) for o in str(cfg["pred_offsets"]).split(",")])
        backbone.load_state_dict(ck["model"]); log.info(f"loaded pretrained trunk from {args.pretrained} (step {ck['step']})")
    else:
        tok = AmplitudeTokenizer(512, 5.0)
        backbone = EEGGPT(512, 20, 128, 4, 4, 512, int(round(args.window_s * args.fs)), args.dropout, (1,))
        log.info("no --pretrained: from-scratch control")
    tr, ev = make_datasets(args, tok)
    log.info(f"train {tr.counts()}"); log.info(f"eval  {ev.counts()}")
    C = tr.n_channels
    model = SeizureModel(backbone, C, len(tr.type_classes), args.context_blocks, args.dropout, args.pool).to(device)
    log.info(f"SeizureModel params {count_params(model) / 1e6:.2f}M | context {args.context_blocks} blocks | types {tr.type_classes}")
    if args.freeze_backbone:
        for p_ in model.encoder.backbone.parameters():
            p_.requires_grad = False
    sampler = BalancedBatchSampler(tr.df, args.batch_size, args.min_ictal_per_batch, seed=args.seed)
    tl = make_loader(tr, batch_sampler=sampler, num_workers=args.num_workers)
    el = make_loader(ev, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    tl_eval = make_loader(tr, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) if args.baseline else None
    if args.eval_only:
        if args.load:
            model.load_state_dict(torch.load(args.load, map_location=device)["model"])
        res = run_eval(model, el, args, device, ev, args.baseline, tl_eval); log_eval(res, "EVAL", ev)
        json.dump({k: v for k, v in res.items() if k != "per_session"}, open(os.path.join(args.out, "eval.json"), "w"), indent=1)
        return
    # class weights
    d = tr.df; ypos = ((d["_cat"] == "preictal")).sum(); yneg = (d["_cat"] == "background").sum()
    pos_weight = torch.tensor([min(args.pos_weight_cap, max(1.0, yneg / max(1, ypos)))], device=device)
    tcount = np.array([(d["_type_idx"] == i).sum() for i in range(len(tr.type_classes))], dtype=np.float64)
    tw = torch.tensor(np.where(tcount > 0, np.sqrt(tcount.sum() / np.maximum(tcount, 1)), 0.0), dtype=torch.float32, device=device)
    tw = tw / tw[tw > 0].mean() if (tw > 0).any() else tw
    log.info(f"pos_weight {pos_weight.item():.2f} | type weights {tw.tolist()}")
    bb = [p_ for n, p_ in model.named_parameters() if n.startswith("encoder.backbone") and p_.requires_grad]
    rest = [p_ for n, p_ in model.named_parameters() if not n.startswith("encoder.backbone") and p_.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr}, {"params": rest, "lr": args.head_lr}], weight_decay=args.weight_decay)
    total = args.epochs * (args.steps_per_epoch or len(sampler)) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); agg = {"loss": 0.0, "bin": 0.0, "type": 0.0}; n = 0
        for i, b in enumerate(tl):
            if args.steps_per_epoch and i >= args.steps_per_epoch:
                break
            tokens = b["tokens"].to(device, non_blocking=torch.cuda.is_available())
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(tokens)
            y = binary_targets(b, args.label).to(device, non_blocking=torch.cuda.is_available()); v = b["occ_valid"].to(device, non_blocking=torch.cuda.is_available()) > 0.5
            lb = F.binary_cross_entropy_with_logits(out["binary_logit"].float()[v], y[v], pos_weight=pos_weight) if v.any() else out["binary_logit"].sum() * 0
            ti = b["type_idx"].to(device); ii = b["is_ictal"].to(device) > 0.5
            wrow = torch.where(ii, 1.0, args.type_preictal_weight) * (ti >= 0).float()
            if (wrow > 0).any():
                ce = F.cross_entropy(out["type_logits"].float(), ti.clamp_min(0), weight=tw, reduction="none")
                lt = (ce * wrow).sum() / wrow.sum()
            else:
                lt = out["type_logits"].sum() * 0
            loss = (args.binary_weight * lb + args.type_weight * lt) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            agg["loss"] += float(loss) * args.accum; agg["bin"] += float(lb); agg["type"] += float(lt); n += 1
        log.info(f"epoch {ep} | " + " ".join(f"{k}={v / max(1, n):.4f}" for k, v in agg.items()) + f" | {time.time() - t0:.0f}s")
        res = run_eval(model, el, args, device, ev, args.baseline and ep == 1, tl_eval); log_eval(res, f"epoch {ep}", ev)
        hist.append({"epoch": ep, **{k: v for k, v in res.items() if k not in ("per_session", "type_cm_ictal")}})
        torch.save({"model": model.state_dict(), "epoch": ep, "config": vars(args)}, os.path.join(args.out, "last.pt"))
        if res["auroc"] == res["auroc"] and res["auroc"] > best:
            best = res["auroc"]; torch.save({"model": model.state_dict(), "epoch": ep, "config": vars(args)}, os.path.join(args.out, "best.pt"))
        json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
    log.info(f"best binary AUROC {best:.4f}")


if __name__ == "__main__":
    main()