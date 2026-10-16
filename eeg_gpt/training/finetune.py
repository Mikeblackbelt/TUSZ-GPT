"""Stage 2 -- supervised fine-tuning of the pretrained GPT trunk with two heads.

    python -m eeg_gpt.training.finetune --master-csv ../master_full.csv --checkpoint-dir ../checkpoints \
        --pretrained pretrain_run/best.pt --out ft_run --context-blocks 8 --type-grouping focal_generalized

Binary head   : --label preictal  (your SPH/SOP rows: preictal=1, background=0, ictal masked)
                --label horizon   (Mahdi 2026: onset within --horizon-s after the window; ictal rows still masked)
Multitype head: seizure type on ictal rows (--type-preictal-weight > 0 adds preictal rows with the
                upcoming seizure's type); classes from --type-grouping (default focal/generalized).
Evaluation    : AUROC, threshold-sweep balanced accuracy, sensitivity@fixed-specificity,
                per-session AUROC, firing-power smoothing,
                the log band-power logistic baseline on the same windows, type confusion on ictal rows.
Protocols     : default = TUSZ patient-disjoint splits (--train-split train --eval-split dev).
                --adapt-patient <id>: Mahdi's patient-adaptive setting done honestly -- fine-tune on the
                patient's EARLIER sessions, evaluate on the LATER ones (chronological, no overlap).

Accuracy improvements vs the original:
  * focal loss (--focal-gamma, default 2.0) for the binary head concentrates
    gradient on hard/uncertain preictal windows rather than the easy background;
  * pos_weight_cap raised to 50 so severe imbalance is handled more aggressively;
  * meanmax pooling is the new default (captures both average and peak activation);
  * context_blocks default raised to 4 (16 s of history at 4 s windows);
  * head_lr raised to 1e-3 (3x) since the randomly-initialised heads need
    larger updates than the pretrained backbone;
  * warmup is a fraction of the run (--warmup-frac, 5 %) instead of a fixed 200
    optimizer steps, which was most of a short run and nothing of a long one;
  * sensitivity@specificity{0.80,0.90} is now reported alongside balanced-accuracy.
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

from eeg_gpt.data.datasets import LabeledWindows, patient_id, resolve_master_csv, session_number
from eeg_gpt.data.tokenizer import AmplitudeTokenizer
from eeg_gpt.models.gpt import EEGGPT, SeizureModel, count_params, focal_bce
from eeg_gpt.training.device import (adapt_num_workers, get_amp_scaler, get_autocast_kwargs, get_device, make_adamw, parse_gpu_usage, resolve_amp,
                                     setup_device_and_hardware)
from eeg_gpt.training.event_metrics import event_metrics, events_from_dataset
from eeg_gpt.training.metrics import (auroc, bandpower_baseline, bandpower_features, causal_session_normalise, confusion, macro_f1,
                                      per_session_auroc, score_stats, sensitivity_at_specificity, smoothed_auroc, threshold_sweep)

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
    p.add_argument("--patient", default=None, help="alias for --adapt-patient: target patient for patient-specific training/CV")
    p.add_argument("--patient-base", choices=["pretrained", "general", "both"], default="pretrained",
                   help="Patient-specific base model: 'pretrained' (Stage 1 EEGGPT trunk) or 'general' (Stage 2 non-patient-specific model)")
    p.add_argument("--general-checkpoint", default=None, help="Stage 2 non-patient-specific model checkpoint (.pt) for patient fine-tuning")
    p.add_argument("--cv-folds", type=int, default=None,
                   help="Number of CV folds for patient-specific training (default 5 for --patient; 0 or 1 for chronological single split)")
    p.add_argument("--cv-strategy", choices=["auto", "session", "event"], default="auto",
                   help="CV folds: whole sessions ('session'), whole seizures + 10-min background segments ('event'), "
                        "or session when >= cv_folds sessions have seizures else event ('auto'). Row-stratified folds were removed (leakage).")
    p.add_argument("--sweep-patients", action="store_true", help="Sweep all patients in master CSV to identify ideal candidates and benchmark")
    p.add_argument("--sweep-profile-only", action="store_true", help="Profile and rank candidates in sweep without training")
    p.add_argument("--sweep-top-k", type=int, default=5, help="Number of top ideal candidates to evaluate during sweep (0 for all)")
    p.add_argument("--min-preictal", type=int, default=5, help="Minimum preictal windows for an ideal candidate (default 5 for 5-fold CV)")
    p.add_argument("--min-sessions", type=int, default=2, help="Minimum sessions for an ideal candidate")
    p.add_argument("--min-background", type=int, default=20, help="Minimum background windows for an ideal candidate")
    p.add_argument("--adapt-train-frac", type=float, default=0.5)
    p.add_argument("--label", choices=["preictal", "horizon"], default="preictal")
    p.add_argument("--horizon-s", type=float, default=30.0)
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--context-blocks", type=int, default=4,
                   help="temporal context window (default 4 = 16 s history @ 4 s windows)")
    p.add_argument("--type-grouping", choices=["none", "focal_generalized", "fnsz_gnsz_other"], default="focal_generalized")
    p.add_argument("--type-weight", type=float, default=1.0)
    p.add_argument("--type-preictal-weight", type=float, default=0.0)
    p.add_argument("--min-ictal-per-batch", type=int, default=4)
    p.add_argument("--binary-weight", type=float, default=1.0)
    p.add_argument("--pos-weight-cap", type=float, default=50.0,
                   help="max pos_weight for binary BCE (default 50, up from 20 for severe imbalance)")
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="focal loss gamma for the binary head: 0 = weighted BCE (balanced batches make focal damping unnecessary), 2 = today's focal loss")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0, help="0 = full pass")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5, help="trunk lr at the top block after unfreezing (see --llrd)")
    p.add_argument("--head-lr", type=float, default=3e-4,
                   help="learning rate for heads (default 3e-4)")
    p.add_argument("--warmup-frac", type=float, default=0.05, help="fraction of optimizer steps spent in linear warmup")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pool", choices=["mean", "last", "meanmax", "tail", "attn", "meanstd"], default="tail",
                   help="pooling of the trunk's positions per block (tail = mean over the last --pool-tail-frac of positions)")
    p.add_argument("--smooth-k", type=int, default=15)
    p.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                   help="fit the feature (band-power + log sd) baseline on the training rows every fold; the floor the model must beat")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-capacity", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=0, help="0 = 4x --batch-size (no graph is kept in eval)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="mixed precision (bf16 where supported); default: on for CUDA")
    p.add_argument("--device", default=None, help="Device to use (cuda, cuda:0, mps, cpu). Defaults to auto-detecting GPU.")
    p.add_argument("--max-gpu-usage", type=parse_gpu_usage, default=None, help="Float between 0.0 and 1.0 inclusive specifying maximum fraction of GPU memory/usage.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--load", default=None, help="fine-tuned checkpoint for --eval-only or patient fine-tuning")
    p.add_argument("--d", type=int, default=128, help="backbone hidden dimension (when trained from scratch)")
    p.add_argument("--layers", type=int, default=4, help="backbone transformer layers (when trained from scratch)")
    p.add_argument("--heads", type=int, default=4, help="backbone attention heads (when trained from scratch)")
    p.add_argument("--ffn", type=int, default=512, help="backbone ffn dimension (when trained from scratch)")
    p.add_argument("--n-levels", type=int, default=512, help="amplitude quantization levels")
    p.add_argument("--clip-k", type=float, default=4.0, help="tokenizer clip threshold")
    # --- recipe flags (owner: recipe implementer; insert new add_argument lines directly below this marker) ---
    p.add_argument("--preictal-frac", type=float, default=0.10, help="fraction of every training batch reserved for preictal rows (0 = none reserved, today's sampler)")
    p.add_argument("--pos-weight", default="auto", help="'auto' = background / preictal per batch after balancing (capped by --pos-weight-cap), or a fixed number")
    p.add_argument("--freeze-epochs", type=int, default=3, help="LP-FT: epochs with the trunk frozen before unfreezing it with layer-wise lr decay (0 = train everything from epoch 1)")
    p.add_argument("--llrd", type=float, default=0.65, help="layer-wise lr decay of the trunk after unfreezing: block i of L gets --lr * llrd^(L-1-i), embeddings --lr * llrd^L (1 = off)")
    p.add_argument("--agg-lr", type=float, default=1e-4, help="lr of the channel aggregator / temporal context (everything that is neither trunk nor head)")
    p.add_argument("--backbone-dropout", type=float, default=0.1, help="dropout inside the pretrained trunk during fine-tuning (it was pretrained with 0)")
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay of the weights used for evaluation and best.pt (0 = off)")
    p.add_argument("--type-head", choices=["auto", "on", "off"], default="auto", help="auto = off when the training fold has < 2 seizure types")
    p.add_argument("--select", choices=["inner", "none", "oracle"], default="inner",
                   help="epoch selection: inner validation split (falls back to none without one), last epoch, or max on the eval fold (optimistic; always reported as auroc_oracle)")
    # --- protocol flags (owner: protocol implementer; insert new add_argument lines directly below this marker) ---
    p.add_argument("--postictal-exclude-s", type=float, default=0.0,
                   help="seconds after the end of a seizure whose rows are not scored as background (occ_valid 0); 0 = off")
    p.add_argument("--inner-val", action=argparse.BooleanOptionalAction, default=True,
                   help="hold one seizure session / seizure group out of each fold's training part for epoch and threshold "
                        "selection (needs >= 3 in the training part); --no-inner-val selects on the last epoch")
    p.add_argument("--head", choices=["finetune", "probe"], default="finetune",
                   help="'probe': frozen trunk + logistic probe on cached embeddings per fold instead of fine-tuning")
    # --- architecture flags (owner: architecture implementer; insert new add_argument lines directly below this marker) ---
    # old behaviour: --feature-layers last --pool meanmax --agg cls --agg-layers 2 --agg-dim 256 --drop-path 0
    #                --context-mode transformer --feat-branch none --encode-context block --encode-tail 0
    p.add_argument("--feature-layers", default="mix", help="trunk layers fed to the classifier: last | mix (learned softmax over layers) | 3,4,5,6")
    p.add_argument("--pool-tail-frac", type=float, default=0.25, help="--pool tail: fraction of positions (from the end) that are averaged")
    p.add_argument("--pool-queries", type=int, default=1, help="--pool attn: number of learned pooling queries")
    p.add_argument("--agg", choices=["cls", "mean", "attn"], default="attn", help="aggregation over channels")
    p.add_argument("--agg-layers", type=int, default=1, help="transformer blocks of the channel aggregator")
    p.add_argument("--agg-dim", type=int, default=128, help="--agg attn: width of the channel aggregator")
    p.add_argument("--drop-path", type=float, default=0.1, help="stochastic depth in the aggregator / context blocks")
    p.add_argument("--context-mode", choices=["transformer", "mean", "concat", "delta"], default="delta",
                   help="how the N block vectors become one: delta = newest + W(newest - mean of the previous ones)")
    p.add_argument("--encode-context", choices=["block", "whole"], default="block",
                   help="block: every block is its own trunk sequence; whole: the N blocks of a channel form one sequence")
    p.add_argument("--feat-branch", choices=["none", "bandpower", "full", "feat_only"], default="none",
                   help="hand-crafted features (data/features.py) fused with the trunk embedding; feat_only = features alone (the floor)")
    p.add_argument("--encode-tail", type=int, default=0, help="> 0: only the last N samples of every block go through the trunk")
    p.add_argument("--head-hidden", type=int, default=None, help="hidden width of the two heads (default None = linear head)")
    # --- reporting flags (owner: reporting implementer; insert new add_argument lines directly below this marker) ---
    p.add_argument("--session-norm-k", type=int, default=150, help="windows of causal per-session score normalisation for the report-only auroc_norm (0 = off)")
    p.add_argument("--eval-threshold", type=float, default=None, help="alarm threshold for the event metrics (default: the threshold-sweep optimum of the evaluated set)")
    p.add_argument("--fp-window", type=int, default=0, help="windows in the firing-power moving average (0 = one SOP of windows)")
    return p


class BalancedBatchSampler(Sampler):
    """Session-blocked batches (k sessions open at a time, cache-friendly) with
    `min_ictal` slots reserved for ictal rows so the type head sees every batch and
    `preictal_frac` of the batch reserved for preictal rows (cycled + reshuffled), so
    every batch carries positives: with 3 % positives an unreserved batch of 16 holds
    0.35 preictal rows and 70 % of batches hold none."""

    def __init__(self, df, batch_size, min_ictal=0, sessions_per_batch=4, seed=0, preictal_frac=0.0):
        self.groups = [g.index.tolist() for _, g in df.groupby("session_key", sort=False)]
        self.ictal = df.index[df["_cat"] == "ictal"].tolist()
        self.preictal = df.index[df["_cat"] == "preictal"].tolist()
        self.n_bg = int((df["_cat"] == "background").sum())
        self.bs, self.spb, self.seed = batch_size, sessions_per_batch, seed
        self.epoch = 0
        self.set_balance(preictal_frac, min_ictal)

    def set_balance(self, preictal_frac=0.0, min_ictal=0):
        """Reserved slots per batch; may be called before the first iteration (the loader keeps a reference)."""
        self.k = min(int(min_ictal), self.bs - 1) if self.ictal else 0
        kp = int(round(float(preictal_frac) * self.bs)) if self.preictal and preictal_frac > 0 else 0
        self.kp = max(0, min(kp, self.bs - self.k - 1))
        self.fill = max(1, self.bs - self.k - self.kp)
        return self

    def composition(self) -> dict:
        """Expected rows per class in a batch and the chance of a batch without any preictal row."""
        n = sum(len(g) for g in self.groups)
        f_pre = len(self.preictal) / max(1, n); f_bg = self.n_bg / max(1, n)
        exp_pre = self.kp + self.fill * f_pre
        return {"batch_size": self.bs, "reserved_preictal": self.kp, "reserved_ictal": self.k, "fill": self.fill,
                "n_rows": n, "n_preictal": len(self.preictal), "n_ictal": len(self.ictal), "n_background": self.n_bg,
                "expected_preictal_per_batch": exp_pre, "expected_background_per_batch": self.fill * f_bg,
                "preictal_frac_expected": exp_pre / max(1, self.bs),
                "p_no_preictal": 0.0 if self.kp > 0 else float((1.0 - f_pre) ** self.fill)}

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        groups = [list(g) for g in self.groups]; rng.shuffle(groups)
        for g in groups:
            rng.shuffle(g)
        ict = list(self.ictal); rng.shuffle(ict); ip = 0
        pre = list(self.preictal); rng.shuffle(pre); pp = 0
        open_g, nxt, batch, fill = [], 0, [], self.fill
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
                    for _ in range(self.kp):
                        if pp >= len(pre):
                            pp = 0; rng.shuffle(pre)
                        batch.append(pre[pp]); pp += 1
                    for _ in range(self.k):
                        if ip >= len(ict):
                            ip = 0; rng.shuffle(ict)
                        batch.append(ict[ip]); ip += 1
                    yield batch; batch = []
            open_g = [g for g in open_g if g]

    def __len__(self):
        return sum(len(g) for g in self.groups) // max(1, self.fill)


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


def make_loader(dataset, *, batch_size=None, batch_sampler=None, num_workers=0, shuffle=False, drop_last=False, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


def _model_forward(model, b, device, non_blocking):
    """Call the model with the dataset's hand-crafted features when both sides support them."""
    tokens = b["tokens"].to(device, non_blocking=non_blocking)
    feats = b.get("feats") if isinstance(b, dict) else None
    if feats is not None and feats.numel() > 0:
        try:
            return model(tokens, feats=feats.to(device, non_blocking=non_blocking))
        except TypeError:        # model built without a feature branch (old signature)
            pass
    return model(tokens)


def _window_features(b, mask, fs):
    """Baseline features of the masked windows of a batch: the dataset's `feats` when it carries them,
    else band powers of z_last plus the real log sd when the dataset returns it."""
    feats = b.get("feats")
    if feats is not None and feats.numel() > 0:
        return feats[mask].float()
    lsd = b.get("log_sd")
    rows = []
    for i in torch.nonzero(mask).flatten().tolist():
        z = b["z_last"][i]
        if lsd is not None and lsd.numel() > 0:
            li = lsd[i]
            li = li[..., -1] if li.ndim > 1 else li          # (C, N) -> last block
            try:
                rows.append(bandpower_features(z, fs, log_sd=li)); continue
            except TypeError:    # bandpower_features without the log_sd argument
                pass
        rows.append(bandpower_features(z, fs))
    return torch.stack(rows) if rows else torch.zeros((0, 0))


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def eval_record(res: dict) -> dict:
    """The JSON-serialisable part of a run_eval result (drops the raw score arrays and the per-session table)."""
    return {k: v for k, v in res.items() if not k.startswith("_") and k not in ("per_session",)}


def run_eval(model, loader, args, device, ds, do_baseline=False, train_loader=None):
    """Window metrics of `loader` (+ the feature baseline fitted on `train_loader` when do_baseline).

    Returns, besides the historical keys (auroc, sweep, n_valid, n_pos, per_session, smooth, sens_at_spec,
    type_cm_ictal, type_macro_f1_ictal, type_pred_hist, baseline): `score_stats` (collapse diagnostics),
    `auroc_smooth15`, `auroc_norm` (causal per-session normalisation, report-only), `event` (seizure-level
    sensitivity and false alarms per hour at --eval-threshold or the sweep threshold) and `_arrays`
    (numpy scores / targets / valid / session_idx / row_idx for callers; never write it to JSON: use eval_record()).
    """
    model.eval()
    non_blocking = (device.type == "cuda")
    autocast_kw = get_autocast_kwargs(device, args.amp)
    S, Y, V, SID, TIME, TP, TT, TI, Z = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            with torch.autocast(**autocast_kw):
                out = _model_forward(model, b, device, non_blocking)
            S.append(torch.sigmoid(out["binary_logit"].float()).cpu()); Y.append(binary_targets(b, args.label)); V.append(b["occ_valid"])
            SID.append(b["session_idx"]); TIME.append(b["row_idx"].float()); TP.append(out["type_logits"].argmax(-1).cpu())
            TT.append(b["type_idx"]); TI.append(b["is_ictal"])
            if do_baseline:
                Z.append(_window_features(b, torch.ones(len(b["occ_valid"]), dtype=torch.bool), args.fs))
    if S:
        S, Y, V, SID, TIME = (torch.cat(x).numpy() for x in (S, Y, V, SID, TIME))
        TP, TT, TI = (torch.cat(x).numpy() for x in (TP, TT, TI))
    else:   # an empty evaluation set (a CV fold with more folds than seizure/background groups): every metric is NaN
        S, Y, V, TIME, TI = (np.zeros(0, dtype=np.float32) for _ in range(5))
        SID, TP, TT = (np.zeros(0, dtype=np.int64) for _ in range(3))
    m = V > 0.5
    res = {"auroc": auroc(S[m], Y[m]), "sweep": threshold_sweep(S[m], Y[m]), "n_valid": int(m.sum()), "n_pos": int(Y[m].sum())}
    res["per_session"] = per_session_auroc(SID[m], S[m], Y[m])
    res["smooth"] = {k: smoothed_auroc(SID[m], TIME[m], S[m], Y[m], k)["auroc_smoothed"] for k in (5, args.smooth_k, 4 * args.smooth_k)}
    res["auroc_smooth15"] = smoothed_auroc(SID[m], TIME[m], S[m], Y[m], 15)["auroc_smoothed"]
    # sensitivity at fixed specificity thresholds (clinical relevance: false-alarm rate)
    res["sens_at_spec"] = {
        f"spec{int(sp*100)}": sensitivity_at_specificity(S[m], Y[m], sp)
        for sp in (0.80, 0.90, 0.95)
    }
    res["score_stats"] = score_stats(S[m], Y[m])
    k_norm = int(getattr(args, "session_norm_k", 150) or 0)
    res["auroc_norm"] = auroc(causal_session_normalise(SID[m], TIME[m], S[m], k_norm), Y[m]) if k_norm > 0 else float("nan")
    ict = TI > 0.5
    n_types = len(ds.type_classes)
    cm = confusion(TP[ict & (TT >= 0)], TT[ict & (TT >= 0)], n_types)
    res["type_cm_ictal"] = cm.tolist(); res["type_macro_f1_ictal"] = macro_f1(cm)
    res["type_pred_hist"] = np.bincount(TP[ict], minlength=n_types).tolist()
    # seizure-level metrics on every window of the set (ictal windows included: alarms inside a seizure are ignored)
    thr_src = "args" if getattr(args, "eval_threshold", None) is not None else "sweep"
    thr = float(args.eval_threshold) if thr_src == "args" else float(res["sweep"]["threshold"])
    try:
        times = (np.asarray(ds.df["stop_time"].values, dtype=np.float64)[TIME.astype(np.int64)] if "stop_time" in ds.df.columns
                 else TIME * args.window_s)
        res["event"] = {**event_metrics(S, SID, times, events_from_dataset(ds), thr, args.window_s, getattr(args, "sop_s", 420.0),
                                        getattr(args, "sph_s", 120.0), getattr(args, "fp_window", 0) or None), "threshold_source": thr_src}
    except Exception as e:   # report-only: never fail an epoch over it
        log.warning(f"event metrics skipped: {e}")
        res["event"] = None
    if do_baseline and train_loader is not None:
        Xtr, ytr = [], []
        with torch.no_grad():
            for b in train_loader:
                mm = b["occ_valid"] > 0.5
                if not bool(mm.any()):
                    continue
                Xtr.append(_window_features(b, mm, args.fs)); ytr.append(binary_targets(b, args.label)[mm].float())
        if Xtr and Z:
            Xva = torch.cat(Z)[torch.from_numpy(m)]
            bl = bandpower_baseline(torch.cat(Xtr), torch.cat(ytr), Xva, torch.from_numpy(Y[m]).float())
            res["baseline"] = {"train_auroc": bl["train_auroc"], "dev_auroc": bl["dev_auroc"],
                               "sens_spec90": sensitivity_at_specificity(np.asarray(bl["dev_scores"]), Y[m], 0.90), "n_feats": int(Xva.shape[1])}
    res["_arrays"] = {"scores": S, "targets": Y, "valid": V, "session_idx": SID, "row_idx": TIME}
    model.train()
    return res


def log_eval(res, tag, ds):
    ps, sw = res["per_session"], res["sweep"]
    ss = res.get("sens_at_spec", {})
    log.info(f"{tag} | binary AUROC {res['auroc']:.4f} | sweep BA {sw['balanced_accuracy']:.3f} (sens {sw['sensitivity']:.2f} spec {sw['specificity']:.2f}) "
             f"| n={res['n_valid']} pos={res['n_pos']}")
    log.info(f"{tag} | sens@spec: " + " ".join(f"{k}={v:.3f}" for k, v in ss.items()))
    log.info(f"{tag} | per-session AUROC: n={ps['n']} median={ps['median']:.3f} frac>0.6={ps['frac_above_0.6']:.2f} | smoothed AUROC " +
             " ".join(f"k={k}:{v:.3f}" for k, v in res["smooth"].items()) + f" | session-normalised {_fmt(res.get('auroc_norm'))}")
    st = res.get("score_stats")
    if st:
        log.info(f"{tag} | scores: std {_fmt(st['std'])} frac>0.5 {_fmt(st['frac_pos'], 2)} separation {_fmt(st['separation'], 2)} "
                 f"(mean logit pos {_fmt(st['mean_logit_pos'], 2)} / neg {_fmt(st['mean_logit_neg'], 2)})")
    ev = res.get("event")
    if ev:
        log.info(f"{tag} | events: {ev['n_hits']}/{ev['n_events']} predicted (sens {_fmt(ev['sens_events'], 2)}, chance {_fmt(ev['chance_sens'], 2)}) | "
                 f"FA/h {_fmt(ev['fa_per_h'], 2)} over {_fmt(ev['background_hours'], 1)} h | thr {_fmt(ev['threshold'])} ({ev['threshold_source']})")
    if "baseline" in res:
        b = res["baseline"]
        log.info(f"{tag} | feature baseline ({b.get('n_feats', '?')} feats): train {_fmt(b['train_auroc'])} dev {_fmt(b['dev_auroc'])} "
                 f"sens@spec90 {_fmt(b.get('sens_spec90'))}")
    log.info(f"{tag} | TYPE (ictal rows) macro-F1 {_fmt(res.get('type_macro_f1_ictal'))} | classes {ds.type_classes} | pred hist {res['type_pred_hist']} | cm {res['type_cm_ictal']}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.master_csv = resolve_master_csv(args.master_csv)
    if args.patient and not args.adapt_patient:
        args.adapt_patient = args.patient
        if args.cv_folds is None:
            args.cv_folds = 5
    elif args.adapt_patient and not args.patient:
        args.patient = args.adapt_patient
        if args.cv_folds is None:
            args.cv_folds = 0
    else:
        if args.cv_folds is None:
            args.cv_folds = 5

    if args.load and not args.general_checkpoint:
        args.general_checkpoint = args.load
    elif args.general_checkpoint and not args.load:
        args.load = args.general_checkpoint

    if args.sweep_patients or args.sweep_profile_only:
        from eeg_gpt.training.patient_sweep import run_patient_sweep
        run_patient_sweep(args)
        return

    if args.patient and args.cv_folds >= 2:
        from eeg_gpt.training.patient_sweep import run_patient_cv
        run_patient_cv(args, args.patient, base_mode=args.patient_base, general_checkpoint=args.general_checkpoint)
        return

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "finetune.log"))])
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device(args.device)
    setup_device_and_hardware(device, args.max_gpu_usage)
    args.num_workers = adapt_num_workers(args.num_workers, n_loaders=3)
    args.amp = resolve_amp(args.amp, device)
    log.info(f"Using device: {device} | amp {args.amp}")

    from eeg_gpt.training.patient_sweep import build_pretrained_seizure_model, fit_feature_stats, get_tokenizer_for_run, load_general_seizure_model
    if args.patient_base == "general" and args.general_checkpoint:
        # the tokenizer settings travel with the checkpoint (n_levels, clip_k, patch, norm_scope)
        tok = get_tokenizer_for_run(args, base_mode="general", general_checkpoint=args.general_checkpoint)
        tr, ev = make_datasets(args, tok)
        log.info(f"train {tr.counts()}"); log.info(f"eval  {ev.counts()}")
        C = tr.n_channels
        model, tok = load_general_seizure_model(
            args.general_checkpoint, device, C, len(tr.type_classes),
            window_s=args.window_s, fs=args.fs, dropout=args.dropout,
            pool=args.pool, context_blocks=args.context_blocks
        )
    else:
        # --- backbone: from the pretraining checkpoint through EEGGPT.from_config (architecture + tokenizer settings
        # come with it), else the from-scratch control; head stack per the --architecture flags ---
        tok = get_tokenizer_for_run(args, base_mode="pretrained")
        if not args.pretrained:
            log.info("no --pretrained: from-scratch control")
        tr, ev = make_datasets(args, tok)
        log.info(f"train {tr.counts()}"); log.info(f"eval  {ev.counts()}")
        C = tr.n_channels
        model, tok = build_pretrained_seizure_model(args.pretrained, device, C, len(tr.type_classes), args)
    # the datasets emit the model's feature set; standardisation statistics from the training rows
    for ds_ in (tr, ev):
        ds_.feat_branch = model.feat_branch
    fit_feature_stats(model, tr, device, args)
    log.info(f"SeizureModel params {count_params(model) / 1e6:.2f}M | context {args.context_blocks} blocks | types {tr.type_classes}")
    log.info(f"  pool={args.pool} | feature_layers={args.feature_layers} | agg={args.agg} | context_mode={args.context_mode} | "
             f"feat_branch={model.feat_branch} | focal_gamma={args.focal_gamma}")
    if args.freeze_backbone:
        model.encoder.freeze()
    sampler = BalancedBatchSampler(tr.df, args.batch_size, args.min_ictal_per_batch, seed=args.seed)
    tl = make_loader(tr, batch_sampler=sampler, num_workers=args.num_workers)
    eval_bs = args.eval_batch_size or 4 * args.batch_size
    el = make_loader(ev, batch_size=eval_bs, num_workers=args.num_workers, shuffle=False)
    tl_eval = make_loader(tr, batch_size=eval_bs, num_workers=args.num_workers, shuffle=False) if args.baseline else None
    if args.eval_only:
        if args.load:
            model.load_state_dict(torch.load(args.load, map_location=device)["model"])
        res = run_eval(model, el, args, device, ev, args.baseline, tl_eval); log_eval(res, "EVAL", ev)
        json.dump(eval_record(res), open(os.path.join(args.out, "eval.json"), "w"), indent=1)
        return
    # class weights / the wave-1 recipe (training/recipe.py): balanced batches, prior-initialised heads,
    # LP-FT with layer-wise lr decay, EMA weights, type head off when the training split has one seizure type.
    from eeg_gpt.training import recipe
    from eeg_gpt.models.gpt import init_heads
    d = tr.df; ypos = int((d["_cat"] == "preictal").sum()); yneg = int((d["_cat"] == "background").sum())
    type_on = recipe.type_head_enabled(getattr(args, "type_head", "on"), d)
    sampler.set_balance(getattr(args, "preictal_frac", 0.0), args.min_ictal_per_batch if type_on else 0)   # before its first iteration
    comp = sampler.composition()
    pos_weight_val = recipe.resolve_pos_weight(recipe.parse_pos_weight(getattr(args, "pos_weight", "auto")), args.pos_weight_cap, comp, ypos, yneg)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    tcount = np.array([(d["_type_idx"] == i).sum() for i in range(len(tr.type_classes))], dtype=np.float64)
    tw = torch.tensor(np.where(tcount > 0, np.sqrt(tcount.sum() / np.maximum(tcount, 1)), 0.0), dtype=torch.float32, device=device)
    tw = tw / tw[tw > 0].mean() if (tw > 0).any() else tw
    log.info(f"recipe: batch = {comp['fill']} session rows + {comp['reserved_preictal']} preictal + {comp['reserved_ictal']} ictal "
             f"(expected preictal/batch {comp['expected_preictal_per_batch']:.2f}) | pos_weight {pos_weight_val:.2f} | focal gamma {args.focal_gamma} | "
             f"type head {'on' if type_on else 'off'} | type weights {tw.tolist()} | ema {getattr(args, 'ema', 0.0)}")
    init_heads(model, comp["preictal_frac_expected"], tcount if type_on else None)
    freeze_epochs = 0 if args.freeze_backbone else min(int(getattr(args, "freeze_epochs", 0)), args.epochs)
    if freeze_epochs > 0:
        recipe.freeze_trunk(model)
    steps_per_epoch = max(1, (args.steps_per_epoch or len(sampler)) // args.accum)

    def build_opt(n_epochs):
        o = make_adamw(recipe.param_groups(model, args), device, weight_decay=args.weight_decay)
        return o, recipe.make_sched(o, n_epochs * steps_per_epoch, args.warmup_frac)

    opt, sched = build_opt(freeze_epochs if freeze_epochs > 0 else args.epochs)
    log.info(f"freeze {freeze_epochs} epochs then llrd {getattr(args, 'llrd', 1.0)} | param groups: "
             + ", ".join(f"{g.get('name')}={g['lr']:.1e}" for g in opt.param_groups))
    ema = recipe.EMA(model, getattr(args, "ema", 0.0)); eval_model = ema.model
    scaler = get_amp_scaler(device, args.amp)
    autocast_kw = get_autocast_kwargs(device, args.amp)
    non_blocking = (device.type == "cuda")
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        if freeze_epochs and ep == freeze_epochs + 1:
            recipe.unfreeze_trunk(model)
            opt, sched = build_opt(args.epochs - freeze_epochs)
            log.info("trunk unfrozen; param groups: " + ", ".join(f"{g.get('name')}={g['lr']:.1e}" for g in opt.param_groups))
        model.train(); t0 = time.time(); agg = {"loss": 0.0, "bin": 0.0, "type": 0.0}; n = 0
        for i, b in enumerate(tl):
            if args.steps_per_epoch and i >= args.steps_per_epoch:
                break
            with torch.autocast(**autocast_kw):
                out = _model_forward(model, b, device, non_blocking)
            y = binary_targets(b, args.label).to(device, non_blocking=non_blocking)
            v = (b["occ_valid"].to(device, non_blocking=non_blocking) > 0.5)
            # (focal) BCE with p_t from the unweighted loss; gamma 0 = weighted BCE on the balanced batch
            if v.any():
                lb = focal_bce(out["binary_logit"].float()[v], y[v], gamma=args.focal_gamma, pos_weight=pos_weight)
            else:
                lb = out["binary_logit"].sum() * 0
            if type_on:
                ti = b["type_idx"].to(device, non_blocking=non_blocking)
                ii = (b["is_ictal"].to(device, non_blocking=non_blocking) > 0.5)
                wrow = torch.where(ii, 1.0, args.type_preictal_weight) * (ti >= 0).float()
                if (wrow > 0).any():
                    ce = F.cross_entropy(out["type_logits"].float(), ti.clamp_min(0), weight=tw, reduction="none")
                    lt = (ce * wrow).sum() / wrow.sum()
                else:
                    lt = out["type_logits"].sum() * 0
            else:
                lt = out["type_logits"].sum() * 0
            loss = (args.binary_weight * lb + args.type_weight * lt) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); ema.update(model)
            # accumulated on the device: no GPU sync per step
            agg["loss"] = agg["loss"] + loss.detach().float() * args.accum
            agg["bin"] = agg["bin"] + lb.detach().float()
            agg["type"] = agg["type"] + lt.detach().float()
            n += 1
        log.info(f"epoch {ep} | " + " ".join(f"{k}={float(v) / max(1, n):.4f}" for k, v in agg.items()) + f" | {time.time() - t0:.0f}s")
        res = run_eval(eval_model, el, args, device, ev, args.baseline and ep == 1, tl_eval)
        if not type_on:
            res["type_macro_f1_ictal"] = None; res["type_cm_ictal"] = None; res["type_pred_hist"] = None
        log_eval(res, f"epoch {ep}", ev)
        for k_ in [k_ for k_ in res if str(k_).startswith("_")]:   # run_eval's raw arrays: never serialised
            res.pop(k_)
        hist.append({"epoch": ep, **{k: v for k, v in res.items() if k not in ("per_session", "type_cm_ictal")}})
        torch.save({"model": eval_model.state_dict(), "epoch": ep, "config": vars(args)}, os.path.join(args.out, "last.pt"))
        # best.pt by the dev-split AUROC: the dev split is patient-disjoint from training and is not the test set
        if res["auroc"] == res["auroc"] and res["auroc"] > best:
            best = res["auroc"]; torch.save({"model": eval_model.state_dict(), "epoch": ep, "config": vars(args)}, os.path.join(args.out, "best.pt"))
        json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
    log.info(f"best binary AUROC {best:.4f}")


if __name__ == "__main__":
    main()