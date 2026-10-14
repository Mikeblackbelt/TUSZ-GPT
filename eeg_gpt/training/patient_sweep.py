"""Patient-specific training, 5-fold cross-validation, and candidate patient sweep.

Enables:
1. Patient-specific seizure prediction training with 5-fold cross-validation.
2. Two patient-specific initialization modes:
   - 'pretrained': fine-tune from Stage 1 pretrained EEGGPT trunk only.
   - 'general':    fine-tune from Stage 2 non-patient-specific prediction model.
   - 'both':       runs both base initializations and compares transfer gain side-by-side.
3. Candidate patient list sweep:
   - Profiles all patients in master_full.csv (sessions, preictal/bg windows, durations, ratios).
   - Determines and ranks 'ideal candidates' based on 5-fold CV viability, event counts, and recording diversity.
   - Evaluates candidate patients across 5-fold cross-validation and reports ranked candidate benchmarks.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_gpt.data.datasets import LabeledWindows, _SPLIT_PREFIXES, patient_id, session_number
from eeg_gpt.data.label_utils import (
    LABEL_CATEGORY_BACKGROUND, LABEL_CATEGORY_EXCLUDED, LABEL_CATEGORY_ICTAL, LABEL_CATEGORY_PREICTAL,
    classify_label, extract_seizure_type,
)
from eeg_gpt.data.tokenizer import AmplitudeTokenizer
from eeg_gpt.models.gpt import EEGGPT, SeizureModel, count_params, focal_bce
from eeg_gpt.training.device import (
    adapt_num_workers, get_amp_scaler, get_autocast_kwargs, get_device, make_adamw, parse_gpu_usage, resolve_amp,
    setup_device_and_hardware, shutdown_loader,
)
from eeg_gpt.training.metrics import (
    auroc, bandpower_baseline, bandpower_features, confusion, macro_f1,
    per_session_auroc, sensitivity_at_specificity, smoothed_auroc, threshold_sweep,
)

log = logging.getLogger("patient_sweep")


def generate_patient_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    strategy: str = "auto",
    seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Generates K-fold cross validation train/val split indices for a patient's DataFrame.

    Strategies:
      - 'session': splits across recording sessions (zero cross-session leakage).
      - 'stratified': stratified K-fold across rows (balances preictal/background).
      - 'auto': uses session-based splitting if every fold can have >= 1 preictal window,
                otherwise falls back to stratified window-based splitting.
    """
    n_rows = len(df)
    if n_rows < n_splits:
        raise ValueError(f"Patient has {n_rows} rows, which is fewer than n_splits={n_splits}")

    is_preictal = (df["_cat"] == LABEL_CATEGORY_PREICTAL).values
    n_pos = int(is_preictal.sum())
    sessions = sorted(df["session_key"].unique().tolist())
    n_sessions = len(sessions)

    # Check whether session-based splitting can place >= 1 preictal row in each fold
    sess_pos = {s: int(((df["session_key"] == s) & is_preictal).sum()) for s in sessions}
    pos_sessions = [s for s in sessions if sess_pos[s] > 0]
    neg_sessions = [s for s in sessions if sess_pos[s] == 0]

    can_session_split = (n_sessions >= n_splits) and (len(pos_sessions) >= n_splits)

    use_session = (strategy == "session") or (strategy == "auto" and can_session_split)

    if use_session:
        log.info(f"CV splitting strategy: session-based ({n_splits} folds across {n_sessions} sessions, {len(pos_sessions)} with seizures)")
        rng = np.random.default_rng(seed)
        shuffled_pos = list(pos_sessions)
        shuffled_neg = list(neg_sessions)
        rng.shuffle(shuffled_pos)
        rng.shuffle(shuffled_neg)

        fold_sessions: list[list[str]] = [[] for _ in range(n_splits)]
        for i, s in enumerate(shuffled_pos):
            fold_sessions[i % n_splits].append(s)
        for i, s in enumerate(shuffled_neg):
            fold_sessions[i % n_splits].append(s)

        splits = []
        session_col = df["session_key"].values
        for k in range(n_splits):
            val_sess_set = set(fold_sessions[k])
            val_idx = [i for i, s in enumerate(session_col) if s in val_sess_set]
            train_idx = [i for i, s in enumerate(session_col) if s not in val_sess_set]
            splits.append((train_idx, val_idx))
        return splits

    # Stratified window/chunk splitting
    log.info(f"CV splitting strategy: stratified window ({n_splits} folds, balancing preictal & background)")
    label_map = {LABEL_CATEGORY_BACKGROUND: 0, LABEL_CATEGORY_PREICTAL: 1, LABEL_CATEGORY_ICTAL: 2}
    cat_labels = df["_cat"].map(label_map).fillna(0).astype(int).values

    # Check if every category with samples has >= n_splits
    counts = np.bincount(cat_labels)
    valid_classes = np.where(counts > 0)[0]
    min_count = min(counts[c] for c in valid_classes)

    if min_count >= n_splits:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = [(tr.tolist(), val.tolist()) for tr, val in skf.split(df, cat_labels)]
    else:
        # Fall back to standard KFold if a class has fewer instances than n_splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = [(tr.tolist(), val.tolist()) for tr, val in kf.split(df)]

    return splits


def profile_all_patients(
    master_csv: str,
    window_s: float = 4.0,
    cv_folds: int = 5,
    min_preictal: int = 5,
    min_sessions: int = 2,
    min_background: int = 20,
    split: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Analyzes all patients in the master CSV and ranks ideal candidates."""
    df = pd.read_csv(master_csv)
    if "channel" in df.columns:
        df = df[df["channel"].astype(str).str.upper() == "TERM"]
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1", "1.0"]) | (df["is_valid"] == True)]  # noqa: E712
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == split]

    df = df.copy()
    df["_cat"] = df["label"].apply(classify_label)

    # Always from session_key: the pipeline holds patients out of pretraining by patient_id(<parquet name>), so a
    # CSV patient column (read as int it loses leading zeros: 00000254 -> 254) would silently not match it
    df["_patient_id"] = df["session_key"].apply(patient_id)

    # Use explicit session_id / session column if present, else use session_key
    if "session_id" in df.columns:
        df["_session_id"] = df["session_id"].astype(str)
    elif "session" in df.columns:
        df["_session_id"] = df["session"].astype(str)
    else:
        df["_session_id"] = df["session_key"].astype(str)

    profiles = []
    for pid, pdf in df.groupby("_patient_id"):
        pid_str = str(pid).strip()
        if not pid_str or pid_str.lower() in _SPLIT_PREFIXES or pid_str.lower() in {"unknown", "none", "proc", "raw", "nan"}:
            log.warning(f"skipping {len(pdf)} rows whose session_key gives no patient id {pid_str!r} "
                        f"(e.g. {pdf['session_key'].iloc[0]!r})")
            continue
        sess_list = sorted(pdf["_session_id"].unique().tolist(), key=session_number)
        n_sessions = len(sess_list)
        n_rows = len(pdf)
        n_pre = int((pdf["_cat"] == LABEL_CATEGORY_PREICTAL).sum())
        n_bg = int((pdf["_cat"] == LABEL_CATEGORY_BACKGROUND).sum())
        n_ict = int((pdf["_cat"] == LABEL_CATEGORY_ICTAL).sum())
        dur_mins = round((n_rows * window_s) / 60.0, 1)
        dur_hours = round(dur_mins / 60.0, 2)
        total_valid_binary = n_pre + n_bg
        ratio = round(n_pre / max(1, total_valid_binary), 4)

        # seizure amount and variety: an event = a run of ictal rows without a gap, typed by its first row
        seizure_types: dict[str, int] = {}
        sessions_with_seizure = 0
        for _, sdf in pdf[pdf["_cat"] == LABEL_CATEGORY_ICTAL].sort_values("start_time").groupby("_session_id"):
            sessions_with_seizure += 1
            new_event = (sdf["start_time"] - sdf["stop_time"].shift()).fillna(np.inf).values > 0.5 * window_s
            for lab in sdf["label"].values[new_event]:
                t = extract_seizure_type(lab)
                t = "unknown" if t == LABEL_CATEGORY_BACKGROUND else t
                seizure_types[t] = seizure_types.get(t, 0) + 1
        n_seizures = sum(seizure_types.values())

        # 5-fold CV viability: need enough preictal and background rows to populate folds
        cv_viable = (n_pre >= cv_folds) and (n_bg >= cv_folds)

        reasons = []
        if n_sessions < min_sessions:
            reasons.append(f"sessions={n_sessions} < {min_sessions}")
        if n_pre < min_preictal:
            reasons.append(f"preictal={n_pre} < {min_preictal}")
        if n_bg < min_background:
            reasons.append(f"background={n_bg} < {min_background}")
        if not cv_viable:
            reasons.append(f"cannot partition {n_pre} preictal across {cv_folds} folds")

        is_ideal = (len(reasons) == 0)

        # Composite suitability score
        if n_pre == 0 or not cv_viable:
            score = 0.0
        else:
            pre_term = math.log1p(n_pre)
            sess_term = math.log1p(n_sessions)
            bg_term = math.log1p(n_bg / 10.0)
            # Penalize extreme imbalance
            balance_term = max(0.1, 1.0 - abs(0.2 - min(0.8, max(0.01, ratio))))
            # more seizures (ideally one per CV fold) and more seizure types make a better benchmark patient
            sz_term = math.log1p(n_seizures) + 0.5 * max(0, len(seizure_types) - 1)
            score = round((pre_term * 2.5 + sess_term * 2.0 + bg_term * 0.5 + sz_term) * balance_term, 2)

        profiles.append({
            "patient_id": str(pid),
            "n_sessions": n_sessions,
            "sessions": sess_list,
            "n_rows": n_rows,
            "n_preictal": n_pre,
            "n_background": n_bg,
            "n_ictal": n_ict,
            "n_seizures": n_seizures,
            "n_seizure_types": len(seizure_types),
            "seizure_types": dict(sorted(seizure_types.items())),
            "sessions_with_seizure": sessions_with_seizure,
            "preictal_ratio": ratio,
            "duration_mins": dur_mins,
            "duration_hours": dur_hours,
            "cv_viable": cv_viable,
            "is_ideal_candidate": is_ideal,
            "reasons_not_ideal": reasons,
            "suitability_score": score,
        })

    profiles.sort(key=lambda p: (p["is_ideal_candidate"], p["suitability_score"], p["n_preictal"]), reverse=True)
    return profiles


def format_candidates_table(profiles: list[dict[str, Any]]) -> str:
    """Formats candidate profiles into a summary table string."""
    lines = [
        "=" * 140,
        f"{'Rank':<5} {'Patient ID':<12} {'Sessions':<9} {'Preictal':<9} {'Background':<11} {'Ictal':<7} {'Dur(h)':<8} {'5-Fold Viable':<14} {'Ideal?':<8} {'Score':<7} "
        f"{'Seizures':<9} Seizure types",
        "-" * 140,
    ]
    for idx, p in enumerate(profiles, start=1):
        viable_str = "Yes" if p["cv_viable"] else "No"
        ideal_str = "YES" if p["is_ideal_candidate"] else "No"
        types_str = ", ".join(f"{t} x{n}" for t, n in p.get("seizure_types", {}).items()) or "-"
        lines.append(
            f"{idx:<5} {p['patient_id']:<12} {p['n_sessions']:<9} {p['n_preictal']:<9} {p['n_background']:<11} "
            f"{p['n_ictal']:<7} {p['duration_hours']:<8} {viable_str:<14} {ideal_str:<8} {p['suitability_score']:<7.2f} "
            f"{p.get('n_seizures', 0):<9} {types_str}"
        )
    lines.append("=" * 140)
    ideal_count = sum(1 for p in profiles if p["is_ideal_candidate"])
    lines.append(f"Summary: {ideal_count} of {len(profiles)} patients identified as ideal candidates for patient-specific modeling.")
    return "\n".join(lines)


def load_general_seizure_model(
    checkpoint_path: str,
    device: torch.device,
    n_channels: int,
    n_types: int,
    window_s: float = 4.0,
    fs: float = 256.0,
    dropout: float = 0.2,
    pool: str = "meanmax",
    context_blocks: int = 4,
) -> tuple[SeizureModel, AmplitudeTokenizer]:
    """Loads a full Stage 2 non-patient-specific SeizureModel from checkpoint."""
    ck = torch.load(checkpoint_path, map_location="cpu")
    cfg = ck.get("config", {})
    state_dict = ck.get("model", ck)

    # Resolve backbone hyperparams from saved config or state dict
    tok_weight = state_dict.get("encoder.backbone.tok_emb.weight")
    if tok_weight is not None:
        n_levels, d = tok_weight.shape
    else:
        n_levels = cfg.get("n_levels", 512)
        d = cfg.get("d", 256)

    # Clip k
    clip_k = cfg.get("clip_k", 4.0)
    tok = AmplitudeTokenizer(n_levels, clip_k)

    # Count layers
    layer_keys = [k for k in state_dict.keys() if k.startswith("encoder.backbone.blocks.")]
    if layer_keys:
        layers = max(int(k.split(".")[3]) for k in layer_keys) + 1
    else:
        layers = cfg.get("layers", 6)

    heads = cfg.get("heads", 4)
    ffn_w = state_dict.get("encoder.backbone.blocks.0.ffn.0.weight")
    ffn = ffn_w.shape[0] if ffn_w is not None else cfg.get("ffn", 4 * d)
    block_size = cfg.get("block_size", int(round(window_s * fs)))
    rope = cfg.get("rope", "encoder.backbone.pos_emb.weight" not in state_dict)
    value_emb = cfg.get("value_emb", "encoder.backbone.val_proj.weight" in state_dict)
    tie_weights = cfg.get("tie_weights", True)
    pred_offsets_raw = cfg.get("pred_offsets", [1])
    pred_offsets = [int(o) for o in str(pred_offsets_raw).split(",")] if isinstance(pred_offsets_raw, str) else list(pred_offsets_raw)

    # Check if pretrained checkpoint is available for original pretrain config
    pre_path = cfg.get("pretrained")
    if pre_path and os.path.exists(pre_path):
        pre_ck = torch.load(pre_path, map_location="cpu")
        pre_cfg = pre_ck.get("config", {})
        d = pre_cfg.get("d", d)
        layers = pre_cfg.get("layers", layers)
        heads = pre_cfg.get("heads", heads)
        ffn = pre_cfg.get("ffn", ffn)
        block_size = pre_cfg.get("block_size", block_size)
        rope = pre_cfg.get("rope", rope)
        value_emb = pre_cfg.get("value_emb", value_emb)
        tie_weights = pre_cfg.get("tie_weights", tie_weights)
        if "pred_offsets" in pre_cfg:
            po = pre_cfg["pred_offsets"]
            pred_offsets = [int(o) for o in str(po).split(",")] if isinstance(po, str) else list(po)

    backbone = EEGGPT(
        n_levels=n_levels,
        n_channels=n_channels,
        d=d,
        layers=layers,
        heads=heads,
        ffn=ffn,
        block_size=block_size,
        dropout=dropout,
        pred_offsets=pred_offsets,
        rope=rope,
        value_emb=value_emb,
        tie_weights=tie_weights,
    )

    ctx_blocks = cfg.get("context_blocks", context_blocks)
    model_pool = cfg.get("pool", pool)
    model = SeizureModel(backbone, n_channels, n_types, context_blocks=ctx_blocks, dropout=dropout, pool=model_pool)

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"General model loaded with missing keys: {missing}")
    if unexpected:
        log.warning(f"General model loaded with unexpected keys: {unexpected}")

    model.to(device)
    log.info(f"Loaded general non-patient-specific model from {checkpoint_path} (epoch {ck.get('epoch', '?')})")
    return model, tok


def build_pretrained_seizure_model(
    pretrained_path: Optional[str],
    device: torch.device,
    n_channels: int,
    n_types: int,
    args: Any,
) -> tuple[SeizureModel, AmplitudeTokenizer]:
    """Builds a SeizureModel from Stage 1 pretrained EEGGPT trunk or from scratch."""
    if pretrained_path and os.path.exists(pretrained_path):
        ck = torch.load(pretrained_path, map_location="cpu")
        cfg = ck.get("config", {})
        n_levels = cfg.get("n_levels", 512)
        clip_k = cfg.get("clip_k", 4.0)
        tok = AmplitudeTokenizer(n_levels, clip_k)
        pred_offsets_raw = cfg.get("pred_offsets", [1, 8, 32])
        pred_offsets = [int(o) for o in str(pred_offsets_raw).split(",")] if isinstance(pred_offsets_raw, str) else list(pred_offsets_raw)
        backbone = EEGGPT(
            n_levels=n_levels,
            n_channels=cfg.get("n_channels", n_channels),
            d=cfg.get("d", 256),
            layers=cfg.get("layers", 6),
            heads=cfg.get("heads", 4),
            ffn=cfg.get("ffn", 1024),
            block_size=cfg.get("block_size", 1024),
            dropout=args.dropout,
            pred_offsets=pred_offsets,
            rope=cfg.get("rope", True),
            value_emb=cfg.get("value_emb", True),
            tie_weights=cfg.get("tie_weights", True),
        )
        backbone.load_state_dict(ck["model"])
        log.info(f"Loaded pretrained trunk from {pretrained_path} (step {ck.get('step', '?')})")
    else:
        tok = AmplitudeTokenizer(getattr(args, "n_levels", 512), getattr(args, "clip_k", 4.0))
        d = getattr(args, "d", 128)
        layers = getattr(args, "layers", 4)
        heads = getattr(args, "heads", 4)
        ffn = getattr(args, "ffn", 512)
        backbone = EEGGPT(
            getattr(args, "n_levels", 512), n_channels, d, layers, heads, ffn,
            int(round(args.window_s * args.fs)), args.dropout, (1,)
        )
        log.info("Initialized backbone from scratch (no pretrained checkpoint)")

    model = SeizureModel(backbone, n_channels, n_types, args.context_blocks, args.dropout, args.pool).to(device)
    return model, tok


def run_one_fold_training(
    fold_idx: int,
    tr_ds: LabeledWindows,
    ev_ds: LabeledWindows,
    model: SeizureModel,
    args: Any,
    device: torch.device,
    fold_out_dir: str,
) -> dict[str, Any]:
    """Runs supervised fine-tuning and evaluation for a single fold."""
    from eeg_gpt.training.finetune import BalancedBatchSampler, make_loader, run_eval

    os.makedirs(fold_out_dir, exist_ok=True)
    if args.freeze_backbone:
        model.encoder.freeze()

    sampler = BalancedBatchSampler(tr_ds.df, args.batch_size, args.min_ictal_per_batch, seed=args.seed + fold_idx)
    tl = make_loader(tr_ds, batch_sampler=sampler, num_workers=args.num_workers)
    eval_bs = args.eval_batch_size or 4 * args.batch_size
    el = make_loader(ev_ds, batch_size=eval_bs, num_workers=args.num_workers, shuffle=False)
    tl_eval = make_loader(tr_ds, batch_size=eval_bs, num_workers=args.num_workers, shuffle=False) if args.baseline else None
    try:
        return _fit_fold(tr_ds, ev_ds, model, args, device, fold_out_dir, sampler, tl, el, tl_eval, run_eval)
    finally:
        # persistent workers would otherwise pile up (and hold their pipes open) across folds and patients
        for loader in (tl, el, tl_eval):
            shutdown_loader(loader)


def _fit_fold(tr_ds, ev_ds, model, args, device, fold_out_dir, sampler, tl, el, tl_eval, run_eval) -> dict[str, Any]:
    # Class weights
    d = tr_ds.df
    ypos = ((d["_cat"] == LABEL_CATEGORY_PREICTAL)).sum()
    yneg = (d["_cat"] == LABEL_CATEGORY_BACKGROUND).sum()
    pos_weight = torch.tensor([min(args.pos_weight_cap, max(1.0, yneg / max(1, ypos)))], device=device)
    tcount = np.array([(d["_type_idx"] == i).sum() for i in range(len(tr_ds.type_classes))], dtype=np.float64)
    tw = torch.tensor(np.where(tcount > 0, np.sqrt(tcount.sum() / np.maximum(tcount, 1)), 0.0), dtype=torch.float32, device=device)
    tw = tw / tw[tw > 0].mean() if (tw > 0).any() else tw

    bb = [p_ for n, p_ in model.named_parameters() if n.startswith("encoder.backbone") and p_.requires_grad]
    rest = [p_ for n, p_ in model.named_parameters() if not n.startswith("encoder.backbone") and p_.requires_grad]
    groups = [{"params": rest, "lr": args.head_lr}] + ([{"params": bb, "lr": args.lr}] if bb else [])
    opt = make_adamw(groups, device, weight_decay=args.weight_decay)

    total_steps = max(1, args.epochs * (args.steps_per_epoch or len(sampler)) // args.accum)
    warm_steps = max(1, int(round(args.warmup_frac * total_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )
    scaler = get_amp_scaler(device, args.amp)
    autocast_kw = get_autocast_kwargs(device, args.amp)
    non_blocking = (device.type == "cuda")

    best_auroc = -1.0
    best_res: dict[str, Any] = {}

    for ep in range(1, args.epochs + 1):
        model.train()
        for i, b in enumerate(tl):
            if args.steps_per_epoch and i >= args.steps_per_epoch:
                break
            tokens = b["tokens"].to(device, non_blocking=non_blocking)
            with torch.autocast(**autocast_kw):
                out = model(tokens)

            y = b["horizon_pos" if args.label == "horizon" else "occ"].to(device, non_blocking=non_blocking)
            v = (b["occ_valid"].to(device, non_blocking=non_blocking) > 0.5)
            if v.any():
                lb = focal_bce(out["binary_logit"].float()[v], y[v], gamma=args.focal_gamma, pos_weight=pos_weight)
            else:
                lb = out["binary_logit"].sum() * 0

            ti = b["type_idx"].to(device, non_blocking=non_blocking)
            ii = (b["is_ictal"].to(device, non_blocking=non_blocking) > 0.5)
            wrow = torch.where(ii, 1.0, args.type_preictal_weight) * (ti >= 0).float()
            if (wrow > 0).any():
                ce = F.cross_entropy(out["type_logits"].float(), ti.clamp_min(0), weight=tw, reduction="none")
                lt = (ce * wrow).sum() / wrow.sum()
            else:
                lt = out["type_logits"].sum() * 0

            loss = (args.binary_weight * lb + args.type_weight * lt) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        # Evaluate fold
        res = run_eval(model, el, args, device, ev_ds, args.baseline and ep == 1, tl_eval)
        cur_auroc = res.get("auroc", 0.0)
        if not math.isnan(cur_auroc) and cur_auroc > best_auroc:
            best_auroc = cur_auroc
            best_res = copy.deepcopy(res)
            best_res["best_epoch"] = ep
            torch.save({"model": model.state_dict(), "epoch": ep, "config": vars(args)}, os.path.join(fold_out_dir, "best.pt"))

    if not best_res:
        best_res = run_eval(model, el, args, device, ev_ds, False)
        best_res["best_epoch"] = args.epochs

    return best_res


def get_tokenizer_for_run(args: Any, base_mode: str = "pretrained", general_checkpoint: Optional[str] = None) -> AmplitudeTokenizer:
    """Derives tokenizer configuration matching the model checkpoint or args."""
    if base_mode == "general":
        ckpt_path = general_checkpoint or getattr(args, "general_checkpoint", None) or getattr(args, "load", None)
        if ckpt_path and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location="cpu")
            cfg = ck.get("config", {})
            state_dict = ck.get("model", ck)
            tok_w = state_dict.get("encoder.backbone.tok_emb.weight")
            n_levels = tok_w.shape[0] if tok_w is not None else cfg.get("n_levels", 512)
            clip_k = cfg.get("clip_k", 4.0)
            return AmplitudeTokenizer(n_levels, clip_k)
    elif base_mode == "pretrained":
        if getattr(args, "pretrained", None) and os.path.exists(args.pretrained):
            ck = torch.load(args.pretrained, map_location="cpu")
            cfg = ck.get("config", {})
            state_dict = ck.get("model", ck)
            tok_w = state_dict.get("tok_emb.weight")
            n_levels = tok_w.shape[0] if tok_w is not None else cfg.get("n_levels", 512)
            clip_k = cfg.get("clip_k", 4.0)
            return AmplitudeTokenizer(n_levels, clip_k)
    return AmplitudeTokenizer(getattr(args, "n_levels", 512), getattr(args, "clip_k", 4.0))


def run_patient_cv(
    args: Any,
    target_patient: str,
    base_mode: str = "pretrained",
    general_checkpoint: Optional[str] = None,
    out_dir: Optional[str] = None,
    pretrained_per_fold: Optional[list] = None,
) -> dict[str, Any]:
    """Runs 5-fold cross validation for a single patient.

    `pretrained_per_fold` (training/pipeline.py) starts fold k from the k-th Stage 1
    checkpoint (cycled) instead of the single `args.pretrained`.
    """
    if pretrained_per_fold:
        args.pretrained = pretrained_per_fold[0]   # tokenizer settings; identical across pretraining folds
    out_dir = out_dir or args.out
    os.makedirs(out_dir, exist_ok=True)
    device = get_device(args.device)
    setup_device_and_hardware(device, args.max_gpu_usage)
    args.num_workers = adapt_num_workers(args.num_workers, n_loaders=3)
    args.amp = resolve_amp(args.amp, device)

    log.info(f"=== Starting {args.cv_folds}-Fold CV for Patient {target_patient} [Base: {base_mode}] ===")

    tok = get_tokenizer_for_run(args, base_mode=base_mode, general_checkpoint=general_checkpoint)

    # Load full dataset for this patient to establish schema and folds
    kw = dict(stage=args.stage, window_s=args.window_s, fs=args.fs, context_blocks=args.context_blocks,
              horizon_s=args.horizon_s, type_grouping=args.type_grouping, tokenizer=tok, cache_capacity=args.cache_capacity)

    patient_full_ds = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, patient=target_patient, **kw)
    counts = patient_full_ds.counts()
    log.info(f"Patient {target_patient} summary: {counts}")

    if counts["rows"] < args.cv_folds:
        raise ValueError(f"Patient {target_patient} has {counts['rows']} rows, cannot do {args.cv_folds}-fold CV.")

    splits = generate_patient_cv_splits(
        patient_full_ds.df,
        n_splits=args.cv_folds,
        strategy=getattr(args, "cv_strategy", "auto"),
        seed=args.seed
    )

    n_channels = patient_full_ds.n_channels
    n_types = len(patient_full_ds.type_classes)

    fold_results: list[dict[str, Any]] = []

    for fold_k, (tr_idx, ev_idx) in enumerate(splits, start=1):
        log.info(f"--- Patient {target_patient} | Fold {fold_k}/{args.cv_folds} (train={len(tr_idx)}, eval={len(ev_idx)}) ---")
        tr_ds = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, patient=target_patient, row_indices=tr_idx, **kw)
        ev_ds = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, patient=target_patient, row_indices=ev_idx, **kw)

        # Build fresh model instance for this fold according to base mode
        if base_mode == "general":
            ckpt_path = general_checkpoint or getattr(args, "general_checkpoint", None) or args.load
            if not ckpt_path:
                raise ValueError("base_mode='general' requires --general-checkpoint or --load")
            model, _ = load_general_seizure_model(
                ckpt_path, device, n_channels, n_types,
                window_s=args.window_s, fs=args.fs, dropout=args.dropout,
                pool=args.pool, context_blocks=args.context_blocks
            )
        else:  # 'pretrained' or scratch
            fold_pretrained = (pretrained_per_fold[(fold_k - 1) % len(pretrained_per_fold)] if pretrained_per_fold
                               else getattr(args, "pretrained", None))
            model, _ = build_pretrained_seizure_model(fold_pretrained, device, n_channels, n_types, args)

        fold_dir = os.path.join(out_dir, f"fold_{fold_k}")
        res = run_one_fold_training(fold_k, tr_ds, ev_ds, model, args, device, fold_dir)

        auroc_val = res.get("auroc", float("nan"))
        sw = res.get("sweep", {})
        ba_val = sw.get("balanced_accuracy", float("nan"))
        ss = res.get("sens_at_spec", {})
        log.info(f"Fold {fold_k} Result | AUROC={auroc_val:.4f} | Sweep BA={ba_val:.3f} | sens@spec: {ss}")

        fold_record = {
            "fold": fold_k,
            "train_rows": len(tr_idx),
            "eval_rows": len(ev_idx),
            "eval_pos": res.get("n_pos", 0),
            "auroc": auroc_val,
            "sweep_ba": ba_val,
            "sweep_sens": sw.get("sensitivity", float("nan")),
            "sweep_spec": sw.get("specificity", float("nan")),
            "sens_at_spec": ss,
            "type_macro_f1": res.get("type_macro_f1_ictal", float("nan")),
            "best_epoch": res.get("best_epoch"),
        }
        if base_mode != "general":
            fold_record["pretrained"] = fold_pretrained
        fold_results.append(fold_record)

    # Compute aggregate cross-validation statistics
    valid_aurocs = [r["auroc"] for r in fold_results if not math.isnan(r["auroc"])]
    valid_bas = [r["sweep_ba"] for r in fold_results if not math.isnan(r["sweep_ba"])]
    valid_s80 = [r["sens_at_spec"].get("spec80", float("nan")) for r in fold_results if "spec80" in r["sens_at_spec"] and not math.isnan(r["sens_at_spec"]["spec80"])]
    valid_s90 = [r["sens_at_spec"].get("spec90", float("nan")) for r in fold_results if "spec90" in r["sens_at_spec"] and not math.isnan(r["sens_at_spec"]["spec90"])]

    summary = {
        "patient_id": target_patient,
        "base_mode": base_mode,
        "cv_folds": args.cv_folds,
        "mean_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else float("nan"),
        "std_auroc": float(np.std(valid_aurocs)) if valid_aurocs else float("nan"),
        "mean_sweep_ba": float(np.mean(valid_bas)) if valid_bas else float("nan"),
        "std_sweep_ba": float(np.std(valid_bas)) if valid_bas else float("nan"),
        "mean_sens_at_spec80": float(np.mean(valid_s80)) if valid_s80 else float("nan"),
        "mean_sens_at_spec90": float(np.mean(valid_s90)) if valid_s90 else float("nan"),
        "folds": fold_results,
    }

    log.info(
        f"=== Patient {target_patient} [{base_mode}] 5-Fold CV Summary: "
        f"AUROC = {summary['mean_auroc']:.4f} ± {summary['std_auroc']:.4f} | "
        f"BA = {summary['mean_sweep_ba']:.3f} ± {summary['std_sweep_ba']:.3f} ==="
    )

    json.dump(summary, open(os.path.join(out_dir, "cv_summary.json"), "w"), indent=2)
    return summary


def run_patient_sweep(args: Any) -> list[dict[str, Any]]:
    """Sweeps all patients in the master CSV, profiles ideal candidates, and benchmarks them."""
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "sweep.log"))],
    )

    log.info("Starting candidate patient sweep...")
    profiles = profile_all_patients(
        master_csv=args.master_csv,
        window_s=args.window_s,
        cv_folds=args.cv_folds,
        min_preictal=args.min_preictal,
        min_sessions=args.min_sessions,
        min_background=args.min_background,
        split=getattr(args, "split", getattr(args, "train_split", None)),
    )

    # Save profiles
    json.dump(profiles, open(os.path.join(args.out, "candidates_profile.json"), "w"), indent=2)
    pd.DataFrame(profiles).to_csv(os.path.join(args.out, "candidates_profile.csv"), index=False)

    tbl = format_candidates_table(profiles)
    print("\n" + tbl + "\n")
    log.info("\n" + tbl)

    if getattr(args, "sweep_profile_only", False):
        log.info("Profile-only flag enabled. Exiting sweep without training.")
        return profiles

    # Filter candidates to evaluate
    ideal_candidates = [p for p in profiles if p["is_ideal_candidate"]]
    if not ideal_candidates:
        log.warning("No patients met strict ideal candidate criteria. Evaluating top scoring candidates instead.")
        candidates_to_run = profiles[: max(1, getattr(args, "sweep_top_k", 3))]
    else:
        top_k = getattr(args, "sweep_top_k", 0)
        candidates_to_run = ideal_candidates[:top_k] if top_k > 0 else ideal_candidates

    log.info(f"Evaluating {len(candidates_to_run)} candidate patients: {[c['patient_id'] for c in candidates_to_run]}")

    base_modes = []
    if args.patient_base in ["pretrained", "both"]:
        base_modes.append("pretrained")
    if args.patient_base in ["general", "both"]:
        base_modes.append("general")

    benchmark_records: list[dict[str, Any]] = []

    for c in candidates_to_run:
        pid = c["patient_id"]
        c_res: dict[str, Any] = {
            "patient_id": pid,
            "n_sessions": c["n_sessions"],
            "n_preictal": c["n_preictal"],
            "n_background": c["n_background"],
            "suitability_score": c["suitability_score"],
        }

        for mode in base_modes:
            mode_out = os.path.join(args.out, f"patient_{pid}", mode)
            res = run_patient_cv(
                args=args,
                target_patient=pid,
                base_mode=mode,
                general_checkpoint=getattr(args, "general_checkpoint", None),
                out_dir=mode_out,
            )
            c_res[f"{mode}_auroc_mean"] = res["mean_auroc"]
            c_res[f"{mode}_auroc_std"] = res["std_auroc"]
            c_res[f"{mode}_ba_mean"] = res["mean_sweep_ba"]
            c_res[f"{mode}_ba_std"] = res["std_sweep_ba"]

        if "pretrained" in base_modes and "general" in base_modes:
            diff = c_res.get("general_auroc_mean", 0.0) - c_res.get("pretrained_auroc_mean", 0.0)
            c_res["delta_auroc (gen - pre)"] = round(diff, 4)
            c_res["best_base"] = "general" if diff > 0 else "pretrained"

        benchmark_records.append(c_res)

    # Save benchmark records
    json.dump(benchmark_records, open(os.path.join(args.out, "patient_sweep_results.json"), "w"), indent=2)
    pd.DataFrame(benchmark_records).to_csv(os.path.join(args.out, "patient_sweep_results.csv"), index=False)

    # Log summary benchmark table
    log.info("\n" + "=" * 90)
    log.info("                         PATIENT BENCHMARK SWEEP SUMMARY (5-FOLD CV)")
    log.info("=" * 90)
    for r in benchmark_records:
        pid = r["patient_id"]
        pre_str = f"{r.get('pretrained_auroc_mean', 0.0):.3f}±{r.get('pretrained_auroc_std', 0.0):.3f}" if "pretrained" in base_modes else "N/A"
        gen_str = f"{r.get('general_auroc_mean', 0.0):.3f}±{r.get('general_auroc_std', 0.0):.3f}" if "general" in base_modes else "N/A"
        delta_str = f"{r.get('delta_auroc (gen - pre)', 0.0):+.3f}" if "delta_auroc (gen - pre)" in r else "N/A"
        winner = r.get("best_base", "")
        log.info(f"Patient {pid:<10} | Pretrained: {pre_str:<12} | General: {gen_str:<12} | Delta: {delta_str:<8} | Winner: {winner}")
    log.info("=" * 90)

    return benchmark_records


def build_parser():
    p = argparse.ArgumentParser(description="Sweep patient list and perform patient-specific 5-fold cross-validation")
    p.add_argument("--master-csv", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--stage", default="proc")
    p.add_argument("--out", required=True)

    # Base model configuration
    p.add_argument("--patient-base", choices=["pretrained", "general", "both"], default="pretrained",
                   help="Patient-specific initialization: 'pretrained' (Stage 1 backbone only), 'general' (Stage 2 multi-patient model), or 'both'.")
    p.add_argument("--pretrained", default=None, help="Stage 1 pretrained backbone checkpoint (best.pt)")
    p.add_argument("--general-checkpoint", default=None, help="Stage 2 general non-patient-specific model checkpoint")
    p.add_argument("--load", default=None, help="Checkpoint to load (alias for general-checkpoint or eval-only)")

    # Patient target or sweep mode
    p.add_argument("--patient", default=None, help="Target a specific patient ID for training/CV")
    p.add_argument("--adapt-patient", default=None, help="Alias for --patient")
    p.add_argument("--sweep-patients", action="store_true", help="Sweep all patients in master CSV to find ideal candidates")
    p.add_argument("--sweep-profile-only", action="store_true", help="Profile and rank candidates without training")
    p.add_argument("--sweep-top-k", type=int, default=5, help="Number of top candidates to benchmark in sweep (0 for all)")
    p.add_argument("--split", default=None, help="Filter master CSV to a specific split (e.g. 'train' or 'trn')")

    # Candidate criteria
    p.add_argument("--min-preictal", type=int, default=5, help="Minimum preictal windows for an ideal candidate (default 5 for 5-fold CV)")
    p.add_argument("--min-sessions", type=int, default=2, help="Minimum recording sessions for an ideal candidate")
    p.add_argument("--min-background", type=int, default=20, help="Minimum background windows for an ideal candidate")

    # Cross validation settings
    p.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default 5)")
    p.add_argument("--cv-strategy", choices=["auto", "session", "stratified"], default="auto")

    # Task & architecture settings
    p.add_argument("--label", choices=["preictal", "horizon"], default="preictal")
    p.add_argument("--horizon-s", type=float, default=30.0)
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--context-blocks", type=int, default=4)
    p.add_argument("--type-grouping", choices=["none", "focal_generalized", "fnsz_gnsz_other"], default="focal_generalized")
    p.add_argument("--type-weight", type=float, default=1.0)
    p.add_argument("--type-preictal-weight", type=float, default=0.0)
    p.add_argument("--min-ictal-per-batch", type=int, default=4)
    p.add_argument("--binary-weight", type=float, default=1.0)
    p.add_argument("--pos-weight-cap", type=float, default=50.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pool", choices=["mean", "last", "meanmax"], default="meanmax")
    p.add_argument("--smooth-k", type=int, default=15)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-capacity", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-gpu-usage", type=parse_gpu_usage, default=None)
    p.add_argument("--seed", type=int, default=0)

    # Scratch model parameters
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn", type=int, default=512)
    p.add_argument("--n-levels", type=int, default=512)
    p.add_argument("--clip-k", type=float, default=4.0)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.adapt_patient and not args.patient:
        args.patient = args.adapt_patient
    if args.load and not args.general_checkpoint:
        args.general_checkpoint = args.load

    if args.patient:
        run_patient_cv(args, args.patient, base_mode=args.patient_base, general_checkpoint=args.general_checkpoint)
    else:
        run_patient_sweep(args)


if __name__ == "__main__":
    main()
