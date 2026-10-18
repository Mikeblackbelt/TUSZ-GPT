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
import inspect
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_gpt.data.datasets import LabeledWindows, _SPLIT_PREFIXES, load_master, patient_id, session_number
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


BACKGROUND_SEGMENT_S = 600.0     # event strategy: contiguous background is cut into groups of at most 10 min


def _cv_groups(df: pd.DataFrame, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Group id per row for the event strategy, and which groups are positive (carry preictal rows).

    A positive group is one seizure: its ictal rows (`_event_id`) plus the preictal rows that lead to it
    (`_next_event_start` == the event's onset). Preictal rows with no following event (seizure rows missing from the
    CSV) form their own positive groups per contiguous run. Background rows form contiguous segments per session,
    cut at gaps and every BACKGROUND_SEGMENT_S seconds, so no fold gets one unbroken hour of a single state.
    """
    n = len(df)
    sess = df["session_key"].astype(str).values
    start = df["start_time"].values.astype(np.float64)
    stop = df["stop_time"].values.astype(np.float64)
    cat = df["_cat"].values
    eid = df["_event_id"].values.astype(np.int64)
    estart = df["_event_start"].values.astype(np.float64)
    nxt = df["_next_event_start"].values.astype(np.float64)
    gap = 0.5 * float(window_s)
    group = np.full(n, -1, dtype=np.int64)
    positive: list[bool] = []
    key_of_event: dict = {}
    order = np.lexsort((start, sess))
    bounds = np.flatnonzero(np.r_[True, sess[order][1:] != sess[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        pos = order[a:b]
        # ictal rows -> the event's group; preictal rows -> the group of the event they precede
        starts = {}
        for i in pos[eid[pos] >= 0]:
            key = ("event", int(eid[i]))
            if key not in key_of_event:
                key_of_event[key] = len(positive); positive.append(True)
            group[i] = key_of_event[key]
            starts.setdefault(float(estart[i]), key)
        run_prev = None
        for i in pos:
            if group[i] >= 0:
                run_prev = None
                continue
            if cat[i] == LABEL_CATEGORY_PREICTAL and np.isfinite(nxt[i]) and nxt[i] in starts:
                group[i] = key_of_event[starts[nxt[i]]]
                run_prev = None
                continue
            is_pos = cat[i] == LABEL_CATEGORY_PREICTAL
            # background segments (and orphan preictal runs): new group at a gap, a state change or every 10 min
            if run_prev is None or run_prev[0] != is_pos or start[i] - run_prev[1] > gap or start[i] - run_prev[2] >= BACKGROUND_SEGMENT_S:
                run_prev = (is_pos, stop[i], start[i], len(positive)); positive.append(is_pos)
            group[i] = run_prev[3]
            run_prev = (is_pos, stop[i], run_prev[2], run_prev[3])
    return group, np.array(positive, dtype=bool)


def _assign_groups(group: np.ndarray, positive: np.ndarray, is_pre: np.ndarray, n_splits: int, rng) -> np.ndarray:
    """Fold per group: positive groups first (most preictal rows first) to the fold with the fewest preictal rows,
    then background groups to the fold with the fewest rows. Ties broken by the seeded shuffle."""
    n_groups = len(positive)
    rows = np.bincount(group, minlength=n_groups)
    pre = np.bincount(group, weights=is_pre.astype(np.float64), minlength=n_groups)
    fold_of = np.full(n_groups, -1, dtype=np.int64)
    fold_pre = np.zeros(n_splits); fold_rows = np.zeros(n_splits)
    pos_ids = np.flatnonzero(positive); rng.shuffle(pos_ids)
    for g in sorted(pos_ids, key=lambda g: -pre[g]):
        k = int(np.argmin(fold_pre + 1e-6 * fold_rows))
        fold_of[g] = k; fold_pre[k] += pre[g]; fold_rows[k] += rows[g]
    neg_ids = np.flatnonzero(~positive); rng.shuffle(neg_ids)
    for g in sorted(neg_ids, key=lambda g: -rows[g]):
        k = int(np.argmin(fold_rows))
        fold_of[g] = k; fold_rows[k] += rows[g]
    return fold_of


def _context_margin(df: pd.DataFrame, train_idx: np.ndarray, held_idx: np.ndarray, margin_s: float) -> np.ndarray:
    """Rows of `held_idx` whose context (the `margin_s` seconds before the window) overlaps a training row of the
    same session; they are dropped from the held-out side so no evaluated input contains training-time signal."""
    if margin_s <= 0 or len(train_idx) == 0 or len(held_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    sess = df["session_key"].astype(str).values
    start = df["start_time"].values.astype(np.float64)
    stop = df["stop_time"].values.astype(np.float64)
    drop = []
    tr_by_sess: dict = {}
    for i in train_idx:
        tr_by_sess.setdefault(sess[i], []).append(i)
    for s, idx in tr_by_sess.items():
        idx = np.array(idx); o = np.argsort(start[idx]); idx = idx[o]
        tr_start = start[idx]; tr_stop_cummax = np.maximum.accumulate(stop[idx])
        held = held_idx[sess[held_idx] == s]
        for i in held:
            j = np.searchsorted(tr_start, start[i], side="left") - 1     # training rows starting before this one
            if j >= 0 and tr_stop_cummax[j] > start[i] - margin_s:       # ...ending inside this row's context
                drop.append(i)
    return np.array(sorted(drop), dtype=np.int64)


def generate_patient_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    strategy: str = "auto",
    seed: int = 0,
    context_blocks: int = 1,
    window_s: float = 4.0,
    inner_val: bool = True,
) -> list[dict[str, Any]]:
    """K-fold cross-validation plan for one patient's LabeledWindows frame (positional row indices).

    Returns one dict per fold: {"train": [...], "val": [...] | None, "eval": [...], "strategy": "session"|"event",
    "info": {...}}. `val` is an INNER validation part taken out of the training rows (epoch and threshold selection
    happen there, never on `eval`); it is None when the training part has fewer than 3 seizure sessions (session
    strategy) or 3 seizure groups (event strategy), or with inner_val=False.

    Strategies:
      - 'session': folds are whole recording sessions (no leakage at all); needs >= n_splits sessions with seizures.
      - 'event':   folds are whole seizures (preictal + ictal rows of one event) and 10-minute background segments,
                   so overlapping windows of one seizure never sit on both sides; rows whose context
                   (context_blocks * window_s seconds) would reach into training time are dropped from the held-out
                   side.
      - 'auto':    'session' when possible, else 'event'.
    The old row-stratified fallback leaked (neighbouring 4 s windows of the same seizure on both sides, context
    blocks on top) and was removed; asking for it raises.
    """
    if strategy not in ("auto", "session", "event"):
        raise ValueError(f"cv strategy {strategy!r}: row-stratified folds leak between neighbouring windows of one seizure "
                         f"and were removed; use 'session', 'event' or 'auto'")
    n_rows = len(df)
    if n_rows < n_splits:
        raise ValueError(f"Patient has {n_rows} rows, which is fewer than n_splits={n_splits}")
    if "_event_id" not in df.columns:      # frames not produced by load_master (tests, old callers)
        from eeg_gpt.data.events import events_from_rows
        df = events_from_rows(df, window_s)
    rng = np.random.default_rng(seed)
    is_pre = (df["_cat"] == LABEL_CATEGORY_PREICTAL).values
    sess_col = df["session_key"].astype(str).values
    sessions = sorted(set(sess_col.tolist()))
    sess_pos = {s: int(is_pre[sess_col == s].sum()) for s in sessions}
    pos_sessions = [s for s in sessions if sess_pos[s] > 0]
    neg_sessions = [s for s in sessions if sess_pos[s] == 0]
    use_session = strategy == "session" or (strategy == "auto" and len(pos_sessions) >= n_splits)
    margin_s = max(0, int(context_blocks) - 1) * float(window_s)
    splits: list[dict[str, Any]] = []

    if use_session:
        log.info(f"CV strategy: session ({n_splits} folds over {len(sessions)} sessions, {len(pos_sessions)} with seizures)")
        if len(pos_sessions) < n_splits:
            log.warning(f"only {len(pos_sessions)} sessions with seizures for {n_splits} folds: some folds have no positives (AUROC NaN)")
        shuffled_pos = list(pos_sessions); shuffled_neg = list(neg_sessions)
        rng.shuffle(shuffled_pos); rng.shuffle(shuffled_neg)
        fold_sessions: list[list[str]] = [[] for _ in range(n_splits)]
        for i, s in enumerate(shuffled_pos):
            fold_sessions[i % n_splits].append(s)
        for i, s in enumerate(shuffled_neg):
            fold_sessions[i % n_splits].append(s)
        for k in range(n_splits):
            ev_set = set(fold_sessions[k])
            train_sess = [s for s in sessions if s not in ev_set]
            val_set: set = set()
            if inner_val:
                # >= 2 sessions so the validation AUROC ranks across sessions (one session's AUROC is that
                # session's quirks, and it drove selection to epoch 1): 2 seizure sessions when >= 4 remain,
                # else 1, plus one seizure-free session when there is one
                cand = [s for s in train_sess if sess_pos[s] > 0]
                negs = [s for s in train_sess if sess_pos[s] == 0]
                if len(cand) >= 3:
                    n_val = 2 if len(cand) >= 4 else 1
                    val_set = {cand[int(j)] for j in rng.choice(len(cand), n_val, replace=False)}
                    if negs:
                        val_set.add(negs[int(rng.integers(len(negs)))])
            eval_idx = [i for i, s in enumerate(sess_col) if s in ev_set]
            val_idx = [i for i, s in enumerate(sess_col) if s in val_set]
            train_idx = [i for i, s in enumerate(sess_col) if s not in ev_set and s not in val_set]
            splits.append({"train": train_idx, "val": val_idx if val_set else None, "eval": eval_idx, "strategy": "session",
                           "info": {"eval_sessions": sorted(ev_set), "val_sessions": sorted(val_set), "dropped_margin": 0}})
        return splits

    group, positive = _cv_groups(df, window_s)
    n_pos_groups = int(positive.sum())
    log.info(f"CV strategy: event ({n_splits} folds over {n_pos_groups} seizure groups and {int((~positive).sum())} background "
             f"segments; {len(pos_sessions)} sessions with seizures < {n_splits})")
    if n_pos_groups < n_splits:
        log.warning(f"only {n_pos_groups} seizure groups for {n_splits} folds: some folds have no positives (AUROC NaN)")
    fold_of = _assign_groups(group, positive, is_pre, n_splits, rng)
    row_fold = fold_of[group]
    all_idx = np.arange(n_rows)
    for k in range(n_splits):
        eval_idx = all_idx[row_fold == k]
        rest = all_idx[row_fold != k]
        val_idx = np.zeros(0, dtype=np.int64)
        if inner_val:
            train_pos_groups = [g for g in np.flatnonzero(positive) if fold_of[g] != k]
            if len(train_pos_groups) >= 3:
                vg = {int(train_pos_groups[int(rng.integers(len(train_pos_groups)))])}
                # plus a share of the training background segments so the validation AUROC is defined
                neg_groups = [g for g in np.flatnonzero(~positive) if fold_of[g] != k]
                rng.shuffle(neg_groups)
                want = sum(int((group[rest] == g).sum()) for g in neg_groups) / max(1, len(train_pos_groups))
                got = 0
                for g in neg_groups:
                    if got >= want:
                        break
                    vg.add(int(g)); got += int((group == g).sum())
                val_idx = rest[np.isin(group[rest], list(vg))]
        train_idx = np.setdiff1d(rest, val_idx)
        dropped = 0
        d_eval = _context_margin(df, train_idx, eval_idx, margin_s)
        eval_idx = np.setdiff1d(eval_idx, d_eval); dropped += len(d_eval)
        if len(val_idx):
            d_val = _context_margin(df, train_idx, val_idx, margin_s)
            val_idx = np.setdiff1d(val_idx, d_val); dropped += len(d_val)
        ev_events = sorted(set(int(e) for e in df["_event_id"].values[eval_idx] if e >= 0))
        va_events = sorted(set(int(e) for e in df["_event_id"].values[val_idx] if e >= 0)) if len(val_idx) else []
        splits.append({"train": train_idx.tolist(), "val": val_idx.tolist() if len(val_idx) else None, "eval": eval_idx.tolist(),
                       "strategy": "event",
                       "info": {"eval_events": ev_events, "val_events": va_events, "eval_sessions": sorted(set(sess_col[eval_idx].tolist())),
                                "eval_groups": int(len(set(group[eval_idx].tolist()))), "dropped_margin": int(dropped)}})
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
    t0 = time.time()
    df = load_master(master_csv)
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == split]

    # Always from session_key: the pipeline holds patients out of pretraining by patient_id(<parquet name>), so a
    # CSV patient column (read as int it loses leading zeros: 00000254 -> 254) would silently not match it
    df = df.assign(_patient_id=df["_patient"])

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

        # seizure amount and variety from the event columns (data/events.py): one event = one seizure, typed by
        # its first ictal row
        ict = pdf[pdf["_event_id"] >= 0].sort_values("start_time")
        seizure_types: dict[str, int] = {}
        for lab in ict.groupby("_event_id", sort=True)["label"].first().values:
            t = extract_seizure_type(lab)
            t = "unknown" if t == LABEL_CATEGORY_BACKGROUND else t
            seizure_types[t] = seizure_types.get(t, 0) + 1
        n_events = int(ict["_event_id"].nunique())
        n_seizures = n_events
        sessions_with_seizure = int(ict["_session_id"].nunique())

        # which CV the patient gets (generate_patient_cv_splits 'auto'): whole sessions when enough sessions carry a
        # seizure, else whole seizures; a patient with fewer seizures than folds leaves folds without positives
        cv_strategy = "session" if sessions_with_seizure >= cv_folds else "event" if n_events >= cv_folds else "none"
        cv_viable = cv_strategy != "none" and n_pre >= cv_folds and n_bg >= cv_folds

        reasons = []
        if n_sessions < min_sessions:
            reasons.append(f"sessions={n_sessions} < {min_sessions}")
        if n_pre < min_preictal:
            reasons.append(f"preictal={n_pre} < {min_preictal}")
        if n_bg < min_background:
            reasons.append(f"background={n_bg} < {min_background}")
        if not cv_viable:
            reasons.append(f"{n_events} seizures in {sessions_with_seizure} sessions cannot fill {cv_folds} folds")

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
            "n_events": n_events,
            "n_seizure_types": len(seizure_types),
            "seizure_types": dict(sorted(seizure_types.items())),
            "sessions_with_seizure": sessions_with_seizure,
            "preictal_ratio": ratio,
            "duration_mins": dur_mins,
            "duration_hours": dur_hours,
            "cv_strategy": cv_strategy,
            "cv_viable": cv_viable,
            "is_ideal_candidate": is_ideal,
            "reasons_not_ideal": reasons,
            "suitability_score": score,
        })

    profiles.sort(key=lambda p: (p["is_ideal_candidate"], p["suitability_score"], p["n_preictal"]), reverse=True)
    log.info(f"profiled {len(profiles)} patients ({sum(p['is_ideal_candidate'] for p in profiles)} ideal) in {time.time() - t0:.1f}s")
    return profiles


def format_candidates_table(profiles: list[dict[str, Any]]) -> str:
    """Formats candidate profiles into a summary table string."""
    lines = [
        "=" * 150,
        f"{'Rank':<5} {'Patient ID':<12} {'Sessions':<9} {'Preictal':<9} {'Background':<11} {'Ictal':<7} {'Dur(h)':<8} {'CV':<8} {'Ideal?':<8} {'Score':<7} "
        f"{'Seizures':<9} {'Sz sess':<8} Seizure types",
        "-" * 150,
    ]
    for idx, p in enumerate(profiles, start=1):
        cv_str = p.get("cv_strategy", "yes" if p["cv_viable"] else "none")
        ideal_str = "YES" if p["is_ideal_candidate"] else "No"
        types_str = ", ".join(f"{t} x{n}" for t, n in p.get("seizure_types", {}).items()) or "-"
        lines.append(
            f"{idx:<5} {p['patient_id']:<12} {p['n_sessions']:<9} {p['n_preictal']:<9} {p['n_background']:<11} "
            f"{p['n_ictal']:<7} {p['duration_hours']:<8} {cv_str:<8} {ideal_str:<8} {p['suitability_score']:<7.2f} "
            f"{p.get('n_events', p.get('n_seizures', 0)):<9} {p.get('sessions_with_seizure', 0):<8} {types_str}"
        )
    lines.append("=" * 150)
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
    """Loads a full Stage 2 non-patient-specific SeizureModel from checkpoint.

    The architecture knobs (feature_layers, pool, agg, ..., feat_branch) are read from the checkpoint's config with
    the OLD defaults (models/gpt.py OLD_ARCH), so a checkpoint saved before those knobs existed rebuilds the exact
    module tree it was trained with; strict=False only covers genuinely new buffers.
    """
    from eeg_gpt.models.gpt import OLD_ARCH
    ck = torch.load(checkpoint_path, map_location="cpu")
    cfg = ck.get("config", {})
    state_dict = ck.get("model", ck)

    # trunk config: the pretraining checkpoint's config when it is still around, else inferred from the state dict
    trunk_cfg = {}
    pre_path = cfg.get("pretrained")
    if pre_path and os.path.exists(pre_path):
        trunk_cfg = dict(torch.load(pre_path, map_location="cpu").get("config", {}))
    else:
        tok_weight = state_dict.get("encoder.backbone.tok_emb.weight")
        if tok_weight is not None:
            trunk_cfg["n_levels"], trunk_cfg["d"] = tok_weight.shape
        else:
            trunk_cfg["n_levels"], trunk_cfg["d"] = cfg.get("n_levels", 512), cfg.get("d", 256)
        layer_keys = [k for k in state_dict.keys() if k.startswith("encoder.backbone.blocks.")]
        trunk_cfg["layers"] = (max(int(k.split(".")[3]) for k in layer_keys) + 1) if layer_keys else cfg.get("layers", 6)
        trunk_cfg["heads"] = cfg.get("heads", 4)
        ffn_w = state_dict.get("encoder.backbone.blocks.0.ffn.0.weight")
        trunk_cfg["ffn"] = ffn_w.shape[0] if ffn_w is not None else cfg.get("ffn", 4 * trunk_cfg["d"])
        trunk_cfg["block_size"] = cfg.get("block_size", int(round(window_s * fs)))
        trunk_cfg["rope"] = cfg.get("rope", "encoder.backbone.pos_emb.weight" not in state_dict)
        trunk_cfg["value_emb"] = cfg.get("value_emb", "encoder.backbone.val_proj.weight" in state_dict)
        tok_w, head_w = state_dict.get("encoder.backbone.tok_emb.weight"), state_dict.get("encoder.backbone.heads.1.weight")
        tied = (torch.equal(tok_w, head_w) if (tok_w is not None and head_w is not None and tok_w.shape == head_w.shape) else True)
        trunk_cfg["tie_weights"] = cfg.get("tie_weights", tied)
        trunk_cfg["pred_offsets"] = cfg.get("pred_offsets", [1])
        for k in ("patch", "objective", "norm_scope", "causal", "clip_k"):
            if k in cfg:
                trunk_cfg[k] = cfg[k]
    trunk_cfg.setdefault("clip_k", cfg.get("clip_k", 4.0))
    trunk_cfg["n_channels"] = n_channels
    backbone = _trunk_from_config(trunk_cfg, dropout, n_channels)
    tok = _tokenizer_from_config(trunk_cfg)

    ctx_blocks = cfg.get("context_blocks", context_blocks)
    model_pool = cfg.get("pool", pool)
    arch = {k: cfg.get(k, v) for k, v in OLD_ARCH.items()}
    feat_branch = cfg.get("feat_branch", "none") or "none"
    feat_dim = int(cfg.get("feat_dim", 0) or 0)
    if feat_branch != "none" and feat_dim <= 0:
        from eeg_gpt.data.features import feat_dim as _fd
        feat_dim = _fd(n_channels)
    bp_idx = None
    if feat_branch == "bandpower":
        from eeg_gpt.data.features import bandpower_indices
        bp_idx = bandpower_indices(n_channels)
    model = SeizureModel(backbone, n_channels, n_types, context_blocks=ctx_blocks, dropout=dropout, pool=model_pool,
                         feat_branch=feat_branch, feat_dim=feat_dim, arch=arch, bandpower_idx=bp_idx)

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"General model loaded with missing keys: {missing}")
    if unexpected:
        log.warning(f"General model loaded with unexpected keys: {unexpected}")

    model.to(device)
    log.info(f"Loaded general non-patient-specific model from {checkpoint_path} (epoch {ck.get('epoch', '?')})")
    return model, tok


def _trunk_from_config(cfg: dict, dropout: float, n_channels: int) -> EEGGPT:
    """EEGGPT from a (pretraining) config dict; EEGGPT.from_config when the trunk offers it, else today's kwargs."""
    cfg = dict(cfg)
    cfg.setdefault("n_channels", n_channels)
    if hasattr(EEGGPT, "from_config"):
        return EEGGPT.from_config(cfg, dropout=dropout)
    po = cfg.get("pred_offsets", [1, 8, 32])
    po = [int(o) for o in str(po).split(",")] if isinstance(po, str) else list(po)
    return EEGGPT(cfg.get("n_levels", 512), cfg["n_channels"], cfg.get("d", 256), cfg.get("layers", 6), cfg.get("heads", 4),
                  cfg.get("ffn", 1024), cfg.get("block_size", 1024), dropout, po, rope=cfg.get("rope", True),
                  value_emb=cfg.get("value_emb", True), tie_weights=cfg.get("tie_weights", True))


def _tokenizer_from_config(cfg: dict) -> AmplitudeTokenizer:
    if hasattr(AmplitudeTokenizer, "from_config"):
        return AmplitudeTokenizer.from_config(cfg)
    return AmplitudeTokenizer(int(cfg.get("n_levels", 512)), float(cfg.get("clip_k", 4.0)))


def arch_from_args(args: Any) -> dict:
    """The WindowEncoder / TemporalContext knobs of the --architecture flags (missing attributes -> DEFAULT_ARCH)."""
    from eeg_gpt.models.gpt import DEFAULT_ARCH
    names = {"feature_layers": "feature_layers", "agg": "agg", "agg_layers": "agg_layers", "agg_dim": "agg_dim",
             "pool_tail_frac": "pool_tail_frac", "pool_queries": "pool_queries", "drop_path": "drop_path",
             "context_mode": "context_mode", "encode_context": "encode_context", "encode_tail": "encode_tail"}
    arch = {k: getattr(args, a, DEFAULT_ARCH[k]) for k, a in names.items()}
    arch["head_hidden"] = getattr(args, "head_hidden", None)
    return arch


def build_pretrained_seizure_model(
    pretrained_path: Optional[str],
    device: torch.device,
    n_channels: int,
    n_types: int,
    args: Any,
) -> tuple[SeizureModel, AmplitudeTokenizer]:
    """Builds a SeizureModel from Stage 1 pretrained EEGGPT trunk or from scratch.

    The trunk comes from the checkpoint's config through EEGGPT.from_config (strict load), with
    `args.backbone_dropout` (the trunk was pretrained without dropout). The head stack follows the
    --architecture flags (arch_from_args) and `args.feat_branch`; call fit_feature_stats(model, train_ds, ...)
    once per fold when a feature branch is on.
    """
    backbone_dropout = getattr(args, "backbone_dropout", None)
    if backbone_dropout is None:
        backbone_dropout = args.dropout
    if pretrained_path and os.path.exists(pretrained_path):
        ck = torch.load(pretrained_path, map_location="cpu")
        cfg = dict(ck.get("config", {}))
        cfg.setdefault("n_channels", n_channels)
        tok = _tokenizer_from_config(cfg)
        backbone = _trunk_from_config(cfg, backbone_dropout, n_channels)
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
            int(round(args.window_s * args.fs)), backbone_dropout, (1,)
        )
        log.info("Initialized backbone from scratch (no pretrained checkpoint)")

    feat_branch = getattr(args, "feat_branch", "none") or "none"
    feat_dim, bp_idx = 0, None
    if feat_branch != "none":
        from eeg_gpt.data.features import bandpower_indices, feat_dim as _fd
        feat_dim = _fd(n_channels)
        if feat_branch == "bandpower":
            bp_idx = bandpower_indices(n_channels)
    model = SeizureModel(backbone, n_channels, n_types, args.context_blocks, args.dropout, args.pool,
                         feat_branch=feat_branch, feat_dim=feat_dim, arch=arch_from_args(args), bandpower_idx=bp_idx).to(device)
    n_scratch = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder.backbone"))
    log.info(f"SeizureModel: {n_scratch / 1e6:.2f}M parameters outside the trunk | arch {arch_from_args(args)} | "
             f"feat_branch {feat_branch} (feat_dim {feat_dim})")
    return model, tok


def fit_feature_stats(model: SeizureModel, ds, device, args: Any, max_rows: int = 2000) -> None:
    """Standardisation statistics of the `feats` vectors from (a random subset of) the training dataset.

    Also switches the dataset to emit the model's feature set (ds.feat_branch), so a dataset built without
    `feat_branch` still works. No-op without a feature branch. Call once per fold before training.
    """
    if model.feat_branch == "none":
        return
    if getattr(ds, "feat_branch", "none") != model.feat_branch:
        ds.feat_branch = model.feat_branch
    n = len(ds)
    rng = np.random.default_rng(getattr(args, "seed", 0))
    idx = np.sort(rng.choice(n, size=min(n, max_rows), replace=False))
    sub = torch.utils.data.Subset(ds, idx.tolist())
    bs = getattr(args, "eval_batch_size", 0) or 4 * getattr(args, "batch_size", 16)
    dl = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=getattr(args, "num_workers", 0))
    s = s2 = None; cnt = 0
    for b in dl:
        f = b["feats"].double()
        s = f.sum(0) if s is None else s + f.sum(0)
        s2 = (f * f).sum(0) if s2 is None else s2 + (f * f).sum(0)
        cnt += f.shape[0]
    mean = s / max(1, cnt)
    sd = (s2 / max(1, cnt) - mean * mean).clamp_min(0).sqrt()
    model.set_feature_stats(mean.float().to(device), sd.float().to(device))
    log.info(f"feature stats fitted on {cnt} training rows (feat_dim {model.feat_dim})")


@torch.no_grad()
def embed_rows(model: SeizureModel, ds, device, batch_size: int = 64, num_workers: int = 0, amp: bool = False) -> dict:
    """Frozen-trunk embeddings of every row of `ds`.

    Returns {"X": (rows, d + feat_dim) float32 = trunk embedding (after the temporal context) ++ standardised feats,
    "d": d, "y": occ, "valid": occ_valid > 0.5, "session_idx", "row_idx", "base": (rows, Fb) the band-power baseline's
    inputs (feats when the dataset emits them, else bandpower_features(z_last, log_sd))}. feat_only models never
    run the trunk (the trunk part of X is all zeros).
    """
    from eeg_gpt.training.device import get_autocast_kwargs
    from eeg_gpt.training.metrics import bandpower_features
    model.eval()
    was = getattr(ds, "feat_branch", "none")
    if model.feat_branch != "none" and was != model.feat_branch:
        ds.feat_branch = model.feat_branch
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    autocast_kw = get_autocast_kwargs(device, amp)
    d = model.encoder.d
    X, Y, V, S, R, Bse = [], [], [], [], [], []
    for b in dl:
        feats = b["feats"]
        n = feats.shape[0]
        if model.feat_branch == "feat_only":
            e = torch.zeros(n, d)
        else:
            with torch.autocast(**autocast_kw):
                v = model.encoder(b["tokens"].to(device, non_blocking=True))
                e = model.context(v) if (model.context is not None and v.shape[1] > 1) else v[:, -1]
            e = e.float().cpu()
        if model.feat_branch != "none" and feats.numel():
            fz = ((feats.float().to(device) - model.feat_mean) / model.feat_sd).cpu()
            e = torch.cat([e, fz], 1)
        X.append(e)
        if feats.numel():
            Bse.append(feats.float())
        else:
            Bse.append(torch.stack([bandpower_features(z, ds.fs, ls[:, -1]) for z, ls in zip(b["z_last"], b["log_sd"])]))
        Y.append(b["occ"].float()); V.append(b["occ_valid"] > 0.5); S.append(b["session_idx"]); R.append(b["row_idx"])
    return {"X": torch.cat(X).numpy().astype(np.float32), "d": d, "y": torch.cat(Y).numpy(), "valid": torch.cat(V).numpy(),
            "session_idx": torch.cat(S).numpy(), "row_idx": torch.cat(R).numpy().astype(np.float64),
            "base": torch.cat(Bse).numpy().astype(np.float32)}


def run_probe_fold(fold_idx: int, tr_ds, ev_ds, model: SeizureModel, args: Any, device: torch.device, fold_out_dir: str,
                   val_ds=None) -> dict[str, Any]:
    """`--head probe`: no training of the model. Cache the frozen-trunk embeddings of every row, fit a logistic
    probe (training/probe.py) and report the same record keys as run_one_fold_training.

    `auroc` is the probe on trunk embedding ++ feats when the model has a feature branch, else on the trunk alone;
    `probe_auroc` (trunk only), `probe_feat_auroc` (trunk ++ feats), `feat_only_auroc` (feats only) and
    `baseline_auroc` (bandpower_baseline on the same inputs) are always there (NaN when not applicable).
    Thresholds come from the inner-validation rows when given, else from the training rows (in-sample).
    """
    from eeg_gpt.training.metrics import (auroc, per_session_auroc, sensitivity_at_specificity, smoothed_auroc,
                                          threshold_sweep, bandpower_baseline)
    from eeg_gpt.training.probe import fit_probe
    os.makedirs(fold_out_dir, exist_ok=True)
    t0 = time.time()
    fit_feature_stats(model, tr_ds, device, args)
    bs = getattr(args, "eval_batch_size", 0) or 4 * getattr(args, "batch_size", 16)
    nw = getattr(args, "num_workers", 0)
    tr = embed_rows(model, tr_ds, device, bs, nw, getattr(args, "amp", False))
    ev = embed_rows(model, ev_ds, device, bs, nw, getattr(args, "amp", False))
    va = embed_rows(model, val_ds, device, bs, nw, getattr(args, "amp", False)) if val_ds is not None else None
    log.info(f"probe fold {fold_idx}: embedded {len(tr['y'])} train / {len(ev['y'])} eval rows in {time.time() - t0:.0f}s")
    d = tr["d"]; F = tr["X"].shape[1] - d
    mtr, mev = tr["valid"], ev["valid"]
    ytr, yev = tr["y"][mtr], ev["y"][mev]
    seed = getattr(args, "seed", 0)

    def probe(cols):
        Xtr = tr["X"][mtr][:, cols]
        s_ev = fit_probe(Xtr, ytr, ev["X"][mev][:, cols], seed=seed)
        s_tr = fit_probe(Xtr, ytr, Xtr, seed=seed)
        s_va = fit_probe(Xtr, ytr, va["X"][va["valid"]][:, cols], seed=seed) if va is not None else None
        return np.asarray(s_ev, dtype=np.float64), np.asarray(s_tr, dtype=np.float64), (None if s_va is None else np.asarray(s_va, dtype=np.float64))

    trunk_cols = np.arange(d)
    res_trunk = probe(trunk_cols) if model.feat_branch != "feat_only" else None
    res_both = probe(np.arange(d + F)) if (F > 0 and model.feat_branch != "feat_only") else None
    res_feat = probe(np.arange(d, d + F)) if F > 0 else None
    main = res_both or res_feat or res_trunk
    s_ev, s_tr, s_va = main
    base = bandpower_baseline(torch.from_numpy(tr["base"][mtr]), torch.from_numpy(ytr), torch.from_numpy(ev["base"][mev]), torch.from_numpy(yev))

    # thresholds: inner validation when there is one, else the training rows (in-sample, optimistic)
    if s_va is not None:
        yva = va["y"][va["valid"]]
        neg = s_va[yva < 0.5]
        thr_spec90 = float(np.quantile(neg, 0.9)) if len(neg) else None
        thr_ba = threshold_sweep(s_va, yva)["threshold"] if len(yva) else None
    else:
        neg = s_tr[ytr < 0.5]
        thr_spec90 = float(np.quantile(neg, 0.9)) if len(neg) else None
        thr_ba = threshold_sweep(s_tr, ytr)["threshold"]
    pos_ev = yev > 0.5
    sens_innerthr = float((s_ev[pos_ev] >= thr_spec90).mean()) if (thr_spec90 is not None and pos_ev.any()) else float("nan")

    sid, tim = ev["session_idx"][mev], ev["row_idx"][mev]
    a = auroc(s_ev, yev)
    res = {
        "auroc": a, "auroc_oracle": a, "auroc_last": a, "selected_epoch": None, "selection": "probe",
        "thresholds": {"spec90": thr_spec90, "ba": thr_ba}, "history": [],
        "train_auroc": auroc(s_tr, ytr),
        "probe_auroc": auroc(res_trunk[0], yev) if res_trunk else float("nan"),
        "probe_feat_auroc": auroc(res_both[0], yev) if res_both else float("nan"),
        "feat_only_auroc": auroc(res_feat[0], yev) if res_feat else float("nan"),
        "baseline_auroc": base["dev_auroc"], "baseline_train_auroc": base["train_auroc"],
        "baseline_sens_spec90": sensitivity_at_specificity(base["dev_scores"], yev, 0.90),
        "sweep": threshold_sweep(s_ev, yev), "n_valid": int(mev.sum()), "n_pos": int(yev.sum()),
        "sens_at_spec": {f"spec{int(sp * 100)}": sensitivity_at_specificity(s_ev, yev, sp) for sp in (0.80, 0.90, 0.95)},
        "sens_spec90_innerthr": sens_innerthr,
        "per_session": per_session_auroc(sid, s_ev, yev),
        "smooth": {k: smoothed_auroc(sid, tim, s_ev, yev, k)["auroc_smoothed"] for k in (5, getattr(args, "smooth_k", 15), 4 * getattr(args, "smooth_k", 15))},
        "type_macro_f1_ictal": None, "type_cm_ictal": None, "type_pred_hist": None,
        "feat_dim": int(F), "n_train_rows": int(mtr.sum()),
    }
    try:
        from eeg_gpt.training.metrics import score_stats
        res["score_stats"] = score_stats(s_ev, yev)
    except ImportError:
        res["score_stats"] = {"std": float(np.std(s_ev)), "frac_pos": float((s_ev >= 0.5).mean())}
    res["auroc_smooth15"] = res["smooth"].get(15, float("nan"))
    with open(os.path.join(fold_out_dir, "probe.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "per_session"}, f, indent=1, default=float)
    log.info(f"probe fold {fold_idx} | auroc {a:.4f} | trunk {res['probe_auroc']:.4f} | trunk+feat {res['probe_feat_auroc']:.4f} | "
             f"feat only {res['feat_only_auroc']:.4f} | bandpower baseline {res['baseline_auroc']:.4f} | train {res['train_auroc']:.4f}")
    return res


def run_one_fold_training(
    fold_idx: int,
    tr_ds: LabeledWindows,
    ev_ds: LabeledWindows,
    model: SeizureModel,
    args: Any,
    device: torch.device,
    fold_out_dir: str,
    val_ds: Optional[LabeledWindows] = None,
) -> dict[str, Any]:
    """Supervised fine-tuning and evaluation of one fold (the wave-1 recipe, see training/recipe.py).

    `val_ds` (inner validation split from the training part) drives epoch selection and the decision
    thresholds when --select inner; without it the last epoch is reported. Returns the eval-fold metrics at
    the selected epoch plus `auroc_oracle` / `auroc_last` / `history` / diagnostics (keys documented in _fit_fold).
    """
    from eeg_gpt.training.finetune import BalancedBatchSampler, make_loader, run_eval
    from eeg_gpt.training import recipe

    os.makedirs(fold_out_dir, exist_ok=True)
    fit_feature_stats(model, tr_ds, device, args)     # feature-branch standardisation from the training rows (no-op without a branch)
    for ds_ in (ev_ds, val_ds):                       # every loader must emit the model's feature set (datasets built without feat_branch)
        if ds_ is not None and model.feat_branch != "none" and getattr(ds_, "feat_branch", "none") != model.feat_branch:
            ds_.feat_branch = model.feat_branch
    type_on = recipe.type_head_enabled(getattr(args, "type_head", "on"), tr_ds.df)
    sampler = BalancedBatchSampler(tr_ds.df, args.batch_size, args.min_ictal_per_batch if type_on else 0,
                                   seed=args.seed + fold_idx, preictal_frac=getattr(args, "preictal_frac", 0.0))
    tr_aug = copy.copy(tr_ds)              # augmentation on the training loader only (the diagnostic / baseline loaders share tr_ds)
    tr_aug.augment = float(getattr(args, "augment", 0.0))
    tl = make_loader(tr_aug, batch_sampler=sampler, num_workers=args.num_workers)
    eval_bs = args.eval_batch_size or 4 * args.batch_size
    el = make_loader(ev_ds, batch_size=eval_bs, num_workers=args.num_workers, shuffle=False)
    tl_eval = make_loader(tr_ds, batch_size=eval_bs, num_workers=args.num_workers, shuffle=False) if args.baseline else None
    side_workers = min(2, args.num_workers)
    vl = make_loader(val_ds, batch_size=eval_bs, num_workers=side_workers, shuffle=False) if val_ds is not None and len(val_ds) else None
    # fixed training subset for the train-AUROC diagnostic (optimisation vs generalisation failure)
    dl = make_loader(torch.utils.data.Subset(tr_ds, recipe.diag_subset(len(tr_ds), 1000, args.seed + fold_idx)),
                     batch_size=eval_bs, num_workers=side_workers, shuffle=False)
    try:
        return _fit_fold(tr_ds, ev_ds, model, args, device, fold_out_dir, sampler, tl, el, tl_eval, run_eval,
                         type_on=type_on, vl=vl, dl=dl)
    finally:
        # persistent workers would otherwise pile up (and hold their pipes open) across folds and patients
        for loader in (tl, el, tl_eval, vl, dl):
            shutdown_loader(loader)


def _fit_fold(tr_ds, ev_ds, model, args, device, fold_out_dir, sampler, tl, el, tl_eval, run_eval,
              type_on: bool = True, vl=None, dl=None) -> dict[str, Any]:
    """Returns (contract with run_patient_cv):
      auroc (eval fold at the selected epoch), auroc_oracle (max over epochs), auroc_last, selected_epoch,
      selection ('inner'|'none'|'oracle'), thresholds {'spec90', 'ba'} chosen on the selection set,
      thresholds_source, at_threshold {'spec90': {sens, spec, ba}, 'ba': {...}} applied to the eval fold,
      sens_spec90_innerthr, spec_spec90_innerthr, history [per-epoch dicts], train_auroc, score_stats,
      baseline_auroc (epoch-1 band-power baseline or None), type_head (bool), batch_composition, pos_weight,
      best_epoch (= selected_epoch), and every key run_eval returned at the selected epoch
      (type_macro_f1_ictal is None when the type head is off)."""
    from eeg_gpt.training import recipe
    from eeg_gpt.training.finetune import _model_forward
    from eeg_gpt.models.gpt import init_heads

    d = tr_ds.df
    ypos = int((d["_cat"] == LABEL_CATEGORY_PREICTAL).sum())
    yneg = int((d["_cat"] == LABEL_CATEGORY_BACKGROUND).sum())
    comp = sampler.composition() if hasattr(sampler, "composition") else {}
    pos_weight_val = recipe.resolve_pos_weight(recipe.parse_pos_weight(getattr(args, "pos_weight", "auto")),
                                               args.pos_weight_cap, comp, ypos, yneg)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    tcount = np.array([(d["_type_idx"] == i).sum() for i in range(len(tr_ds.type_classes))], dtype=np.float64)
    tw = torch.tensor(np.where(tcount > 0, np.sqrt(tcount.sum() / np.maximum(tcount, 1)), 0.0), dtype=torch.float32, device=device)
    tw = tw / tw[tw > 0].mean() if (tw > 0).any() else tw

    select = getattr(args, "select", "oracle")
    if select == "inner" and vl is None:
        log.info("no inner validation split for this fold: --select inner falls back to 'none' (last epoch)")
        select = "none"
    freeze_epochs = 0 if args.freeze_backbone else min(int(getattr(args, "freeze_epochs", 0)), args.epochs)
    init_heads(model, comp.get("preictal_frac_expected", ypos / max(1, ypos + yneg)), tcount if type_on else None)
    if args.freeze_backbone or freeze_epochs > 0:
        recipe.freeze_trunk(model)

    steps_per_epoch = max(1, (args.steps_per_epoch or len(sampler)) // args.accum)

    def build_opt(n_epochs):
        o = make_adamw(recipe.param_groups(model, args), device, weight_decay=args.weight_decay)
        return o, recipe.make_sched(o, n_epochs * steps_per_epoch, args.warmup_frac)

    opt, sched = build_opt(freeze_epochs if freeze_epochs > 0 else args.epochs)
    ema = recipe.EMA(model, getattr(args, "ema", 0.0))
    eval_model = ema.model
    scaler = get_amp_scaler(device, args.amp)
    autocast_kw = get_autocast_kwargs(device, args.amp)
    non_blocking = (device.type == "cuda")

    n_batches = args.steps_per_epoch or len(sampler)
    log.info(f"fold training: {len(tr_ds)} train rows ({ypos} preictal / {yneg} background), {len(ev_ds)} eval rows"
             + (f", {len(vl.dataset)} inner-val rows" if vl is not None else "")
             + f" | {args.epochs} epochs x {n_batches} batches of {args.batch_size} (accum {args.accum}, {steps_per_epoch} optimizer steps/epoch)")
    log.info(f"recipe: batch = {comp.get('fill')} session rows + {comp.get('reserved_preictal')} preictal + {comp.get('reserved_ictal')} ictal "
             f"(expected preictal/batch {comp.get('expected_preictal_per_batch', float('nan')):.2f}, P(no preictal) {comp.get('p_no_preictal', float('nan')):.2f}) | "
             f"pos_weight {pos_weight_val:.2f} ({getattr(args, 'pos_weight', 'auto')}) | focal gamma {args.focal_gamma} | "
             f"type head {'on' if type_on else 'off'} | freeze {freeze_epochs} epochs then llrd {getattr(args, 'llrd', 1.0)} | "
             f"ema {getattr(args, 'ema', 0.0)} | select {select}")
    log.info("param groups: " + ", ".join(f"{g.get('name')}={g['lr']:.1e}" for g in opt.param_groups))
    log_every = max(1, n_batches // 5)

    history: list = []
    eval_res: dict = {}; eval_sy: dict = {}; val_sy: dict = {}; train_sy: dict = {}
    val_aurocs: list = []; eval_aurocs: list = []
    snaps: dict = {}                      # the last two epochs' weights on CPU (inner selection lags one epoch)
    oracle_epoch, oracle_auroc = None, -1.0
    selected, best_smooth = None, -float("inf")
    fb_arg = int(getattr(args, "fallback_epoch", 0) or 0)
    fallback_epoch = min(args.epochs, fb_arg if fb_arg > 0 else freeze_epochs + 2)
    baseline_auroc = None

    def save_best(ep):
        torch.save({"model": snaps[ep], "epoch": ep, "config": vars(args), "selection": select}, os.path.join(fold_out_dir, "best.pt"))

    for ep in range(1, args.epochs + 1):
        phase = "frozen" if ep <= freeze_epochs else "full"
        if freeze_epochs and ep == freeze_epochs + 1:
            recipe.unfreeze_trunk(model)
            opt, sched = build_opt(args.epochs - freeze_epochs)
            log.info("trunk unfrozen; param groups: " + ", ".join(f"{g.get('name')}={g['lr']:.1e}" for g in opt.param_groups))
        model.train()
        t_ep = time.time(); t_wait = 0.0; t_last = t_ep
        sum_lb = sum_lt = 0.0; n_seen = 0
        zero_pos = torch.zeros((), device=device); gn_sum: dict = {}; n_opt = 0
        for i, b in enumerate(tl):
            if args.steps_per_epoch and i >= args.steps_per_epoch:
                break
            t_wait += time.time() - t_last
            with torch.autocast(**autocast_kw):
                out = _model_forward(model, b, device, non_blocking)   # passes the hand-crafted feats when the model has a branch

            y = b["horizon_pos" if args.label == "horizon" else "occ"].to(device, non_blocking=non_blocking)
            v = (b["occ_valid"].to(device, non_blocking=non_blocking) > 0.5)
            if v.any():
                lb = focal_bce(out["binary_logit"].float()[v], y[v], gamma=args.focal_gamma, pos_weight=pos_weight)
            else:
                lb = out["binary_logit"].sum() * 0
            zero_pos = zero_pos + ((y * v.float()).sum() == 0).float()

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
                scaler.unscale_(opt)
                for k_, v_ in recipe.grad_norms(opt).items():
                    gn_sum[k_] = gn_sum.get(k_, 0.0) + v_
                n_opt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)
            sum_lb += float(lb.detach()); sum_lt += float(lt.detach()); n_seen = i + 1
            if n_seen % log_every == 0:
                el_s = time.time() - t_ep
                log.info(f"  ep {ep} batch {n_seen}/{n_batches} | binary {sum_lb / n_seen:.4f} | type {sum_lt / n_seen:.4f} | "
                         f"{n_seen * args.batch_size / el_s:.0f} windows/s | waiting on data {100 * t_wait / el_s:.0f}%"
                         + (f" | gpu peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f}G" if device.type == "cuda" else ""))
            t_last = time.time()
        t_train = time.time() - t_ep

        # ---- evaluation of the (EMA) weights: eval fold, inner val, training subset
        t_ev = time.time()
        res = run_eval(eval_model, el, args, device, ev_ds, args.baseline and ep == 1, tl_eval)
        if not type_on:
            res["type_macro_f1_ictal"] = None; res["type_cm_ictal"] = None; res["type_pred_hist"] = None
        if ep == 1 and isinstance(res.get("baseline"), dict):
            baseline_auroc = res["baseline"].get("dev_auroc")
        arrays = res.pop("_arrays", None)     # run_eval's raw (unmasked) scores; never serialised
        for k_ in [k_ for k_ in res if str(k_).startswith("_")]:
            res.pop(k_)
        if arrays is not None:
            S_e, Y_e, V_e = (np.asarray(arrays[k_], dtype=np.float64) for k_ in ("scores", "targets", "valid"))
        else:
            S_e, Y_e, V_e = recipe.collect_scores(eval_model, el, args, device)
        m = V_e > 0.5; S_e, Y_e = S_e[m], Y_e[m]
        eval_auroc = float(res.get("auroc", float("nan")))
        e_stats = recipe.score_stats(S_e, Y_e)
        val_auroc = float("nan")
        if vl is not None:
            S_v, Y_v, V_v, SID_v, T_v = recipe.collect_scores(eval_model, vl, args, device, with_meta=True); m = V_v > 0.5
            val_sy[ep] = (S_v[m], Y_v[m])
            k_sel = int(getattr(args, "select_smooth_k", 0))
            val_auroc = (smoothed_auroc(SID_v[m], T_v[m], S_v[m], Y_v[m], k_sel)["auroc_smoothed"] if k_sel > 1
                         else auroc(S_v[m], Y_v[m]))
        diag = {}
        if dl is not None:
            S_t, Y_t, V_t = recipe.collect_scores(eval_model, dl, args, device); m = V_t > 0.5
            train_sy[ep] = (S_t[m], Y_t[m])
            diag = {"train_auroc": auroc(S_t[m], Y_t[m]), "train_score_stats": recipe.score_stats(S_t[m], Y_t[m])}
        eval_res[ep] = res; eval_sy[ep] = (S_e, Y_e)
        eval_aurocs.append(eval_auroc); val_aurocs.append(val_auroc)
        snaps[ep] = recipe.cpu_state(eval_model)
        if ep - 2 != fallback_epoch:
            snaps.pop(ep - 2, None)

        # ---- selection
        if not math.isnan(eval_auroc) and eval_auroc > oracle_auroc:
            oracle_auroc, oracle_epoch = eval_auroc, ep
        if select == "oracle":
            if oracle_epoch == ep:
                selected = ep; save_best(ep)
        elif select == "inner":
            sm = recipe.smoothed(val_aurocs, 3)      # the value for epoch e-1 is final once epoch e is in
            cand = ep - 1 if ep >= 2 else None
            if cand is not None and not math.isnan(sm[cand - 1]) and sm[cand - 1] > best_smooth:
                best_smooth, selected = sm[cand - 1], cand; save_best(cand)
            if ep == args.epochs and not math.isnan(sm[ep - 1]) and sm[ep - 1] > best_smooth:
                best_smooth, selected = sm[ep - 1], ep; save_best(ep)
            if ep == args.epochs:
                sm_ok = [v_ for v_ in sm if not math.isnan(v_)]
                flat = len(sm_ok) < 2 or (max(sm_ok) - min(sm_ok)) < getattr(args, "select_flat_tol", 0.02)
                if selected is None or flat:
                    fb = fallback_epoch if fallback_epoch in snaps else ep
                    if flat:
                        log.info(f"inner-val curve is flat (range < {getattr(args, 'select_flat_tol', 0.02)}): "
                                 f"selecting the fallback epoch {fb} instead of {selected}")
                    selected = fb; save_best(fb)
        else:
            if ep == args.epochs:
                selected = ep; save_best(ep)

        gnorm = {k_: float(v_) / max(1, n_opt) for k_, v_ in gn_sum.items()}
        rec = {"epoch": ep, "phase": phase, "train_binary": sum_lb / max(1, n_seen), "train_type": sum_lt / max(1, n_seen),
               "eval_auroc": eval_auroc, "val_auroc": val_auroc, "eval_score_stats": e_stats,
               "zero_positive_batches": int(zero_pos.item()), "grad_norms": gnorm,
               "lr": {g.get("name", "g"): g["lr"] for g in opt.param_groups}, **diag,
               "sens_at_spec": res.get("sens_at_spec"), "sweep_ba": res.get("sweep", {}).get("balanced_accuracy")}
        history.append(rec)
        log.info(f"epoch {ep}/{args.epochs} [{phase}] | train binary {rec['train_binary']:.4f} type {rec['train_type']:.4f} | "
                 f"eval AUROC {recipe._f(eval_auroc)} BA {recipe._f(rec['sweep_ba'])} | val AUROC {recipe._f(val_auroc)} | "
                 f"train-subset AUROC {recipe._f(diag.get('train_auroc'))} | oracle {recipe._f(oracle_auroc)} (ep {oracle_epoch}) | "
                 f"score std {recipe._f(e_stats.get('std'))} frac>0.5 {recipe._f(e_stats.get('frac_pos'))} "
                 f"sep {recipe._f(e_stats.get('separation'))} | zero-pos batches {rec['zero_positive_batches']} | "
                 f"grad " + " ".join(f"{k_}={v_:.2e}" for k_, v_ in gnorm.items()) +
                 f" | train {t_train:.0f}s eval {time.time() - t_ev:.0f}s")

    # ---- the selected epoch: eval-fold metrics, thresholds from the selection set, applied to the eval fold
    sel = selected if selected is not None else args.epochs
    out = copy.deepcopy(eval_res[sel])
    if select == "inner" and sel in val_sy:
        thr, src = recipe.thresholds_from(*val_sy[sel]), "inner_val"
    elif select == "oracle":
        thr, src = recipe.thresholds_from(*eval_sy[sel]), "eval_fold (optimistic)"
    else:
        thr, src = (recipe.thresholds_from(*train_sy[sel]) if sel in train_sy else {"spec90": None, "ba": None}), "train_subset"
    S_e, Y_e = eval_sy[sel]
    at = {k_: recipe.apply_threshold(S_e, Y_e, t_) for k_, t_ in thr.items()}
    out.update({
        "auroc": float(out.get("auroc", float("nan"))),
        "auroc_oracle": float(oracle_auroc) if oracle_epoch is not None else float("nan"),
        "auroc_last": float(eval_aurocs[-1]) if eval_aurocs else float("nan"),
        "selected_epoch": sel, "best_epoch": sel, "selection": select,
        "thresholds": thr, "thresholds_source": src, "at_threshold": at,
        # protocol contract: None unless the threshold really came from the inner validation set
        "sens_spec90_innerthr": at["spec90"]["sens"] if src == "inner_val" else None,
        "spec_spec90_innerthr": at["spec90"]["spec"] if src == "inner_val" else None,
        "history": history, "train_auroc": history[sel - 1].get("train_auroc"),
        "score_stats": history[sel - 1].get("eval_score_stats"),
        "baseline_auroc": baseline_auroc, "type_head": type_on, "batch_composition": comp, "pos_weight": pos_weight_val,
    })
    log.info(f"selected epoch {sel} ({select}): eval AUROC {recipe._f(out['auroc'])} | oracle {recipe._f(out['auroc_oracle'])} | "
             f"last {recipe._f(out['auroc_last'])} | sens@spec90 at {src} threshold {recipe._f(at['spec90']['sens'])} "
             f"(spec {recipe._f(at['spec90']['spec'])}) | baseline {recipe._f(baseline_auroc)}")
    return out


def get_tokenizer_for_run(args: Any, base_mode: str = "pretrained", general_checkpoint: Optional[str] = None) -> AmplitudeTokenizer:
    """Derives tokenizer configuration matching the model checkpoint or args (AmplitudeTokenizer.from_config:
    n_levels, clip_k, patch and norm_scope travel with the pretraining checkpoint)."""
    if base_mode == "general":
        ckpt_path = general_checkpoint or getattr(args, "general_checkpoint", None) or getattr(args, "load", None)
        if ckpt_path and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location="cpu")
            cfg = dict(ck.get("config", {}))
            pre_path = cfg.get("pretrained")
            if pre_path and os.path.exists(pre_path):       # the trunk's own tokenizer settings win
                cfg.update({k: v for k, v in torch.load(pre_path, map_location="cpu").get("config", {}).items()
                            if k in ("n_levels", "clip_k", "patch", "norm_scope")})
            state_dict = ck.get("model", ck)
            tok_w = state_dict.get("encoder.backbone.tok_emb.weight")
            if tok_w is not None:
                cfg["n_levels"] = tok_w.shape[0]
            return _tokenizer_from_config(cfg)
    elif base_mode == "pretrained":
        if getattr(args, "pretrained", None) and os.path.exists(args.pretrained):
            ck = torch.load(args.pretrained, map_location="cpu")
            cfg = dict(ck.get("config", {}))
            state_dict = ck.get("model", ck)
            tok_w = state_dict.get("tok_emb.weight")
            if tok_w is not None:
                cfg["n_levels"] = tok_w.shape[0]
            return _tokenizer_from_config(cfg)
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
              horizon_s=args.horizon_s, type_grouping=args.type_grouping, tokenizer=tok, cache_capacity=args.cache_capacity,
              feat_branch=getattr(args, "feat_branch", "none"))   # every split's loader must emit the model's feats
    # postictal exclusion is applied by LabeledWindows (occ_valid = 0) when it takes the keyword
    postictal_s = float(getattr(args, "postictal_exclude_s", 0.0) or 0.0)
    if postictal_s > 0:
        if "postictal_exclude_s" in inspect.signature(LabeledWindows.__init__).parameters:
            kw["postictal_exclude_s"] = postictal_s
        else:
            log.warning("--postictal-exclude-s ignored: LabeledWindows has no postictal_exclude_s keyword yet")

    patient_full_ds = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, patient=target_patient, **kw)
    counts = patient_full_ds.counts()
    log.info(f"Patient {target_patient} summary: {counts}")

    if counts["rows"] < args.cv_folds:
        raise ValueError(f"Patient {target_patient} has {counts['rows']} rows, cannot do {args.cv_folds}-fold CV.")

    splits = generate_patient_cv_splits(
        patient_full_ds.df,
        n_splits=args.cv_folds,
        strategy=getattr(args, "cv_strategy", "auto"),
        seed=args.seed,
        context_blocks=args.context_blocks,
        window_s=args.window_s,
        inner_val=bool(getattr(args, "inner_val", True)),
    )
    head = getattr(args, "head", "finetune")

    n_channels = patient_full_ds.n_channels
    n_types = len(patient_full_ds.type_classes)

    fold_results: list[dict[str, Any]] = []

    for fold_k, sp in enumerate(splits, start=1):
        tr_idx, va_idx, ev_idx = sp["train"], sp["val"], sp["eval"]
        info = sp.get("info", {})
        log.info(f"--- Patient {target_patient} | Fold {fold_k}/{args.cv_folds} [{sp['strategy']}] "
                 f"(train={len(tr_idx)}, val={len(va_idx) if va_idx else 0}, eval={len(ev_idx)}) ---")
        t_fold = time.time()
        tr_ds = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, patient=target_patient, row_indices=tr_idx, **kw)
        ev_ds = LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, patient=target_patient, row_indices=ev_idx, **kw)
        val_ds = (LabeledWindows(args.master_csv, args.checkpoint_dir, split=None, patient=target_patient, row_indices=va_idx, **kw)
                  if va_idx else None)
        log.info(f"fold datasets built in {time.time() - t_fold:.1f}s | eval sessions: {ev_ds.sessions}"
                 + (f" | eval events: {info['eval_events']}" if "eval_events" in info else "")
                 + (f" | val sessions: {val_ds.sessions}" if val_ds is not None else " | no inner validation (fewer than 3 seizure "
                    "sessions/groups in the training part): selection falls back to the last epoch")
                 + (f" | {info['dropped_margin']} held-out rows dropped (context reaches training time)" if info.get("dropped_margin") else ""))

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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        res = _run_fold(head, fold_k, tr_ds, ev_ds, model, args, device, fold_dir, val_ds)
        log.info(f"fold {fold_k} finished in {(time.time() - t_fold) / 60:.1f} min")

        fold_record = _fold_record(res, fold_k, sp, ev_ds, args)
        if base_mode != "general":
            fold_record["pretrained"] = fold_pretrained
        fold_results.append(fold_record)
        log.info(f"Fold {fold_k} Result | AUROC={_f(fold_record['auroc'])} (oracle {_f(fold_record['auroc_oracle'])}, "
                 f"baseline {_f(fold_record['baseline_auroc'])}, smooth15 {_f(fold_record['auroc_smooth15'])}) | "
                 f"sens@spec90={_f(fold_record['sens_spec90'])} [{fold_record['threshold_source']}] | BA={_f(fold_record['sweep_ba'])} | "
                 f"epoch {fold_record['selected_epoch']} by {fold_record['selection']}")

    summary = _cv_summary(target_patient, base_mode, args.cv_folds, fold_results)
    log.info(
        f"=== Patient {target_patient} [{base_mode}] {args.cv_folds}-Fold CV Summary: "
        f"AUROC = {summary['mean_auroc']:.4f} ± {summary['std_auroc']:.4f} ({summary['n_folds_valid']}/{len(fold_results)} folds valid; "
        f"oracle {_f(summary['mean_auroc_oracle'])}, baseline {_f(summary['mean_baseline_auroc'])}) | "
        f"BA = {summary['mean_sweep_ba']:.3f} ± {summary['std_sweep_ba']:.3f} ==="
    )
    json.dump(summary, open(os.path.join(out_dir, "cv_summary.json"), "w"), indent=2)
    return summary


def _f(v, nd=4) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _run_fold(head, fold_k, tr_ds, ev_ds, model, args, device, fold_dir, val_ds):
    """One fold through the fine-tuning recipe or the frozen-trunk probe; `val_ds` only where the callee takes it."""
    if head == "probe":
        fn = globals().get("run_probe_fold")
        if fn is None:
            raise SystemExit("--head probe needs patient_sweep.run_probe_fold (architecture implementer)")
    elif head == "finetune":
        fn = run_one_fold_training
    else:
        raise ValueError(f"unknown --head {head!r}")
    if "val_ds" in inspect.signature(fn).parameters:
        return fn(fold_k, tr_ds, ev_ds, model, args, device, fold_dir, val_ds=val_ds)
    if val_ds is not None:
        log.warning(f"{fn.__name__} takes no val_ds: inner validation rows are unused in this fold")
    return fn(fold_k, tr_ds, ev_ds, model, args, device, fold_dir)


def _num(v):
    """JSON-safe float or None (NaN -> None is NOT done: NaN marks an undefined metric and stats.py maps it)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _jsonable(v):
    """JSON-safe copy: keys starting with '_' (run_eval's `_arrays`) and values json can't take are dropped,
    numpy scalars become Python scalars."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if not str(k).startswith("_") and _jsonable(x) is not _DROP}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v if _jsonable(x) is not _DROP]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
        return v.item()
    return _DROP


_DROP = object()


def _fold_record(res: dict, fold_k: int, sp: dict, ev_ds, args) -> dict[str, Any]:
    """The per-fold record of cv_summary.json, read defensively from whatever the fold function returned
    (only named scalar metrics are copied; run_eval's `_arrays` and `per_session` tables are not)."""
    g = res.get
    sw = g("sweep") or {}
    ss = g("sens_at_spec") or {}
    smooth = g("smooth") or {}
    ps = g("per_session") or {}
    bl = g("baseline") or {}
    thr = g("thresholds") or {}
    selection = g("selection") or ("oracle" if "best_epoch" in res else "none")
    inner = g("sens_spec90_innerthr")
    if selection == "inner" and inner is not None:
        sens90, thr_src = inner, "inner_val"
    else:
        sens90, thr_src = ss.get("spec90"), "eval_sweep"
    ev_df = ev_ds.df
    n_bg = int((ev_df["_cat"] == LABEL_CATEGORY_BACKGROUND).sum())
    rec = {
        "fold": fold_k,
        "train_rows": len(sp["train"]),
        "val_rows": len(sp["val"]) if sp.get("val") else 0,
        "eval_rows": len(sp["eval"]),
        "eval_pos": int(g("n_pos") or 0),
        "strategy": sp.get("strategy"),
        "auroc": _num(g("auroc", float("nan"))),
        "auroc_oracle": _num(g("auroc_oracle")),
        "auroc_last": _num(g("auroc_last")),
        "selected_epoch": g("selected_epoch", g("best_epoch")),
        "selection": selection,
        "train_auroc": _num(g("train_auroc")),
        "baseline_auroc": _num(g("baseline_auroc", bl.get("dev_auroc"))),
        "baseline_sens_spec90": _num(g("baseline_sens_spec90", bl.get("sens_spec90"))),
        "auroc_smooth15": _num(smooth.get(15, smooth.get("15"))),
        "auroc_norm": _num(g("auroc_norm")),
        "per_session_median_auroc": _num(ps.get("median")),
        "per_session_n": ps.get("n"),
        "sens_spec90": _num(sens90),
        "threshold_source": thr_src,
        "threshold_spec90": _num(thr.get("spec90")),
        "sens_at_spec": _jsonable(ss),
        "sweep_ba": _num(sw.get("balanced_accuracy", float("nan"))),
        "sweep_sens": _num(sw.get("sensitivity", float("nan"))),
        "sweep_spec": _num(sw.get("specificity", float("nan"))),
        "type_macro_f1": _num(g("type_macro_f1_ictal")),
        "background_hours": round(n_bg * float(args.window_s) / 3600.0, 3),
        "event": _jsonable(g("event")) if isinstance(g("event"), dict) else None,
        "best_epoch": g("best_epoch", g("selected_epoch")),
        "info": _jsonable(sp.get("info", {})),
    }
    return rec


def _cv_summary(patient_id_: str, base_mode: str, cv_folds: int, fold_results: list[dict]) -> dict[str, Any]:
    def finite(key):
        return [r[key] for r in fold_results if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
    aur, bas = finite("auroc"), finite("sweep_ba")
    s80 = [r["sens_at_spec"]["spec80"] for r in fold_results
           if isinstance(r.get("sens_at_spec"), dict) and r["sens_at_spec"].get("spec80") is not None and not math.isnan(r["sens_at_spec"]["spec80"])]
    s90 = finite("sens_spec90")
    mean = lambda v: float(np.mean(v)) if v else float("nan")   # noqa: E731
    std = lambda v: float(np.std(v)) if v else float("nan")     # noqa: E731
    return {
        "patient_id": patient_id_,
        "base_mode": base_mode,
        "cv_folds": cv_folds,
        "n_folds_valid": len(aur),
        "strategy_per_fold": [r.get("strategy") for r in fold_results],
        "mean_auroc": mean(aur), "std_auroc": std(aur),
        "mean_auroc_oracle": mean(finite("auroc_oracle")),
        "mean_auroc_last": mean(finite("auroc_last")),
        "mean_baseline_auroc": mean(finite("baseline_auroc")),
        "mean_auroc_smooth15": mean(finite("auroc_smooth15")),
        "mean_train_auroc": mean(finite("train_auroc")),
        "mean_sweep_ba": mean(bas), "std_sweep_ba": std(bas),
        "mean_sens_at_spec80": mean(s80),
        "mean_sens_at_spec90": mean(s90),
        "background_hours": float(sum(r.get("background_hours") or 0.0 for r in fold_results)),
        "folds": fold_results,
    }


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
    p.add_argument("--cv-strategy", choices=["auto", "session", "event"], default="auto",
                   help="CV folds: whole sessions ('session'), whole seizures + 10-min background segments ('event'), "
                        "or session when >= cv_folds sessions have seizures else event ('auto'). Row-stratified folds were removed (leakage).")

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
    p.add_argument("--focal-gamma", type=float, default=0.0, help="0 = weighted BCE (balanced batches make focal damping unnecessary); 2 = today's focal loss")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6, help="trunk lr at the top block after unfreezing (see --llrd)")
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pool", choices=["mean", "last", "meanmax", "tail", "attn", "meanstd"], default="tail",
                   help="pooling of the trunk's positions per block (tail = mean over the last --pool-tail-frac of positions)")
    p.add_argument("--smooth-k", type=int, default=15)
    p.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                   help="fit the feature (band-power + log sd) baseline on the training rows every fold; the floor the model must beat")
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

    # --- recipe flags (owner: recipe implementer; insert new add_argument lines directly below this marker) ---
    p.add_argument("--preictal-frac", type=float, default=0.10, help="fraction of every training batch reserved for preictal rows (0 = none reserved, today's sampler)")
    p.add_argument("--pos-weight", default="auto", help="'auto' = background / preictal per batch after balancing (capped by --pos-weight-cap), or a fixed number")
    p.add_argument("--freeze-epochs", type=int, default=5, help="LP-FT: epochs with the trunk frozen before unfreezing it with layer-wise lr decay (0 = train everything from epoch 1)")
    p.add_argument("--llrd", type=float, default=0.65, help="layer-wise lr decay of the trunk after unfreezing: block i of L gets --lr * llrd^(L-1-i), embeddings --lr * llrd^L (1 = off)")
    p.add_argument("--agg-lr", type=float, default=1e-4, help="lr of the channel aggregator / temporal context (everything that is neither trunk nor head)")
    p.add_argument("--backbone-dropout", type=float, default=0.1, help="dropout inside the pretrained trunk during fine-tuning (it was pretrained with 0)")
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay of the weights used for evaluation and best.pt (0 = off)")
    p.add_argument("--type-head", choices=["auto", "on", "off"], default="auto", help="auto = off when the training fold has < 2 seizure types")
    p.add_argument("--select", choices=["inner", "none", "oracle"], default="inner",
                   help="epoch selection: inner validation split (falls back to none without one), last epoch, or max on the eval fold (optimistic; always reported as auroc_oracle)")
    p.add_argument("--augment", type=float, default=0.1,
                   help="training augmentation strength: per-channel amplitude scale +-x, white noise x/2 of channel sd, window shift <= 0.5 s (0 = off)")
    p.add_argument("--select-smooth-k", type=int, default=15,
                   help="inner selection on the per-session moving-averaged val AUROC over this many windows (<= 1 = raw AUROC)")
    p.add_argument("--fallback-epoch", type=int, default=0,
                   help="epoch chosen when the inner-val curve is flat or missing (0 = --freeze-epochs + 2)")
    p.add_argument("--select-flat-tol", type=float, default=0.02, help="inner-val curve range below which it counts as flat")
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
