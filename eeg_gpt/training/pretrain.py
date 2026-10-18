"""Stage 1 -- self-supervised GPT pretraining on ALL recordings (no labels).

    python -m eeg_gpt.training.pretrain --checkpoint-dir ../checkpoints --out pretrain_run \
        --block-size 1024 --steps 100000 --batch-size 64 --num-workers 8

    # patch objective (0.25 s patches, 32 s crops, leak-free normalisation) with a linear probe
    python -m eeg_gpt.training.pretrain --checkpoint-dir ../checkpoints --out pretrain_patch \
        --patch 64 --block-size 8192 --batch-size 512 --steps 10000 --probe-csv ../master_full.csv

    # floors / probe of an existing checkpoint, no training
    python -m eeg_gpt.training.pretrain --checkpoint-dir ../checkpoints --out /tmp/x --eval-only runs/pre/best.pt

What is different from Mahdi 2026: cross-patient corpus (every session in the
directory, all of it, including unlabeled stretches), one sequence per channel
with a channel embedding, held-out SESSIONS for validation (never the same
recording), and an optional multi-offset objective (--pred-offsets 1,8,32) so
the trunk must model more than one-sample autocorrelation.

Two objectives (the trunk is a causal transformer either way):
  token (--patch 1, today's model): amplitude levels in, next level out. `ce@1` is the plain
    next-token cross-entropy in nats (`ppl@1 = exp(ce@1)`); `best.pt` is selected on it. A
    0.5-40 Hz signal at 256 Hz is almost linearly predictable one sample ahead, so ce@1 sits
    close to its floor within a few hundred steps: read it next to the context-free floors that
    are logged beside it -- `ce_copy` (repeat the previous level) and `ce_ar8` (least-squares
    AR(8), Gaussian residual quantised onto the bins). ce@1 well below ce_ar8 means the trunk
    learned something non-linear; ce@1 ~ ce_ar8 means it learned the linear predictor and stopped.
  patch (--patch k > 1): float z in, k samples per position; the heads predict the next patch
    (`mse_patch`, selection metric), its log band energies (`mse_spec`) and optionally the patch
    h positions ahead (`mse_coarse@h`). Floors: `mse_copy_patch` (repeat the last patch) and
    `mse_ar8` (AR(8) extrapolated over the next patch).

The number that matters downstream is neither: with --probe-csv a linear probe on pooled trunk
features of labelled preictal/background windows of the fold's held-out patients is fitted every
--probe-every steps (`probe_auroc`), next to the same probe on band-power features of the same
windows (`bandpower_auroc`). If probe_auroc never rises above bandpower_auroc, Stage 1 is not
helping the seizure task whatever the loss does.

Other things that make it converge lower / faster than the first version:
  * rotary positions, amplitude (Fourier) features under the token embedding and
    a tied output head -- the model no longer has to learn from scratch that
    level 300 sits next to level 301;
  * no dropout and no label smoothing: every batch is new data (one pass over
    the corpus), so both only held the loss up;
  * a trunk sized for the corpus (d 256 / 6 layers, ~5 M params);
  * far offsets (8, 32) kept as auxiliary heads at weight 0.5;
  * pooled session sampling, bf16 autocast, fused AdamW and no per-step GPU
    sync for speed; --compile adds torch.compile on top.
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
from torch.utils.data import DataLoader

from eeg_gpt.data.datasets import PretrainSessions, patient_id
from eeg_gpt.data.tokenizer import AmplitudeTokenizer
from eeg_gpt.models.gpt import EEGGPT, count_params, gpt_loss
from eeg_gpt.training.device import (adapt_num_workers, get_amp_scaler, get_autocast_kwargs, get_device, make_adamw, parse_gpu_usage, resolve_amp,
                                     setup_device_and_hardware, shutdown_loader)

log = logging.getLogger("pretrain")

PROBE_WINDOW = 1024          # samples per probe window: 4 s at 256 Hz, the fine-tuning block
PROBE_FS = 256.0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--stage", default="proc")
    p.add_argument("--out", required=True)
    p.add_argument("--n-levels", type=int, default=512)
    p.add_argument("--clip-k", type=float, default=4.0,
                   help="z-score clipping range (default 4.0; smaller = more dynamic range per bin)")
    p.add_argument("--norm-stats", choices=["block", "session"], default="block")
    p.add_argument("--block-size", type=int, default=1024, help="SAMPLES per sequence (4 s at 256 Hz); with --patch k the trunk sees block_size/k positions")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="0 by default: pretraining is a single pass over the corpus, there is nothing to overfit")
    p.add_argument("--rope", action=argparse.BooleanOptionalAction, default=True,
                   help="rotary positions instead of a learned absolute position embedding")
    p.add_argument("--value-emb", action=argparse.BooleanOptionalAction, default=True,
                   help="add Fourier features of the token's amplitude to the token embedding")
    p.add_argument("--tie-weights", action=argparse.BooleanOptionalAction, default=True,
                   help="share the next-token head with the token embedding")
    p.add_argument("--pred-offsets", default="1,8,32",
                   help="comma list of prediction offsets; multi-offset forces the model to learn "
                        "structure at multiple timescales (default: 1,8,32 = 4ms, 31ms, 125ms @ 256Hz)")
    p.add_argument("--lambda-mse", type=float, default=0.5,
                   help="MSE weight on dequantised amplitude (default 0.5, up from 0.1)")
    p.add_argument("--aux-weight", type=float, default=0.5,
                   help="loss weight of the offset > 1 heads (token CE is normalized by the sum of CE head weights) / coarse-horizon heads (patch)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="label smoothing on CE heads (off: it floors the loss at ~0.95 nats per head)")
    p.add_argument("--n-channels", type=int, default=20)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="cosine LR floor (default 1e-5, up from 1e-6 for non-zero late updates)")
    p.add_argument("--warmup", type=int, default=1000,
                   help="linear warmup steps (default 1000, doubled for smoother early training)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--val-sessions", type=int, default=8, help="held-out sessions (by patient) for validation")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-capacity", type=int, default=8)
    p.add_argument("--session-pool", type=int, default=16,
                   help="recordings mixed in a batch and held in each worker's cache")
    p.add_argument("--samples-per-session", type=int, default=256,
                   help="draws from a recording before its pool slot loads another one")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="mixed precision (bf16 where supported); default: on for CUDA")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (needs a working compiler toolchain)")
    p.add_argument("--device", default=None, help="Device to use (cuda, cuda:0, mps, cpu). Defaults to auto-detecting GPU.")
    p.add_argument("--max-gpu-usage", type=parse_gpu_usage, default=None, help="Float between 0.0 and 1.0 inclusive specifying maximum fraction of GPU memory/usage.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    # --- stage1 flags (owner: stage1 implementer; insert new add_argument lines directly below this marker) ---
    p.add_argument("--patch", type=int, default=1, help="samples per trunk position; 1 = amplitude tokens (today), k > 1 = float patches (64 = 0.25 s)")
    p.add_argument("--objective", choices=["token", "patch", "auto"], default="auto", help="auto: token for --patch 1, patch otherwise")
    p.add_argument("--norm-scope", choices=["block", "lead4s", "auto"], default="auto",
                   help="z-score statistics: per block (today) or from the first 4 s of a crop/context only (leak-free, keeps the amplitude trend); auto = block for --patch 1 else lead4s. Written into the checkpoint: fine-tuning uses the same scope")
    p.add_argument("--spec-weight", type=float, default=1.0, help="patch objective: weight of the next-patch log band-energy MSE")
    p.add_argument("--n-bands", type=int, default=8, help="patch objective: rfft bins of a patch grouped into this many log band energies")
    p.add_argument("--coarse-horizons", default="", help="patch objective: comma list of horizons in patches with an extra next-patch head each (weight --aux-weight)")
    p.add_argument("--tie-mode", choices=["tok", "full"], default="tok", help="token path: tie the next-level head to the embedding table (tok, today) or to the table plus the amplitude features (full)")
    p.add_argument("--dequant", choices=["edge", "center"], default="edge", help="token path: dequantise levels at bin lower edges (today) or bin centres")
    p.add_argument("--mse-units", choices=["sigma", "bin"], default="sigma", help="token path: Mahdi MSE in sigma^2 units (today, numerically tiny) or in bin^2 units")
    p.add_argument("--probe-csv", default="", help="master CSV: fit a linear probe (preictal vs background) on pooled trunk features of held-out patients' windows every --probe-every steps, next to the same probe on band-power features")
    p.add_argument("--probe-patients", type=int, default=4, help="held-out patients used by the probe")
    p.add_argument("--probe-windows", type=int, default=600, help="labelled windows for the probe (class-balanced), split by session into train/test halves")
    p.add_argument("--probe-every", type=int, default=None, help="steps between probes (default: --val-every)")
    p.add_argument("--eval-only", default=None, help="checkpoint: build the model from its config, log the validation metrics, floors and probe, write <out>/eval.json, and exit")
    return p


def split_sessions(sessions, n_val, seed):
    pats = sorted({patient_id(s) for s in sessions})
    rng = np.random.default_rng(seed); rng.shuffle(pats)
    val_p, val = set(), []
    for pt in pats:
        if len(val) >= n_val:
            break
        val_p.add(pt); val = [s for s in sessions if patient_id(s) in val_p]
    return [s for s in sessions if patient_id(s) not in val_p], val


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * step / max(1, args.warmup)
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * min(1.0, t)))


def resolve_objective(args):
    """'auto' choices depend on --patch; resolve them once so the checkpoint config carries the real values."""
    args.patch = max(1, int(getattr(args, "patch", 1) or 1))
    if getattr(args, "objective", "auto") in (None, "auto"):
        args.objective = "patch" if args.patch > 1 else "token"
    if getattr(args, "norm_scope", "auto") in (None, "auto"):
        args.norm_scope = "block" if args.patch == 1 else "lead4s"
    if getattr(args, "probe_every", None) is None:
        args.probe_every = args.val_every
    if args.block_size % args.patch:
        raise SystemExit(f"--block-size {args.block_size} must be a multiple of --patch {args.patch}")
    return args


def selection_key(model) -> str:
    return "mse_patch" if getattr(model, "objective", "token") == "patch" else "ce@1"


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


@torch.no_grad()
def evaluate(model, loader, tok, args, device, n_batches):
    model.eval(); agg = {}
    non_blocking = (device.type == "cuda")
    autocast_kw = get_autocast_kwargs(device, args.amp)
    for i, (tokens, z, ch) in enumerate(loader):
        if i >= n_batches:
            break
        with torch.autocast(**autocast_kw):
            out = gpt_loss(
                model,
                tokens.to(device, non_blocking=non_blocking),
                z.to(device, non_blocking=non_blocking),
                ch.to(device, non_blocking=non_blocking),
                tok,
                args.lambda_mse,
                args.label_smoothing,
                args.aux_weight,
                getattr(args, "spec_weight", 1.0),
            )
        for k, v in out.items():
            agg[k] = agg.get(k, 0.0) + v.detach().float()
    model.train()
    return {k: float(v) / max(1, min(n_batches, len(loader))) for k, v in agg.items()}


# ------------------------------------------------------------------------------------------------ linear probe

_PROBE_CACHE: dict = {}


def _probe_windows(args, sessions):
    """Class-balanced preictal/background windows of up to --probe-patients held-out patients, split by
    session into train/test halves. Cached per (csv, sessions). Returns None when nothing usable."""
    import pandas as pd
    from eeg_gpt.data.datasets import LABEL_CATEGORY_BACKGROUND, LABEL_CATEGORY_PREICTAL, SessionStore, load_master
    key = (os.path.abspath(args.probe_csv), tuple(sorted(sessions)), args.probe_patients, args.probe_windows, args.seed)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    df = load_master(args.probe_csv)
    df = df[df["session_key"].isin(set(sessions)) & df["_cat"].isin([LABEL_CATEGORY_PREICTAL, LABEL_CATEGORY_BACKGROUND])]
    rng = np.random.default_rng(args.seed)
    usable = [p for p, g in df.groupby("_patient") if (g["_cat"] == LABEL_CATEGORY_PREICTAL).any() and (g["_cat"] == LABEL_CATEGORY_BACKGROUND).any()]
    if not usable:
        log.warning("probe: no held-out patient has both preictal and background rows in %s", args.probe_csv)
        _PROBE_CACHE[key] = None
        return None
    rng.shuffle(usable)
    df = df[df["_patient"].isin(set(usable[: max(1, args.probe_patients)]))]
    per_class = max(4, args.probe_windows // 2)
    parts = []
    for cat in (LABEL_CATEGORY_PREICTAL, LABEL_CATEGORY_BACKGROUND):
        g = df[df["_cat"] == cat]
        parts.append(g.sample(n=min(per_class, len(g)), random_state=int(rng.integers(0, 2**31 - 1))))
    rows = pd.concat(parts).reset_index(drop=True)
    y = (rows["_cat"] == LABEL_CATEGORY_PREICTAL).values.astype(np.float32)
    sess = sorted(rows["session_key"].unique().tolist())
    test_sess = set(sess[1::2])
    is_te = rows["session_key"].isin(test_sess).values
    ok = lambda m: m.sum() > 0 and y[m].sum() > 0 and (1 - y[m]).sum() > 0  # noqa: E731
    if not (ok(is_te) and ok(~is_te)):
        log.warning("probe: session split leaves a half without both classes; falling back to a random row split")
        is_te = rng.random(len(rows)) < 0.5
        if not (ok(is_te) and ok(~is_te)):
            _PROBE_CACHE[key] = None
            return None
    store = SessionStore(args.checkpoint_dir, args.stage, capacity=4)
    xs = []
    for k_, st in zip(rows["session_key"].values, rows["stop_time"].values):
        arr = store.get(k_)
        C, Ttot = arr.shape
        e = min(max(int(round(float(st) * PROBE_FS)), PROBE_WINDOW), Ttot)
        w = np.zeros((C, PROBE_WINDOW), dtype=np.float32)
        seg = arr[:, max(0, e - PROBE_WINDOW): e]
        w[:, PROBE_WINDOW - seg.shape[1]:] = seg
        xs.append(w)
    X = np.stack(xs)                                                                   # (n, C, T) raw
    res = {"x": X, "y": y, "is_te": is_te, "patients": sorted(set(rows["_patient"])), "n": len(rows)}
    _PROBE_CACHE[key] = res
    log.info(f"probe windows: {len(rows)} ({int(y.sum())} preictal) of patients {res['patients']} | "
             f"train {int((~is_te).sum())} / test {int(is_te.sum())} by session")
    return res


def run_probe(model, tok, args, device, sessions) -> dict:
    """probe_auroc (pooled trunk features: mean over positions, channels concatenated) and bandpower_auroc
    (eeg_gpt.data.features.window_features) of the same held-out windows. {} when the probe cannot run;
    a failing probe is logged, never raised (it must not take the pretraining run down)."""
    if not getattr(args, "probe_csv", ""):
        return {}
    try:
        return _run_probe(model, tok, args, device, sessions)
    except Exception as e:
        log.warning(f"probe failed ({e!r}); probe_auroc skipped")
        return {}


@torch.no_grad()
def _run_probe(model, tok, args, device, sessions) -> dict:
    from eeg_gpt.training.probe import probe_auroc
    pw = _probe_windows(args, sessions)
    if pw is None:
        return {}
    X, y, is_te = pw["x"], pw["y"], pw["is_te"]
    was_training = model.training
    model.eval()
    autocast_kw = get_autocast_kwargs(device, args.amp)
    # the trunk sees at most block_size samples (a multiple of its patch): the newest ones of the 4 s window,
    # normalised with the whole window's statistics
    t_keep = min(PROBE_WINDOW, int(getattr(model, "block_size", PROBE_WINDOW)))
    t_keep -= t_keep % max(1, int(getattr(model, "patch", 1)))
    feats = []
    bs = 32
    for i in range(0, len(X), bs):
        xb = X[i: i + bs]                                                              # (b, C, T)
        inp = np.stack([tok.encode_blocks(x[:, None, :])["inp"][:, 0, :] for x in xb])  # (b, C, T)
        inp = np.ascontiguousarray(inp[..., -t_keep:])
        b, C, T = inp.shape
        t = torch.from_numpy(inp).reshape(b * C, T).to(device)
        ch = torch.arange(C, device=device).repeat(b)
        with torch.autocast(**autocast_kw):
            h = model.hidden(t, ch)                                                    # (b*C, P, d)
        feats.append(h.float().mean(1).reshape(b, -1).cpu())
    F_ = torch.cat(feats).numpy()
    out = {"probe_auroc": probe_auroc(F_[~is_te], y[~is_te], F_[is_te], y[is_te], seed=args.seed), "probe_n": float(len(y))}
    try:
        from eeg_gpt.data.features import window_features
        bf = np.stack([np.asarray(window_features(x[:, None, :], PROBE_FS), dtype=np.float32).reshape(-1) for x in X])
        out["bandpower_auroc"] = probe_auroc(bf[~is_te], y[~is_te], bf[is_te], y[is_te], seed=args.seed)
    except ImportError:
        log.warning("probe: eeg_gpt.data.features.window_features not available; bandpower_auroc skipped")
    except Exception as e:  # the baseline must never take the pretraining run down
        log.warning(f"probe: band-power baseline failed ({e!r}); bandpower_auroc skipped")
    if was_training:
        model.train()
    return out


# ------------------------------------------------------------------------------------------------ training

def build_from_args(args):
    """Tokenizer and trunk from the (resolved) args; the same dict is what the checkpoint stores as `config`."""
    cfg = {k: v for k, v in vars(args).items()}
    return AmplitudeTokenizer.from_config(cfg), EEGGPT.from_config(cfg), cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "pretrain.log"))])
    if args.eval_only:
        return run_eval_only(args)
    return run_pretrain(args)


def run_eval_only(args, train_sessions=None, val_sessions=None) -> dict:
    """Validation metrics (with floors) and the probe of an existing checkpoint, no training.
    The validation sessions are the --val-sessions/--seed split of the checkpoint dir unless given."""
    ck = torch.load(args.eval_only, map_location="cpu")
    cfg = dict(ck.get("config", {}))
    # model/objective keys come from the checkpoint; data, device and probe settings from the command line
    for k in ("checkpoint_dir", "stage", "val_sessions", "val_batches", "batch_size", "num_workers", "cache_capacity",
              "session_pool", "samples_per_session", "device", "max_gpu_usage", "amp", "seed", "probe_csv",
              "probe_patients", "probe_windows", "probe_every", "out", "norm_stats", "eval_only"):
        cfg[k] = getattr(args, k)
    a = argparse.Namespace(**{**vars(args), **cfg})
    resolve_objective(a)
    device = get_device(a.device)
    setup_device_and_hardware(device, a.max_gpu_usage)
    a.num_workers = adapt_num_workers(a.num_workers, n_loaders=1)
    a.amp = resolve_amp(a.amp, device)
    tok, model, _ = build_from_args(a)
    model.load_state_dict(ck["model"]); model.to(device)
    if train_sessions is None or val_sessions is None:
        from eeg_gpt.data.datasets import SessionStore
        _, val_s = split_sessions(SessionStore(a.checkpoint_dir, a.stage).list_sessions(), a.val_sessions, a.seed)
    else:
        val_s = list(val_sessions)
    va = PretrainSessions(a.checkpoint_dir, a.block_size, a.val_batches * a.batch_size, a.stage, a.cache_capacity, tok, val_s,
                          a.norm_stats, a.seed + 1, a.session_pool, 32)
    vl = make_loader(va, batch_size=a.batch_size, num_workers=min(2, a.num_workers), drop_last=False)
    ev = evaluate(model, vl, tok, a, device, a.val_batches)
    ev.update(run_probe(model, tok, a, device, val_s))
    ev["step"] = ck.get("step"); ev["checkpoint"] = a.eval_only; ev["objective"] = model.objective; ev["patch"] = model.patch
    log.info("EVAL " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in ev.items()))
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    shutdown_loader(vl)
    return ev


def run_pretrain(args, train_sessions=None, val_sessions=None) -> dict:
    """One pretraining run. With explicit session lists (training/pipeline.py: one CV fold)
    the built-in --val-sessions split is skipped. Returns the best validation metrics."""
    os.makedirs(args.out, exist_ok=True)
    resolve_objective(args)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device(args.device)
    setup_device_and_hardware(device, args.max_gpu_usage)
    args.num_workers = adapt_num_workers(args.num_workers, n_loaders=2)
    args.amp = resolve_amp(args.amp, device)
    log.info(f"Using device: {device} | amp {args.amp}")
    tok, model, cfg = build_from_args(args)
    model = model.to(device)
    if train_sessions is None or val_sessions is None:
        from eeg_gpt.data.datasets import SessionStore
        all_sessions = SessionStore(args.checkpoint_dir, args.stage).list_sessions()
        train_s, val_s = split_sessions(all_sessions, args.val_sessions, args.seed)
    else:
        train_s, val_s = list(train_sessions), list(val_sessions)
    log.info(f"{len(train_s) + len(val_s)} sessions | train {len(train_s)} | val {len(val_s)} (patient-disjoint)")
    per_epoch = args.steps * args.batch_size * args.accum
    tr = PretrainSessions(args.checkpoint_dir, args.block_size, per_epoch, args.stage, args.cache_capacity, tok, train_s, args.norm_stats, args.seed,
                          args.session_pool, args.samples_per_session)
    # short stints per recording so the fixed validation set covers many of the held-out sessions
    va = PretrainSessions(args.checkpoint_dir, args.block_size, args.val_batches * args.batch_size, args.stage, args.cache_capacity, tok, val_s,
                          args.norm_stats, args.seed + 1, args.session_pool, 32)
    tl = make_loader(tr, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    vl = make_loader(va, batch_size=args.batch_size, num_workers=min(2, args.num_workers), drop_last=False)
    sel_key = selection_key(model)
    log.info(f"EEGGPT params: {count_params(model) / 1e6:.2f}M | objective {model.objective} | patch {model.patch} "
             f"({model.n_positions} positions of {args.block_size} samples) | norm_scope {args.norm_scope} | "
             f"offsets {list(model.pred_offsets)} | levels {args.n_levels} | clip_k {args.clip_k} | selection on {sel_key}")
    log.info(f"  rope={args.rope} | value_emb={args.value_emb} | tie_weights={args.tie_weights} | tie_mode={args.tie_mode} | "
             f"dequant={args.dequant} | mse_units={args.mse_units} | dropout={args.dropout} | lambda_mse={args.lambda_mse} | "
             f"aux_weight={args.aux_weight} | spec_weight={args.spec_weight} | coarse_horizons={args.coarse_horizons!r} | "
             f"label_smoothing={args.label_smoothing} | warmup={args.warmup} | lr_min={args.lr_min} | "
             f"probe={'every %d steps on %s' % (args.probe_every, args.probe_csv) if args.probe_csv else 'off'}")
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "emb" not in n)]
    opt = make_adamw([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], device,
                     lr=args.lr, betas=(0.9, 0.95))
    scaler = get_amp_scaler(device, args.amp)
    autocast_kw = get_autocast_kwargs(device, args.amp)
    non_blocking = (device.type == "cuda")
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    # the compiled wrapper shares parameters with `model`; checkpoints are saved from `model`
    net = torch.compile(model) if args.compile else model
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    model.train(); t0 = time.time(); step0 = step; it = iter(tl); best = float("inf"); best_val = {}
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            tokens, z, ch = next(it)
            with torch.autocast(**autocast_kw):
                out = gpt_loss(
                    net,
                    tokens.to(device, non_blocking=non_blocking),
                    z.to(device, non_blocking=non_blocking),
                    ch.to(device, non_blocking=non_blocking),
                    tok,
                    args.lambda_mse,
                    args.label_smoothing,
                    args.aux_weight,
                    args.spec_weight,
                )
            scaler.scale(out["loss"] / args.accum).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if step % 50 == 0:
            # the only GPU sync of the training loop: metrics of the last micro-batch
            log.info(f"step {step} | " + " ".join(f"{k}={float(v.detach()):.4f}" for k, v in out.items()) +
                     f" | lr {lr_at(step, args):.2e} | {(time.time() - t0) / (step - step0):.3f}s/step")
        if step % args.val_every == 0 or step == args.steps:
            ev = evaluate(net, vl, tok, args, device, args.val_batches)
            if args.probe_csv and (step % args.probe_every == 0 or step == args.steps):
                ev.update(run_probe(model, tok, args, device, val_s))
            sel = ev.get(sel_key, ev.get("loss", float("inf")))
            extra = f" | ppl@1 {math.exp(min(20.0, ev['ce@1'])):.1f}" if "ce@1" in ev else ""
            log.info(f"VAL step {step} | " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()) + extra)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg, "val": ev}
            torch.save(ck, os.path.join(args.out, "last.pt"))
            if sel < best:
                best = sel; best_val = {"step": step, **ev}; torch.save(ck, os.path.join(args.out, "best.pt"))
    del it
    shutdown_loader(tl); shutdown_loader(vl)
    log.info("done")
    return {"best_path": os.path.join(args.out, "best.pt"), "best_val": best_val, "steps": step,
            "train_sessions": len(train_s), "val_sessions": len(val_s)}


if __name__ == "__main__":
    main()
