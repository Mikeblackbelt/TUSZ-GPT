"""Stage 1 -- self-supervised GPT pretraining on ALL recordings (no labels).

    python -m eeg_gpt.training.pretrain --checkpoint-dir ../checkpoints --out pretrain_run \
        --block-size 1024 --steps 100000 --batch-size 64 --num-workers 8

What is different from Mahdi 2026: cross-patient corpus (every session in the
directory, all of it, including unlabeled stretches), one sequence per channel
with a channel embedding, held-out SESSIONS for validation (never the same
recording), and an optional multi-offset objective (--pred-offsets 1,8,32) so
the trunk must model more than one-sample autocorrelation.

Reading the log: `ce@1` is the plain next-token cross-entropy in nats and is
the number to compare between runs (`ppl@1 = exp(ce@1)`); `best.pt` is selected
on it. `loss` is what is optimized (ce@1 + aux_weight * ce@8 + ... + lambda * mse)
and moves with the weights. ce@1 has a floor set by the tokenizer: a residual of
r bins RMS costs about ln(4.13 * r) nats, so 2.0 nats is already ~1.8 bins
(0.03 sigma) of next-sample error at 512 levels over +-4 sigma.

What makes it converge lower / faster than the first version:
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


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--stage", default="proc")
    p.add_argument("--out", required=True)
    p.add_argument("--n-levels", type=int, default=512)
    p.add_argument("--clip-k", type=float, default=4.0,
                   help="z-score clipping range (default 4.0; smaller = more dynamic range per bin)")
    p.add_argument("--norm-stats", choices=["block", "session"], default="block")
    p.add_argument("--block-size", type=int, default=1024, help="tokens per sequence (4 s at 256 Hz)")
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
    p.add_argument("--aux-weight", type=float, default=0.5, help="loss weight of the offset > 1 heads")
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
            )
        for k, v in out.items():
            agg[k] = agg.get(k, 0.0) + v.detach().float()
    model.train()
    return {k: float(v) / max(1, min(n_batches, len(loader))) for k, v in agg.items()}


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "pretrain.log"))])
    run_pretrain(args)


def run_pretrain(args, train_sessions=None, val_sessions=None) -> dict:
    """One pretraining run. With explicit session lists (training/pipeline.py: one CV fold)
    the built-in --val-sessions split is skipped. Returns the best validation metrics."""
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device(args.device)
    setup_device_and_hardware(device, args.max_gpu_usage)
    args.num_workers = adapt_num_workers(args.num_workers, n_loaders=2)
    args.amp = resolve_amp(args.amp, device)
    log.info(f"Using device: {device} | amp {args.amp}")
    tok = AmplitudeTokenizer(args.n_levels, args.clip_k)
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
    offsets = [int(o) for o in args.pred_offsets.split(",")]
    model = EEGGPT(args.n_levels, args.n_channels, args.d, args.layers, args.heads, args.ffn, args.block_size, args.dropout, offsets,
                   rope=args.rope, value_emb=args.value_emb, tie_weights=args.tie_weights).to(device)
    log.info(f"EEGGPT params: {count_params(model) / 1e6:.2f}M | offsets {offsets} | block {args.block_size} | levels {args.n_levels} | clip_k {args.clip_k}")
    log.info(f"  rope={args.rope} | value_emb={args.value_emb} | tie_weights={args.tie_weights} | dropout={args.dropout} | "
             f"lambda_mse={args.lambda_mse} | aux_weight={args.aux_weight} | label_smoothing={args.label_smoothing} | "
             f"warmup={args.warmup} | lr_min={args.lr_min}")
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
    cfg = {k: v for k, v in vars(args).items()}
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
            sel = ev.get("ce@1", ev.get("loss", float("inf")))
            ppl = math.exp(min(20.0, ev.get("ce@1", float("nan"))))
            log.info(f"VAL step {step} | " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()) + f" | ppl@1 {ppl:.1f}")
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