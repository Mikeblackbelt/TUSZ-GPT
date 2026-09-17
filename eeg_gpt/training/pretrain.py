"""Stage 1 -- self-supervised GPT pretraining on ALL recordings (no labels).

    python -m eeg_gpt.training.pretrain --checkpoint-dir ../checkpoints --out pretrain_run \
        --block-size 1024 --d 128 --layers 4 --heads 4 --steps 20000 --batch-size 64

What is different from Mahdi 2026: cross-patient corpus (every session in the
directory, all of it, including unlabeled stretches), one sequence per channel
with a channel embedding, held-out SESSIONS for validation (never the same
recording), and an optional multi-offset objective (--pred-offsets 1,8,32) so
the trunk must model more than one-sample autocorrelation.
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

log = logging.getLogger("pretrain")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--stage", default="proc")
    p.add_argument("--out", required=True)
    p.add_argument("--n-levels", type=int, default=512)
    p.add_argument("--clip-k", type=float, default=5.0)
    p.add_argument("--norm-stats", choices=["block", "session"], default="block")
    p.add_argument("--block-size", type=int, default=1024, help="tokens per sequence (4 s at 256 Hz)")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pred-offsets", default="1", help="comma list; 1 = Mahdi next-sample; e.g. 1,8,32")
    p.add_argument("--lambda-mse", type=float, default=0.1)
    p.add_argument("--n-channels", type=int, default=20)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--val-sessions", type=int, default=8, help="held-out sessions (by patient) for validation")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-capacity", type=int, default=8)
    p.add_argument("--amp", action="store_true")
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
    return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model, loader, tok, args, device, n_batches):
    model.eval(); agg = {}
    for i, (tokens, z, ch) in enumerate(loader):
        if i >= n_batches:
            break
        out = gpt_loss(model, tokens.to(device), z.to(device), ch.to(device), tok, args.lambda_mse)
        for k, v in out.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    model.train()
    return {k: v / max(1, min(n_batches, len(loader))) for k, v in agg.items()}


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "pretrain.log"))])
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AmplitudeTokenizer(args.n_levels, args.clip_k)
    from eeg_gpt.data.datasets import SessionStore
    all_sessions = SessionStore(args.checkpoint_dir, args.stage).list_sessions()
    train_s, val_s = split_sessions(all_sessions, args.val_sessions, args.seed)
    log.info(f"{len(all_sessions)} sessions | train {len(train_s)} | val {len(val_s)} (patient-disjoint)")
    per_epoch = args.steps * args.batch_size * args.accum
    tr = PretrainSessions(args.checkpoint_dir, args.block_size, per_epoch, args.stage, args.cache_capacity, tok, train_s, args.norm_stats, args.seed)
    va = PretrainSessions(args.checkpoint_dir, args.block_size, args.val_batches * args.batch_size, args.stage, args.cache_capacity, tok, val_s, args.norm_stats, args.seed + 1)
    tl = DataLoader(tr, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, num_workers=min(2, args.num_workers))
    offsets = [int(o) for o in args.pred_offsets.split(",")]
    model = EEGGPT(args.n_levels, args.n_channels, args.d, args.layers, args.heads, args.ffn, args.block_size, args.dropout, offsets).to(device)
    log.info(f"EEGGPT params: {count_params(model) / 1e6:.2f}M | offsets {offsets} | block {args.block_size} | levels {args.n_levels}")
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    cfg = {k: v for k, v in vars(args).items()}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    model.train(); t0 = time.time(); it = iter(tl); best = float("inf")
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        opt.zero_grad(set_to_none=True)
        agg = {}
        for _ in range(args.accum):
            tokens, z, ch = next(it)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = gpt_loss(model, tokens.to(device), z.to(device), ch.to(device), tok, args.lambda_mse)
            scaler.scale(out["loss"] / args.accum).backward()
            for k, v in out.items():
                agg[k] = agg.get(k, 0.0) + float(v) / args.accum
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if step % 50 == 0:
            log.info(f"step {step} | " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()) + f" | lr {lr_at(step, args):.2e} | {(time.time() - t0) / step:.2f}s/step")
        if step % args.val_every == 0 or step == args.steps:
            ev = evaluate(model, vl, tok, args, device, args.val_batches)
            ppl = math.exp(min(20.0, ev.get("ce@1", float("nan"))))
            log.info(f"VAL step {step} | " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()) + f" | ppl@1 {ppl:.1f}")
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg, "val": ev}
            torch.save(ck, os.path.join(args.out, "last.pt"))
            if ev.get("loss", float("inf")) < best:
                best = ev["loss"]; torch.save(ck, os.path.join(args.out, "best.pt"))
    log.info("done")


if __name__ == "__main__":
    main()
