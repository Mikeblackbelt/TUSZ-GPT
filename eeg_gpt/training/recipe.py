"""Fine-tuning recipe helpers shared by finetune.main() and patient_sweep._fit_fold().

Wave-1 recipe (why each piece exists):
  * batch balancing     -- the sampler reserves preictal slots (--preictal-frac); with 3 % positives a batch of
                           16 otherwise holds 0.35 preictal rows and 70 % of batches have none.
  * pos_weight 'auto'   -- background / preictal per batch AFTER balancing (a 30x weight on top of balancing
                           would overshoot); a number fixes it.
  * head init           -- last layer zeroed, bias = log-odds of the presented prior, so step 0 predicts the prior
                           instead of a random constant (Lin et al. 2017).
  * LP-FT               -- trunk frozen for --freeze-epochs (linear-probe phase), then unfrozen with layer-wise
                           lr decay (--llrd) and a small trunk lr (Kumar et al. 2022, LaBraM recipe).
  * EMA                 -- evaluation and best.pt use an exponential moving average of the weights.
  * type head 'auto'    -- off when the training fold has < 2 seizure types (a 1-class head trains to 0 loss).
  * selection 'inner'   -- the epoch is picked on an inner validation split (val_ds), never on the eval fold;
                           'oracle' (today's max over epochs on the eval fold) is always reported beside it.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from eeg_gpt.training.device import get_autocast_kwargs
from eeg_gpt.training.metrics import auroc, threshold_sweep

log = logging.getLogger("recipe")


# ----------------------------------------------------------------------------------------------- class balance

def parse_pos_weight(s: Any) -> Any:
    """'auto' or a float (argparse type)."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    if s == "auto":
        return "auto"
    return float(s)


def resolve_pos_weight(setting: Any, cap: float, composition: dict, n_pos: int, n_neg: int) -> float:
    """pos_weight for the binary loss. 'auto' = expected background / expected preictal per batch after the
    sampler's balancing (today's yneg/ypos rule when nothing is reserved), capped at `cap`, floored at 1."""
    if setting != "auto":
        return float(setting)
    pre = float(composition.get("expected_preictal_per_batch", 0.0))
    bg = float(composition.get("expected_background_per_batch", 0.0))
    if pre <= 0 or bg <= 0:
        pre, bg = float(max(1, n_pos)), float(n_neg)
    raw = bg / pre
    if float(composition.get("reserved_preictal", 0.0)) > 0:
        return float(min(cap, min(2.0, max(1.0, raw))))
    return float(min(cap, max(1.0, raw)))


def type_head_enabled(mode: str, df) -> bool:
    """'on' / 'off' / 'auto' (= on only when >= 2 seizure types occur among the training fold's ictal rows)."""
    mode = (mode or "on").lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    ict = df[df["_cat"] == "ictal"]
    n_types = int(ict.loc[ict["_type_idx"] >= 0, "_type_idx"].nunique()) if len(ict) else 0
    return n_types >= 2


def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
    lin = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return lin[-1] if lin else None


@torch.no_grad()
def init_heads(model: nn.Module, pos_frac: float, type_counts=None) -> None:
    """Zero the last Linear of both heads; binary bias = log-odds of the preictal fraction the sampler presents,
    type bias = log class frequencies (classes with no rows get a large negative bias)."""
    p = float(min(max(pos_frac, 1e-4), 1 - 1e-4))
    lin = _last_linear(getattr(model, "binary_head", model))
    if lin is not None:
        lin.weight.zero_()
        if lin.bias is not None:
            lin.bias.fill_(math.log(p / (1 - p)))
    th = getattr(model, "type_head", None)
    lin = _last_linear(th) if th is not None else None
    if lin is not None:
        lin.weight.zero_()
        if lin.bias is not None:
            if type_counts is not None and len(type_counts) == lin.bias.numel():
                c = np.asarray(type_counts, dtype=np.float64)
                freq = c / max(1.0, c.sum())
                lin.bias.copy_(torch.tensor(np.where(freq > 0, np.log(np.maximum(freq, 1e-12)), -10.0), dtype=lin.bias.dtype))
            else:
                lin.bias.zero_()


# ----------------------------------------------------------------------------------------------- LP-FT / LLRD

def freeze_trunk(model: nn.Module) -> None:
    model.encoder.freeze()


def unfreeze_trunk(model: nn.Module) -> None:
    """WindowEncoder.unfreeze() (architecture implementer) when present, else the inverse of freeze()."""
    enc = model.encoder
    if hasattr(enc, "unfreeze"):
        enc.unfreeze()
        return
    enc.frozen = False
    for p in enc.backbone.parameters():
        p.requires_grad = True
    enc.backbone.train(enc.training)


def param_groups(model: nn.Module, args: Any) -> list:
    """Heads at --head-lr, aggregator/context at --agg-lr, trunk blocks at --lr * llrd^(depth from the top),
    trunk embeddings at --lr * llrd^L. Frozen parameters are left out, shared parameters counted once."""
    head_lr = float(args.head_lr)
    agg_lr = float(getattr(args, "agg_lr", head_lr))
    lr = float(args.lr)
    llrd = float(getattr(args, "llrd", 1.0))
    wd = float(getattr(args, "weight_decay", 0.0))
    seen, heads, agg, emb, top = set(), [], [], [], []
    layers: dict = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if n.startswith("encoder.backbone."):
            sub = n[len("encoder.backbone."):]
            if sub.startswith("blocks."):
                layers.setdefault(int(sub.split(".")[1]), []).append(p)
            elif sub.startswith(("ln_f", "heads")):
                top.append(p)
            else:
                emb.append(p)
        elif n.startswith(("binary_head", "type_head")):
            heads.append(p)
        else:
            agg.append(p)
    groups = []
    if heads:
        groups.append({"params": heads, "lr": head_lr, "name": "heads", "weight_decay": wd})
    if agg:
        groups.append({"params": agg, "lr": agg_lr, "name": "agg", "weight_decay": wd})
    L = (max(layers) + 1) if layers else 0
    if top:
        groups.append({"params": top, "lr": lr, "name": "trunk.top", "weight_decay": wd})
    for i in sorted(layers):
        groups.append({"params": layers[i], "lr": lr * llrd ** (L - 1 - i), "name": f"trunk.block{i}", "weight_decay": wd})
    if emb:
        groups.append({"params": emb, "lr": lr * llrd ** L, "name": "trunk.emb", "weight_decay": wd})
    return groups


def make_sched(opt, total_steps: int, warmup_frac: float):
    total = max(1, int(total_steps))
    warm = max(1, int(round(float(warmup_frac) * total)))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))


def grad_norms(opt) -> dict:
    """Pre-clip gradient norm per named param group, as device tensors (no sync)."""
    out = {}
    for g in opt.param_groups:
        sq = [p.grad.detach().float().pow(2).sum() for p in g["params"] if p.grad is not None]
        if sq:
            out[g.get("name", "group")] = torch.stack(sq).sum().sqrt()
    return out


# ----------------------------------------------------------------------------------------------- EMA

class EMA:
    """Exponential moving average of the weights (decay warms up as (1+n)/(10+n) so the first epochs are not
    dominated by the initialisation). `model` is the averaged copy to evaluate; disabled => the live model."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.enabled = self.decay > 0
        self.n = 0
        if self.enabled:
            self.model = copy.deepcopy(model)
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = model

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for pe, pm in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for be, bm in zip(self.model.buffers(), model.buffers()):
            be.copy_(bm)


def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


# ----------------------------------------------------------------------------------------------- scores / thresholds

@torch.no_grad()
def collect_scores(model: nn.Module, loader, args: Any, device: torch.device, with_meta: bool = False):
    """(scores, targets, valid) numpy arrays of the binary head over a loader; with_meta adds
    (session_idx, row_idx) for per-session smoothing."""
    was_training = model.training
    model.eval()
    kw = get_autocast_kwargs(device, args.amp)
    nb = device.type == "cuda"
    from eeg_gpt.training.finetune import _model_forward   # lazy: finetune imports this module
    S, Y, V, SID, R = [], [], [], [], []
    key = "horizon_pos" if getattr(args, "label", "preictal") == "horizon" else "occ"
    for b in loader:
        with torch.autocast(**kw):
            out = _model_forward(model, b, device, nb)
        S.append(torch.sigmoid(out["binary_logit"].float()).cpu())
        Y.append(b[key].float())
        V.append(b["occ_valid"].float())
        if with_meta:
            SID.append(b["session_idx"]); R.append(b["row_idx"].float())
    if was_training:
        model.train()
    if not S:
        return (np.zeros(0),) * (5 if with_meta else 3)
    out = (torch.cat(S).numpy(), torch.cat(Y).numpy(), torch.cat(V).numpy())
    return out + (torch.cat(SID).numpy(), torch.cat(R).numpy()) if with_meta else out


def thresholds_from(scores, targets, spec: float = 0.90) -> dict:
    """{'spec90': lowest threshold holding specificity >= spec (max sensitivity), 'ba': best balanced-accuracy
    threshold}; None where the set has no positives or no negatives."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return {"spec90": None, "ba": None}
    thr = np.unique(s)
    tn = np.searchsorted(np.sort(s[y == 0]), thr, side="left")
    ok = tn / n_neg >= spec
    return {"spec90": float(thr[ok].min()) if ok.any() else None, "ba": float(threshold_sweep(s, y)["threshold"])}


def apply_threshold(scores, targets, thr: Optional[float]) -> dict:
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    if thr is None or len(s) == 0:
        return {"sens": float("nan"), "spec": float("nan"), "ba": float("nan")}
    pred = s >= thr
    tp = float((pred & (y == 1)).sum()); fn = float((~pred & (y == 1)).sum())
    tn = float((~pred & (y == 0)).sum()); fp = float((pred & (y == 0)).sum())
    sens = tp / max(1.0, tp + fn); spec = tn / max(1.0, tn + fp)
    return {"sens": sens, "spec": spec, "ba": 0.5 * (sens + spec)}


def score_stats(scores, targets) -> dict:
    """metrics.score_stats (reporting implementer) when available; else the same keys computed here."""
    try:
        from eeg_gpt.training import metrics as _m
        fn = getattr(_m, "score_stats", None)
        if fn is not None:
            return fn(scores, targets)
    except Exception:
        pass
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    if len(s) == 0:
        return {"std": float("nan"), "frac_pos": float("nan"), "mean_logit_pos": float("nan"),
                "mean_logit_neg": float("nan"), "separation": float("nan")}
    p = np.clip(s, 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))
    mp = float(lg[y == 1].mean()) if (y == 1).any() else float("nan")
    mn = float(lg[y == 0].mean()) if (y == 0).any() else float("nan")
    sd = float(lg.std()) if len(lg) > 1 else float("nan")
    sep = (mp - mn) / sd if sd and sd > 0 and not (math.isnan(mp) or math.isnan(mn)) else float("nan")
    return {"std": float(s.std()), "frac_pos": float((s > 0.5).mean()), "mean_logit_pos": mp,
            "mean_logit_neg": mn, "separation": float(sep)}


def smoothed(values: list, k: int = 3) -> list:
    """NaN-aware moving average over the k neighbouring epochs (centred)."""
    v = np.asarray(values, dtype=np.float64); out = []
    h = k // 2
    for i in range(len(v)):
        w = v[max(0, i - h): i + h + 1]
        w = w[~np.isnan(w)]
        out.append(float(w.mean()) if len(w) else float("nan"))
    return out


def diag_subset(n_rows: int, n: int = 1000, seed: int = 0) -> list:
    """Fixed random subset of training rows for the train-AUROC diagnostic."""
    rng = np.random.default_rng(seed)
    if n_rows <= n:
        return list(range(n_rows))
    return sorted(rng.choice(n_rows, size=n, replace=False).tolist())


def _f(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
