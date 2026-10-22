"""Models.

EEGGPT           -- causal transformer over amplitude tokens of ONE channel.
                    token embedding (+ optional Fourier features of the amplitude
                    the token stands for, so the 512 levels start out ordered)
                    + channel-ID embedding (electrode-wise modeling, BrainGPT/EEGPT);
                    positions are rotary (RoPE) or a learned absolute embedding.
                    LM head predicts the next token (cross-entropy, optionally tied
                    to the token embedding) with the dual MSE term of Mahdi 2026
                    §III-E on the dequantized expectation; optional extra heads
                    for offsets > 1 (multi-token prediction).
WindowEncoder    -- runs the backbone on every channel of a block, mean-pools
                    over time (Mahdi §III-F eq. 17), then aggregates the C channel
                    embeddings with a small transformer over channels (learned
                    channel embeddings + CLS) -> one vector per block.
TemporalContext  -- causal transformer over the N block vectors of a context
                    (oldest -> newest); returns the newest position.
SeizureModel     -- WindowEncoder (+ TemporalContext) + binary head (preictal /
                    horizon) + multitype head, both LN -> GELU -> Dropout -> FC
                    (Mahdi §III-F 2).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _rotate(x, cos, sin):
    """Rotary position embedding: rotate each (even, odd) feature pair of x (..., T, dh)."""
    rot = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return x * cos + rot * sin


class Attention(nn.Module):
    """Multi-head self-attention on scaled_dot_product_attention with is_causal,
    so flash / memory-efficient kernels are used on GPU (no T x T mask tensor).
    rope_len > 0 rotates q/k by position (RoPE) for sequences up to that length."""

    def __init__(self, d: int, heads: int, dropout: float, rope_len: int = 0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = dropout
        self.out_drop = nn.Dropout(dropout)
        self.rope = rope_len > 0
        if self.rope:
            assert self.dh % 2 == 0, "RoPE needs an even head dimension"
            inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2).float() / self.dh))
            ang = torch.arange(rope_len).float()[:, None] * inv[None]            # (T, dh/2)
            self.register_buffer("rope_cos", ang.cos().repeat_interleave(2, -1), persistent=False)
            self.register_buffer("rope_sin", ang.sin().repeat_interleave(2, -1), persistent=False)

    def forward(self, x, causal: bool):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.rope:
            cos, sin = self.rope_cos[:T].to(q.dtype), self.rope_sin[:T].to(q.dtype)
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=self.drop if self.training else 0.0)
        return self.out_drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float, rope_len: int = 0, drop_path: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout, rope_len)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, d), nn.Dropout(dropout),
        )
        self.drop_path = float(drop_path)   # stochastic depth: drop a whole residual branch per sample (train only)

    def _drop_path(self, y):
        if self.drop_path <= 0.0 or not self.training:
            return y
        keep = 1.0 - self.drop_path
        mask = y.new_empty((y.shape[0],) + (1,) * (y.ndim - 1)).bernoulli_(keep) / keep
        return y * mask

    def forward(self, x, causal: bool = True):
        x = x + self._drop_path(self.attn(self.ln1(x), causal))
        return x + self._drop_path(self.ffn(self.ln2(x)))


def _parse_offsets(v):
    if v is None or v == "":
        return ()
    if isinstance(v, str):
        return tuple(int(o) for o in v.split(",") if o.strip())
    return tuple(int(o) for o in v)


class EEGGPT(nn.Module):
    """Causal transformer trunk over one channel.

    Token path (patch == 1): amplitude-level tokens in, next-level logits out (today's model).
    Patch path (patch > 1): float z in, `patch` samples per position (Linear patch->d plus a
    projection of the patch's log band energies), heads predict the next patch (MSE), its log
    band energies (`spec`) and optionally the patch `h` positions ahead (`coarse@h`).
    `block_size` is in SAMPLES on both paths; positions = block_size // patch.
    Construct from a checkpoint/args config with `EEGGPT.from_config` (the only cfg->model path;
    every key defaults to today's value so old checkpoints reconstruct bit-for-bit).
    """

    CONFIG_DEFAULTS = dict(n_levels=512, n_channels=20, d=256, layers=6, heads=4, ffn=1024, block_size=1024,
                           pred_offsets="1,8,32", rope=True, value_emb=True, tie_weights=True, dropout=0.0,
                           patch=1, objective="token", norm_scope="block", causal=True, tie_mode="tok",
                           dequant="edge", mse_units="sigma", n_bands=8, coarse_horizons="", drop_path=0.0)

    def __init__(self, n_levels: int = 512, n_channels: int = 20, d: int = 128, layers: int = 4, heads: int = 4,
                 ffn: int = 512, block_size: int = 1024, dropout: float = 0.1, pred_offsets: Sequence[int] = (1,),
                 rope: bool = False, value_emb: bool = False, tie_weights: bool = False, n_value_freqs: int = 8,
                 patch: int = 1, objective: str = "auto", causal: bool = True, tie_mode: str = "tok",
                 dequant: str = "edge", mse_units: str = "sigma", n_bands: int = 8, coarse_horizons=(),
                 drop_path: float = 0.0):
        super().__init__()
        self.L, self.d, self.block_size = n_levels, d, block_size
        self.patch = max(1, int(patch))
        if block_size % self.patch:
            raise ValueError(f"block_size {block_size} must be a multiple of patch {self.patch}")
        self.objective = ("patch" if self.patch > 1 else "token") if objective in (None, "auto") else objective
        if self.objective == "patch" and self.patch == 1:
            raise ValueError("objective 'patch' needs patch > 1")
        self.causal = bool(causal)
        self.tie_mode, self.dequant, self.mse_units = tie_mode, dequant, mse_units
        n_pos = block_size // self.patch
        # Submodules are registered in the order of the pre-patch constructor (tok_emb, pos_emb, ch_emb, val_proj, drop,
        # blocks, ln_f, heads): named_parameters() then names the tied table "tok_emb.weight" (the no-decay group of
        # pretrain.py, as before) and the same seed gives the same initial weights as an old run.
        self.val_proj = None
        self.pred_offsets = ()
        self.coarse_horizons = ()
        if self.patch == 1:
            self.tok_emb = nn.Embedding(n_levels, d)
        else:
            # one patch = `patch` z-scored samples; its log band energies (rfft bins grouped into
            # n_bands) are added so the position starts out with an explicit spectral code
            self.n_bands = max(1, min(int(n_bands), self.patch // 2 + 1))
            self.patch_proj = nn.Linear(self.patch, d)
            self.fft_proj = nn.Linear(self.n_bands, d)
        # rope: relative positions inside attention (pretraining crops start at random
        # offsets, so absolute position carries no signal); otherwise learned absolute.
        self.pos_emb = None if rope else nn.Embedding(n_pos, d)
        self.ch_emb = nn.Embedding(n_channels, d)
        # value_emb: the tokens are ordered amplitudes, which a free embedding table has
        # to rediscover level by level. Fourier features of the amplitude give every
        # level a smooth, ordered code from step 0.
        if self.patch == 1 and value_emb:
            u = torch.linspace(-1.0, 1.0, n_levels)[:, None]
            f = math.pi * 2.0 ** torch.arange(n_value_freqs)[None]
            self.register_buffer("val_feats", torch.cat([u, torch.sin(u * f), torch.cos(u * f)], dim=-1), persistent=False)
            self.val_proj = nn.Linear(self.val_feats.shape[1], d)
        self.drop = nn.Dropout(dropout)
        # stochastic depth rate linearly increases from 0 to drop_path with depth (same as timm / DeiT)
        dp_rates = [drop_path * i / max(1, layers - 1) for i in range(layers)]
        self.blocks = nn.ModuleList([Block(d, heads, ffn, dropout, n_pos if rope else 0, dp_rates[i]) for i in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        if self.patch == 1:
            self.pred_offsets = tuple(int(o) for o in pred_offsets)
            self.heads = nn.ModuleDict({str(o): nn.Linear(d, n_levels, bias=False) for o in self.pred_offsets})
        else:
            self.heads = nn.ModuleDict()
            self.next_patch = nn.Linear(d, self.patch)
            self.spec_head = nn.Linear(d, self.n_bands)
            self.coarse_horizons = tuple(h for h in _parse_offsets(coarse_horizons) if h > 1)
            self.coarse = nn.ModuleDict({str(h): nn.Linear(d, self.patch) for h in self.coarse_horizons})
        self.apply(self._init)
        if self.patch == 1 and tie_weights and "1" in self.heads:
            self.heads["1"].weight = self.tok_emb.weight
        # GPT-2-style scaled init for residual projections: std /= sqrt(2 * n_layers).
        # Prevents hidden-state variance from growing with depth, which otherwise
        # causes early representation collapse and slows convergence.
        scale = (2 * len(self.blocks)) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 * scale)
            nn.init.normal_(block.ffn[-2].weight, std=0.02 * scale)

    @classmethod
    def from_config(cls, cfg: dict, dropout: Optional[float] = None) -> "EEGGPT":
        """Build from a checkpoint's `config` (or vars(args)); missing keys take today's defaults,
        so a checkpoint written before a key existed reconstructs the same module tree."""
        g = lambda k: cfg.get(k, cls.CONFIG_DEFAULTS[k])  # noqa: E731
        patch = int(g("patch") or 1)
        objective = g("objective") or "auto"
        if objective == "auto":
            objective = "patch" if patch > 1 else "token"
        return cls(int(g("n_levels")), int(g("n_channels")), int(g("d")), int(g("layers")), int(g("heads")), int(g("ffn")),
                   int(g("block_size")), float(g("dropout") if dropout is None else dropout), _parse_offsets(g("pred_offsets")),
                   rope=bool(g("rope")), value_emb=bool(g("value_emb")), tie_weights=bool(g("tie_weights")),
                   patch=patch, objective=objective, causal=bool(g("causal")), tie_mode=g("tie_mode") or "tok",
                   dequant=g("dequant") or "edge", mse_units=g("mse_units") or "sigma", n_bands=int(g("n_bands") or 8),
                   coarse_horizons=g("coarse_horizons") or "", drop_path=float(g("drop_path") or 0.0))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def n_positions(self) -> int:
        return self.block_size // self.patch

    def _emb_table(self) -> torch.Tensor:
        emb = self.tok_emb.weight
        if self.val_proj is not None:
            emb = emb + self.val_proj(self.val_feats)
        return emb

    def patch_spec(self, zp: torch.Tensor) -> torch.Tensor:
        """zp (..., patch) float z -> (..., n_bands) log band energies (rfft power summed in n_bands contiguous groups)."""
        dev = zp.device.type if zp.device.type in ("cuda", "cpu", "xpu") else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            p = torch.fft.rfft(zp.float(), dim=-1).abs() ** 2                     # (..., patch//2 + 1)
            nb = p.shape[-1]
            edges = torch.linspace(0, nb, self.n_bands + 1).round().long().tolist()
            bands = [p[..., edges[i]: max(edges[i] + 1, edges[i + 1])].sum(-1, keepdim=True) for i in range(self.n_bands)]
            return torch.log(torch.cat(bands, dim=-1) + 1e-6)

    def hidden(self, tokens: torch.Tensor, channel: torch.Tensor, return_layers: bool = False):
        """tokens (B, T) long levels (token path) or float z (patch path), channel (B,) long
        -> (B, P, d) hidden states, P = T // patch. With return_layers=True a list with the
        output of every block (ln_f applied to the last entry only)."""
        B, T = tokens.shape
        if T > self.block_size:
            raise ValueError(f"sequence {T} samples > block_size {self.block_size}")
        if self.patch == 1:
            if tokens.dtype != torch.long:
                raise TypeError(f"token path expects int64 levels, got {tokens.dtype}")
            x = F.embedding(tokens, self._emb_table())
        else:
            if not tokens.is_floating_point():
                raise TypeError(f"patch path expects float z, got {tokens.dtype}")
            if T % self.patch:
                raise ValueError(f"sequence {T} samples is not a multiple of patch {self.patch}")
            zp = tokens.view(B, T // self.patch, self.patch)
            x = self.patch_proj(zp.to(self.patch_proj.weight.dtype)) + self.fft_proj(self.patch_spec(zp).to(self.fft_proj.weight.dtype))
        x = x + self.ch_emb(channel).unsqueeze(1)
        if self.pos_emb is not None:
            x = x + self.pos_emb.weight[: x.shape[1]].unsqueeze(0)
        x = self.drop(x)
        layers = []
        for b in self.blocks:
            x = b(x, causal=self.causal)
            if return_layers:
                layers.append(x)
        if return_layers:
            layers[-1] = self.ln_f(x)
            return layers
        return self.ln_f(x)

    def forward(self, tokens, channel):
        h = self.hidden(tokens, channel)
        if self.patch == 1:
            out = {}
            for o in self.pred_offsets:
                if o == 1 and self.tie_mode == "full":
                    out["1"] = F.linear(h, self._emb_table().to(h.dtype))   # tie to the amplitude-aware table
                else:
                    out[str(o)] = self.heads[str(o)](h)
            return out, h
        out = {"next_patch": self.next_patch(h), "spec": self.spec_head(h)}
        for hz in self.coarse_horizons:
            out[f"coarse@{hz}"] = self.coarse[str(hz)](h)
        return out, h


def _gaussian_bin_ce(pred_z: torch.Tensor, sigma: torch.Tensor, tgt: torch.Tensor, tokenizer) -> torch.Tensor:
    """Cross-entropy of a Gaussian predictive density N(pred_z, sigma) quantised onto the tokenizer's
    bins, evaluated at the target levels `tgt`. Used for the copy-last and AR(8) floors."""
    edges = tokenizer.level_values(pred_z.device, "edge").float()
    w = tokenizer.bin_width
    lo = edges[tgt]; hi = lo + w
    lo = torch.where(tgt == 0, torch.full_like(lo, -1e4), lo)
    hi = torch.where(tgt == tokenizer.L - 1, torch.full_like(hi, 1e4), hi)
    sig = sigma.clamp_min(1e-4)
    cdf = lambda v: 0.5 * (1.0 + torch.erf((v - pred_z) / (sig * math.sqrt(2.0))))  # noqa: E731
    p = (cdf(hi) - cdf(lo)).clamp_min(1e-12)
    return -(p.log()).mean()


def _ar_fit(z: torch.Tensor, order: int = 8):
    """Least-squares AR(order) on z (B, T) pooled over the batch -> coefficients (order,) and one-step
    predictions (B, T - order) for positions order..T-1 (predicting z[:, t] from z[:, t-order:t])."""
    zf = z.float()
    B, T = zf.shape
    X = torch.stack([zf[:, order - 1 - i: T - 1 - i] for i in range(order)], dim=-1).reshape(-1, order)   # lag 1..order
    y = zf[:, order:].reshape(-1)
    coef = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)
    return coef, (X @ coef).view(B, T - order)


def _token_floors(tokens: torch.Tensor, z: torch.Tensor, tokenizer) -> dict:
    """Context-free floors for the token objective: copy the previous level, and a least-squares AR(8)
    on z with a Gaussian residual quantised onto the bins. In nats, comparable to ce@1."""
    out = {}
    prev, nxt = tokens[:, :-1], tokens[:, 1:]
    out["acc_copy"] = (prev == nxt).float().mean()
    zf = z.float()
    d = zf[:, 1:] - zf[:, :-1]
    out["ce_copy"] = _gaussian_bin_ce(zf[:, :-1].reshape(-1), d.std(), nxt.reshape(-1), tokenizer)
    order = 8
    if zf.shape[1] > order + 8:
        _, pred = _ar_fit(zf, order)
        tgt_z = zf[:, order:]
        res = (pred - tgt_z)
        out["mse_ar8"] = (res ** 2).mean()
        out["ce_ar8"] = _gaussian_bin_ce(pred.reshape(-1), res.std(), tokens[:, order:].reshape(-1), tokenizer)
    return out


def _patch_floors(z: torch.Tensor, patch: int) -> dict:
    """Floors for the patch objective: repeat the last patch, and recursive AR(8) extrapolation over
    the next patch (coefficients fitted by least squares on the batch). MSE in z units, comparable to mse_patch."""
    out = {}
    B, T = z.shape
    P = T // patch
    zp = z.float()[:, : P * patch].view(B, P, patch)
    out["mse_copy_patch"] = ((zp[:, 1:] - zp[:, :-1]) ** 2).mean()
    order = 8
    if T > order + patch + 8:
        coef, _ = _ar_fit(z, order)
        # recursive forecast of patch p from the `order` samples before it, for every p whose history exists
        starts = torch.arange(1, P, device=z.device) * patch
        starts = starts[starts >= order]
        if len(starts):
            idx = starts[:, None] - order + torch.arange(order, device=z.device)[None]
            hist = z.float()[:, idx]                                                   # (B, n, order) oldest..newest
            preds = []
            for _ in range(patch):
                nxt = (hist * coef.flip(0)[None, None, :]).sum(-1)                      # newest sample <-> lag-1 coef
                preds.append(nxt)
                hist = torch.cat([hist[:, :, 1:], nxt.unsqueeze(-1)], dim=-1)
            pred = torch.stack(preds, dim=-1)                                          # (B, n, patch)
            out["mse_ar8"] = ((pred - zp[:, starts // patch]) ** 2).mean()
    return out


def gpt_loss(model: EEGGPT, tokens: torch.Tensor, z: torch.Tensor, channel: torch.Tensor, tokenizer,
             lambda_mse: float = 0.1, label_smoothing: float = 0.0, aux_weight: float = 1.0,
             spec_weight: float = 1.0, compute_floors: bool = True) -> dict:
    """Token path -- tokens/z: (B, block+1). Input = tokens[:, :-1]; the head for offset o at
    position t predicts tokens[:, t+o]. CE on every offset (offsets > 1 weighted
    by aux_weight); composite cross-entropy is normalized by the sum of CE head weights;
    the Mahdi dual MSE (dequantized expected level vs true z) on the offset-1 head.

    `ce@o` is always the plain cross-entropy, so runs stay comparable whatever
    the loss weights are. label_smoothing only changes the optimized `loss`:
    uniform smoothing over L ordered levels puts a floor of ~0.95 nats per head
    (eps 0.1, L 512) under it, so it is off by default.

    Patch path -- tokens is the float z sequence (B, block + patch): input = z[:, :-patch] as
    P = block/patch patches; `mse_patch` = MSE of the next patch, `mse_spec` = MSE of its log band
    energies (weight spec_weight), `mse_coarse@h` = MSE of the patch h positions ahead (weight
    aux_weight). loss = mse_patch + spec_weight * mse_spec + aux_weight * sum(mse_coarse@h).

    Both paths report context-free floors under no_grad (`ce_copy`, `acc_copy`, `ce_ar8`, `mse_ar8`
    on the token path; `mse_copy_patch`, `mse_ar8` on the patch path).
    """
    if getattr(model, "objective", "token") == "patch":
        return _patch_loss(model, tokens, z, channel, aux_weight, spec_weight, compute_floors=compute_floors)
    inp = tokens[:, :-1]
    logits, _ = model(inp, channel)
    T = inp.shape[1]
    out = {}
    dequant = getattr(model, "dequant", "edge")
    level_vals = tokenizer.level_values(tokens.device, dequant)  # (L,)
    ce_loss_sum = inp.new_zeros((), dtype=torch.float32)
    ce_weight_sum = 0.0
    for o in model.pred_offsets:
        lg = logits[str(o)][:, : T - (o - 1)]          # positions with a target o steps ahead
        tgt = tokens[:, o: o + lg.shape[1]]
        flat_lg, flat_tgt = lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1)
        ce = F.cross_entropy(flat_lg, flat_tgt)
        out[f"ce@{o}"] = ce
        opt_ce = F.cross_entropy(flat_lg, flat_tgt, label_smoothing=label_smoothing) if label_smoothing > 0 else ce
        w_o = 1.0 if o == 1 else aux_weight
        ce_loss_sum = ce_loss_sum + w_o * opt_ce
        ce_weight_sum += w_o
        if compute_floors:
            with torch.no_grad():
                out[f"acc@{o}"] = (lg.argmax(-1) == tgt).float().mean()
        if o == 1 and lambda_mse > 0:
            p = lg.float().softmax(-1)
            if getattr(model, "mse_units", "sigma") == "bin":
                lv_hat = (p * torch.arange(tokenizer.L, device=p.device, dtype=p.dtype)[None, None, :]).sum(-1)
                mse = F.mse_loss(lv_hat, tgt.float())                     # bin^2 units
            else:
                z_hat = (p * level_vals[None, None, :]).sum(-1)
                mse = F.mse_loss(z_hat, z[:, 1: 1 + lg.shape[1]].float())  # sigma^2 units (historical)
            out["mse"] = mse
    if ce_weight_sum > 0:
        out["loss"] = ce_loss_sum / ce_weight_sum
    else:
        out["loss"] = ce_loss_sum
    if "mse" in out and lambda_mse > 0:
        out["loss"] = out["loss"] + lambda_mse * out["mse"]
    if compute_floors:
        with torch.no_grad():
            out.update(_token_floors(tokens, z, tokenizer))
    return out


def _patch_loss(model: EEGGPT, zseq: torch.Tensor, z: torch.Tensor, channel: torch.Tensor,
                aux_weight: float, spec_weight: float, compute_floors: bool = True) -> dict:
    k = model.patch
    zf = zseq.float()
    B, Tp = zf.shape
    P = (Tp - k) // k                                   # input patches; patch P (the last) is only a target
    inp = zf[:, : P * k]
    tgt = zf[:, k: (P + 1) * k].view(B, P, k)           # next patch for every input position
    preds, _ = model(inp.to(zseq.dtype), channel)
    out = {}
    out["mse_patch"] = F.mse_loss(preds["next_patch"].float(), tgt)
    spec_t = model.patch_spec(tgt)
    out["mse_spec"] = F.mse_loss(preds["spec"].float(), spec_t)
    out["loss"] = out["mse_patch"] + spec_weight * out["mse_spec"]
    for hz in model.coarse_horizons:
        if P - hz + 1 <= 0:
            continue
        pr = preds[f"coarse@{hz}"][:, : P - hz + 1].float()
        tg = zf[:, hz * k: (P + 1) * k].view(B, P - hz + 1, k)
        m = F.mse_loss(pr, tg)
        out[f"mse_coarse@{hz}"] = m
        out["loss"] = out["loss"] + aux_weight * m
    if compute_floors:
        with torch.no_grad():
            out.update(_patch_floors(zf, k))
    return out

def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
              pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Focal binary cross-entropy loss (Lin et al. 2017, RetinaNet).

    Downweights easy examples by (1-p_t)^gamma, so the model concentrates
    gradient on hard / rare positive examples.  gamma=0 recovers (weighted) BCE.

    p_t is taken from the UNWEIGHTED cross-entropy: computing it from the
    pos_weighted one gave p_t = p^pos_weight for positives, i.e. a positive at
    p = 0.9 looked "hard" (p_t = 0.9^29 ~ 0.05) while a negative at the same
    confidence looked easy, which made a constant predictor a fixed point of
    the loss (the class collapse). pos_weight multiplies the positive terms
    exactly as F.binary_cross_entropy_with_logits does.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    w = torch.ones_like(bce)
    if gamma > 0:
        p_t = torch.exp(-bce)   # p_t = sigmoid(logit)*t + (1-sigmoid(logit))*(1-t)
        w = w * (1.0 - p_t) ** gamma
    if pos_weight is not None:
        w = w * (targets * pos_weight + (1.0 - targets))
    return (w * bce).mean()


@torch.no_grad()
def init_heads(model: nn.Module, pos_frac: float, type_counts=None) -> None:
    """Zero the last Linear of both heads; binary bias = log-odds of the preictal fraction the sampler
    presents, type bias = log class frequencies. Step 0 then predicts the prior (Lin et al. 2017) instead
    of a random constant, which is where the collapse used to start."""
    def last_linear(m):
        lin = [x for x in m.modules() if isinstance(x, nn.Linear)]
        return lin[-1] if lin else None
    p = float(min(max(pos_frac, 1e-4), 1 - 1e-4))
    lin = last_linear(model.binary_head) if hasattr(model, "binary_head") else None
    if lin is not None:
        lin.weight.zero_()
        if lin.bias is not None:
            lin.bias.fill_(math.log(p / (1 - p)))
    lin = last_linear(model.type_head) if hasattr(model, "type_head") else None
    if lin is not None:
        lin.weight.zero_()
        if lin.bias is not None:
            lin.bias.zero_()
            if type_counts is not None and len(type_counts) == lin.bias.numel():
                c = torch.tensor([float(x) for x in type_counts], dtype=lin.bias.dtype)
                freq = c / c.sum().clamp_min(1.0)
                lin.bias.copy_(torch.where(freq > 0, torch.log(freq.clamp_min(1e-12)), torch.full_like(freq, -10.0)))


#: fine-tuning architecture knobs of SeizureModel / WindowEncoder / TemporalContext (new defaults). The values that
#: reproduce the architecture before these knobs existed are in OLD_ARCH; load_general_seizure_model() reads a
#: checkpoint's config with those as the fallbacks so old Stage 2 checkpoints rebuild the same module tree.
DEFAULT_ARCH = {
    "feature_layers": "mix", "agg": "attn", "agg_layers": 1, "agg_dim": 128, "pool_tail_frac": 0.25, "pool_queries": 1,
    "drop_path": 0.0, "context_mode": "delta", "encode_context": "block", "encode_tail": 0, "head_hidden": None,
}
OLD_ARCH = {
    "feature_layers": "last", "agg": "cls", "agg_layers": 2, "agg_dim": None, "pool_tail_frac": 0.25, "pool_queries": 1,
    "drop_path": 0.0, "context_mode": "transformer", "encode_context": "block", "encode_tail": 0, "head_hidden": None,
}


def _parse_feature_layers(spec, n_layers: int):
    """'last' | 'mix' | '3,4,5,6' (1-based block indices) -> ('last' | 'mix' | 'select', [0-based indices])."""
    spec = "last" if spec is None else str(spec).strip().lower()
    if spec in ("last", "mix"):
        return spec, []
    idx = sorted({min(n_layers, max(1, int(t))) - 1 for t in spec.split(",") if t.strip()})
    if not idx:
        return "last", []
    return "select", idx


class WindowEncoder(nn.Module):
    """The pretrained trunk on every channel of every block, pooled over positions, then aggregated over channels.

    tokens (B, C, N, T) -> (B, N, d); a (B, C, T) input is treated as N = 1 and returns (B, d).

    feature_layers : "last" (ln_f of the last block, as before) | "mix" (softmax-weighted sum of all layer outputs,
                     ELMo-style) | "3,4,5,6" (mean of those 1-based layers). Needs EEGGPT.hidden(..., return_layers=True);
                     falls back to "last" with a warning when the trunk does not offer it.
    pool           : mean | last | meanmax | tail (mean over the last pool_tail_frac of positions) |
                     attn (pool_queries learned queries over positions) | meanstd (mean ++ std -> Linear)
    agg            : cls (CLS + agg_layers Blocks at width d, as before) | mean (mean over channels) |
                     attn (channels projected to agg_dim, one Block with a learned query token, projected back)
    encode_context : block (each block is its own trunk sequence) | whole (the N blocks of a channel form one sequence
                     of N*T samples; positions are pooled per block, so the trunk sees the 16 s amplitude trend)
    encode_tail    : > 0 -> only the last encode_tail samples of every block go through the trunk (block mode)
    The layer mix and the pooling run inside _pooled, i.e. inside the activation checkpoint, so only (b, n_blocks, d)
    per chunk survives the forward pass. Checkpointing is skipped for patch trunks (few positions, cheap).
    """

    def __init__(self, backbone: EEGGPT, n_channels: int, agg_heads: int = 4, agg_layers: int = 1,
                 dropout: float = 0.1, pool: str = "tail", seq_chunk: int = 256, agg: str = "attn",
                 agg_dim: Optional[int] = 128, pool_tail_frac: float = 0.25, pool_queries: int = 1,
                 feature_layers: str = "mix", drop_path: float = 0.0, encode_context: str = "block", encode_tail: int = 0):
        super().__init__()
        self.backbone = backbone
        self.seq_chunk = seq_chunk   # channel-sequences per backbone call; with grad, each chunk is recomputed in backward
        d = backbone.d
        self.d = d
        self.pool = pool
        self.pool_tail_frac = float(pool_tail_frac)
        self.frozen = False          # set by freeze(): backbone runs in eval mode without a graph
        self.agg_mode = agg
        self.encode_context = encode_context
        self.encode_tail = int(encode_tail)
        self.patch = int(getattr(backbone, "patch", 1))
        if self.encode_tail and self.encode_tail % self.patch:
            raise ValueError(f"--encode-tail {self.encode_tail} must be a multiple of the trunk's patch {self.patch}")
        # which trunk layers feed the classifier
        n_layers = len(backbone.blocks)
        self.layer_mode, self.layer_idx = _parse_feature_layers(feature_layers, n_layers)
        import inspect
        self._hidden_has_layers = "return_layers" in inspect.signature(backbone.hidden).parameters
        if self.layer_mode != "last" and not self._hidden_has_layers:
            import warnings
            warnings.warn("EEGGPT.hidden() has no return_layers: feature_layers falls back to 'last'")
            self.layer_mode, self.layer_idx = "last", []
        self.layer_w = nn.Parameter(torch.zeros(n_layers)) if self.layer_mode == "mix" else None
        # pooling over positions
        if pool == "attn":
            self.pool_q = nn.Parameter(torch.zeros(int(pool_queries), d))
            self.pool_k = nn.Linear(d, d)
            nn.init.normal_(self.pool_q, std=0.02)
        elif pool == "meanstd":
            self.pool_proj = nn.Linear(2 * d, d)
        elif pool not in ("mean", "last", "meanmax", "tail"):
            raise ValueError(f"unknown pool {pool!r}")
        # aggregation over channels
        if agg == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.ch_pos = nn.Embedding(n_channels, d)
            self.agg = nn.ModuleList([Block(d, agg_heads, 4 * d, dropout, drop_path=drop_path) for _ in range(agg_layers)])
            self.ln = nn.LayerNorm(d)
            nn.init.normal_(self.cls, std=0.02)
        elif agg == "attn":
            a = int(agg_dim or d)
            self.agg_dim = a
            self.ch_pos = nn.Embedding(n_channels, a)
            self.agg_in = nn.Linear(d, a)
            self.cls = nn.Parameter(torch.zeros(1, 1, a))
            self.agg = nn.ModuleList([Block(a, agg_heads, 4 * a, dropout, drop_path=drop_path) for _ in range(max(1, agg_layers))])
            self.agg_ln = nn.LayerNorm(a)
            self.agg_out = nn.Linear(a, d)
            self.ln = nn.LayerNorm(d)
            nn.init.normal_(self.cls, std=0.02)
        elif agg == "mean":
            self.ch_pos = nn.Embedding(n_channels, d)
            self.ln = nn.LayerNorm(d)
        else:
            raise ValueError(f"unknown agg {agg!r}")

    def freeze(self):
        self.frozen = True
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze(self):
        """Undo freeze(): the trunk gets gradients again and follows the module's train/eval mode."""
        self.frozen = False
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def _hidden(self, tokens, ch):
        """(b, T) -> (b, P, d): the trunk's positions, after the layer selection / mix.

        Only the trunk runs without a graph while frozen: the layer-mix weights and the pooling
        parameters below still train during the linear-probe epochs."""
        trunk_grad = torch.is_grad_enabled() and not self.frozen
        if self.layer_mode == "last":
            with torch.set_grad_enabled(trunk_grad):
                return self.backbone.hidden(tokens, ch)
        with torch.set_grad_enabled(trunk_grad):
            layers = self.backbone.hidden(tokens, ch, return_layers=True)
        if isinstance(layers, torch.Tensor):
            return layers
        n = len(self.backbone.blocks)
        layers = list(layers)[-n:] if len(layers) > n else list(layers)
        if self.layer_mode == "mix":
            w = torch.softmax(self.layer_w[: len(layers)], 0)
            return sum(w[i] * layers[i] for i in range(len(layers)))
        idx = [i for i in self.layer_idx if i < len(layers)] or [len(layers) - 1]
        return sum(layers[i] for i in idx) / len(idx)

    def _pool(self, h):
        """(m, P, d) -> (m, d)"""
        if self.pool == "last":
            return h[:, -1]
        if self.pool == "meanmax":
            return 0.5 * (h.mean(1) + h.amax(1))
        if self.pool == "tail":
            k = max(1, int(round(h.shape[1] * self.pool_tail_frac)))
            return h[:, -k:].mean(1)
        if self.pool == "attn":
            att = torch.softmax(self.pool_k(h) @ self.pool_q.t() / math.sqrt(self.d), dim=1)   # (m, P, q)
            return torch.einsum("mpq,mpd->mqd", att, h).mean(1)
        if self.pool == "meanstd":
            return self.pool_proj(torch.cat([h.mean(1), h.std(1, unbiased=False)], -1))
        return h.mean(1)                                                     # eq. 17

    def _pooled(self, tokens, ch, n_blocks: int = 1):
        """(b, n_blocks * T) sequences -> (b, n_blocks, d): layer mix + pooling per block, inside the checkpoint."""
        h = self._hidden(tokens, ch)                                         # (b, P, d)
        b, P, d = h.shape
        h = h.reshape(b * n_blocks, P // n_blocks, d)
        return self._pool(h).view(b, n_blocks, d)

    def _encode(self, seq, ch, n_blocks: int) -> torch.Tensor:
        use_ckpt = torch.is_grad_enabled() and not self.frozen and self.patch == 1
        # under autograd every chunk's activations would stay alive until backward (B*C*N sequences of T tokens);
        # checkpointing keeps only the pooled (b, n_blocks, d) output per chunk and recomputes the chunk in backward.
        # While frozen, _hidden runs the trunk without a graph and only the mix / pooling parameters build one.
        pooled = (lambda t, c: checkpoint(self._pooled, t, c, n_blocks, use_reentrant=False)) if use_ckpt \
            else (lambda t, c: self._pooled(t, c, n_blocks))
        return torch.cat([pooled(seq[i: i + self.seq_chunk], ch[i: i + self.seq_chunk])
                          for i in range(0, seq.shape[0], self.seq_chunk)], 0)

    def _aggregate(self, v: torch.Tensor) -> torch.Tensor:
        """v (M, C, d) channel vectors -> (M, d)"""
        M, C, _ = v.shape
        if self.agg_mode == "mean":
            return self.ln((v + self.ch_pos.weight[:C].unsqueeze(0)).mean(1))
        if self.agg_mode == "attn":
            x = self.agg_in(v) + self.ch_pos.weight[:C].unsqueeze(0)
            x = torch.cat([self.cls.expand(M, -1, -1), x], dim=1)           # (M, 1+C, a)
            for b in self.agg:
                x = b(x, causal=False)
            return self.ln(self.agg_out(self.agg_ln(x[:, 0])))
        v = v + self.ch_pos.weight[:C].unsqueeze(0)
        x = torch.cat([self.cls.expand(M, -1, -1), v], dim=1)               # (M, 1+C, d)
        for b in self.agg:
            x = b(x, causal=False)
        return self.ln(x[:, 0])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, C, N, T) -> (B, N, d); tokens (B, C, T) -> (B, d)"""
        squeeze = tokens.ndim == 3
        if squeeze:
            tokens = tokens.unsqueeze(2)
        B, C, N, T = tokens.shape
        if self.encode_tail and self.encode_tail < T:
            tokens = tokens[..., -self.encode_tail:]
            T = self.encode_tail
        if self.encode_context == "whole" and N > 1:
            seq = tokens.reshape(B * C, N * T)                               # rows ordered (b, c)
            ch = torch.arange(C, device=tokens.device).repeat(B)
            v = self._encode(seq, ch, N).view(B, C, N, -1).permute(0, 2, 1, 3)   # (B, N, C, d)
        else:
            seq = tokens.permute(0, 2, 1, 3).reshape(B * N * C, T)          # rows ordered (b, n, c)
            ch = torch.arange(C, device=tokens.device).repeat(B * N)
            v = self._encode(seq, ch, 1).view(B, N, C, -1)
        e = self._aggregate(v.reshape(B * N, C, -1)).view(B, N, -1)
        return e[:, 0] if squeeze else e


class TemporalContext(nn.Module):
    """The N block vectors of a context (oldest -> newest) -> one vector.

    mode: transformer (causal Blocks over the N positions, as before) | mean | concat (all N concatenated, Linear) |
          delta (newest + W(newest - mean of the previous ones), W zero-initialised: starts as "newest block only")
    """

    def __init__(self, d: int, max_blocks: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1,
                 mode: str = "transformer", drop_path: float = 0.0):
        super().__init__()
        self.mode = mode
        self.max_blocks = int(max_blocks)
        if mode == "transformer":
            self.pos = nn.Embedding(max_blocks, d)
            self.blocks = nn.ModuleList([Block(d, heads, 4 * d, dropout, drop_path=drop_path) for _ in range(layers)])
        elif mode == "delta":
            self.delta = nn.Linear(d, d)
            nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        elif mode == "concat":
            self.proj = nn.Linear(self.max_blocks * d, d)
        elif mode != "mean":
            raise ValueError(f"unknown context mode {mode!r}")
        self.ln = nn.LayerNorm(d)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v (B, N, d) oldest->newest -> (B, d) at the newest block"""
        B, N, d = v.shape
        if self.mode == "mean":
            return self.ln(v.mean(1))
        if self.mode == "delta":
            prev = v[:, :-1].mean(1) if N > 1 else torch.zeros_like(v[:, -1])
            return self.ln(v[:, -1] + self.delta(v[:, -1] - prev))
        if self.mode == "concat":
            if N < self.max_blocks:                                          # oldest blocks missing: zero-pad in front
                v = torch.cat([v.new_zeros(B, self.max_blocks - N, d), v], 1)
            return self.ln(self.proj(v[:, -self.max_blocks:].reshape(B, -1)))
        x = v + self.pos.weight[:N].unsqueeze(0)
        for b in self.blocks:
            x = b(x, causal=True)
        return self.ln(x[:, -1])


class Head(nn.Module):
    """LayerNorm -> Dropout -> Linear (or with hidden MLP layer if hidden > 0; Mahdi §III-F 2).

    Linear head by default for robust generalization on small seizure sets without
    overparameterized non-linear memorization; hidden > 0 adds an intermediate MLP layer.
    """

    def __init__(self, d: int, n_out: int, dropout: float = 0.2, hidden: Optional[int] = None):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(dropout),
                nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_out),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(dropout),
                nn.Linear(d, n_out),
            )

    def forward(self, x):
        return self.net(x)


class SeizureModel(nn.Module):
    """WindowEncoder (+ TemporalContext) (+ hand-crafted feature branch) -> binary head + type head.

    feat_branch: none | bandpower (only the band-power features of `feats`) | full (all of `feats`) | feat_only
    (the feature branch alone; the trunk is never run -- the floor / ablation). The branch is LayerNorm -> Linear -> GELU
    on `feats` standardised with training-fold statistics (set_feature_stats(); buffers in the state dict, so a
    checkpoint carries them). With a branch, forward() takes `feats` (B, feat_dim) from the dataset's batch key `feats`
    and fuses Linear(2d -> d) with the trunk embedding. `arch` holds the WindowEncoder / TemporalContext knobs
    (see DEFAULT_ARCH); `bandpower_idx` are the indices of the band powers inside `feats` (data/features.py).
    """

    def __init__(self, backbone: EEGGPT, n_channels: int, n_types: int, context_blocks: int = 1, dropout: float = 0.2,
                 pool: str = "tail", feat_branch: str = "none", feat_dim: int = 0, arch: Optional[dict] = None,
                 bandpower_idx=None):
        super().__init__()
        a = dict(DEFAULT_ARCH, **(arch or {}))
        self.arch = a
        self.feat_branch = feat_branch
        self.feat_dim = int(feat_dim)
        d = backbone.d
        self.encoder = WindowEncoder(backbone, n_channels, dropout=dropout, pool=pool, agg=a["agg"], agg_layers=a["agg_layers"],
                                     agg_dim=a["agg_dim"], pool_tail_frac=a["pool_tail_frac"], pool_queries=a["pool_queries"],
                                     feature_layers=a["feature_layers"], drop_path=a["drop_path"],
                                     encode_context=a["encode_context"], encode_tail=a["encode_tail"])
        self.context = (TemporalContext(d, max_blocks=max(2, context_blocks), dropout=dropout, mode=a["context_mode"],
                                        drop_path=a["drop_path"]) if context_blocks > 1 else None)
        if feat_branch != "none":
            if self.feat_dim <= 0:
                raise ValueError("feat_branch needs feat_dim > 0 (data/features.py feat_dim(n_channels))")
            self.register_buffer("feat_mean", torch.zeros(self.feat_dim))
            self.register_buffer("feat_sd", torch.ones(self.feat_dim))
            if feat_branch == "bandpower":
                if bandpower_idx is None:
                    raise ValueError("feat_branch 'bandpower' needs bandpower_idx")
                self.register_buffer("feat_idx", torch.as_tensor(bandpower_idx, dtype=torch.long))
                fin = int(self.feat_idx.numel())
            else:
                self.feat_idx = None
                fin = self.feat_dim
            self.feat_net = nn.Sequential(nn.LayerNorm(fin), nn.Dropout(dropout), nn.Linear(fin, d), nn.GELU(), nn.Dropout(dropout))
            self.fuse = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, d)) if feat_branch != "feat_only" else None
        hidden = a.get("head_hidden")
        self.binary_head = Head(d, 1, dropout, hidden=hidden)
        self.type_head = Head(d, n_types, dropout, hidden=hidden)

    @torch.no_grad()
    def set_feature_stats(self, mean, sd):
        """Standardisation of `feats` (training-fold statistics); call once per fold before training."""
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).reshape(-1))
        self.feat_sd.copy_(torch.as_tensor(sd, dtype=torch.float32).reshape(-1).clamp_min(1e-6))

    def _features(self, feats: torch.Tensor) -> torch.Tensor:
        x = (feats.float() - self.feat_mean) / self.feat_sd
        if self.feat_idx is not None:
            x = x[:, self.feat_idx]
        return self.feat_net(x)

    def embed(self, tokens: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tokens (B, C, N, T) [, feats (B, feat_dim)] -> (B, d)"""
        if self.feat_branch == "feat_only":
            if feats is None:
                raise ValueError("feat_branch 'feat_only' needs feats")
            return self._features(feats)
        v = self.encoder(tokens)                                             # (B, N, d)
        e = self.context(v) if (self.context is not None and v.shape[1] > 1) else v[:, -1]
        if self.feat_branch != "none":
            if feats is None:
                raise ValueError(f"feat_branch {self.feat_branch!r} needs feats")
            e = self.fuse(torch.cat([e, self._features(feats)], -1))
        return e

    def forward(self, tokens, feats: Optional[torch.Tensor] = None):
        e = self.embed(tokens, feats)
        return {"binary_logit": self.binary_head(e).squeeze(-1), "type_logits": self.type_head(e), "embedding": e}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())