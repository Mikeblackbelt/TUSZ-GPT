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
    def __init__(self, d: int, heads: int, ffn: int, dropout: float, rope_len: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout, rope_len)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, d), nn.Dropout(dropout),
        )

    def forward(self, x, causal: bool = True):
        x = x + self.attn(self.ln1(x), causal)
        return x + self.ffn(self.ln2(x))


class EEGGPT(nn.Module):
    def __init__(self, n_levels: int = 512, n_channels: int = 20, d: int = 128, layers: int = 4, heads: int = 4,
                 ffn: int = 512, block_size: int = 1024, dropout: float = 0.1, pred_offsets: Sequence[int] = (1,),
                 rope: bool = False, value_emb: bool = False, tie_weights: bool = False, n_value_freqs: int = 8):
        super().__init__()
        self.L, self.d, self.block_size = n_levels, d, block_size
        self.tok_emb = nn.Embedding(n_levels, d)
        # rope: relative positions inside attention (pretraining crops start at random
        # offsets, so absolute position carries no signal); otherwise learned absolute.
        self.pos_emb = None if rope else nn.Embedding(block_size, d)
        self.ch_emb = nn.Embedding(n_channels, d)
        # value_emb: the tokens are ordered amplitudes, which a free embedding table has
        # to rediscover level by level. Fourier features of the amplitude give every
        # level a smooth, ordered code from step 0.
        self.val_proj = None
        if value_emb:
            u = torch.linspace(-1.0, 1.0, n_levels)[:, None]
            f = math.pi * 2.0 ** torch.arange(n_value_freqs)[None]
            self.register_buffer("val_feats", torch.cat([u, torch.sin(u * f), torch.cos(u * f)], dim=-1), persistent=False)
            self.val_proj = nn.Linear(self.val_feats.shape[1], d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, ffn, dropout, block_size if rope else 0) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.pred_offsets = tuple(int(o) for o in pred_offsets)
        self.heads = nn.ModuleDict({str(o): nn.Linear(d, n_levels, bias=False) for o in self.pred_offsets})
        self.apply(self._init)
        if tie_weights and "1" in self.heads:
            self.heads["1"].weight = self.tok_emb.weight
        # GPT-2-style scaled init for residual projections: std /= sqrt(2 * n_layers).
        # Prevents hidden-state variance from growing with depth, which otherwise
        # causes early representation collapse and slows convergence.
        scale = (2 * len(self.blocks)) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 * scale)
            nn.init.normal_(block.ffn[-2].weight, std=0.02 * scale)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden(self, tokens: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        """tokens (B, T) long, channel (B,) long -> (B, T, d) causal hidden states."""
        B, T = tokens.shape
        if T > self.block_size:
            raise ValueError(f"sequence {T} > block_size {self.block_size}")
        emb = self.tok_emb.weight
        if self.val_proj is not None:
            emb = emb + self.val_proj(self.val_feats)
        x = F.embedding(tokens, emb) + self.ch_emb(channel).unsqueeze(1)
        if self.pos_emb is not None:
            x = x + self.pos_emb.weight[:T].unsqueeze(0)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x, causal=True)
        return self.ln_f(x)

    def forward(self, tokens, channel):
        h = self.hidden(tokens, channel)
        return {str(o): self.heads[str(o)](h) for o in self.pred_offsets}, h


def gpt_loss(model: EEGGPT, tokens: torch.Tensor, z: torch.Tensor, channel: torch.Tensor, tokenizer,
             lambda_mse: float = 0.1, label_smoothing: float = 0.0, aux_weight: float = 1.0) -> dict:
    """tokens/z: (B, block+1). Input = tokens[:, :-1]; the head for offset o at
    position t predicts tokens[:, t+o]. CE on every offset (offsets > 1 weighted
    by aux_weight); the Mahdi dual MSE (dequantized expected level vs true z) on
    the offset-1 head.

    `ce@o` is always the plain cross-entropy, so runs stay comparable whatever
    the loss weights are. label_smoothing only changes the optimized `loss`:
    uniform smoothing over L ordered levels puts a floor of ~0.95 nats per head
    (eps 0.1, L 512) under it, so it is off by default.
    """
    inp = tokens[:, :-1]
    logits, _ = model(inp, channel)
    T = inp.shape[1]
    out = {"loss": inp.new_zeros((), dtype=torch.float32)}
    level_vals = tokenizer.level_values(tokens.device)  # (L,)
    for o in model.pred_offsets:
        lg = logits[str(o)][:, : T - (o - 1)]          # positions with a target o steps ahead
        tgt = tokens[:, o: o + lg.shape[1]]
        flat_lg, flat_tgt = lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1)
        ce = F.cross_entropy(flat_lg, flat_tgt)
        out[f"ce@{o}"] = ce
        opt_ce = F.cross_entropy(flat_lg, flat_tgt, label_smoothing=label_smoothing) if label_smoothing > 0 else ce
        out["loss"] = out["loss"] + (1.0 if o == 1 else aux_weight) * opt_ce
        with torch.no_grad():
            out[f"acc@{o}"] = (lg.argmax(-1) == tgt).float().mean()
        if o == 1 and lambda_mse > 0:
            p = lg.float().softmax(-1)
            z_hat = (p * level_vals[None, None, :]).sum(-1)
            mse = F.mse_loss(z_hat, z[:, 1: 1 + lg.shape[1]].float())
            out["mse"] = mse
            out["loss"] = out["loss"] + lambda_mse * mse
    return out


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
              pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Focal binary cross-entropy loss (Lin et al. 2017, RetinaNet).

    Downweights easy negatives by (1-p_t)^gamma, so the model concentrates
    gradient on hard / rare positive examples.  gamma=0 recovers standard BCE.
    pos_weight is applied identically to F.binary_cross_entropy_with_logits so
    class imbalance and example difficulty are handled jointly.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p_t = torch.exp(-bce)   # p_t = sigmoid(logit)*t + (1-sigmoid(logit))*(1-t)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


class WindowEncoder(nn.Module):
    def __init__(self, backbone: EEGGPT, n_channels: int, agg_heads: int = 4, agg_layers: int = 2,
                 dropout: float = 0.1, pool: str = "mean", seq_chunk: int = 256):
        super().__init__()
        self.backbone = backbone
        self.seq_chunk = seq_chunk   # channel-sequences per backbone call (bounds activation memory)
        d = backbone.d
        self.pool = pool
        self.frozen = False          # set by freeze(): backbone runs in eval mode without a graph
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.ch_pos = nn.Embedding(n_channels, d)
        # 2 aggregation layers (up from 1) for richer cross-channel modelling
        self.agg = nn.ModuleList([Block(d, agg_heads, 4 * d, dropout) for _ in range(agg_layers)])
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.cls, std=0.02)

    def freeze(self):
        self.frozen = True
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def _pooled(self, tokens, ch):
        h = self.backbone.hidden(tokens, ch)                                 # (b, T, d)
        if self.pool == "last":
            return h[:, -1]
        if self.pool == "meanmax":
            return 0.5 * (h.mean(1) + h.amax(1))
        return h.mean(1)                                                     # eq. 17

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, C, T) -> (B, d)"""
        B, C, T = tokens.shape
        ch = torch.arange(C, device=tokens.device).repeat(B)                 # (B*C,)
        flat = tokens.reshape(B * C, T)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            v = torch.cat([self._pooled(flat[i: i + self.seq_chunk], ch[i: i + self.seq_chunk])
                           for i in range(0, B * C, self.seq_chunk)], 0)
        v = v.view(B, C, -1) + self.ch_pos.weight[:C].unsqueeze(0)
        x = torch.cat([self.cls.expand(B, -1, -1), v], dim=1)               # (B, 1+C, d)
        for b in self.agg:
            x = b(x, causal=False)
        return self.ln(x[:, 0])


class TemporalContext(nn.Module):
    def __init__(self, d: int, max_blocks: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pos = nn.Embedding(max_blocks, d)
        self.blocks = nn.ModuleList([Block(d, heads, 4 * d, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v (B, N, d) oldest->newest -> (B, d) at the newest block"""
        B, N, _ = v.shape
        x = v + self.pos.weight[:N].unsqueeze(0)
        for b in self.blocks:
            x = b(x, causal=True)
        return self.ln(x[:, -1])


class Head(nn.Module):
    """LayerNorm -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear (Mahdi §III-F 2).

    hidden defaults to 2*d (wider than the paper's d) for stronger non-linear
    capacity at low parameter cost relative to the backbone.
    """

    def __init__(self, d: int, n_out: int, dropout: float = 0.2, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or 2 * d
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class SeizureModel(nn.Module):
    def __init__(self, backbone: EEGGPT, n_channels: int, n_types: int, context_blocks: int = 1, dropout: float = 0.2,
                 pool: str = "mean"):
        super().__init__()
        self.encoder = WindowEncoder(backbone, n_channels, dropout=dropout, pool=pool)
        d = backbone.d
        self.context = TemporalContext(d, max_blocks=max(2, context_blocks), dropout=dropout) if context_blocks > 1 else None
        self.binary_head = Head(d, 1, dropout)
        self.type_head = Head(d, n_types, dropout)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, C, N, T) -> (B, d)"""
        B, C, N, T = tokens.shape
        v = self.encoder(tokens.permute(0, 2, 1, 3).reshape(B * N, C, T)).view(B, N, -1)
        if self.context is not None and N > 1:
            return self.context(v)
        return v[:, -1]

    def forward(self, tokens):
        e = self.embed(tokens)
        return {"binary_logit": self.binary_head(e).squeeze(-1), "type_logits": self.type_head(e), "embedding": e}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())