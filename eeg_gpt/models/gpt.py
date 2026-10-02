"""Models.

EEGGPT           -- causal transformer over amplitude tokens of ONE channel.
                    token embedding + learned positional embedding + channel-ID
                    embedding (electrode-wise modeling, BrainGPT/EEGPT). LM head
                    predicts the next token (cross-entropy) with the dual MSE
                    term of Mahdi 2026 §III-E on the dequantized expectation;
                    optional extra heads for offsets > 1 (multi-token prediction).
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


class Attention(nn.Module):
    """Multi-head self-attention on scaled_dot_product_attention with is_causal,
    so flash / memory-efficient kernels are used on GPU (no T x T mask tensor)."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = dropout
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, causal: bool):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=self.drop if self.training else 0.0)
        return self.out_drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d), nn.Dropout(dropout))

    def forward(self, x, causal: bool = True):
        x = x + self.attn(self.ln1(x), causal)
        return x + self.ffn(self.ln2(x))


class EEGGPT(nn.Module):
    def __init__(self, n_levels: int = 512, n_channels: int = 20, d: int = 128, layers: int = 4, heads: int = 4,
                 ffn: int = 512, block_size: int = 1024, dropout: float = 0.1, pred_offsets: Sequence[int] = (1,)):
        super().__init__()
        self.L, self.d, self.block_size = n_levels, d, block_size
        self.tok_emb = nn.Embedding(n_levels, d)
        self.pos_emb = nn.Embedding(block_size, d)
        self.ch_emb = nn.Embedding(n_channels, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, ffn, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.pred_offsets = tuple(int(o) for o in pred_offsets)
        self.heads = nn.ModuleDict({str(o): nn.Linear(d, n_levels, bias=False) for o in self.pred_offsets})
        self.apply(self._init)

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
        x = self.tok_emb(tokens) + self.pos_emb.weight[:T].unsqueeze(0) + self.ch_emb(channel).unsqueeze(1)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x, causal=True)
        return self.ln_f(x)

    def forward(self, tokens, channel):
        h = self.hidden(tokens, channel)
        return {str(o): self.heads[str(o)](h) for o in self.pred_offsets}, h


def gpt_loss(model: EEGGPT, tokens: torch.Tensor, z: torch.Tensor, channel: torch.Tensor, tokenizer,
             lambda_mse: float = 0.1) -> dict:
    """tokens/z: (B, block+1). Input = tokens[:, :-1]; the head for offset o at
    position t predicts tokens[:, t+o]. CE on every offset; the Mahdi dual MSE
    (dequantized expected level vs true z) on the offset-1 head."""
    inp = tokens[:, :-1]
    logits, _ = model(inp, channel)
    T = inp.shape[1]
    out = {"loss": inp.new_zeros((), dtype=torch.float32)}
    level_vals = tokenizer.level_values(tokens.device)  # (L,)
    for o in model.pred_offsets:
        lg = logits[str(o)][:, : T - (o - 1)]          # positions with a target o steps ahead
        tgt = tokens[:, o: o + lg.shape[1]]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
        out[f"ce@{o}"] = ce
        out["loss"] = out["loss"] + ce
        with torch.no_grad():
            out[f"acc@{o}"] = (lg.argmax(-1) == tgt).float().mean()
        if o == 1 and lambda_mse > 0:
            p = lg.float().softmax(-1)
            z_hat = (p * level_vals[None, None, :]).sum(-1)
            mse = F.mse_loss(z_hat, z[:, 1: 1 + lg.shape[1]].float())
            out["mse"] = mse
            out["loss"] = out["loss"] + lambda_mse * mse
    return out


class WindowEncoder(nn.Module):
    def __init__(self, backbone: EEGGPT, n_channels: int, agg_heads: int = 4, agg_layers: int = 1, dropout: float = 0.1,
                 pool: str = "mean", seq_chunk: int = 256):
        super().__init__()
        self.backbone = backbone
        self.seq_chunk = seq_chunk   # channel-sequences per backbone call (bounds activation memory)
        d = backbone.d
        self.pool = pool
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.ch_pos = nn.Embedding(n_channels, d)
        self.agg = nn.ModuleList([Block(d, agg_heads, 4 * d, dropout) for _ in range(agg_layers)])
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, C, T) -> (B, d)"""
        B, C, T = tokens.shape
        ch = torch.arange(C, device=tokens.device).repeat(B)                 # (B*C,)
        flat = tokens.reshape(B * C, T)
        if B * C <= self.seq_chunk:
            h = self.backbone.hidden(flat, ch)
            if self.pool == "last":
                v = h[:, -1]
            elif self.pool == "meanmax":
                v = 0.5 * (h.mean(1) + h.amax(1))
            else:
                v = h.mean(1)
        else:
            vs = []
            for i in range(0, B * C, self.seq_chunk):
                h = self.backbone.hidden(flat[i: i + self.seq_chunk], ch[i: i + self.seq_chunk])   # (b, T, d)
                if self.pool == "last":
                    vs.append(h[:, -1])
                elif self.pool == "meanmax":
                    vs.append(0.5 * (h.mean(1) + h.amax(1)))
                else:
                    vs.append(h.mean(1))                                           # eq. 17
            v = torch.cat(vs, 0)
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
    """LayerNorm -> GELU -> Dropout -> Linear -> Linear (Mahdi §III-F 2)."""

    def __init__(self, d: int, n_out: int, dropout: float = 0.2, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or d
        self.net = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_out))

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