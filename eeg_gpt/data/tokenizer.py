"""Amplitude tokenizer -- Mahdi & Baghdadi 2026 (arXiv:2603.26821) §III-D.

    z      = (x - mu) / sigma            (mu, sigma: per channel over the block, or per session)
    z_clip = clip(z, -k, +k)             (k = 4 here; the paper uses 5)
    u      = (z_clip + k) / (2k)  in [0, 1]
    token  = floor(u * (L - 1))    in {0 .. L-1}     (L = 512)

Dequantize (for the dual MSE loss and the band-power baseline):
    z_hat  = (level / (L - 1)) * 2k - k              ("edge": bin lower edges, the historical default)
    z_hat  = ((level + 0.5) / (L - 1)) * 2k - k      ("center": bin centres, no -half-bin bias)

Differences from the paper, on purpose:
  * one token stream PER CHANNEL (the paper interleaves all channels into one
    stream, so a 512-token block covers ~15 time steps); channel identity goes
    into the model as an embedding instead (BrainGPT/EEGPT electrode-wise modeling).
  * normalization statistics are per block (window) per channel by default,
    which is what the seizure loader has always used; `stats="session"` gives
    the paper's per-recording statistics.

Patch path (`patch > 1`): the model consumes the z-scored signal itself (float
patches of `patch` samples) instead of amplitude levels, so `encode_blocks`
returns z as the model input and the quantization is not used.

`norm_scope` (both stages must use the same one; it is a checkpoint config key):
  "block"  -- every block (window) is z-scored on its own: today's behaviour. Note
              that this uses statistics of samples the causal model has not seen
              yet and erases the amplitude trend across context blocks.
  "lead4s" -- mean/sd of the FIRST block of a context (4 s at 256 Hz) are applied
              to every block of that context: leak-free for the causal model and
              the amplitude trend over the context survives (z of later blocks is
              clipped at +-8 on the patch path, +-k on the token path).
`log_sd` (log of each block's own raw sd per channel) is returned in both scopes:
it is the absolute-amplitude feature that z-scoring removes.
"""
from __future__ import annotations

import numpy as np
import torch

PATCH_Z_CLIP = 8.0       # z clip on the patch path (continuous input; robustness only)
LEAD_BLOCK_SAMPLES = 1024  # "lead4s": 4 s at 256 Hz


class AmplitudeTokenizer:
    def __init__(self, n_levels: int = 512, clip_k: float = 4.0, patch: int = 1, norm_scope: str = "block"):
        self.L = int(n_levels)
        self.k = float(clip_k)
        self.patch = max(1, int(patch))
        if norm_scope not in ("block", "lead4s"):
            raise ValueError(f"norm_scope must be 'block' or 'lead4s', got {norm_scope!r}")
        self.norm_scope = norm_scope
        self._cached_level_values: dict = {}

    @classmethod
    def from_config(cls, cfg: dict) -> "AmplitudeTokenizer":
        """From a checkpoint / args config dict; missing keys take today's defaults."""
        patch = int(cfg.get("patch", 1) or 1)
        scope = cfg.get("norm_scope", "block") or "block"
        if scope == "auto":
            scope = "block" if patch == 1 else "lead4s"
        return cls(int(cfg.get("n_levels", 512)), float(cfg.get("clip_k", 4.0)), patch, scope)

    @property
    def z_clip(self) -> float:
        return PATCH_Z_CLIP if self.patch > 1 else self.k

    # ---- numpy path (loaders) -------------------------------------------------
    def zscore(self, x: np.ndarray, mu=None, sd=None):
        """x: (..., T). Returns z-scored signal and the stats used."""
        if mu is None:
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-8
        return (x - mu) / sd, (mu, sd)

    def quantize_z(self, z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, -self.k, self.k)
        u = (zc + self.k) / (2.0 * self.k)
        return np.floor(u * (self.L - 1)).astype(np.int64)

    def encode(self, x: np.ndarray, mu=None, sd=None):
        """x: (..., T) -> (tokens int64, z float32, (mu, sd)). On the patch path z is clipped at +-8
        (tokens are still returned for callers that want them)."""
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        z, stats = self.zscore(x, mu, sd)
        if self.patch > 1:
            z = np.clip(z, -PATCH_Z_CLIP, PATCH_Z_CLIP)
        return self.quantize_z(z), z.astype(np.float32, copy=False), stats

    def lead_stats(self, x: np.ndarray, n: int = LEAD_BLOCK_SAMPLES):
        """(mu, sd) per channel of the first `n` samples of x (..., T): the leak-free 'lead4s' statistics."""
        lead = x[..., : max(2, min(n, x.shape[-1]))]
        return lead.mean(axis=-1, keepdims=True), lead.std(axis=-1, keepdims=True) + 1e-8

    def encode_blocks(self, x: np.ndarray) -> dict:
        """x: (C, N, T) float32 raw signal, N context blocks oldest first.

        Returns {"inp": model input (int64 levels (C, N, T) on the token path, float32 z (C, N, T)
        on the patch path), "z": float32 z (C, N, T), "log_sd": float32 (C, N) log raw sd per block}.
        """
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        C, N, T = x.shape
        log_sd = np.log(x.std(axis=-1) + 1e-8).astype(np.float32)                  # (C, N): a feature, always per block
        if self.norm_scope == "lead4s":
            mu, sd = self.lead_stats(x[:, 0, :])                                    # (C, 1)
            z = (x - mu[:, None, :]) / sd[:, None, :]
        else:
            mu = x.mean(axis=-1, keepdims=True); sd = x.std(axis=-1, keepdims=True) + 1e-8
            z = (x - mu) / sd
        z = z.astype(np.float32, copy=False)
        if self.patch > 1:
            z = np.clip(z, -PATCH_Z_CLIP, PATCH_Z_CLIP)
            return {"inp": z, "z": z, "log_sd": log_sd}
        return {"inp": self.quantize_z(z), "z": z, "log_sd": log_sd}

    # ---- torch path (model / losses) --------------------------------------------
    def dequantize(self, level: torch.Tensor, dequant: str = "edge") -> torch.Tensor:
        """level: float or long tensor of quantization levels (can be an
        expectation over levels). Returns the z-scored amplitude ("edge" = bin
        lower edge, the historical default; "center" = bin centre)."""
        lv = level.float() + (0.5 if dequant == "center" else 0.0)
        return lv / (self.L - 1) * (2.0 * self.k) - self.k

    @property
    def bin_width(self) -> float:
        return 2.0 * self.k / (self.L - 1)

    def level_values(self, device=None, dequant: str = "edge") -> torch.Tensor:
        """z value of each of the L levels, shape (L,)."""
        key = (str(device), dequant)
        if key not in self._cached_level_values:
            self._cached_level_values[key] = self.dequantize(torch.arange(self.L, device=device), dequant)
        return self._cached_level_values[key]
