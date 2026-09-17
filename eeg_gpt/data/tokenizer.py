"""Amplitude tokenizer -- Mahdi & Baghdadi 2026 (arXiv:2603.26821) §III-D.

    z      = (x - mu) / sigma            (mu, sigma: per channel over the block, or per session)
    z_clip = clip(z, -k, +k)             (k = 5)
    u      = (z_clip + k) / (2k)  in [0, 1]
    token  = floor(u * (L - 1))    in {0 .. L-1}     (L = 512)

Dequantize (for the dual MSE loss and the band-power baseline):
    z_hat  = (level / (L - 1)) * 2k - k

Differences from the paper, on purpose:
  * one token stream PER CHANNEL (the paper interleaves all channels into one
    stream, so a 512-token block covers ~15 time steps); channel identity goes
    into the model as an embedding instead (BrainGPT/EEGPT electrode-wise modeling).
  * normalization statistics are per block (window) per channel by default,
    which is what the seizure loader has always used; `stats="session"` gives
    the paper's per-recording statistics.
"""
from __future__ import annotations

import numpy as np
import torch


class AmplitudeTokenizer:
    def __init__(self, n_levels: int = 512, clip_k: float = 5.0):
        self.L = int(n_levels)
        self.k = float(clip_k)

    # ---- numpy path (loaders) -------------------------------------------------
    def zscore(self, x: np.ndarray, mu=None, sd=None):
        """x: (C, T). Returns z-scored signal and the stats used."""
        if mu is None:
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-8
        return (x - mu) / sd, (mu, sd)

    def quantize_z(self, z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, -self.k, self.k)
        u = (zc + self.k) / (2.0 * self.k)
        return np.floor(u * (self.L - 1)).astype(np.int64)

    def encode(self, x: np.ndarray, mu=None, sd=None):
        z, stats = self.zscore(x.astype(np.float32), mu, sd)
        return self.quantize_z(z), z.astype(np.float32), stats

    # ---- torch path (model / losses) --------------------------------------------
    def dequantize(self, level: torch.Tensor) -> torch.Tensor:
        """level: float or long tensor of quantization levels (can be an
        expectation over levels). Returns the z-scored amplitude."""
        return level.float() / (self.L - 1) * (2.0 * self.k) - self.k

    def level_values(self, device=None) -> torch.Tensor:
        """z value of each of the L levels (bin lower edges), shape (L,)."""
        return self.dequantize(torch.arange(self.L, device=device))
