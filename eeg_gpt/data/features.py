"""Hand-crafted per-window EEG features on the RAW (not z-scored) signal.

One function feeds both the band-power baseline (training/metrics.py) and the
fine-tuning feature branch (models/gpt.py SeizureModel), so "model beats
baseline" always compares the same inputs.

Per block and channel, in this order (N_FEATS_PER_CHANNEL = 10):
    0  log absolute variance            (the amplitude information the per-block
                                         z-scoring removes from the tokens)
    1-5 log relative band power          delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz
    6  spectral edge frequency 90 %      (Hz, over 0.5-45 Hz)
    7  line length                       mean |x[t] - x[t-1]|, log
    8  Hjorth mobility
    9  Hjorth complexity
Layout of a block's vector is channel-major: feature k of channel c sits at c * 10 + k.
"""
from __future__ import annotations

import numpy as np

N_FEATS_PER_CHANNEL = 10
BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
BANDPOWER_OFFSETS = (1, 2, 3, 4, 5)        # indices of the band-power features inside a channel's 10
_EPS = 1e-12


def feat_dim(n_channels: int, n_parts: int = 2) -> int:
    """Length of context_features(): last block + delta against the previous blocks."""
    return n_parts * N_FEATS_PER_CHANNEL * int(n_channels)


def feature_names(n_channels: int) -> list[str]:
    base = ["log_var", "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma", "sef90", "line_length", "hjorth_mob", "hjorth_cmp"]
    return [f"ch{c}_{n}" for c in range(n_channels) for n in base]


def bandpower_indices(n_channels: int, n_parts: int = 2) -> np.ndarray:
    """Indices of the band-power features inside a context_features() vector (both parts)."""
    F = N_FEATS_PER_CHANNEL * n_channels
    one = np.array([c * N_FEATS_PER_CHANNEL + k for c in range(n_channels) for k in BANDPOWER_OFFSETS], dtype=np.int64)
    return np.concatenate([one + p * F for p in range(n_parts)])


def window_features(x_raw: np.ndarray, fs: float) -> np.ndarray:
    """x_raw (C, N, T) float32 raw signal -> (N, C * 10) float32, channel-major (see module docstring)."""
    x = np.asarray(x_raw, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    C, N, T = x.shape
    xc = x - x.mean(-1, keepdims=True)
    var = (xc * xc).mean(-1)                                                 # (C, N)
    log_var = np.log(var + _EPS)

    spec = np.abs(np.fft.rfft(xc, axis=-1)) ** 2                             # (C, N, T//2+1)
    f = np.fft.rfftfreq(T, d=1.0 / float(fs))
    in_range = (f >= 0.5) & (f < 45.0)
    total = spec[..., in_range].sum(-1) + _EPS                               # (C, N)
    bp = np.stack([np.log(spec[..., (f >= lo) & (f < hi)].sum(-1) / total + _EPS) for lo, hi in BANDS], -1)  # (C, N, 5)

    cum = np.cumsum(spec[..., in_range], axis=-1) / total[..., None]        # (C, N, K)
    f_in = f[in_range]
    sef_idx = np.minimum((cum >= 0.9).argmax(-1), len(f_in) - 1)
    sef90 = f_in[sef_idx]                                                    # (C, N)

    d1 = np.diff(xc, axis=-1)
    line_length = np.log(np.abs(d1).mean(-1) + _EPS)
    v1 = (d1 * d1).mean(-1)
    d2 = np.diff(d1, axis=-1)
    v2 = (d2 * d2).mean(-1)
    mobility = np.sqrt(v1 / (var + _EPS))
    complexity = np.sqrt(v2 / (v1 + _EPS)) / (mobility + _EPS)

    feats = np.concatenate([log_var[..., None], bp, sef90[..., None], line_length[..., None],
                            mobility[..., None], complexity[..., None]], -1)   # (C, N, 10)
    return np.ascontiguousarray(feats.transpose(1, 0, 2).reshape(N, C * N_FEATS_PER_CHANNEL)).astype(np.float32)


def context_features(x_raw: np.ndarray, fs: float) -> np.ndarray:
    """x_raw (C, N, T) -> (2F,): features of the last block, then last - mean(previous blocks) (zeros when N == 1)."""
    f = window_features(x_raw, fs)                                           # (N, F)
    last = f[-1]
    delta = last - f[:-1].mean(0) if f.shape[0] > 1 else np.zeros_like(last)
    return np.concatenate([last, delta]).astype(np.float32)
