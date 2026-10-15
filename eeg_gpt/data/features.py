"""Hand-crafted per-window EEG features on the RAW (not z-scored) signal.

One function feeds both the band-power baseline (training/metrics.py) and the
fine-tuning feature branch (models/gpt.py SeizureModel), so "model beats
baseline" always compares the same inputs.

Per block and channel, in this order (N_FEATS_PER_CHANNEL = 24):
    0     log absolute variance            (the amplitude information the per-block
                                            z-scoring removes from the tokens)
    1-5   log relative band power          delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz
    6     spectral edge frequency 90 %     (Hz, over 0.5-45 Hz)
    7     line length                      mean |x[t] - x[t-1]|, log
    8     Hjorth mobility
    9     Hjorth complexity
    10    skewness                         3rd standardized moment
    11    kurtosis                         4th standardized moment (excess kurtosis)
    12    spectral edge frequency 50 %     median spectral frequency (Hz)
    13    spectral edge frequency 75 %     75% spectral edge (Hz)
    14    spectral entropy                 normalized Shannon spectral entropy
    15    spectral flatness                Wiener entropy (geometric / arithmetic mean)
    16    spectral centroid                mean frequency (Hz)
    17    spectral spread                  spectral standard deviation (Hz)
    18    slow/fast ratio                  log (delta + theta) / (alpha + beta)
    19    theta/alpha ratio                log (theta / alpha)
    20    zero-crossing rate               mean zero-crossing rate of centered signal
    21    log Teager-Kaiser energy         mean |x[t]^2 - x[t-1]x[t+1]|, log
    22    second difference line length    mean |x[t] - 2x[t-1] + x[t-2]|, log
    23    log peak-to-peak amplitude       log (max(x) - min(x))

Layout of a block's vector is channel-major: feature k of channel c sits at c * 24 + k.
"""
from __future__ import annotations

import numpy as np

N_FEATS_PER_CHANNEL = 24
BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
BANDPOWER_OFFSETS = (1, 2, 3, 4, 5)        # indices of the band-power features inside a channel's 24
_EPS = 1e-12


def feat_dim(n_channels: int, n_parts: int = 2) -> int:
    """Length of context_features(): last block + delta against the previous blocks."""
    return n_parts * N_FEATS_PER_CHANNEL * int(n_channels)


def feature_names(n_channels: int) -> list[str]:
    base = [
        "log_var", "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
        "sef90", "line_length", "hjorth_mob", "hjorth_cmp",
        "skewness", "kurtosis", "sef50", "sef75",
        "spectral_entropy", "spectral_flatness", "spectral_centroid", "spectral_spread",
        "slow_fast_ratio", "theta_alpha_ratio", "zero_crossings", "log_tkeo",
        "diff2_line_length", "peak_to_peak"
    ]
    return [f"ch{c}_{n}" for c in range(n_channels) for n in base]


def bandpower_indices(n_channels: int, n_parts: int = 2) -> np.ndarray:
    """Indices of the band-power features inside a context_features() vector (both parts)."""
    F = N_FEATS_PER_CHANNEL * n_channels
    one = np.array([c * N_FEATS_PER_CHANNEL + k for c in range(n_channels) for k in BANDPOWER_OFFSETS], dtype=np.int64)
    return np.concatenate([one + p * F for p in range(n_parts)])


def window_features(x_raw: np.ndarray, fs: float) -> np.ndarray:
    """x_raw (C, N, T) float32 raw signal -> (N, C * 24) float32, channel-major (see module docstring)."""
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
    spec_in = spec[..., in_range]
    f_in = f[in_range]
    total = spec_in.sum(-1) + _EPS                                           # (C, N)
    bp = np.stack([np.log(spec[..., (f >= lo) & (f < hi)].sum(-1) / total + _EPS) for lo, hi in BANDS], -1)  # (C, N, 5)

    cum = np.cumsum(spec_in, axis=-1) / total[..., None]                    # (C, N, K)
    K = len(f_in)
    sef50_idx = np.minimum((cum >= 0.50).argmax(-1), K - 1)
    sef50 = f_in[sef50_idx]                                                  # (C, N)
    sef75_idx = np.minimum((cum >= 0.75).argmax(-1), K - 1)
    sef75 = f_in[sef75_idx]                                                  # (C, N)
    sef_idx = np.minimum((cum >= 0.90).argmax(-1), K - 1)
    sef90 = f_in[sef_idx]                                                    # (C, N)

    d1 = np.diff(xc, axis=-1)
    line_length = np.log(np.abs(d1).mean(-1) + _EPS)
    v1 = (d1 * d1).mean(-1)
    d2 = np.diff(d1, axis=-1)
    v2 = (d2 * d2).mean(-1)
    mobility = np.sqrt(v1 / (var + _EPS))
    complexity = np.sqrt(v2 / (v1 + _EPS)) / (mobility + _EPS)
    diff2_ll = np.log(np.abs(d2).mean(-1) + _EPS)

    std = np.sqrt(var + _EPS)
    m3 = (xc ** 3).mean(-1)
    skewness = m3 / (std ** 3 + _EPS)
    m4 = (xc ** 4).mean(-1)
    kurtosis = np.clip(m4 / (var ** 2 + _EPS) - 3.0, -3.0, 50.0)

    p_norm = spec_in / (total[..., None])
    log_K = np.log(float(max(2, K)))
    spec_entropy = - (p_norm * np.log(p_norm + _EPS)).sum(-1) / log_K
    spec_entropy = np.clip(spec_entropy, 0.0, 1.0)

    log_spec = np.log(spec_in + _EPS)
    geom_mean = np.exp(log_spec.mean(-1))
    arith_mean = spec_in.mean(-1) + _EPS
    spec_flatness = np.clip(geom_mean / arith_mean, 0.0, 1.0)

    spec_centroid = (f_in[None, None, :] * p_norm).sum(-1)
    spec_spread = np.sqrt(np.maximum(0.0, ((f_in[None, None, :] - spec_centroid[..., None]) ** 2 * p_norm).sum(-1)) + _EPS)

    p_delta = np.exp(bp[..., 0])
    p_theta = np.exp(bp[..., 1])
    p_alpha = np.exp(bp[..., 2])
    p_beta = np.exp(bp[..., 3])
    slow_fast_ratio = np.log((p_delta + p_theta + _EPS) / (p_alpha + p_beta + _EPS))
    theta_alpha_ratio = np.log((p_theta + _EPS) / (p_alpha + _EPS))

    s1 = xc[..., 1:]
    s0 = xc[..., :-1]
    zero_crossings = ((s1 * s0) < 0).mean(-1)

    tkeo = xc[..., 1:-1] ** 2 - xc[..., :-2] * xc[..., 2:]
    log_tkeo = np.log(np.abs(tkeo).mean(-1) + _EPS)
    peak_to_peak = np.log(np.ptp(x, axis=-1) + _EPS)

    feats = np.concatenate([
        log_var[..., None],           # 0
        bp,                           # 1-5
        sef90[..., None],             # 6
        line_length[..., None],       # 7
        mobility[..., None],          # 8
        complexity[..., None],        # 9
        skewness[..., None],          # 10
        kurtosis[..., None],          # 11
        sef50[..., None],             # 12
        sef75[..., None],             # 13
        spec_entropy[..., None],      # 14
        spec_flatness[..., None],     # 15
        spec_centroid[..., None],     # 16
        spec_spread[..., None],       # 17
        slow_fast_ratio[..., None],   # 18
        theta_alpha_ratio[..., None], # 19
        zero_crossings[..., None],    # 20
        log_tkeo[..., None],          # 21
        diff2_ll[..., None],          # 22
        peak_to_peak[..., None],      # 23
    ], -1)                            # (C, N, 24)

    return np.ascontiguousarray(feats.transpose(1, 0, 2).reshape(N, C * N_FEATS_PER_CHANNEL)).astype(np.float32)


def context_features(x_raw: np.ndarray, fs: float) -> np.ndarray:
    """x_raw (C, N, T) -> (2F,): features of the last block, then last - mean(previous blocks) (zeros when N == 1)."""
    f = window_features(x_raw, fs)                                           # (N, F)
    last = f[-1]
    delta = last - f[:-1].mean(0) if f.shape[0] > 1 else np.zeros_like(last)
    return np.concatenate([last, delta]).astype(np.float32)
