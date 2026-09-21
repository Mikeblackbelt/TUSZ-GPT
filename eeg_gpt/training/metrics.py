"""Evaluation utilities (ported from the previous repo's metrics/posthoc/baseline).

auroc                 rank-based AUROC (ties averaged)
threshold_sweep       best balanced accuracy over thresholds
per_session_auroc     the distribution that a chance-level mean can hide
smoothed_auroc        causal moving average ("firing power") inside each session
bandpower_baseline    logistic regression on log relative band powers of the
                      last block -- the floor every model has to beat
"""
from __future__ import annotations

import numpy as np
import torch


def auroc(scores, targets) -> float:
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_sweep(scores, targets, n: int = 99) -> dict:
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    best = {"threshold": 0.5, "balanced_accuracy": 0.5, "sensitivity": 0.0, "specificity": 0.0}
    for thr in np.quantile(s, np.linspace(0.01, 0.99, n)):
        pred = (s >= thr).astype(np.float64)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
        ba = 0.5 * (sens + spec)
        if ba > best["balanced_accuracy"]:
            best = {"threshold": float(thr), "balanced_accuracy": float(ba), "sensitivity": float(sens), "specificity": float(spec)}
    return best


def per_session_auroc(session_idx, scores, targets, min_pos: int = 5, min_neg: int = 5) -> dict:
    sid = np.asarray(session_idx); s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    rows = []
    for k in np.unique(sid):
        m = sid == k
        if y[m].sum() < min_pos or (1 - y[m]).sum() < min_neg:
            continue
        rows.append((int(k), int(y[m].sum()), int((1 - y[m]).sum()), auroc(s[m], y[m])))
    a = np.array([r[3] for r in rows]) if rows else np.array([])
    return {"sessions": sorted(rows, key=lambda r: -r[3]), "n": len(rows),
            "median": float(np.median(a)) if len(a) else float("nan"), "mean": float(a.mean()) if len(a) else float("nan"),
            "frac_above_0.6": float((a > 0.6).mean()) if len(a) else float("nan")}


def smoothed_auroc(session_idx, order_key, scores, targets, k: int = 15) -> dict:
    sid = np.asarray(session_idx); t = np.asarray(order_key, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    sm = np.empty_like(s)
    for key in np.unique(sid):
        idx = np.nonzero(sid == key)[0]
        o = idx[np.argsort(t[idx], kind="stable")]
        c = np.cumsum(np.concatenate([[0.0], s[o]]))
        tt = np.arange(1, len(o) + 1); lo = np.maximum(0, tt - k)
        sm[o] = (c[tt] - c[lo]) / (tt - lo)
    return {"k": k, "auroc_raw": auroc(s, y), "auroc_smoothed": auroc(sm, y), "scores": sm}


# ---------------- band-power baseline -----------------------------------------------------
_BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def bandpower_features(z: torch.Tensor, fs: float) -> torch.Tensor:
    """z (C, T) z-scored -> (C*5 + C,) log relative band power + log variance."""
    x = z.float()
    spec = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1).abs() ** 2
    f = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    total = spec.sum(-1, keepdim=True) + 1e-12
    feats = [torch.log(spec[:, (f >= lo) & (f < hi)].sum(-1, keepdim=True) / total + 1e-12) for lo, hi in _BANDS]
    feats.append(torch.log(x.var(-1, keepdim=True) + 1e-12))
    return torch.cat(feats, dim=-1).flatten()


def bandpower_baseline(Xtr, ytr, Xva, yva, steps: int = 400, l2: float = 1e-3) -> dict:
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    pw = ((1 - ytr).sum() / ytr.sum().clamp_min(1)).clamp(1, 50)
    w = torch.zeros(Xtr.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr, pos_weight=pw) + l2 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return {"train_auroc": auroc((Xtr @ w + b).numpy(), ytr.numpy()), "dev_auroc": auroc((Xva @ w + b).numpy(), yva.numpy()),
                "dev_scores": torch.sigmoid(Xva @ w + b).numpy()}


def confusion(pred, target, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(np.asarray(target), np.asarray(pred)):
        if 0 <= t < n:
            cm[int(t), int(p)] += 1
    return cm


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        if cm[i].sum() == 0:
            continue
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else float("nan")