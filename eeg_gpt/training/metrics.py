"""Evaluation utilities (ported from the previous repo's metrics/posthoc/baseline).

auroc                     rank-based AUROC (ties averaged)
threshold_sweep           best balanced accuracy over thresholds
sensitivity_at_specificity sensitivity achieved while holding specificity >= target
per_session_auroc         the distribution that a chance-level mean can hide
smoothed_auroc            causal moving average ("firing power") inside each session
bandpower_baseline        logistic regression on log relative band powers of the
                          last block -- the floor every model has to beat
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def auroc(scores, targets) -> float:
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]          # 1-based rank, ties averaged
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_sweep(scores, targets, n: int = 99) -> dict:
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    if len(s) == 0:
        return {"threshold": 0.5, "balanced_accuracy": 0.5, "sensitivity": 0.0, "specificity": 0.0}
    best = None
    for thr in np.quantile(s, np.linspace(0.01, 0.99, n)):
        pred = (s >= thr).astype(np.float64)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
        ba = 0.5 * (sens + spec)
        if best is None or ba > best["balanced_accuracy"]:
            best = {"threshold": float(thr), "balanced_accuracy": float(ba), "sensitivity": float(sens), "specificity": float(spec)}
    return best if best is not None else {"threshold": float(np.median(s)), "balanced_accuracy": 0.5, "sensitivity": 0.0, "specificity": 0.0}


def sensitivity_at_specificity(scores, targets, target_specificity: float = 0.90) -> float:
    """Return the sensitivity achieved by the highest threshold that keeps
    specificity >= target_specificity.

    This is clinically more meaningful than the balanced-accuracy optimum
    because seizure prediction requires a bounded false-alarm rate.  If no
    threshold satisfies the constraint, returns 0.0.
    """
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    thr = np.unique(s)                                        # pred = s >= thr
    tn = np.searchsorted(np.sort(s[y == 0]), thr, side="left")
    tp = n_pos - np.searchsorted(np.sort(s[y == 1]), thr, side="left")
    ok = tn / n_neg >= target_specificity
    return float(tp[ok].max() / n_pos) if ok.any() else 0.0


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


def bandpower_features(z: torch.Tensor, fs: float, log_sd: Optional[torch.Tensor] = None) -> torch.Tensor:
    """z (C, T) z-scored -> (C*5 [+ C],) log relative band power [+ log raw sd per channel].

    The former `log var(z)` term was identically ~0 (z is z-scored per block) and is gone; the absolute amplitude
    comes from `log_sd` (the dataset's batch key, (C,) for the last block) when given. For the full feature set
    (band powers, SEF, line length, Hjorth, deltas) use the dataset's `feats` (data/features.py) instead.
    """
    x = z.float()
    spec = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1).abs() ** 2
    f = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    total = spec.sum(-1, keepdim=True) + 1e-12
    feats = [torch.log(spec[:, (f >= lo) & (f < hi)].sum(-1, keepdim=True) / total + 1e-12) for lo, hi in _BANDS]
    if log_sd is not None:
        feats.append(torch.as_tensor(log_sd).float().reshape(x.shape[0], -1)[:, -1:])
    return torch.cat(feats, dim=-1).flatten()


def bandpower_baseline(Xtr, ytr, Xva, yva, steps: int = 400, l2: float = 1e-3) -> dict:
    """Logistic regression on standardised features: the floor every model has to beat.

    Xtr/Xva are (rows, F) float tensors: bandpower_features() per row, or directly the dataset's `feats`
    (data/features.py) when the loader provides them -- the feature branch of SeizureModel uses the same vectors,
    so the comparison is like for like.
    """
    Xtr, Xva = torch.as_tensor(Xtr).float(), torch.as_tensor(Xva).float()
    ytr, yva = torch.as_tensor(ytr).float(), torch.as_tensor(yva).float()
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


# ---------------- collapse diagnostics and report-only score normalisation -------------------

def score_stats(scores, targets) -> dict:
    """Shape of the score distribution: tells an optimisation failure (everything at one value,
    separation ~0) from a generalisation failure (separated on train, not on eval)."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(targets, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    eps = 1e-6
    logit = np.log(np.clip(s, eps, 1 - eps)) - np.log(np.clip(1 - s, eps, 1 - eps))
    lp, ln = logit[y == 1], logit[y == 0]
    pooled = math.sqrt(0.5 * (float(pos.var()) if len(pos) > 1 else 0.0) + 0.5 * (float(neg.var()) if len(neg) > 1 else 0.0))
    if pooled < 1e-9:      # a constant predictor: rounding noise must not become a huge "separation"
        pooled = 0.0
    return {"std": float(s.std()) if len(s) else float("nan"),
            "frac_pos": float((s > 0.5).mean()) if len(s) else float("nan"),
            "mean_logit_pos": float(lp.mean()) if len(lp) else float("nan"),
            "mean_logit_neg": float(ln.mean()) if len(ln) else float("nan"),
            "separation": float((pos.mean() - neg.mean()) / pooled) if len(pos) and len(neg) and pooled > 0 else float("nan"),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def causal_session_normalise(session_idx, order_key, scores, k: int = 150) -> np.ndarray:
    """Each score minus the mean, over the std, of the previous `k` windows of the same session
    (causal: a window only sees its past). Removes a per-session offset of the classifier; the
    first window of a session is returned as 0. Report-only (`auroc_norm`), never used for training."""
    sid = np.asarray(session_idx); t = np.asarray(order_key, dtype=np.float64); s = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(s)
    for key in np.unique(sid):
        idx = np.nonzero(sid == key)[0]
        o = idx[np.argsort(t[idx], kind="stable")]
        x = s[o]
        c1 = np.cumsum(np.concatenate([[0.0], x])); c2 = np.cumsum(np.concatenate([[0.0], x * x]))
        tt = np.arange(len(o)); lo = np.maximum(0, tt - k); n = (tt - lo).astype(np.float64)   # strictly previous windows
        mean = np.where(n > 0, (c1[tt] - c1[lo]) / np.maximum(n, 1), 0.0)
        var = np.where(n > 1, (c2[tt] - c2[lo]) / np.maximum(n, 1) - mean ** 2, 0.0)
        sd = np.sqrt(np.maximum(var, 0.0)) + 1e-6
        out[o] = np.where(n > 0, (x - mean) / sd, 0.0)
    return out