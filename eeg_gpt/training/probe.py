"""Linear probe: how much seizure information is linearly readable from a representation.

    scores = fit_probe(X_train, y_train, X_test)          # L2 logistic regression, lambda by inner split
    probe_auroc(X_train, y_train, X_test, y_test)         # the number to compare with the band-power baseline

Used by pretrain.py (every --probe-every steps on held-out patients' windows: pooled trunk
features vs band-power features of the same windows) and by the fine-tuner's `--head probe`.
Pure torch, so it runs wherever the model runs; no sklearn dependency.
"""
from __future__ import annotations

import numpy as np
import torch

from eeg_gpt.training.metrics import auroc

_L2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _as_tensor(a) -> torch.Tensor:
    t = a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))
    return t.detach().float().cpu()


def _fit(X: torch.Tensor, y: torch.Tensor, l2: float, steps: int = 400, lr: float = 0.05) -> tuple:
    n, d = X.shape
    pw = ((1 - y).sum() / y.sum().clamp_min(1)).clamp(1.0, 50.0)
    w = torch.zeros(d, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    with torch.enable_grad():   # callers (pretrain.run_probe, the fine-tuner's evaluation) run under no_grad
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, pos_weight=pw) + l2 * (w * w).sum() / d
            opt.zero_grad(); loss.backward(); opt.step()
    return w.detach(), b.detach()


def fit_probe(X_tr, y_tr, X_te, seed: int = 0, l2_grid=_L2_GRID, steps: int = 400):
    """Standardise with the training statistics, pick the L2 strength on a 25 % inner split of the
    training rows (by AUROC), refit on all training rows, return sigmoid scores for X_te (numpy)."""
    X_tr, y_tr, X_te = _as_tensor(X_tr), _as_tensor(y_tr).reshape(-1), _as_tensor(X_te)
    mu, sd = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp_min(1e-6)
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    n = len(y_tr)
    best_l2, best_a = l2_grid[0], -1.0
    if n >= 20 and len(l2_grid) > 1:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_val = max(4, n // 4)
        va, tr = perm[:n_val], perm[n_val:]
        if y_tr[va].sum() > 0 and (1 - y_tr[va]).sum() > 0 and y_tr[tr].sum() > 0:
            for l2 in l2_grid:
                w, b = _fit(X_tr[tr], y_tr[tr], l2, steps)
                a = auroc((X_tr[va] @ w + b).numpy(), y_tr[va].numpy())
                if a == a and a > best_a:
                    best_a, best_l2 = a, l2
    w, b = _fit(X_tr, y_tr, best_l2, steps)
    with torch.no_grad():
        return torch.sigmoid(X_te @ w + b).numpy()


def probe_auroc(X_tr, y_tr, X_te, y_te, seed: int = 0) -> float:
    s = fit_probe(X_tr, y_tr, X_te, seed=seed)
    return auroc(s, _as_tensor(y_te).reshape(-1).numpy())
