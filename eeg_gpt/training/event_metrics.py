"""Seizure-level (event-level) prediction metrics from per-window scores.

Window AUROC says how well preictal windows rank above background ones; a clinician cares
whether each seizure was announced in time and how often the device cries wolf. This module
turns a score trace into alarms with the usual "firing power" post-processing and reports:

    sens_events       fraction of seizures with an alarm inside their preictal span
                      [onset - sop_s - sph_s, onset - sph_s]
    fa_per_h          false alarms per hour of time outside any preictal span
    chance_sens       sensitivity a random predictor with the same alarm rate would get
                      (Snyder et al. 2008: 1 - exp(-rate * SOP), with the refractory period)

Alarm rule: fp[t] = causal moving average of the window scores over the previous `fp_window`
windows (default: one SOP of windows); an alarm is raised when fp >= thr, and no new alarm can
be raised for sop_s + sph_s seconds after one (refractory period). Alarms raised inside
[onset - sph_s, event stop] are neither hits nor false alarms (too late to act, but not wrong).
"""
from __future__ import annotations

import math

import numpy as np


def _alarms(times, scores, thr, fp_window, refractory_s):
    """Alarm times of one session; `times` ascending."""
    s = np.asarray(scores, dtype=np.float64)
    c = np.cumsum(np.concatenate([[0.0], s]))
    tt = np.arange(1, len(s) + 1); lo = np.maximum(0, tt - fp_window)
    fp = (c[tt] - c[lo]) / (tt - lo)
    alarms, blocked_until = [], -math.inf
    for t, f in zip(times, fp):
        if f >= thr and t >= blocked_until:
            alarms.append(float(t)); blocked_until = t + refractory_s
    return alarms, fp


def event_metrics(scores, session_idx, times, events, thr, window_s, sop_s: float = 420.0, sph_s: float = 120.0,
                  fp_window=None) -> dict:
    """scores / session_idx / times: one entry per window (times = window end in seconds, any order);
    events: list of (session_idx, onset_s) or (session_idx, onset_s, stop_s); thr: alarm threshold on the
    firing power (same scale as the scores); fp_window: windows in the moving average (None = SOP / window_s)."""
    s = np.asarray(scores, dtype=np.float64); sid = np.asarray(session_idx); t = np.asarray(times, dtype=np.float64)
    fp_window = int(fp_window) if fp_window else max(1, int(round(sop_s / max(window_s, 1e-6))))
    ev = [(e[0], float(e[1]), float(e[2]) if len(e) > 2 else float(e[1])) for e in events]
    spans = {}
    for k, onset, stop in ev:
        spans.setdefault(k, []).append((onset - sop_s - sph_s, onset - sph_s, stop))
    hit = [False] * len(ev)
    n_false = n_alarms = 0
    bg_windows = 0
    for key in np.unique(sid):
        idx = np.nonzero(sid == key)[0]
        o = idx[np.argsort(t[idx], kind="stable")]
        tk = t[o]
        sess_spans = spans.get(key, [])
        in_pre = np.zeros(len(o), dtype=bool)
        for a, b, _ in sess_spans:
            in_pre |= (tk >= a) & (tk <= b)
        bg_windows += int((~in_pre).sum())
        alarms, _ = _alarms(tk, s[o], thr, fp_window, sop_s + sph_s)
        n_alarms += len(alarms)
        for at in alarms:
            matched = ignored = False
            for j, (k2, onset, stop) in enumerate(ev):
                if k2 != key:
                    continue
                if onset - sop_s - sph_s <= at <= onset - sph_s:
                    hit[j] = True; matched = True
                elif onset - sph_s < at <= stop:
                    ignored = True
            if not matched and not ignored:
                n_false += 1
    background_hours = bg_windows * window_s / 3600.0
    fa_per_h = n_false / background_hours if background_hours > 0 else float("nan")
    chance = 1.0 - math.exp(-fa_per_h * sop_s / 3600.0) if not math.isnan(fa_per_h) else float("nan")
    return {"n_events": len(ev), "n_hits": int(sum(hit)), "sens_events": float(sum(hit) / len(ev)) if ev else float("nan"),
            "n_alarms": int(n_alarms), "n_false_alarms": int(n_false), "fa_per_h": float(fa_per_h),
            "background_hours": float(background_hours), "chance_sens": float(chance), "threshold": float(thr),
            "fp_window": fp_window}


def events_from_dataset(ds) -> list:
    """(session_idx, onset_s, stop_s) per seizure event of a LabeledWindows dataset: from the protocol
    owner's `_event_id/_event_start/_event_stop` columns when present, else from `ds.onsets` (ictal row
    starts merged when closer than one window)."""
    df = ds.df
    out = []
    if "_event_id" in df.columns and "_event_start" in df.columns:
        sub = df[df["_event_id"].astype(float) >= 0]
        stop_col = "_event_stop" if "_event_stop" in df.columns else "_event_start"
        for (key, eid), g in sub.groupby(["session_key", "_event_id"], sort=False):
            if key in ds.session_idx:
                out.append((ds.session_idx[key], float(g["_event_start"].iloc[0]), float(g[stop_col].iloc[0])))
        return out
    gap = getattr(ds, "T", 1024) / max(getattr(ds, "fs", 256.0), 1e-6)
    for key, onsets in getattr(ds, "onsets", {}).items():
        if key not in ds.session_idx or not onsets:
            continue
        start = prev = onsets[0]
        for o in onsets[1:]:
            if o - prev > gap:
                out.append((ds.session_idx[key], float(start), float(prev))); start = o
            prev = o
        out.append((ds.session_idx[key], float(start), float(prev)))
    return out
