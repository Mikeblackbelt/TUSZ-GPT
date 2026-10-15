"""Seizure events from the master CSV rows.

The CSV holds one row per window: a seizure is chopped into several window-sized ictal rows. An EVENT is one
seizure: consecutive ictal rows of a session whose gap is <= 0.5 * window_s. Everything that needs a seizure
rather than a window (the event-grouped CV folds, the horizon / time-to-onset labels, postictal exclusion,
event-level sensitivity, the candidate ranking) reads the columns `events_from_rows` adds:

    _event_id            int   id of the event an ictal row belongs to (-1 for non-ictal rows); unique per CSV
    _event_start         float onset of that event (NaN for non-ictal rows)
    _event_stop          float end of that event (NaN for non-ictal rows)
    _next_event_start    float onset of the first event of the same session starting at or after the row's
                               stop_time (NaN if none): time_to_onset = _next_event_start - stop_time
    _prev_event_stop     float end of the last event of the same session ending at or before the row's
                               start_time (NaN if none): postictal distance = start_time - _prev_event_stop

`load_master` (datasets.py) calls `events_from_rows` once on the full frame, before the excluded rows (SOP buffer,
postictal...) are dropped, so the columns describe the recording as a whole.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .label_utils import LABEL_CATEGORY_ICTAL

EVENT_COLUMNS = ["_event_id", "_event_start", "_event_stop", "_next_event_start", "_prev_event_stop"]


def events_from_rows(df: pd.DataFrame, window_s: float = 4.0) -> pd.DataFrame:
    """Return a copy of `df` with the EVENT_COLUMNS added (see the module docstring).

    Needs `session_key`, `start_time`, `stop_time` and `_cat`. Row order and index are preserved.
    """
    n = len(df)
    event_id = np.full(n, -1, dtype=np.int64)
    event_start = np.full(n, np.nan)
    event_stop = np.full(n, np.nan)
    next_start = np.full(n, np.nan)
    prev_stop = np.full(n, np.nan)
    if n:
        sess = df["session_key"].astype(str).values
        start = df["start_time"].values.astype(np.float64)
        stop = df["stop_time"].values.astype(np.float64)
        ict = (df["_cat"].values == LABEL_CATEGORY_ICTAL)
        gap = 0.5 * float(window_s)
        # positions grouped by session, each group sorted by start time
        order = np.lexsort((start, sess))
        bounds = np.flatnonzero(np.r_[True, sess[order][1:] != sess[order][:-1], True])
        next_id = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            pos = order[a:b]
            ipos = pos[ict[pos]]
            if len(ipos):
                s_i, e_i = start[ipos], stop[ipos]
                # a new event wherever the gap to the previous ictal row of the session exceeds half a window
                new = np.r_[True, (s_i[1:] - np.maximum.accumulate(e_i)[:-1]) > gap]
                local = np.cumsum(new) - 1
                starts = np.full(local[-1] + 1, np.inf)
                stops = np.full(local[-1] + 1, -np.inf)
                np.minimum.at(starts, local, s_i)
                np.maximum.at(stops, local, e_i)
                event_id[ipos] = local + next_id
                event_start[ipos] = starts[local]
                event_stop[ipos] = stops[local]
                next_id += len(starts)
                # starts/stops are increasing (events do not overlap within a session)
                j = np.searchsorted(starts, stop[pos], side="left")        # first event starting at/after the row's end
                ok = j < len(starts)
                next_start[pos[ok]] = starts[j[ok]]
                k = np.searchsorted(stops, start[pos], side="right") - 1   # last event ending at/before the row's start
                ok = k >= 0
                prev_stop[pos[ok]] = stops[k[ok]]
    out = df.copy()
    out["_event_id"] = event_id
    out["_event_start"] = event_start
    out["_event_stop"] = event_stop
    out["_next_event_start"] = next_start
    out["_prev_event_stop"] = prev_stop
    return out


def postictal_mask(df: pd.DataFrame, exclude_s: float) -> pd.Series:
    """True for rows that start within `exclude_s` seconds after the end of the previous seizure of their session
    (postictal EEG labelled as background). All False when exclude_s <= 0 or the event columns are missing."""
    if exclude_s is None or exclude_s <= 0 or "_prev_event_stop" not in df.columns:
        return pd.Series(False, index=df.index)
    dist = df["start_time"].astype(float) - df["_prev_event_stop"].astype(float)
    return (dist.notna() & (dist < float(exclude_s))).astype(bool)


def event_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per event: `_event_id`, `session_key`, `_event_start`, `_event_stop`, `label` and `_type_raw`
    (when present) of the event's first ictal row, `n_rows`."""
    ict = df[df["_event_id"] >= 0].sort_values(["_event_id", "start_time"])
    if not len(ict):
        cols = ["_event_id", "session_key", "_event_start", "_event_stop", "n_rows"]
        return pd.DataFrame(columns=cols)
    first = ict.groupby("_event_id", sort=True).first()
    keep = [c for c in ("session_key", "_event_start", "_event_stop", "label", "_type_raw") if c in first.columns]
    out = first[keep].copy()
    out["n_rows"] = ict.groupby("_event_id", sort=True).size()
    return out.reset_index()
