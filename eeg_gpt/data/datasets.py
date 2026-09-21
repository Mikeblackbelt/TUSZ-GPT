"""Datasets.

SessionStore      -- LRU cache of `<session_key>_<stage>.parquet` recordings
                     (columns = channels, rows = samples), exactly the layout
                     written by TUSZ-Seizure-Predict.
PretrainSessions  -- UNLABELED corpus for GPT pretraining: random
                     (session, channel, start) draws over every parquet in the
                     checkpoint dir, including stretches the master CSV never
                     labels (postictal, SOP buffers, gaps). This is the data the
                     old pipeline never used.
LabeledWindows    -- the labeled rows of the master CSV (TERM channel, is_valid),
                     mapped with the same label semantics as before
                     (data/label_utils.py), returned as per-channel token
                     sequences with an optional context of N preceding blocks.
"""
from __future__ import annotations

import bisect
import json
import os
import re
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .label_utils import (
    LABEL_CATEGORY_BACKGROUND, LABEL_CATEGORY_EXCLUDED, LABEL_CATEGORY_ICTAL, LABEL_CATEGORY_PREICTAL,
    SEIZURE_TYPE_CLASSES, TYPE_GROUPINGS, classify_label, extract_seizure_type,
)
from .tokenizer import AmplitudeTokenizer

_PAT_RE = re.compile(r"^(?:trn|dev|eval|train|test)?_?([a-z]{8})_s(\d+)_", re.I)


def patient_id(session_key: str) -> str:
    m = _PAT_RE.match(str(session_key))
    return m.group(1) if m else str(session_key).split("_")[0]


def session_number(session_key: str) -> int:
    m = _PAT_RE.match(str(session_key))
    return int(m.group(2)) if m else 0


class SessionStore:
    def __init__(self, checkpoint_dir: str, stage: str = "proc", capacity: int = 8, channels: Optional[list] = None):
        self.dir = checkpoint_dir
        self.stage = stage
        self.capacity = max(1, int(capacity))
        self.channels = channels
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def path(self, session_key: str) -> str:
        return os.path.join(self.dir, f"{session_key}_{self.stage}.parquet")

    def list_sessions(self) -> list:
        suf = f"_{self.stage}.parquet"
        return sorted(f[: -len(suf)] for f in os.listdir(self.dir) if f.endswith(suf))

    def n_samples(self, session_key: str) -> int:
        return pq.ParquetFile(self.path(session_key)).metadata.num_rows

    def get(self, session_key: str) -> np.ndarray:
        """(C, T) float32 array of the whole recording."""
        if session_key in self._cache:
            self._cache.move_to_end(session_key)
            return self._cache[session_key]
        tbl = pq.read_table(self.path(session_key), columns=self.channels)
        # int16 + per-channel scale (EDF-style storage written by the preprocessing
        # repo's checkpoint_io / tools/shrink_checkpoints.py) or plain float columns
        meta = tbl.schema.metadata or {}
        scales = json.loads(meta[b"eeg_scales_json"].decode()) if b"eeg_scales_json" in meta else {}
        cols = []
        for name, col in zip(tbl.column_names, tbl.columns):
            v = col.to_numpy(zero_copy_only=False)
            if v.dtype == np.int16 or name in scales:
                v = v.astype(np.float32) * np.float32(scales.get(name, float(meta.get(f"eeg_scale_{name}".encode(), b"1").decode())))
            cols.append(v.astype(np.float32, copy=False))
        arr = np.stack(cols, axis=0)
        if self.channels is None:
            self.channels = list(tbl.column_names)
        self._cache[session_key] = arr
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return arr


class PretrainSessions(Dataset):
    """Yields (tokens[block+1], z[block+1], channel_id) from random positions.

    __len__ is virtual (`samples_per_epoch`); sessions are drawn proportionally
    to their length so every hour of recording is seen equally often. Data
    loading with several workers is fine: each worker keeps its own cache.
    """

    def __init__(self, checkpoint_dir: str, block_size: int = 1024, samples_per_epoch: int = 100_000,
                 stage: str = "proc", cache_capacity: int = 8, tokenizer: Optional[AmplitudeTokenizer] = None,
                 sessions: Optional[list] = None, stats: str = "block", seed: int = 0):
        self.store = SessionStore(checkpoint_dir, stage, cache_capacity)
        self.block = int(block_size)
        self.n = int(samples_per_epoch)
        self.tok = tokenizer or AmplitudeTokenizer()
        self.stats = stats
        self.sessions = sessions if sessions is not None else self.store.list_sessions()
        if not self.sessions:
            raise ValueError(f"no *_{stage}.parquet sessions under {checkpoint_dir}")
        lengths = np.array([max(0, self.store.n_samples(s) - self.block - 1) for s in self.sessions], dtype=np.float64)
        keep = lengths > 0
        self.sessions = [s for s, k in zip(self.sessions, keep) if k]
        lengths = lengths[keep]
        self.p = lengths / lengths.sum()
        self.seed = seed
        self._session_stats: dict = {}

    def __len__(self):
        return self.n

    def _stats(self, key, arr):
        if self.stats != "session":
            return None, None
        if key not in self._session_stats:
            self._session_stats[key] = (arr.mean(axis=1, keepdims=True), arr.std(axis=1, keepdims=True) + 1e-8)
        return self._session_stats[key]

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + i * 7919 + (wi.id if wi else 0))
        s = self.sessions[rng.choice(len(self.sessions), p=self.p)]
        arr = self.store.get(s)
        C, T = arr.shape
        c = int(rng.integers(0, C))
        t0 = int(rng.integers(0, T - self.block - 1))
        x = arr[c: c + 1, t0: t0 + self.block + 1]
        mu, sd = self._stats(s, arr)
        tokens, z, _ = self.tok.encode(x, None if mu is None else mu[c: c + 1], None if sd is None else sd[c: c + 1])
        return torch.from_numpy(tokens[0]), torch.from_numpy(z[0]), torch.tensor(c, dtype=torch.long)


class LabeledWindows(Dataset):
    """Rows of the master CSV -> per-channel tokens.

    item = {
      "tokens":   (C, N, T) long   -- N context blocks (oldest first), the last is the row's window
      "z_last":   (C, T) float     -- z-scored last block (for the band-power baseline)
      "occ":      float            -- 1 preictal / 0 background (ictal rows: occ_valid = 0)
      "occ_valid": float
      "horizon_pos": float         -- 1 if a seizure onset lies within `horizon_s` after the window (Mahdi-style label)
      "type_idx": long             -- seizure type index (grouped), -1 if none
      "is_ictal": float
      "time_to_onset": float       -- seconds, or -1
      "row_idx": long, "session_idx": long
    }
    """

    def __init__(self, master_csv: str, checkpoint_dir: str, split: str, stage: str = "proc",
                 window_s: float = 4.0, fs: float = 256.0, context_blocks: int = 1, horizon_s: float = 30.0,
                 type_grouping: str = "focal_generalized", tokenizer: Optional[AmplitudeTokenizer] = None,
                 cache_capacity: int = 8, keep_ictal: bool = True, sessions: Optional[list] = None):
        df = pd.read_csv(master_csv)
        if "channel" in df.columns:
            df = df[df["channel"].astype(str).str.upper() == "TERM"]
        if "is_valid" in df.columns:
            df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1", "1.0"]) | (df["is_valid"] == True)]  # noqa: E712
        if split and "split" in df.columns:
            df = df[df["split"].astype(str) == split]
        if sessions is not None:
            df = df[df["session_key"].isin(set(sessions))]
        df = df.copy()
        df["_cat"] = df["label"].apply(classify_label)
        df["_type_raw"] = df["label"].apply(extract_seizure_type)
        groups = TYPE_GROUPINGS.get(type_grouping)
        if groups is None:
            self.type_classes = SEIZURE_TYPE_CLASSES
            tmap = {t: i for i, t in enumerate(self.type_classes)}
        else:
            self.type_classes = list(groups.keys())
            tmap = {t: i for i, g in enumerate(groups.values()) for t in g}
        df["_type_idx"] = df["_type_raw"].apply(lambda t: tmap.get(t, -1))
        # seizure onsets per session from the ictal rows (before any filtering)
        self.onsets: dict = {}
        for k, g in df[df["_cat"] == LABEL_CATEGORY_ICTAL].groupby("session_key"):
            self.onsets[k] = sorted(float(v) for v in g["start_time"].values)
        df = df[df["_cat"].isin([LABEL_CATEGORY_BACKGROUND, LABEL_CATEGORY_PREICTAL] + ([LABEL_CATEGORY_ICTAL] if keep_ictal else []))]
        df = df.sort_values(["session_key", "start_time"]).reset_index(drop=True)
        self.df = df
        self.sessions = sorted(df["session_key"].unique().tolist())
        self.session_idx = {s: i for i, s in enumerate(self.sessions)}
        self.store = SessionStore(checkpoint_dir, stage, cache_capacity)
        self.tok = tokenizer or AmplitudeTokenizer()
        self.fs = float(fs)
        self.T = int(round(window_s * fs))
        self.N = max(1, int(context_blocks))
        self.horizon_s = float(horizon_s)

    def __len__(self):
        return len(self.df)

    @property
    def n_channels(self) -> int:
        if self.store.channels is None:
            self.store.get(self.sessions[0])
        return len(self.store.channels)

    def counts(self) -> dict:
        d = self.df
        return {"rows": len(d), "preictal": int((d["_cat"] == LABEL_CATEGORY_PREICTAL).sum()),
                "background": int((d["_cat"] == LABEL_CATEGORY_BACKGROUND).sum()),
                "ictal": int((d["_cat"] == LABEL_CATEGORY_ICTAL).sum()), "sessions": len(self.sessions),
                "patients": len({patient_id(s) for s in self.sessions}),
                "type_hist": {self.type_classes[i]: int((d["_type_idx"] == i).sum()) for i in range(len(self.type_classes))}}

    def __getitem__(self, i):
        r = self.df.iloc[i]
        key = r["session_key"]
        arr = self.store.get(key)
        C, Ttot = arr.shape
        e = int(round(float(r["stop_time"]) * self.fs))
        e = min(max(e, self.T), Ttot)
        s = e - self.T
        blocks_tok, blocks_z = [], []
        for j in range(self.N - 1, -1, -1):
            b0, b1 = e - (j + 1) * self.T, e - j * self.T
            if b1 <= 0:
                x = np.zeros((C, self.T), dtype=np.float32)
            else:
                x = arr[:, max(0, b0): b1]
                if x.shape[1] < self.T:
                    x = np.pad(x, ((0, 0), (self.T - x.shape[1], 0)))
            t, z, _ = self.tok.encode(x)
            blocks_tok.append(t); blocks_z.append(z)
        tokens = torch.from_numpy(np.stack(blocks_tok, axis=1))          # (C, N, T)
        cat = r["_cat"]
        is_pre, is_ict = cat == LABEL_CATEGORY_PREICTAL, cat == LABEL_CATEGORY_ICTAL
        g_stop = float(r["stop_time"])
        ons = self.onsets.get(key, [])
        pos = bisect.bisect_left(ons, g_stop)
        tto = (ons[pos] - g_stop) if pos < len(ons) else -1.0
        horizon_pos = 1.0 if (0.0 <= tto <= self.horizon_s) else 0.0
        return {
            "tokens": tokens,
            "z_last": torch.from_numpy(blocks_z[-1]),
            "occ": torch.tensor(1.0 if is_pre else 0.0),
            "occ_valid": torch.tensor(0.0 if is_ict else 1.0),
            "horizon_pos": torch.tensor(horizon_pos),
            "type_idx": torch.tensor(int(r["_type_idx"]) if (is_ict or is_pre) else -1, dtype=torch.long),
            "is_ictal": torch.tensor(1.0 if is_ict else 0.0),
            "time_to_onset": torch.tensor(float(tto)),
            "row_idx": torch.tensor(i, dtype=torch.long),
            "session_idx": torch.tensor(self.session_idx[key], dtype=torch.long),
        }