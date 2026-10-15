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
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from ..hardware import cache_budget_bytes, reader_threads
from .label_utils import (
    LABEL_CATEGORY_BACKGROUND, LABEL_CATEGORY_EXCLUDED, LABEL_CATEGORY_ICTAL, LABEL_CATEGORY_PREICTAL,
    SEIZURE_TYPE_CLASSES, TYPE_GROUPINGS, classify_label, extract_seizure_type,
)
from .tokenizer import AmplitudeTokenizer
from .events import events_from_rows

log = logging.getLogger("eeg_gpt.data")

_SPLIT_PREFIXES = {
    "trn", "train", "training",
    "dev", "devel", "development",
    "val", "vld", "valid", "validation",
    "eval", "evaluation",
    "tst", "test", "testing",
}

_PAT_RE = re.compile(
    r"^(?:(?:trn|train|training|dev|devel|development|val|vld|valid|validation|eval|evaluation|tst|test|testing)_)?([a-z0-9]+)_s(\d+)",
    re.I,
)


def patient_id(session_key: str) -> str:
    s = str(session_key).strip()
    m = _PAT_RE.match(s)
    if m:
        pid = m.group(1)
        if pid.lower() not in _SPLIT_PREFIXES:
            return pid
    # Fallback: scan underscore-separated tokens for the first token that is not a split prefix or stage suffix
    parts = s.split("_")
    for p in parts:
        if p.lower() not in _SPLIT_PREFIXES and p.lower() not in {"proc", "raw", "parquet"} and p:
            return p
    return "unknown"


def session_number(session_key: str) -> int:
    m = _PAT_RE.match(str(session_key))
    if m:
        return int(m.group(2))
    m2 = re.search(r"_s(\d+)", str(session_key), re.I)
    return int(m2.group(1)) if m2 else 0



_MASTER_CACHE: dict = {}


def _map_unique(col: pd.Series, fn) -> pd.Series:
    """fn over the distinct values only: a few thousand session keys / labels instead of millions of rows."""
    return col.map({v: fn(v) for v in col.unique()})


def _event_window_s(df: pd.DataFrame) -> float:
    """The CSV's window length (median row duration): the gap under which two ictal rows are one seizure."""
    try:
        d = float(np.nanmedian((df["stop_time"] - df["start_time"]).values.astype(np.float64)))
        return d if np.isfinite(d) and d > 0 else 4.0
    except Exception:
        return 4.0


def load_master(master_csv: str) -> pd.DataFrame:
    """The labeled rows of the master CSV (TERM channel, is_valid) with `_patient`, `_cat`, `_type_raw`, the seizure
    event columns of data/events.py (`_event_id`, `_event_start`, `_event_stop`, `_next_event_start`,
    `_prev_event_stop`) and `_postictal` (False here) added.

    Read and parsed once per process (keyed on path, size and mtime): the patient CV builds 11 datasets per
    patient, and each used to re-read and re-parse the whole CSV. Callers must not modify the returned frame.
    """
    st = os.stat(master_csv)
    key = (os.path.abspath(master_csv), st.st_size, st.st_mtime_ns)
    df = _MASTER_CACHE.get(key)
    if df is not None:
        return df
    t0 = time.time()
    df = pd.read_csv(master_csv, dtype={"session_key": str, "label": str}, low_memory=False)
    n_all = len(df)
    if "channel" in df.columns:
        df = df[df["channel"].astype(str).str.upper() == "TERM"]
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1", "1.0"]) | (df["is_valid"] == True)]  # noqa: E712
    df = df.reset_index(drop=True)
    df["_patient"] = _map_unique(df["session_key"], patient_id)
    df["_cat"] = _map_unique(df["label"], classify_label)
    df["_type_raw"] = _map_unique(df["label"], extract_seizure_type)
    # seizure events (data/events.py): computed on the full frame so the SOP-buffer / postictal rows dropped later
    # still count; the window length only sets the merge gap between ictal rows of one seizure
    df = events_from_rows(df, window_s=_event_window_s(df))
    df["_postictal"] = False      # set by LabeledWindows from --postictal-exclude-s via events.postictal_mask
    _MASTER_CACHE.clear()
    _MASTER_CACHE[key] = df
    log.info(f"master CSV {master_csv}: {n_all} rows read, {len(df)} labeled TERM rows kept, "
             f"{df['_patient'].nunique()} patients, {df['session_key'].nunique()} sessions, "
             f"{df.loc[df['_event_id'] >= 0, '_event_id'].nunique()} seizure events "
             f"({time.time() - t0:.1f}s, cached for this process)")
    return df

# per-process cache accounting: every DataLoader worker is its own process with its own share of RAM
_PROC = {"pid": None, "budget": 0, "bytes": 0}


def _proc_state() -> dict:
    pid = os.getpid()
    if _PROC["pid"] != pid:
        wi = torch.utils.data.get_worker_info()
        n = wi.num_workers if wi else 1
        _PROC.update(pid=pid, budget=cache_budget_bytes(n), bytes=0)
        try:
            # pyarrow sizes its pool from the node's core count: bound it to our share of the usable cores
            pa.set_cpu_count(reader_threads(n))
        except Exception:
            pass
    return _PROC


class SessionStore:
    """`capacity` recordings are always kept; beyond that the cache keeps growing while the
    process is inside its RAM budget (eeg_gpt/hardware.py), so a machine with memory to spare
    stops re-reading parquet files altogether."""

    def __init__(self, checkpoint_dir: str, stage: str = "proc", capacity: int = 8, channels: Optional[list] = None):
        self.dir = checkpoint_dir
        self.stage = stage
        self.capacity = max(1, int(capacity))
        self.channels = channels
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._n_samples_cache: dict = {}
        self._disk_index: Optional[dict] = None      # file name -> [size, mtime_ns, rows], persisted next to the parquets
        self._disk_dirty = False

    def path(self, session_key: str) -> str:
        return os.path.join(self.dir, f"{session_key}_{self.stage}.parquet")

    def list_sessions(self) -> list:
        suf = f"_{self.stage}.parquet"
        return sorted(f[: -len(suf)] for f in os.listdir(self.dir) if f.endswith(suf))

    def _index_path(self) -> str:
        return os.path.join(self.dir, f".eeg_gpt_nsamples_{self.stage}.json")

    def n_samples(self, session_key: str) -> int:
        if session_key not in self._n_samples_cache:
            if self._disk_index is None:
                try:
                    with open(self._index_path()) as f:
                        self._disk_index = json.load(f)
                except (OSError, ValueError):
                    self._disk_index = {}
            path = self.path(session_key)
            name = os.path.basename(path)
            st = os.stat(path)
            hit = self._disk_index.get(name)
            if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
                rows = int(hit[2])
            else:
                rows = pq.ParquetFile(path, memory_map=True).metadata.num_rows
                self._disk_index[name] = [st.st_size, st.st_mtime_ns, rows]
                self._disk_dirty = True
            self._n_samples_cache[session_key] = rows
        return self._n_samples_cache[session_key]

    def flush_index(self) -> None:
        """Persist the row counts read so far, so the next run skips the parquet footers."""
        if not self._disk_dirty:
            return
        self._disk_dirty = False
        try:
            merged = {}
            try:
                with open(self._index_path()) as f:
                    merged = json.load(f)
            except (OSError, ValueError):
                pass
            merged.update(self._disk_index)
            tmp = f"{self._index_path()}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(merged, f)
            os.replace(tmp, self._index_path())
        except OSError:
            pass                                     # read-only checkpoint dir: stay with the in-memory cache

    def get(self, session_key: str) -> np.ndarray:
        """(C, T) float32 array of the whole recording."""
        if session_key in self._cache:
            self._cache.move_to_end(session_key)
            return self._cache[session_key]
        proc = _proc_state()
        tbl = pq.read_table(self.path(session_key), columns=self.channels, memory_map=True, use_threads=True)
        # int16 + per-channel scale (EDF-style storage written by the preprocessing
        # repo's checkpoint_io / tools/shrink_checkpoints.py) or plain float columns
        meta = tbl.schema.metadata or {}
        scales = json.loads(meta[b"eeg_scales_json"].decode()) if b"eeg_scales_json" in meta else {}
        num_cols = tbl.num_columns
        num_rows = tbl.num_rows
        arr = np.empty((num_cols, num_rows), dtype=np.float32)
        for i, (name, col) in enumerate(zip(tbl.column_names, tbl.columns)):
            v = col.to_numpy(zero_copy_only=False)
            if v.dtype == np.int16 or name in scales:
                scale = np.float32(scales.get(name, float(meta.get(f"eeg_scale_{name}".encode(), b"1").decode())))
                np.multiply(v, scale, out=arr[i], casting="unsafe")
            else:
                arr[i] = v
        if self.channels is None:
            self.channels = list(tbl.column_names)
        self._cache[session_key] = arr
        proc["bytes"] += arr.nbytes
        while len(self._cache) > self.capacity and proc["bytes"] > proc["budget"]:
            _, old = self._cache.popitem(last=False)
            proc["bytes"] = max(0, proc["bytes"] - old.nbytes)
        return arr


class PretrainSessions(Dataset):
    """Yields (tokens[block+1], z[block+1], channel_id) from random positions.

    __len__ is virtual (`samples_per_epoch`); sessions are drawn proportionally
    to their length so every hour of recording is seen equally often. Data
    loading with several workers is fine: each worker keeps its own cache.

    Sampling is pooled so the parquet cache actually hits: sample i belongs to
    stream `i % session_pool`, and a stream keeps its session for
    `samples_per_session` draws before drawing a new one (streams are staggered,
    so the pool turns over one session at a time). A batch therefore mixes
    `session_pool` recordings while a full-recording read is amortized over
    `samples_per_session` samples instead of paid on almost every sample.
    """

    def __init__(self, checkpoint_dir: str, block_size: int = 1024, samples_per_epoch: int = 100_000,
                 stage: str = "proc", cache_capacity: int = 8, tokenizer: Optional[AmplitudeTokenizer] = None,
                 sessions: Optional[list] = None, stats: str = "block", seed: int = 0,
                 session_pool: int = 16, samples_per_session: int = 256):
        self.pool = max(1, int(session_pool))
        self.per_session = max(1, int(samples_per_session))
        self.store = SessionStore(checkpoint_dir, stage, max(cache_capacity, self.pool))
        self.block = int(block_size)
        self.n = int(samples_per_epoch)
        self.tok = tokenizer or AmplitudeTokenizer()
        self.stats = stats
        self.sessions = sessions if sessions is not None else self.store.list_sessions()
        if not self.sessions:
            raise ValueError(f"no *_{stage}.parquet sessions under {checkpoint_dir}")
        lengths = np.array([max(0, self.store.n_samples(s) - self.block - self.tok.patch) for s in self.sessions], dtype=np.float64)
        self.store.flush_index()
        keep = lengths > 0
        self.sessions = [s for s, k in zip(self.sessions, keep) if k]
        lengths = lengths[keep]
        self.p = lengths / lengths.sum()
        self._cdf = np.cumsum(self.p)
        self.seed = seed
        self._session_stats: dict = {}
        self._stream: dict = {}      # stream -> (turn, session index)

    def __len__(self):
        return self.n

    def _stats(self, key, arr):
        if self.stats != "session":
            return None, None
        if key not in self._session_stats:
            self._session_stats[key] = (arr.mean(axis=1, keepdims=True), arr.std(axis=1, keepdims=True) + 1e-8)
        return self._session_stats[key]

    def _session_for(self, i: int) -> str:
        j = i % self.pool
        turn = (i // self.pool + j * self.per_session // self.pool) // self.per_session
        cur = self._stream.get(j)
        if cur is None or cur[0] != turn:
            u = np.random.default_rng((self.seed, j, turn)).random()
            cur = (turn, min(int(np.searchsorted(self._cdf, u)), len(self.sessions) - 1))
            self._stream[j] = cur
        return self.sessions[cur[1]]

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + i * 7919 + (wi.id if wi else 0))
        s = self._session_for(i)
        arr = self.store.get(s)
        C, T = arr.shape
        c = int(rng.integers(0, C))
        patch = self.tok.patch
        # crop = block + patch samples so every input position has a next-sample / next-patch target
        t0 = int(rng.integers(0, max(1, T - self.block - patch)))
        x = arr[c: c + 1, t0: t0 + self.block + patch]
        mu, sd = self._stats(s, arr)
        if mu is not None:
            mu, sd = mu[c: c + 1], sd[c: c + 1]
        elif self.tok.norm_scope == "lead4s":
            # leak-free: statistics of the first 1024 samples (4 s at 256 Hz) of the crop only, so the
            # causal model never sees a normalisation that used its own future
            mu, sd = self.tok.lead_stats(x)
        tokens, z, _ = self.tok.encode(x, mu, sd)
        inp = torch.from_numpy(z[0]) if patch > 1 else torch.from_numpy(tokens[0])
        return inp, torch.from_numpy(z[0]), torch.tensor(c, dtype=torch.long)


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
      "time_to_onset": float       -- seconds to the next seizure EVENT onset (data/events.py), or -1
      "feats":    (2F,) float      -- data/features.py context_features of the RAW signal (last block, delta vs the
                                      previous blocks); empty tensor when feat_branch == "none"
      "log_sd":   (C, N) float     -- log raw sd per block and channel (the amplitude the z-scoring removes)
      "row_idx": long, "session_idx": long
    }
    `tokens` is whatever the tokenizer's encode_blocks() emits as model input: int64 levels for a token trunk,
    float32 z for a patch trunk. `postictal_exclude_s` > 0 sets occ_valid = 0 on background rows that start within
    that many seconds after the previous seizure of the session (events.postictal_mask).
    """

    def __init__(self, master_csv: str, checkpoint_dir: str, split: Optional[str] = None, stage: str = "proc",
                 window_s: float = 4.0, fs: float = 256.0, context_blocks: int = 1, horizon_s: float = 30.0,
                 type_grouping: str = "focal_generalized", tokenizer: Optional[AmplitudeTokenizer] = None,
                 cache_capacity: int = 8, keep_ictal: bool = True, sessions: Optional[list] = None,
                 patient: Optional[str] = None, row_indices: Optional[list[int] | np.ndarray] = None,
                 feat_branch: str = "none", postictal_exclude_s: float = 0.0):
        t0 = time.time()
        self.feat_branch = feat_branch or "none"
        self.postictal_exclude_s = float(postictal_exclude_s or 0.0)
        df = load_master(master_csv)
        if split and "split" in df.columns:
            df = df[df["split"].astype(str) == split]
        if patient is not None:
            # from session_key, like everywhere else (pretraining folds use the parquet file names)
            df = df[df["_patient"] == str(patient)]
        if sessions is not None:
            df = df[df["session_key"].isin(set(sessions))]
        df = df.copy()
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
        if row_indices is not None:
            df = df.iloc[row_indices].reset_index(drop=True)
        self.df = df
        self.sessions = sorted(df["session_key"].unique().tolist())
        self.session_idx = {s: i for i, s in enumerate(self.sessions)}
        self.store = SessionStore(checkpoint_dir, stage, cache_capacity)
        self.tok = tokenizer or AmplitudeTokenizer()
        self.fs = float(fs)
        self.T = int(round(window_s * fs))
        self.N = max(1, int(context_blocks))
        self.horizon_s = float(horizon_s)

        # Pre-extract columnar metadata into fast NumPy/torch structures to eliminate per-sample DataFrame overhead
        self._session_keys = self.df["session_key"].tolist()
        self._stop_times = self.df["stop_time"].values.astype(np.float64)
        self._cats = self.df["_cat"].values
        self._raw_type_idxs = self.df["_type_idx"].values.astype(np.int64)

        n_rows = len(self.df)
        occ = np.zeros(n_rows, dtype=np.float32)
        occ_valid = np.ones(n_rows, dtype=np.float32)
        horizon_pos = np.zeros(n_rows, dtype=np.float32)
        type_idx = np.full(n_rows, -1, dtype=np.int64)
        is_ictal = np.zeros(n_rows, dtype=np.float32)
        time_to_onset = np.full(n_rows, -1.0, dtype=np.float32)
        session_idx = np.zeros(n_rows, dtype=np.int64)
        # time to the next seizure EVENT onset (events.py columns, computed on the full frame before the SOP-buffer
        # rows were dropped); the per-ictal-row onsets are the fallback for frames without the event columns
        next_onset = (self.df["_next_event_start"].values.astype(np.float64) if "_next_event_start" in self.df.columns
                      else None)

        for i in range(n_rows):
            k = self._session_keys[i]
            g_stop = self._stop_times[i]
            cat = self._cats[i]
            is_pre = (cat == LABEL_CATEGORY_PREICTAL)
            is_ict = (cat == LABEL_CATEGORY_ICTAL)
            if is_pre:
                occ[i] = 1.0
            if is_ict:
                occ_valid[i] = 0.0
                is_ictal[i] = 1.0
            if is_ict or is_pre:
                type_idx[i] = self._raw_type_idxs[i]
            if next_onset is not None:
                tto = (next_onset[i] - g_stop) if np.isfinite(next_onset[i]) else -1.0
            else:
                ons = self.onsets.get(k, [])
                pos = bisect.bisect_left(ons, g_stop)
                tto = (ons[pos] - g_stop) if pos < len(ons) else -1.0
            time_to_onset[i] = float(tto)
            if 0.0 <= tto <= self.horizon_s:
                horizon_pos[i] = 1.0
            session_idx[i] = self.session_idx[k]

        if self.postictal_exclude_s > 0:
            from .events import postictal_mask
            post = postictal_mask(self.df, self.postictal_exclude_s).values & (self._cats == LABEL_CATEGORY_BACKGROUND)
            occ_valid[post] = 0.0
            self.df["_postictal"] = post
            log.info(f"LabeledWindows: {int(post.sum())} background rows within {self.postictal_exclude_s:.0f}s after a "
                     f"seizure excluded from the binary labels (--postictal-exclude-s)")

        self._occ = torch.from_numpy(occ)
        self._occ_valid = torch.from_numpy(occ_valid)
        self._horizon_pos = torch.from_numpy(horizon_pos)
        self._type_idx = torch.from_numpy(type_idx)
        self._is_ictal = torch.from_numpy(is_ictal)
        self._time_to_onset = torch.from_numpy(time_to_onset)
        self._session_idx = torch.from_numpy(session_idx)
        log.debug(f"LabeledWindows: {n_rows} rows, {len(self.sessions)} sessions"
                  f"{f' (patient {patient})' if patient is not None else ''}{f' (split {split})' if split else ''} in {time.time() - t0:.2f}s")

    def __len__(self):
        return len(self.df)

    @property
    def n_channels(self) -> int:
        if self.store.channels is None:
            self.store.get(self.sessions[0])
        return len(self.store.channels)

    @property
    def feat_dim(self) -> int:
        """Length of the `feats` vector (0 when feat_branch == 'none'); reads one recording for the channel count."""
        if self.feat_branch == "none":
            return 0
        from .features import feat_dim
        return feat_dim(self.n_channels)

    def counts(self) -> dict:
        d = self.df
        return {"rows": len(d), "preictal": int((d["_cat"] == LABEL_CATEGORY_PREICTAL).sum()),
                "background": int((d["_cat"] == LABEL_CATEGORY_BACKGROUND).sum()),
                "ictal": int((d["_cat"] == LABEL_CATEGORY_ICTAL).sum()), "sessions": len(self.sessions),
                "patients": len({patient_id(s) for s in self.sessions}),
                "type_hist": {self.type_classes[i]: int((d["_type_idx"] == i).sum()) for i in range(len(self.type_classes))}}

    def __getitem__(self, i):
        key = self._session_keys[i]
        arr = self.store.get(key)
        C, Ttot = arr.shape
        e = int(round(float(self._stop_times[i]) * self.fs))
        e = min(max(e, self.T), Ttot)

        L = self.N * self.T
        if e - L >= 0:
            x_total = arr[:, e - L: e].reshape(C, self.N, self.T)
        else:
            x_total = np.zeros((C, L), dtype=np.float32)
            if e > 0:
                x_total[:, L - e:] = arr[:, :e]
            x_total = x_total.reshape(C, self.N, self.T)

        if hasattr(self.tok, "encode_blocks"):
            enc = self.tok.encode_blocks(x_total)
            inp, z_all, log_sd = enc["inp"], enc["z"], enc["log_sd"]
        else:                                           # older tokenizer: per-block z-score, token input
            inp, z_all, _ = self.tok.encode(x_total)
            log_sd = np.log(x_total.std(axis=-1) + 1e-8).astype(np.float32)
        if self.feat_branch != "none":
            from .features import context_features
            feats = torch.from_numpy(context_features(x_total, self.fs))
        else:
            feats = torch.zeros(0, dtype=torch.float32)
        return {
            "tokens": torch.from_numpy(np.ascontiguousarray(inp)),
            "z_last": torch.from_numpy(np.ascontiguousarray(z_all[:, -1, :])),
            "feats": feats,
            "log_sd": torch.from_numpy(np.ascontiguousarray(log_sd)),
            "occ": self._occ[i],
            "occ_valid": self._occ_valid[i],
            "horizon_pos": self._horizon_pos[i],
            "type_idx": self._type_idx[i],
            "is_ictal": self._is_ictal[i],
            "time_to_onset": self._time_to_onset[i],
            "row_idx": torch.tensor(i, dtype=torch.long),
            "session_idx": self._session_idx[i],
        }