"""CV protocol and seizure-event tests (CPU, seconds; no parquet files needed).

    python -m eeg_gpt.tests.test_protocol

Covers data/events.py (event merge, next/previous event, postictal mask), the session and event CV strategies of
patient_sweep.generate_patient_cv_splits (no seizure on both sides of a fold, context margin, inner validation),
the removal of the row-stratified fallback, the horizon-label sanity, profile_all_patients' cv_strategy column and
the fold-record / summary assembly.
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pandas as pd

from eeg_gpt.data.datasets import load_master
from eeg_gpt.data.events import EVENT_COLUMNS, event_table, events_from_rows, postictal_mask
from eeg_gpt.data.label_utils import LABEL_CATEGORY_BACKGROUND, LABEL_CATEGORY_ICTAL, LABEL_CATEGORY_PREICTAL
from eeg_gpt.training import patient_sweep
from eeg_gpt.training.patient_sweep import generate_patient_cv_splits, profile_all_patients

W = 4.0


def synthetic_master(path: str, patient: str = "aaaaaaaa", n_sessions: int = 6, seizures_per_session: int = 1,
                     session_s: float = 1800.0, typ: str = "fnsz") -> str:
    """A master CSV in the TUSZ-Seizure-Predict conventions: bg / p<type> / p<type>_sopbuffer (invalid) / <type>
    rows of 4 s, `seizures_per_session` seizures per session spaced evenly, SPH 120 s / SOP 420 s."""
    rows = []
    for s in range(1, n_sessions + 1):
        key = f"trn_{patient}_s{s:03d}_2010_ar1"
        onsets = [session_s * (k + 1) / (seizures_per_session + 1) for k in range(seizures_per_session)]
        t = 0.0
        for on in onsets:
            dur = 12.0
            while t + W <= on - 540:
                rows.append((key, t, t + W, "bg", True)); t += W
            while t + W <= on - 120:
                rows.append((key, t, t + W, f"p{typ}", True)); t += W
            rows.append((key, on - 120, on, f"p{typ}_sopbuffer", False))
            t = on
            while t + W <= on + dur:
                rows.append((key, t, t + W, typ, True)); t += W
        while t + W <= session_s:
            rows.append((key, t, t + W, "bg", True)); t += W
    df = pd.DataFrame(rows, columns=["session_key", "start_time", "stop_time", "label", "is_valid"])
    df["channel"] = "TERM"
    df["edf_path"] = "/data/" + df["session_key"] + ".edf"
    df.to_csv(path, index=False)
    return path


def labeled_frame(csv: str, patient: str) -> pd.DataFrame:
    """What LabeledWindows.df looks like for one patient: bg/preictal/ictal rows, sorted, positional index."""
    df = load_master(csv)
    df = df[df["_patient"] == patient]
    df = df[df["_cat"].isin([LABEL_CATEGORY_BACKGROUND, LABEL_CATEGORY_PREICTAL, LABEL_CATEGORY_ICTAL])]
    return df.sort_values(["session_key", "start_time"]).reset_index(drop=True)


def test_events():
    df = pd.DataFrame({
        "session_key": ["s1"] * 8 + ["s2"] * 2,
        "start_time": [0, 96, 100, 104, 110, 200, 300, 304, 0, 50],
        "stop_time":  [4, 100, 104, 108, 114, 204, 304, 308, 4, 54],
        "label": ["bg", "pfnsz", "fnsz", "fnsz", "bg", "pfnsz", "fnsz", "bg", "bg", "gnsz"],
    })
    df["_cat"] = df["label"].map(lambda l: LABEL_CATEGORY_ICTAL if l in ("fnsz", "gnsz") else
                                 LABEL_CATEGORY_PREICTAL if l.startswith("p") else LABEL_CATEGORY_BACKGROUND)
    out = events_from_rows(df, W)
    assert all(c in out.columns for c in EVENT_COLUMNS) and list(out.index) == list(df.index)
    assert out["_event_id"].tolist() == [-1, -1, 0, 0, -1, -1, 1, -1, -1, 2], out["_event_id"].tolist()
    assert out.loc[2, "_event_start"] == 100 and out.loc[3, "_event_stop"] == 108 and out.loc[6, "_event_stop"] == 304
    assert out.loc[1, "_next_event_start"] == 100 and out.loc[5, "_next_event_start"] == 300 and out.loc[0, "_next_event_start"] == 100
    assert out.loc[4, "_prev_event_stop"] == 108 and out.loc[7, "_prev_event_stop"] == 304 and math.isnan(out.loc[0, "_prev_event_stop"])
    assert math.isnan(out.loc[7, "_next_event_start"]) and out.loc[8, "_next_event_start"] == 50 and out.loc[9, "_event_id"] == 2
    m = postictal_mask(out, 60.0)
    assert bool(m[4]) and not bool(m[5]) and not bool(m[0]) and m.dtype == bool
    assert not postictal_mask(out, 0.0).any()
    tab = event_table(out)
    assert len(tab) == 3 and tab["n_rows"].tolist() == [2, 1, 1]
    # two ictal rows separated by more than half a window are two events
    df2 = df.copy(); df2.loc[3, ["start_time", "stop_time"]] = [107, 111]
    assert events_from_rows(df2, W)["_event_id"].tolist()[2:4] == [0, 1]
    print("events ok")


def test_load_master_columns(csv):
    df = load_master(csv)
    assert all(c in df.columns for c in EVENT_COLUMNS) and "_postictal" in df.columns and not df["_postictal"].any()
    pre = df[df["_cat"] == LABEL_CATEGORY_PREICTAL]
    tto = (pre["_next_event_start"] - pre["stop_time"]).values
    # SPH 120 s / SOP 420 s: every preictal row ends 120..540 s before its onset, so --label horizon needs
    # --horizon-s > 120 to have any positive (the 0..120 s rows are the dropped sopbuffer rows)
    assert np.all(tto >= 120 - 1e-6) and np.all(tto <= 540 + 1e-6), (tto.min(), tto.max())
    assert (tto <= 30).sum() == 0 and (tto <= 130).sum() > 0
    print("load_master event columns ok")


def _check_no_event_split(df, splits):
    eid = df["_event_id"].values
    for sp in splits:
        for a, b in (("train", "eval"), ("train", "val"), ("val", "eval")):
            if sp[a] is None or sp[b] is None:
                continue
            ea = {int(e) for e in eid[sp[a]] if e >= 0}; eb = {int(e) for e in eid[sp[b]] if e >= 0}
            assert not (ea & eb), f"event on both sides of {a}/{b}: {ea & eb}"
        parts = [sp["train"], sp["eval"]] + ([sp["val"]] if sp["val"] else [])
        allidx = sum((list(p) for p in parts), [])
        assert len(allidx) == len(set(allidx)), "a row is in two parts"


def _check_margin(df, splits, margin_s):
    sess = df["session_key"].values; start = df["start_time"].values; stop = df["stop_time"].values
    for sp in splits:
        tr = np.array(sp["train"])
        for part in ("eval", "val"):
            if not sp[part]:
                continue
            for i in sp[part]:
                same = tr[sess[tr] == sess[i]]
                bad = same[(start[same] < start[i]) & (stop[same] > start[i] - margin_s)]
                assert len(bad) == 0, f"{part} row {i} has training rows inside its context: {bad[:3]}"


def test_session_strategy(csv):
    df = labeled_frame(csv, "aaaaaaaa")                     # 6 sessions, one seizure each
    splits = generate_patient_cv_splits(df, n_splits=2, strategy="auto", seed=0, context_blocks=4, window_s=W)
    assert len(splits) == 2 and all(sp["strategy"] == "session" for sp in splits)
    sess = df["session_key"].values
    for sp in splits:
        ev_s = set(sess[sp["eval"]]); tr_s = set(sess[sp["train"]])
        assert not (ev_s & tr_s)
        assert sp["val"] is not None, "training part has 3 seizure sessions: inner validation expected"
        va_s = set(sess[sp["val"]]); assert len(va_s) == 1 and not (va_s & ev_s) and not (va_s & tr_s)
        assert len(sp["train"]) + len(sp["val"]) + len(sp["eval"]) == len(df)
    _check_no_event_split(df, splits)
    assert all(sp["val"] is None for sp in generate_patient_cv_splits(df, 2, "auto", 0, 4, W, inner_val=False))
    # 4 sessions in 2 folds leave 2 seizure sessions per training part -> no inner validation
    four = sorted(set(sess))[:4]
    df4 = df[df["session_key"].isin(four)].reset_index(drop=True)
    s4 = generate_patient_cv_splits(df4, n_splits=2, strategy="session", seed=1, context_blocks=4, window_s=W)
    assert all(sp["val"] is None for sp in s4) and all(sp["strategy"] == "session" for sp in s4)
    print("session strategy ok")


def test_event_strategy(csv2):
    df = labeled_frame(csv2, "bbbbbbbb")                    # 2 sessions with 3 seizures each
    is_pre = (df["_cat"] == LABEL_CATEGORY_PREICTAL).values
    assert df.loc[df["_event_id"] >= 0, "_event_id"].nunique() == 6
    # 2 sessions with seizures < 5 folds -> auto falls back to events
    splits = generate_patient_cv_splits(df, n_splits=5, strategy="auto", seed=0, context_blocks=4, window_s=W)
    assert len(splits) == 5 and all(sp["strategy"] == "event" for sp in splits)
    _check_no_event_split(df, splits)
    _check_margin(df, splits, 3 * W)
    for sp in splits:
        assert is_pre[sp["eval"]].sum() > 0, "every fold should hold a seizure's preictal rows"
        assert (~is_pre[sp["eval"]]).sum() > 0, "every fold should hold background"
        assert len(set(df["session_key"].values[sp["eval"]])) >= 1
        assert sp["info"]["eval_events"] and sp["info"]["dropped_margin"] >= 0
    # 2 folds: each training part has 3 seizure groups -> inner validation present, with background in it
    s2 = generate_patient_cv_splits(df, n_splits=2, strategy="event", seed=0, context_blocks=4, window_s=W)
    assert all(sp["val"] is not None for sp in s2)
    for sp in s2:
        assert is_pre[sp["val"]].sum() > 0 and (~is_pre[sp["val"]]).sum() > 0
    _check_no_event_split(df, s2); _check_margin(df, s2, 3 * W)
    # with one context block nothing is dropped
    s1 = generate_patient_cv_splits(df, n_splits=5, strategy="event", seed=0, context_blocks=1, window_s=W)
    assert all(sp["info"]["dropped_margin"] == 0 for sp in s1)
    assert sum(len(sp["eval"]) for sp in s1) == len(df), "with one context block every row is evaluated exactly once"
    try:
        generate_patient_cv_splits(df, n_splits=2, strategy="stratified")
        raise AssertionError("stratified must raise")
    except ValueError as e:
        assert "leak" in str(e)
    # frames without the event columns (old callers) still work
    bare = df.drop(columns=EVENT_COLUMNS)
    sb = generate_patient_cv_splits(bare, n_splits=2, strategy="event", seed=0)
    assert len(sb) == 2 and all(sp["eval"] for sp in sb)
    print("event strategy ok")


def test_profile(csv, csv2):
    for path, pid, strat in ((csv, "aaaaaaaa", "session"), (csv2, "bbbbbbbb", "event")):
        profs = profile_all_patients(path, window_s=W, cv_folds=5, min_preictal=5, min_sessions=2, min_background=20)
        p = {q["patient_id"]: q for q in profs}[pid]
        assert p["n_events"] == 6 and p["n_seizures"] == 6 and p["cv_strategy"] == strat and p["cv_viable"] and p["is_ideal_candidate"], p
        assert p["seizure_types"] == {"fnsz": 6} and p["sessions_with_seizure"] == (6 if strat == "session" else 2)
        tbl = patient_sweep.format_candidates_table(profs)
        assert strat in tbl and "CV" in tbl
    # too few seizures for the folds -> not viable, strategy none
    profs = profile_all_patients(csv2, window_s=W, cv_folds=7, min_preictal=5, min_sessions=2, min_background=20)
    p = profs[0]
    assert p["cv_strategy"] == "none" and not p["cv_viable"] and not p["is_ideal_candidate"]
    print("profile cv_strategy ok")


def test_record_assembly(csv):
    class _DS:
        df = labeled_frame(csv, "aaaaaaaa")

    class _A:
        window_s = W

    sp = {"train": [0, 1], "val": None, "eval": [2, 3], "strategy": "session", "info": {}}
    old = {"auroc": 0.7, "sweep": {"balanced_accuracy": 0.6, "sensitivity": 0.5, "specificity": 0.7},
           "sens_at_spec": {"spec80": 0.4, "spec90": 0.3}, "type_macro_f1_ictal": float("nan"), "best_epoch": 3, "n_pos": 10}
    rec = patient_sweep._fold_record(old, 1, sp, _DS(), _A())
    assert rec["auroc"] == 0.7 and rec["selection"] == "oracle" and rec["selected_epoch"] == 3 and rec["threshold_source"] == "eval_sweep"
    assert rec["sens_spec90"] == 0.3 and rec["auroc_oracle"] is None and rec["baseline_auroc"] is None and rec["background_hours"] > 0
    new = dict(old, auroc_oracle=0.75, auroc_last=0.65, selected_epoch=2, selection="inner", sens_spec90_innerthr=0.25,
               thresholds={"spec90": 0.8}, baseline_auroc=0.6, smooth={15: 0.72}, per_session={"median": 0.7, "n": 2, "sessions": [(0, 1, 2, 0.7)]},
               type_macro_f1_ictal=None, _arrays={"scores": np.zeros(3)},
               event={"n_events": 2, "n_hits": np.int64(1), "fa_per_h": 0.5, "_scores": np.zeros(2), "threshold_source": "inner_val"})
    rec2 = patient_sweep._fold_record(new, 2, sp, _DS(), _A())
    import json
    json.dumps(rec2)                                           # numpy arrays / private keys must not reach the record
    assert rec2["event"] == {"n_events": 2, "n_hits": 1, "fa_per_h": 0.5, "threshold_source": "inner_val"} and "_arrays" not in rec2
    assert rec2["sens_spec90"] == 0.25 and rec2["threshold_source"] == "inner_val" and rec2["auroc_smooth15"] == 0.72
    assert rec2["type_macro_f1"] is None and rec2["per_session_median_auroc"] == 0.7 and rec2["threshold_spec90"] == 0.8
    s = patient_sweep._cv_summary("p", "pretrained", 2, [rec, rec2, dict(rec2, auroc=float("nan"))])
    assert s["n_folds_valid"] == 2 and abs(s["mean_auroc"] - 0.7) < 1e-9 and s["mean_auroc_oracle"] == 0.75
    assert s["strategy_per_fold"] == ["session"] * 3 and s["mean_baseline_auroc"] == 0.6 and s["mean_sens_at_spec90"] > 0
    print("fold record / summary ok")


def main():
    root = tempfile.mkdtemp(prefix="eeggpt_protocol_")
    try:
        csv = synthetic_master(os.path.join(root, "m1.csv"), "aaaaaaaa", n_sessions=6, seizures_per_session=1)
        csv2 = synthetic_master(os.path.join(root, "m2.csv"), "bbbbbbbb", n_sessions=2, seizures_per_session=3, session_s=5400.0)
        test_events()
        test_load_master_columns(csv)
        test_session_strategy(csv)
        test_event_strategy(csv2)
        test_profile(csv, csv2)
        test_record_assembly(csv)
        print("PROTOCOL TESTS PASSED")
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
