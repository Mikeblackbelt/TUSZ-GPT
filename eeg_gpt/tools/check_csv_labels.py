"""What is actually in a master CSV: label categories, postictal rows, split prefixes, usable patients.

    python -m eeg_gpt.tools.check_csv_labels --master-csv ../master_full.csv

No torch needed. Answers the questions the pipeline cannot: does the preprocessing emit postictal
(q*/x*) rows (if not, postictal EEG is labelled background: see --postictal-exclude-s), which labels
are unrecognised, and how many patients have >= 5 sessions with a seizure (the ones that get the
session-based CV instead of the event-grouped fallback).
"""
from __future__ import annotations

import argparse
import collections
import re

import pandas as pd

from eeg_gpt.data.label_utils import LABEL_CATEGORY_ICTAL, classify_label

_SPLIT = {"trn", "train", "training", "dev", "devel", "development", "val", "vld", "valid", "validation",
          "eval", "evaluation", "tst", "test", "testing"}
_PAT = re.compile(r"^(?:(?:trn|train|training|dev|devel|development|val|vld|valid|validation|eval|evaluation|tst|test|testing)_)?([a-z0-9]+)_s(\d+)", re.I)


def _patient(key: str) -> str:
    m = _PAT.match(str(key))
    return m.group(1) if m else str(key).split("_")[0]


def _prefix(key: str) -> str:
    head = str(key).split("_")[0].lower()
    return head if head in _SPLIT else "(none)"


def check(master_csv: str, min_seizure_sessions: int = 5) -> dict:
    df = pd.read_csv(master_csv, dtype={"session_key": str, "label": str}, low_memory=False)
    n_all = len(df)
    if "channel" in df.columns:
        df = df[df["channel"].astype(str).str.upper() == "TERM"]
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    cats, unrecognised, postictal = collections.Counter(), collections.Counter(), collections.Counter()
    by_label = collections.Counter(df["label"].fillna("").astype(str))
    for lab, n in by_label.items():
        s = lab.strip().lower()
        try:
            cats[classify_label(lab)] += n
        except ValueError:
            unrecognised[lab] += n
        if s in ("q", "x", "postictal") or s.startswith("q") or s.startswith("x"):
            postictal[lab] += n
    prefixes = collections.Counter(_prefix(k) for k in df["session_key"])
    df = df.assign(_patient=df["session_key"].map(_patient))
    cat_of = {lab: (None if lab in unrecognised else classify_label(lab)) for lab in by_label}
    ictal = df[df["label"].fillna("").astype(str).map(cat_of) == LABEL_CATEGORY_ICTAL]
    sess_with_sz = ictal.groupby("_patient")["session_key"].nunique()
    usable = sorted(sess_with_sz[sess_with_sz >= min_seizure_sessions].index.tolist())
    return {"rows_total": n_all, "rows_labeled_term": len(df), "categories": dict(cats), "unrecognised": dict(unrecognised),
            "postictal_labels": dict(postictal), "split_prefixes": dict(prefixes), "patients": int(df["_patient"].nunique()),
            "patients_with_seizure_sessions_ge": {"min": min_seizure_sessions, "n": len(usable), "ids": usable},
            "seizure_sessions_per_patient": sess_with_sz.sort_values(ascending=False).head(30).to_dict()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master-csv", required=True)
    p.add_argument("--min-seizure-sessions", type=int, default=5)
    a = p.parse_args(argv)
    r = check(a.master_csv, a.min_seizure_sessions)
    print(f"rows: {r['rows_total']} total, {r['rows_labeled_term']} labelled TERM rows kept | patients {r['patients']}")
    print("label categories:", r["categories"])
    print("postictal-looking labels (q*/x*):", r["postictal_labels"] or "NONE -> postictal EEG is labelled background; consider --postictal-exclude-s")
    print("unrecognised labels (would raise in classify_label):", r["unrecognised"] or "none")
    print("rows per split prefix of session_key:", r["split_prefixes"])
    u = r["patients_with_seizure_sessions_ge"]
    print(f"patients with >= {u['min']} seizure sessions (get session-based CV): {u['n']} {u['ids']}")
    print("seizure sessions per patient (top 30):", r["seizure_sessions_per_patient"])
    return r


if __name__ == "__main__":
    main()
