"""Reporting / metrics tests (wave 1): score_stats, event metrics, causal normalisation, markdown on old records, stats.

    python -m eeg_gpt.tests.test_reporting
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile

import numpy as np


def test_score_stats():
    from eeg_gpt.training.metrics import score_stats
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3]); y = np.array([1, 1, 1, 0, 0, 0])
    r = score_stats(s, y)
    assert r["n_pos"] == 3 and r["n_neg"] == 3 and r["frac_pos"] == 0.5
    assert r["separation"] > 2 and r["mean_logit_pos"] > 0 > r["mean_logit_neg"]
    flat = score_stats(np.full(10, 0.55), np.array([1] * 3 + [0] * 7))
    assert flat["std"] < 1e-9 and math.isnan(flat["separation"])     # constant scores: no separation, not a rounding artefact
    print("score_stats ok")


def test_causal_normalise():
    from eeg_gpt.training.metrics import causal_session_normalise
    rng = np.random.default_rng(0)
    s = rng.random(40); sid = np.array([0] * 20 + [1] * 20); t = np.concatenate([np.arange(20), np.arange(20)])
    a = causal_session_normalise(sid, t, s, k=5)
    s2 = s.copy(); s2[15] += 10.0                      # a later score must not change earlier outputs
    b = causal_session_normalise(sid, t, s2, k=5)
    assert np.allclose(a[:15], b[:15]) and not np.allclose(a[16:20], b[16:20])
    assert a[0] == 0 and a[20] == 0                    # first window of each session has no past
    print("causal_session_normalise ok")


def test_event_metrics():
    from eeg_gpt.training.event_metrics import event_metrics
    w = 4.0
    times = np.arange(1, 601) * w                      # one session, 40 min of 4 s windows
    scores = np.zeros(600)
    onset = 1500.0                                     # preictal span [960, 1380]
    pre = (times >= onset - 540) & (times <= onset - 120)
    scores[pre] = 1.0                                  # perfect alarm inside the span
    scores[50:70] = 1.0                                # a false alarm block at 200-280 s
    r = event_metrics(scores, np.zeros(600, dtype=int), times, [(0, onset, onset + 20)], thr=0.5, window_s=w, fp_window=5)
    assert r["n_events"] == 1 and r["n_hits"] == 1 and r["sens_events"] == 1.0, r
    assert r["n_false_alarms"] == 1, r                 # the 80 s block is one alarm (refractory = SOP + SPH)
    assert r["fa_per_h"] > 0 and 0 <= r["chance_sens"] < 1
    miss = event_metrics(np.zeros(600), np.zeros(600, dtype=int), times, [(0, onset)], thr=0.5, window_s=w)
    assert miss["sens_events"] == 0.0 and miss["n_false_alarms"] == 0 and miss["fa_per_h"] == 0.0
    print("event_metrics ok")


def test_markdown_old_records():
    from eeg_gpt.training.pipeline import finetune_markdown, pretrain_markdown
    old_run = {"finetune_args": "", "cv_folds": 5, "pretrain_select": "paired", "patients": [
        {"patient_id": "aaaaaaaa", "init": "fold 1", "profile": {"n_seizures": 3, "seizure_types": {"fnsz": 3}, "n_preictal": 40, "n_background": 900},
         "cv": {"mean_auroc": 0.7, "std_auroc": 0.05, "mean_sweep_ba": 0.6, "std_sweep_ba": 0.02, "mean_sens_at_spec80": 0.5, "mean_sens_at_spec90": 0.3,
                "folds": [{"fold": 1, "train_rows": 700, "eval_rows": 200, "eval_pos": 8, "auroc": 0.7, "sweep_ba": 0.6, "sweep_sens": 0.5,
                           "sweep_spec": 0.7, "sens_at_spec": {"spec90": 0.3}, "type_macro_f1": None, "best_epoch": 3, "pretrained": "x.pt"}]}},
        {"patient_id": "bbbbbbbb", "init": "fold 2", "profile": {"n_seizures": 1, "seizure_types": {}, "n_preictal": 4, "n_background": 90}, "error": "too few rows"}]}
    md = finetune_markdown(old_run, per_fold=True)
    assert "n/a" in md and "0.700" in md and "failed: too few rows" in md and "## Patient aaaaaaaa" in md
    new_run = json.loads(json.dumps(old_run))
    f = new_run["patients"][0]["cv"]["folds"][0]
    f.update({"auroc_oracle": 0.75, "auroc_last": 0.68, "baseline_auroc": 0.65, "auroc_smooth15": 0.72, "strategy": "session", "selected_epoch": 2,
              "sens_spec90": 0.35, "val_rows": 100, "event": {"sens_events": 1.0, "fa_per_h": 0.5, "chance_sens": 0.1}})
    new_run["patients"][0]["cv"].update({"n_folds_valid": 1, "strategy_per_fold": ["session"], "mean_baseline_auroc": 0.65, "mean_auroc_oracle": 0.75})
    md2 = finetune_markdown(new_run, per_fold=True)
    assert "0.750" in md2 and "0.650" in md2 and "session" in md2 and "1.00 / 0.50" in md2 and "model - baseline" in md2
    plan = {"held_out": ["aaaaaaaa"], "folds": [
        {"fold": 1, "val_patients": ["p"], "result": {"train_patients": 3, "train_sessions": 6, "val_patients": 1, "val_sessions": 2, "best_step": 20,
                                                      "val": {"ce@1": 3.0, "loss": 4.0}}},
        {"fold": 2, "val_patients": ["q"], "result": {"train_patients": 3, "train_sessions": 6, "val_patients": 1, "val_sessions": 2, "best_step": 20,
                                                      "val": {"mse_patch": 0.3, "mse_copy_patch": 0.5, "mse_ar8": 0.4, "probe_auroc": 0.7, "bandpower_auroc": 0.6, "loss": 0.3}}},
        {"fold": 3, "val_patients": ["r"]}]}
    pm = pretrain_markdown(plan)
    assert "ce@1 3.0000" in pm and "mse_patch 0.3000" in pm and "0.700" in pm and "not finished" in pm and "Probe AUROC" in pm
    print("markdown on old/new records ok")


def test_stats_new_metrics():
    from eeg_gpt import stats
    root = tempfile.mkdtemp(prefix="eeggpt_rep_")
    try:
        rng = np.random.default_rng(1)
        for pid in ("p1", "p2", "p3", "p4"):
            d = os.path.join(root, f"patient_{pid}"); os.makedirs(d)
            folds = [{"fold": k, "auroc": float(0.72 + rng.normal(0, 0.02)), "auroc_oracle": 0.8, "baseline_auroc": float(0.6 + rng.normal(0, 0.02)),
                      "sens_at_spec": {"spec90": 0.3}, "sens_spec90": 0.25, "type_macro_f1": None,
                      "event": {"sens_events": 0.5, "fa_per_h": 0.4}} for k in range(1, 4)]
            json.dump({"patient_id": pid, "folds": folds}, open(os.path.join(d, "cv_summary.json"), "w"))
        r = stats.main(["--run", root, "--metric", "baseline_delta"])
        assert r["chance"] == 0.0 and abs(r["vs_chance"]["cohort"]["mean"] - 0.12) < 0.05
        r2 = stats.main(["--run", root, "--metric", "event_sens"])
        assert r2["vs_chance"]["cohort"]["n"] == 4 and abs(r2["vs_chance"]["cohort"]["mean"] - 0.5) < 1e-9
        r3 = stats.main(["--run", root, "--metric", "type_macro_f1"])
        assert r3["vs_chance"]["cohort"]["n"] == 0 or math.isnan(r3["vs_chance"]["cohort"]["mean"])
        r4 = stats.main(["--run", root, "--metric", "sens_spec90"])
        assert abs(r4["vs_chance"]["cohort"]["mean"] - 0.25) < 1e-9
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("stats new metrics ok")


def main():
    test_score_stats()
    test_causal_normalise()
    test_event_metrics()
    test_markdown_old_records()
    test_stats_new_metrics()


if __name__ == "__main__":
    main()
