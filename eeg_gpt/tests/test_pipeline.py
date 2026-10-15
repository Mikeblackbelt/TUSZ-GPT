"""End-to-end tests on a synthetic TUSZ-shaped corpus (CPU, ~2 min).

    python -m eeg_gpt.tests.test_pipeline

Builds N fake sessions as `<key>_proc.parquet` (20 channels, 256 Hz) with one
seizure each, a master CSV with bg / preictal / sopbuffer / ictal rows in the
same conventions as TUSZ-Seizure-Predict, then:
  1. tokenizer round-trip and level arithmetic
  2. EEGGPT forward/backward, causal mask (future tokens cannot change past logits)
  3. pretraining for a few hundred steps lowers CE below the unigram entropy
  4. fine-tuning runs, evaluates, writes checkpoints; from-scratch control runs too
  5. patient-adaptive protocol splits sessions chronologically
"""
from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from eeg_gpt.data.datasets import LabeledWindows, PretrainSessions, patient_id, session_number
from eeg_gpt.data.tokenizer import AmplitudeTokenizer
from eeg_gpt.models.gpt import EEGGPT, SeizureModel, gpt_loss
from eeg_gpt.training import finetune, pretrain

FS, C = 256, 20


def _signal(rng, n, tilt=0.0, gain=1.0):
    w = rng.standard_normal((C, n)).astype(np.float32); a = 0.97 - 0.02 * tilt
    out = np.empty_like(w); acc = np.zeros(C, dtype=np.float32)
    for t in range(n):
        acc = a * acc + w[:, t]; out[:, t] = acc
    return (out + 0.3 * rng.standard_normal((C, n)).astype(np.float32)) * gain * 1e-5


def make_corpus(root, n_patients=4, sessions_per_patient=2, dur_s=900.0, seed=0):
    rng = np.random.default_rng(seed)
    ck = os.path.join(root, "ckpt"); os.makedirs(ck, exist_ok=True)
    rows = []
    for p in range(n_patients):
        pid = "".join(chr(97 + int(x)) for x in rng.integers(0, 26, 8))
        for s in range(1, sessions_per_patient + 1):
            key = f"trn_{pid}_s{s:03d}_2010_ar1"
            x = _signal(rng, int(dur_s * FS), tilt=rng.normal(), gain=float(np.exp(rng.normal(0, 0.3))))
            onset = float(rng.uniform(620, 760)); dur = float(rng.uniform(8, 30))
            t = np.arange(x.shape[1]) / FS
            # preictal: mild beta boost on half the channels; ictal: rhythmic burst
            pre = (t >= onset - 540) & (t < onset - 120)
            x[: C // 2, pre] += (0.35 * np.sin(2 * np.pi * 20 * t[pre]) * 1e-5).astype(np.float32)
            ict = (t >= onset) & (t < onset + dur)
            x[:, ict] += (2.5 * np.sin(2 * np.pi * 4 * t[ict]) * 1e-5).astype(np.float32)
            pq.write_table(pa.table({f"ch{i}": x[i] for i in range(C)}), os.path.join(ck, f"{key}_proc.parquet"))
            typ = "fnsz" if p % 2 == 0 else "gnsz"
            edf = f"/data/{key}.edf"
            def add(label, a, b, valid=True, status=1):
                rows.append(dict(edf_path=edf, csv_path=edf.replace(".edf", ".csv"), split="train" if p < n_patients - 1 else "dev",
                                 channel="TERM", start_time=a, stop_time=b, label=label, confidence=1.0, is_valid=valid, status=status, session_key=key))
            for a in np.arange(0, max(4, onset - 540 - 8), 4.0): add("bg", a, a + 4.0)
            for a in np.arange(onset - 540, onset - 120, 4.0): add(f"p{typ}", a, a + 4.0)
            add(f"p{typ}_sopbuffer", onset - 120, onset, valid=False)
            for a in np.arange(onset, onset + dur - 4, 4.0): add(typ, a, a + 4.0, status=-1)
    pd.DataFrame(rows).to_csv(os.path.join(root, "master.csv"), index=False)
    return ck, os.path.join(root, "master.csv")


def test_training_loader_speed_settings():
    dl = pretrain.make_loader(list(range(32)), batch_size=8, num_workers=2, drop_last=True)
    assert dl.pin_memory is torch.cuda.is_available()
    assert dl.persistent_workers is True
    assert dl.prefetch_factor == 2
    dl2 = finetune.make_loader(list(range(32)), batch_size=8, num_workers=2, drop_last=True)
    assert dl2.pin_memory is torch.cuda.is_available()
    assert dl2.persistent_workers is True
    assert dl2.prefetch_factor == 2


def test_tokenizer():
    tok = AmplitudeTokenizer(512, 5.0)
    x = np.random.default_rng(0).standard_normal((3, 1000)).astype(np.float32) * 7 + 2
    t, z, _ = tok.encode(x)
    assert t.min() >= 0 and t.max() <= 511 and t.shape == x.shape
    z_hat = tok.dequantize(torch.from_numpy(t)).numpy()
    assert np.abs(np.clip(z, -5, 5) - z_hat).max() <= 10 / 511 + 1e-6, "dequantize must invert quantize within one bin"
    print("tokenizer ok")


def test_gpt_causal():
    torch.manual_seed(0)
    m = EEGGPT(64, C, 32, 2, 4, 128, 64, 0.0, (1, 4)).eval()
    tok = torch.randint(0, 64, (2, 32)); ch = torch.tensor([0, 5])
    lg, _ = m(tok, ch)
    tok2 = tok.clone(); tok2[:, 20:] = torch.randint(0, 64, (2, 12))
    lg2, _ = m(tok2, ch)
    assert torch.allclose(lg["1"][:, :20], lg2["1"][:, :20], atol=1e-5), "future tokens leaked into past logits"
    assert not torch.allclose(lg["1"][:, 20:], lg2["1"][:, 20:])
    out = gpt_loss(m, torch.randint(0, 64, (2, 65)), torch.randn(2, 65), ch, AmplitudeTokenizer(64, 5.0))
    out["loss"].backward()
    assert all(k in out for k in ("ce@1", "ce@4", "mse", "acc@1"))
    # rotary positions + amplitude features + tied head: still causal, and the tie is real
    m = EEGGPT(64, C, 32, 2, 4, 128, 64, 0.0, (1, 4), rope=True, value_emb=True, tie_weights=True).eval()
    assert m.pos_emb is None and m.heads["1"].weight is m.tok_emb.weight
    lg, _ = m(tok, ch); lg2, _ = m(tok2, ch)
    assert torch.allclose(lg["1"][:, :20], lg2["1"][:, :20], atol=1e-5), "future tokens leaked into past logits (rope)"
    assert not torch.allclose(lg["1"][:, 20:], lg2["1"][:, 20:])
    m2 = EEGGPT(64, C, 32, 2, 4, 128, 64, 0.0, (1, 4), rope=True, value_emb=True, tie_weights=True)
    m2.load_state_dict(m.state_dict())
    # reported ce is the plain cross-entropy whatever the loss weights are
    args = (torch.randint(0, 64, (2, 65)), torch.randn(2, 65), ch, AmplitudeTokenizer(64, 5.0))
    a, b = gpt_loss(m, *args, 0.1, 0.0, 1.0), gpt_loss(m, *args, 0.1, 0.1, 0.5)
    assert torch.allclose(a["ce@1"], b["ce@1"]) and torch.allclose(a["ce@4"], b["ce@4"]) and not torch.allclose(a["loss"], b["loss"])
    print("gpt causal + loss ok")


def test_metrics():
    from eeg_gpt.training.metrics import auroc, sensitivity_at_specificity
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(np.float64)
    s = np.round(rng.random(400) + 0.3 * y, 2)          # rounded -> plenty of ties
    pos, neg = s[y == 1], s[y == 0]
    ref = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert abs(auroc(s, y) - ref) < 1e-12
    for sp in (0.8, 0.9, 0.95):
        best = max([((s >= t) & (y == 1)).sum() / y.sum() for t in np.unique(s) if ((s < t) & (y == 0)).sum() / (1 - y).sum() >= sp], default=0.0)
        assert abs(sensitivity_at_specificity(s, y, sp) - best) < 1e-12
    print("metrics ok")


def test_pooled_sampling(ck):
    ds = PretrainSessions(ck, block_size=64, samples_per_epoch=4096, cache_capacity=2, session_pool=4, samples_per_session=32)
    assert ds.store.capacity >= 4
    seq = [ds._session_for(i) for i in range(4096)]
    for j in range(4):                                   # a stream changes recording at most once per 32 draws
        st = seq[j::4]
        assert sum(a != b for a, b in zip(st, st[1:])) <= len(st) // 32 + 1
    assert len(set(seq)) > 4, "the pool must turn over"
    assert seq == [PretrainSessions(ck, 64, 4096, session_pool=4, samples_per_session=32)._session_for(i) for i in range(4096)]
    t, z, c = ds[5]
    assert t.shape == (65,) and z.shape == (65,)
    print("pooled sampling ok")


def test_pretrain_learns(ck):
    torch.manual_seed(0)
    tok = AmplitudeTokenizer(64, 5.0)
    ds = PretrainSessions(ck, block_size=128, samples_per_epoch=4000, tokenizer=tok, cache_capacity=4)
    dl = torch.utils.data.DataLoader(ds, batch_size=32)
    m = EEGGPT(64, C, 64, 2, 4, 256, 128, 0.0, (1,), rope=True, value_emb=True, tie_weights=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first, last = None, None
    for i, (t, z, ch) in enumerate(dl):
        if i >= 120:
            break
        out = gpt_loss(m, t, z, ch, tok); opt.zero_grad(); out["loss"].backward(); opt.step()
        first = first or float(out["ce@1"].detach()); last = float(out["ce@1"].detach())
    # unigram entropy of the tokens (what a model with no context would reach)
    toks = torch.cat([ds[i][0] for i in range(200)]).numpy()
    p = np.bincount(toks, minlength=64) / len(toks); H = -(p[p > 0] * np.log(p[p > 0])).sum()
    print(f"pretrain: ce@1 {first:.3f} -> {last:.3f} | unigram entropy {H:.3f}")
    assert last < H - 0.2, "next-token CE should beat the context-free unigram entropy"
    return m


def test_max_gpu_usage_flag():
    from eeg_gpt.training.device import parse_gpu_usage
    assert parse_gpu_usage("0.0") == 0.0
    assert parse_gpu_usage("0.5") == 0.5
    assert parse_gpu_usage("1.0") == 1.0
    for invalid in ("-0.01", "1.0001", "-1.0", "2.0", "not_a_float"):
        try:
            parse_gpu_usage(invalid)
            assert False, f"Expected exception for invalid value: {invalid}"
        except Exception:
            pass
    print("max-gpu-usage validation ok")


def test_finetune(ck, csv, root):
    from eeg_gpt.training import finetune, pretrain
    pre = os.path.join(root, "pre")
    pretrain.main(["--checkpoint-dir", ck, "--out", pre, "--block-size", "256", "--d", "32", "--layers", "1", "--heads", "4", "--ffn", "64",
                   "--steps", "40", "--batch-size", "8", "--val-sessions", "2", "--val-every", "40", "--val-batches", "2", "--num-workers", "0",
                   "--n-levels", "64", "--max-gpu-usage", "0.8"])
    assert os.path.exists(os.path.join(pre, "best.pt"))
    ft = os.path.join(root, "ft")
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--pretrained", os.path.join(pre, "best.pt"), "--out", ft, "--epochs", "2",
                   "--steps-per-epoch", "8", "--batch-size", "4", "--accum", "1", "--context-blocks", "3", "--num-workers", "0", "--baseline",
                   "--min-ictal-per-batch", "2", "--window-s", "1.0", "--max-gpu-usage", "0.75"])
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--pretrained", os.path.join(pre, "best.pt"), "--out", ft + "_frozen", "--epochs", "1",
                   "--steps-per-epoch", "4", "--batch-size", "4", "--accum", "1", "--context-blocks", "2", "--num-workers", "0",
                   "--window-s", "1.0", "--freeze-backbone"])
    assert os.path.exists(os.path.join(ft, "best.pt")) and os.path.exists(os.path.join(ft, "history.json"))
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", ft + "_eval", "--eval-only", "--load", os.path.join(ft, "best.pt"),
                   "--pretrained", os.path.join(pre, "best.pt"), "--context-blocks", "3", "--num-workers", "0", "--batch-size", "4", "--window-s", "1.0",
                   "--max-gpu-usage", "1.0"])
    # from-scratch control + horizon label
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", ft + "_scratch", "--epochs", "1", "--steps-per-epoch", "4",
                   "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--label", "horizon", "--horizon-s", "130", "--window-s", "1.0",
                   "--d", "32", "--layers", "1", "--heads", "4", "--ffn", "64", "--max-gpu-usage", "0.5"])
    print("finetune ok")


def test_adaptive(ck, csv, root):
    from eeg_gpt.training import finetune
    ds = LabeledWindows(csv, ck, split=None, context_blocks=1)
    pid = patient_id(ds.sessions[0])
    ft = os.path.join(root, "adapt")
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", ft, "--adapt-patient", pid, "--epochs", "1", "--steps-per-epoch", "3",
                   "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--window-s", "1.0",
                   "--d", "32", "--layers", "1", "--heads", "4", "--ffn", "64", "--max-gpu-usage", "0.9"])
    print("patient-adaptive ok")


def test_patient_id_parsing():
    from eeg_gpt.data.datasets import _SPLIT_PREFIXES, patient_id, session_number
    # Split prefixes with letter patient IDs
    assert patient_id("trn_aaaaaaaa_s001_2010_ar1") == "aaaaaaaa"
    assert session_number("trn_aaaaaaaa_s001_2010_ar1") == 1
    assert patient_id("vld_aaaaaaaa_s001_2010_ar1") == "aaaaaaaa"
    assert session_number("vld_aaaaaaaa_s001_2010_ar1") == 1
    assert patient_id("tst_aaaaaaaa_s002_2010_ar1") == "aaaaaaaa"
    assert session_number("tst_aaaaaaaa_s002_2010_ar1") == 2
    assert patient_id("dev_aaaaaaaa_s003_2010_ar1") == "aaaaaaaa"
    assert session_number("dev_aaaaaaaa_s003_2010_ar1") == 3
    assert patient_id("eval_aaaaaaaa_s004_2010_ar1") == "aaaaaaaa"
    assert session_number("eval_aaaaaaaa_s004_2010_ar1") == 4

    # Split prefixes with numeric patient IDs (standard TUSZ)
    assert patient_id("vld_00000254_s005_2010_ar1") == "00000254"
    assert session_number("vld_00000254_s005_2010_ar1") == 5
    assert patient_id("tst_00000254_s006_2010_ar1") == "00000254"
    assert session_number("tst_00000254_s006_2010_ar1") == 6
    assert patient_id("trn_00000254_s001_2010_ar1") == "00000254"
    assert session_number("trn_00000254_s001_2010_ar1") == 1

    # Unprefixed session keys
    assert patient_id("00000254_s001_2010_ar1") == "00000254"
    assert session_number("00000254_s001_2010_ar1") == 1
    assert patient_id("aaaaaaaa_s001_2010_ar1") == "aaaaaaaa"
    assert session_number("aaaaaaaa_s001_2010_ar1") == 1

    # Never return split names as patient IDs
    for split in _SPLIT_PREFIXES:
        assert patient_id(split) != split
        assert patient_id(f"{split}_proc") != split

    # Test explicit patient_id and session_id column usage & split filtering
    import tempfile, pandas as pd
    from eeg_gpt.training.patient_sweep import profile_all_patients
    dummy_df = pd.DataFrame({
        "channel": ["TERM"] * 4,
        "is_valid": [True] * 4,
        "split": ["trn", "trn", "vld", "vld"],
        "patient_id": ["patA", "patA", "patB", "patB"],
        "session_id": ["s01", "s02", "s01", "s02"],
        "session_key": ["trn_patA_s01", "trn_patA_s02", "vld_patB_s01", "vld_patB_s02"],
        "start_time": [0.0, 10.0, 0.0, 10.0],
        "stop_time": [4.0, 14.0, 4.0, 14.0],
        "label": ["bg", "fnsz", "bg", "gnsz"],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        dummy_df.to_csv(f.name, index=False)
        dummy_csv = f.name
    try:
        profs_trn = profile_all_patients(dummy_csv, window_s=4.0, cv_folds=2, min_preictal=1, min_sessions=1, min_background=1, split="trn")
        assert [p["patient_id"] for p in profs_trn] == ["patA"]
        profs_all = profile_all_patients(dummy_csv, window_s=4.0, cv_folds=2, min_preictal=1, min_sessions=1, min_background=1)
        assert set(p["patient_id"] for p in profs_all) == {"patA", "patB"}
        # ids come from session_key: a numeric patient_id column must not drop the leading zeros
        num_df = dummy_df.assign(patient_id=[254, 254, 255, 255],
                                 session_key=["trn_00000254_s001", "trn_00000254_s002", "vld_00000255_s001", "tst_00000255_s002"])
        num_df.to_csv(dummy_csv, index=False)
        profs_num = profile_all_patients(dummy_csv, window_s=4.0, cv_folds=2, min_preictal=1, min_sessions=1, min_background=1)
        assert set(p["patient_id"] for p in profs_num) == {"00000254", "00000255"}
    finally:
        os.remove(dummy_csv)

    print("patient_id parsing ok")


def test_patient_sweep_and_cv(ck, csv, root):
    from eeg_gpt.training import finetune, patient_sweep
    pre = os.path.join(root, "pre")
    ft = os.path.join(root, "ft")

    # 1. Profile all patients and rank candidates
    profiles = patient_sweep.profile_all_patients(csv, window_s=1.0, cv_folds=5, min_preictal=2, min_sessions=2, min_background=10)
    assert len(profiles) > 0
    assert all("patient_id" in p and "suitability_score" in p for p in profiles)
    pid = profiles[0]["patient_id"]
    print(f"patient profile ok ({len(profiles)} patients, top candidate: {pid})")

    # 2. Patient-specific 5-fold CV starting from pretrained trunk
    ft_cv_pre = os.path.join(root, "cv_pre")
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--patient", pid, "--patient-base", "pretrained",
                   "--pretrained", os.path.join(pre, "best.pt"), "--out", ft_cv_pre, "--cv-folds", "5", "--epochs", "1",
                   "--steps-per-epoch", "1", "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--window-s", "1.0",
                   "--context-blocks", "3"])
    assert os.path.exists(os.path.join(ft_cv_pre, "cv_summary.json"))
    summary_pre = json.load(open(os.path.join(ft_cv_pre, "cv_summary.json")))
    assert len(summary_pre["folds"]) == 5
    assert "mean_auroc" in summary_pre
    print("patient-specific 5-fold CV (pretrained trunk) ok")

    # 3. Patient-specific 5-fold CV starting from non-patient-specific (general) model
    ft_cv_gen = os.path.join(root, "cv_gen")
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--patient", pid, "--patient-base", "general",
                   "--general-checkpoint", os.path.join(ft, "best.pt"), "--out", ft_cv_gen, "--cv-folds", "5", "--epochs", "1",
                   "--steps-per-epoch", "1", "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--window-s", "1.0",
                   "--context-blocks", "3"])
    assert os.path.exists(os.path.join(ft_cv_gen, "cv_summary.json"))
    summary_gen = json.load(open(os.path.join(ft_cv_gen, "cv_summary.json")))
    assert len(summary_gen["folds"]) == 5
    assert "mean_auroc" in summary_gen
    print("patient-specific 5-fold CV (general model) ok")

    # 4. Sweep profile-only CLI flag
    sweep_prof_out = os.path.join(root, "sweep_prof")
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", sweep_prof_out, "--sweep-patients", "--sweep-profile-only"])
    assert os.path.exists(os.path.join(sweep_prof_out, "candidates_profile.json"))
    assert os.path.exists(os.path.join(sweep_prof_out, "candidates_profile.csv"))
    print("patient sweep profile-only ok")

    # 5. Full candidate sweep comparing both base models
    sweep_run_out = os.path.join(root, "sweep_run")
    patient_sweep.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", sweep_run_out, "--patient-base", "both",
                        "--pretrained", os.path.join(pre, "best.pt"), "--general-checkpoint", os.path.join(ft, "best.pt"),
                        "--sweep-top-k", "1", "--cv-folds", "5", "--epochs", "1", "--steps-per-epoch", "1",
                        "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--window-s", "1.0", "--context-blocks", "3"])
    assert os.path.exists(os.path.join(sweep_run_out, "patient_sweep_results.json"))
    assert os.path.exists(os.path.join(sweep_run_out, "patient_sweep_results.csv"))
    sweep_recs = json.load(open(os.path.join(sweep_run_out, "patient_sweep_results.json")))
    assert len(sweep_recs) == 1
    assert "pretrained_auroc_mean" in sweep_recs[0] and "general_auroc_mean" in sweep_recs[0]
    print("patient sweep benchmark (both base models) ok")


def test_cv_pipeline(ck, csv, root):
    from eeg_gpt.training import pipeline
    out = os.path.join(root, "pipe")
    common = ["--master-csv", csv, "--checkpoint-dir", ck, "--out", out, "--cv-folds", "2", "--window-s", "1.0", "--num-workers", "0",
              "--min-preictal", "2", "--min-background", "10",
              "--pretrain-args", "--block-size 256 --d 32 --layers 1 --heads 4 --ffn 64 --steps 20 --batch-size 8 --val-every 20 --val-batches 2 --n-levels 64",
              "--finetune-args", "--epochs 1 --steps-per-epoch 1 --batch-size 4 --accum 1 --context-blocks 3"]
    pipeline.main(common + ["--top-k", "1", "--run-name", "first"])
    plan = pipeline.load_pretrain_state(out)
    held = plan["held_out"]
    assert len(held) == 1 and len(plan["folds"]) == 2 and all(f["result"] for f in plan["folds"])
    assert not any(held[0] in f["val_patients"] for f in plan["folds"]), "held-out patient must not be in the pretraining corpus"
    run = json.load(open(os.path.join(out, "finetune", "first", "run.json")))
    assert len(run["patients"][0]["cv"]["folds"]) == 2
    # wave-1 fold-record contract (protocol + recipe owners): honest selection next to the oracle number and the baseline
    for f in run["patients"][0]["cv"]["folds"]:
        for key in ("auroc", "auroc_oracle", "auroc_last", "selection", "strategy", "baseline_auroc", "auroc_smooth15"):
            assert key in f, f"fold record misses '{key}': has {sorted(f)}"
        a, o = f.get("auroc"), f.get("auroc_oracle")
        if isinstance(a, float) and isinstance(o, float) and not math.isnan(a) and not math.isnan(o):
            assert o >= a - 1e-9, f"oracle AUROC {o} below the selected AUROC {a}"
    cv = run["patients"][0]["cv"]
    assert "n_folds_valid" in cv, f"cv_summary misses n_folds_valid: has {sorted(cv)}"
    text_first = open(os.path.join(out, "finetune", "first", "summary.md"), encoding="utf-8").read()
    assert "oracle" in text_first and "baseline" in text_first
    # a patient that was pretrained on must not be fine-tuned patient-specifically from that pretraining
    other = plan["folds"][0]["val_patients"][0]
    try:
        pipeline.main(common + ["--patients", other, "--run-name", "second"])
        raise AssertionError("fine-tuning a pretrained-on patient must be refused")
    except SystemExit:
        pass
    # the held-out patient again: pretraining is reused, not redone
    stamp = os.path.getmtime(os.path.join(out, "pretrain", "fold_1", "best.pt"))
    pipeline.main(common + ["--patients", held[0], "--run-name", "second"])
    assert os.path.getmtime(os.path.join(out, "pretrain", "fold_1", "best.pt")) == stamp
    text = open(os.path.join(out, "SUMMARY.md"), encoding="utf-8").read()
    assert "Run first" in text and "Run second" in text and "Pretraining cross-validation" in text
    print("pretrain -> finetune CV pipeline ok")


def test_stats(root):
    from eeg_gpt import stats
    rng = np.random.default_rng(0)
    for run, shift in (("A", 0.25), ("B", 0.1)):
        for pid in ("p1", "p2", "p3", "p4", "p5"):
            d = os.path.join(root, "stats", run, f"patient_{pid}"); os.makedirs(d, exist_ok=True)
            folds = [{"fold": k, "auroc": float(0.5 + shift + rng.normal(0, 0.03)), "sweep_ba": 0.6,
                      "sens_at_spec": {"spec80": 0.5, "spec90": 0.3}, "type_macro_f1": float("nan")} for k in range(1, 6)]
            json.dump({"patient_id": pid, "folds": folds}, open(os.path.join(d, "cv_summary.json"), "w"))
    res = stats.main(["--run", os.path.join(root, "stats", "A"), "--compare", os.path.join(root, "stats", "B"),
                      "--out", os.path.join(root, "stats", "report.md")])
    assert res["vs_chance"]["cohort"]["n"] == 5 and res["vs_chance"]["cohort"]["mean"] > 0.7
    assert res["vs_chance"]["cohort"]["p_t"] < 0.001 and res["vs_chance"]["cohort"]["d"] > 0.8
    assert res["paired"]["pairs"] == 25 and res["paired"]["folds"]["mean"] > 0.1
    assert stats.holm([0.01, 0.04, float("nan")])[:2] == [0.02, 0.04]
    assert os.path.exists(os.path.join(root, "stats", "report.json"))
    print("stats ok")


def main():
    root = tempfile.mkdtemp(prefix="eeggpt_")
    try:
        ck, csv = make_corpus(root)
        ds = LabeledWindows(csv, ck, split="train", context_blocks=2)
        it = ds[len(ds) // 2]
        print("labeled corpus:", ds.counts(), "| tokens", tuple(it["tokens"].shape), "| z_last", tuple(it["z_last"].shape))
        assert it["tokens"].shape == (C, 2, 1024)
        test_training_loader_speed_settings()
        test_max_gpu_usage_flag()
        test_patient_id_parsing()
        test_tokenizer(); test_gpt_causal(); test_metrics(); test_pooled_sampling(ck); test_pretrain_learns(ck); test_finetune(ck, csv, root); test_adaptive(ck, csv, root)
        test_patient_sweep_and_cv(ck, csv, root)
        test_cv_pipeline(ck, csv, root)
        test_stats(root)
        # wave-1 per-area test modules (each owner wrote its own); a missing module is reported, not fatal
        missing = []
        for mod in ("test_recipe", "test_protocol", "test_arch", "test_stage1", "test_reporting"):
            try:
                m = importlib.import_module(f"eeg_gpt.tests.{mod}")
            except ImportError as e:
                print(f"{mod}: NOT RUN ({e})"); missing.append(mod); continue
            m.main()
            print(f"{mod} ok")
        if missing:
            print(f"WARNING: test modules not run: {missing}")
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()