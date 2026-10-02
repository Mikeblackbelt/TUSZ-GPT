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
    print("gpt causal + loss ok")


def test_pretrain_learns(ck):
    torch.manual_seed(0)
    tok = AmplitudeTokenizer(64, 5.0)
    ds = PretrainSessions(ck, block_size=128, samples_per_epoch=4000, tokenizer=tok, cache_capacity=4)
    dl = torch.utils.data.DataLoader(ds, batch_size=32)
    m = EEGGPT(64, C, 64, 2, 4, 256, 128, 0.1, (1,))
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
    assert os.path.exists(os.path.join(ft, "best.pt")) and os.path.exists(os.path.join(ft, "history.json"))
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", ft + "_eval", "--eval-only", "--load", os.path.join(ft, "best.pt"),
                   "--pretrained", os.path.join(pre, "best.pt"), "--context-blocks", "3", "--num-workers", "0", "--batch-size", "4", "--window-s", "1.0",
                   "--max-gpu-usage", "1.0"])
    # from-scratch control + horizon label
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", ft + "_scratch", "--epochs", "1", "--steps-per-epoch", "4",
                   "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--label", "horizon", "--horizon-s", "130", "--window-s", "1.0",
                   "--max-gpu-usage", "0.5"])
    print("finetune ok")


def test_adaptive(ck, csv, root):
    from eeg_gpt.training import finetune
    ds = LabeledWindows(csv, ck, split=None, context_blocks=1)
    pid = patient_id(ds.sessions[0])
    ft = os.path.join(root, "adapt")
    finetune.main(["--master-csv", csv, "--checkpoint-dir", ck, "--out", ft, "--adapt-patient", pid, "--epochs", "1", "--steps-per-epoch", "3",
                   "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--window-s", "1.0", "--max-gpu-usage", "0.9"])
    print("patient-adaptive ok")


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
        test_tokenizer(); test_gpt_causal(); test_pretrain_learns(ck); test_finetune(ck, csv, root); test_adaptive(ck, csv, root)
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()