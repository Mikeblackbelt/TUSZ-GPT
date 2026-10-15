"""CPU-tiny tests of the fine-tuning architecture: models/gpt.py (WindowEncoder / TemporalContext / SeizureModel),
data/features.py, the LabeledWindows batch keys `feats` / `log_sd`, the config-driven loaders and the probe fold.

    python -m eeg_gpt.tests.test_arch
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import numpy as np
import torch

from eeg_gpt.data.datasets import LabeledWindows
from eeg_gpt.data.features import N_FEATS_PER_CHANNEL, bandpower_indices, context_features, feat_dim, window_features
from eeg_gpt.models.gpt import DEFAULT_ARCH, OLD_ARCH, EEGGPT, SeizureModel
from eeg_gpt.tests.test_pipeline import C, FS, make_corpus

N_LEVELS, D, T = 64, 32, 256


def _trunk(d=D, layers=1, block=T, n_ch=C, **kw):
    return EEGGPT(N_LEVELS, n_ch, d, layers, 4, 4 * d, block, 0.0, (1,), **kw)


def _scratch_params(m):
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith("encoder.backbone"))


def test_param_count():
    bb = EEGGPT(512, C, 256, 1, 4, 1024, 1024, 0.0, (1,))
    m = SeizureModel(bb, C, 2, context_blocks=4, dropout=0.2, pool="tail", feat_branch="none")
    n = _scratch_params(m)
    m_full = SeizureModel(bb, C, 2, context_blocks=4, dropout=0.2, pool="tail", feat_branch="full", feat_dim=feat_dim(C))
    m_old = SeizureModel(bb, C, 2, context_blocks=4, dropout=0.2, pool="meanmax", feat_branch="none", arch=OLD_ARCH)
    print(f"from-scratch params at d=256: default arch {n / 1e6:.3f}M | + feature branch {_scratch_params(m_full) / 1e6:.3f}M "
          f"| old arch {_scratch_params(m_old) / 1e6:.3f}M")
    assert n < 700_000, n      # was ~3.4 M before wave 1; ~0.6 M now (a few hundred below 600 k, so leave headroom)
    assert _scratch_params(m_old) > 3_000_000


def test_forward_variants():
    torch.manual_seed(0)
    B, N = 2, 3
    bb = _trunk()
    tokens = torch.randint(0, N_LEVELS, (B, C, N, T))
    feats = torch.randn(B, feat_dim(C))

    def run(m, with_feats=False):
        m.train()
        out = m(tokens, feats=feats) if with_feats else m(tokens)
        assert out["binary_logit"].shape == (B,) and out["type_logits"].shape == (B, 2) and out["embedding"].shape == (B, D)
        out["binary_logit"].sum().backward()
        m.eval()
        with torch.no_grad():
            o2 = m(tokens, feats=feats) if with_feats else m(tokens)
        assert torch.isfinite(o2["binary_logit"]).all()

    for pool in ("mean", "last", "meanmax", "tail", "attn", "meanstd"):
        run(SeizureModel(bb, C, 2, N, 0.1, pool, arch=dict(agg_dim=16)))
    for agg in ("cls", "mean", "attn"):
        run(SeizureModel(bb, C, 2, N, 0.1, "tail", arch=dict(agg=agg, agg_dim=16, agg_layers=1, drop_path=0.2)))
    for ctx in ("transformer", "mean", "concat", "delta"):
        run(SeizureModel(bb, C, 2, N, 0.1, "tail", arch=dict(context_mode=ctx, agg_dim=16)))
    for fl in ("last", "mix", "1"):
        run(SeizureModel(bb, C, 2, N, 0.1, "tail", arch=dict(feature_layers=fl, agg_dim=16)))
    for enc in ("block", "whole"):
        # whole: the N blocks of a channel form ONE trunk sequence of N*T samples, so the trunk's block_size must hold them
        run(SeizureModel(_trunk(block=N * T) if enc == "whole" else bb, C, 2, N, 0.1, "tail", arch=dict(encode_context=enc, agg_dim=16)))
    run(SeizureModel(bb, C, 2, N, 0.1, "tail", arch=dict(encode_tail=128, agg_dim=16)))
    # N = 1 (no context) and the 3-D encoder input path
    run(SeizureModel(bb, C, 2, 1, 0.1, "tail", arch=dict(agg_dim=16)).__class__(bb, C, 2, 1, 0.1, "tail", arch=dict(agg_dim=16)))
    assert SeizureModel(bb, C, 2, 1, 0.1, "tail", arch=dict(agg_dim=16)).encoder(tokens[:, :, 0]).shape == (B, D)
    # feature branches
    bp = bandpower_indices(C)
    for fb in ("bandpower", "full", "feat_only"):
        m = SeizureModel(bb, C, 2, N, 0.1, "tail", feat_branch=fb, feat_dim=feat_dim(C), arch=dict(agg_dim=16), bandpower_idx=bp)
        m.set_feature_stats(np.zeros(feat_dim(C)), np.ones(feat_dim(C)))
        calls = []
        h = m.encoder.backbone.register_forward_pre_hook(lambda *a: calls.append(1))
        run(m, with_feats=True)
        h.remove()
        if fb == "feat_only":
            assert not calls, "feat_only must never run the trunk"
        else:
            assert calls
    # a model with a feature branch refuses to run without feats (loud failure instead of a silent zero branch)
    m = SeizureModel(bb, C, 2, N, 0.1, "tail", feat_branch="full", feat_dim=feat_dim(C), arch=dict(agg_dim=16))
    try:
        m(tokens); raise AssertionError("expected a ValueError without feats")
    except ValueError:
        pass
    # layer mix really mixes: the softmax weights are parameters that get a gradient
    m = SeizureModel(_trunk(layers=2), C, 2, N, 0.1, "tail", arch=dict(feature_layers="mix", agg_dim=16))
    m(tokens)["binary_logit"].sum().backward()
    assert m.encoder.layer_w.grad is not None and m.encoder.layer_w.numel() == 2
    print("forward variants ok")


def test_freeze_unfreeze():
    m = SeizureModel(_trunk(), C, 2, 2, 0.1, "tail", arch=dict(agg_dim=16))
    m.encoder.freeze(); m.train()
    assert not any(p.requires_grad for p in m.encoder.backbone.parameters()) and not m.encoder.backbone.training
    m.encoder.unfreeze()
    assert all(p.requires_grad for p in m.encoder.backbone.parameters()) and m.encoder.backbone.training
    print("freeze/unfreeze ok")


def test_old_checkpoint_loads(root):
    from eeg_gpt.training.patient_sweep import load_general_seizure_model
    old = SeizureModel(_trunk(), C, 2, context_blocks=4, dropout=0.2, pool="meanmax", feat_branch="none", arch=OLD_ARCH)
    path = os.path.join(root, "old_general.pt")
    # a config as the code before the architecture knobs wrote it: no feature_layers / agg / context_mode / feat_branch keys
    torch.save({"model": old.state_dict(), "epoch": 3, "config": {"context_blocks": 4, "pool": "meanmax", "dropout": 0.2}}, path)
    new, tok = load_general_seizure_model(path, torch.device("cpu"), C, 2, window_s=1.0, fs=FS, dropout=0.2, pool="meanmax", context_blocks=4)
    assert set(new.state_dict().keys()) == set(old.state_dict().keys()), \
        sorted(set(new.state_dict().keys()) ^ set(old.state_dict().keys()))
    for k, v in old.state_dict().items():
        assert torch.equal(new.state_dict()[k], v), k
    assert new.encoder.agg_mode == "cls" and new.encoder.layer_mode == "last" and new.context.mode == "transformer" and new.feat_branch == "none"
    print("old general checkpoint rebuilds the old module tree ok")


def test_features_and_dataset(ck, csv):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((C, 3, T)).astype(np.float32) * np.array([1.0, 3.0, 0.5], dtype=np.float32)[None, :, None]
    f = window_features(x, FS)
    assert f.shape == (3, C * N_FEATS_PER_CHANNEL) and np.isfinite(f).all()
    log_var = f[:, 0::N_FEATS_PER_CHANNEL]                                   # feature 0 of every channel
    assert log_var[1].mean() > log_var[0].mean() > log_var[2].mean(), "log variance must follow the raw amplitude"
    cf = context_features(x, FS)
    assert cf.shape == (feat_dim(C),) and np.allclose(cf[: C * N_FEATS_PER_CHANNEL], f[-1])
    assert np.allclose(context_features(x[:, :1], FS)[C * N_FEATS_PER_CHANNEL:], 0.0)
    ds = LabeledWindows(csv, ck, split=None, context_blocks=2, window_s=1.0, feat_branch="full")
    it = ds[len(ds) // 2]
    assert it["feats"].shape == (feat_dim(C),) and it["log_sd"].shape == (C, 2) and it["tokens"].shape == (C, 2, T)
    assert ds.feat_dim == feat_dim(C)
    ds0 = LabeledWindows(csv, ck, split=None, context_blocks=2, window_s=1.0)
    assert ds0[0]["feats"].numel() == 0 and ds0.feat_dim == 0
    vals = np.stack([ds[i]["feats"].numpy()[0::N_FEATS_PER_CHANNEL] for i in range(0, len(ds), max(1, len(ds) // 8))])
    assert vals.std(0).max() > 0, "log variance must differ between windows"
    print("features / dataset keys ok")


def test_probe_fold(ck, csv, root):
    from eeg_gpt.training.patient_sweep import run_probe_fold
    full = LabeledWindows(csv, ck, split=None, context_blocks=2, window_s=1.0, feat_branch="full")
    pid = full.df["_patient"].iloc[0]
    n = int((full.df["_patient"] == pid).sum())
    idx = np.arange(n)
    tr = LabeledWindows(csv, ck, split=None, patient=pid, context_blocks=2, window_s=1.0, row_indices=idx[: n // 2], feat_branch="full")
    ev = LabeledWindows(csv, ck, split=None, patient=pid, context_blocks=2, window_s=1.0, row_indices=idx[n // 2:], feat_branch="full")
    model = SeizureModel(_trunk(), C, 2, 2, 0.1, "tail", feat_branch="full", feat_dim=feat_dim(C), arch=dict(agg_dim=16))
    args = argparse.Namespace(eval_batch_size=64, batch_size=16, num_workers=0, amp=False, seed=0, smooth_k=15)
    out = os.path.join(root, "probe_fold")
    res = run_probe_fold(1, tr, ev, model, args, torch.device("cpu"), out)
    for k in ("auroc", "auroc_oracle", "auroc_last", "selected_epoch", "selection", "thresholds", "history", "train_auroc",
              "score_stats", "baseline_auroc", "probe_auroc", "probe_feat_auroc", "feat_only_auroc", "sens_at_spec", "smooth",
              "per_session", "type_macro_f1_ictal", "sens_spec90_innerthr"):
        assert k in res, k
    assert res["selection"] == "probe" and os.path.exists(os.path.join(out, "probe.json"))
    assert 0.0 <= res["auroc"] <= 1.0 and 0.0 <= res["baseline_auroc"] <= 1.0
    print(f"probe fold ok: auroc {res['auroc']:.3f} trunk {res['probe_auroc']:.3f} feats {res['feat_only_auroc']:.3f} baseline {res['baseline_auroc']:.3f}")


def main():
    root = tempfile.mkdtemp(prefix="eeggpt_arch_")
    try:
        ck, csv = make_corpus(root, n_patients=2, sessions_per_patient=2, dur_s=600.0)
        test_param_count()
        test_forward_variants()
        test_freeze_unfreeze()
        test_old_checkpoint_loads(root)
        test_features_and_dataset(ck, csv)
        test_probe_fold(ck, csv, root)
        print("ARCH TESTS PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
