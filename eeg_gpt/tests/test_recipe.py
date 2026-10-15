"""CPU-tiny tests of the wave-1 fine-tuning recipe (training/recipe.py, the sampler, focal_bce, init_heads,
run_one_fold_training with LP-FT / EMA / inner-val selection, type head auto-off).

    python -m eeg_gpt.tests.test_recipe
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eeg_gpt.tests.test_pipeline import make_corpus


def test_sampler_composition():
    from eeg_gpt.training.finetune import BalancedBatchSampler
    cats = ["background"] * 40 + ["preictal"] * 6 + ["ictal"] * 4
    df = pd.DataFrame({"_cat": cats, "session_key": ["s1"] * 25 + ["s2"] * 25, "_type_idx": 0})
    s = BalancedBatchSampler(df, batch_size=8, min_ictal=1, seed=0, preictal_frac=0.25)
    assert s.kp == 2 and s.k == 1 and s.fill == 5, (s.kp, s.k, s.fill)
    batches = list(iter(s))
    assert len(batches) == len(s) == 50 // 5
    for b in batches:
        assert len(b) == 8
        c = df.loc[b, "_cat"].tolist()
        assert c.count("preictal") >= 2 and c.count("ictal") >= 1, c
    comp = s.composition()
    assert comp["reserved_preictal"] == 2 and comp["p_no_preictal"] == 0.0
    assert abs(comp["expected_preictal_per_batch"] - (2 + 5 * 6 / 50)) < 1e-9
    # today's behaviour: nothing reserved
    s0 = BalancedBatchSampler(df, batch_size=8, min_ictal=0, seed=0, preictal_frac=0.0)
    assert s0.kp == 0 and s0.k == 0 and s0.fill == 8 and 0 < s0.composition()["p_no_preictal"] < 1
    print("sampler composition ok")


def test_focal_bce():
    from eeg_gpt.models.gpt import focal_bce
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(64, generator=g); y = (torch.rand(64, generator=g) < 0.3).float()
    pw = torch.tensor([29.0])
    ref = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
    assert torch.allclose(focal_bce(logits, y, gamma=0.0, pos_weight=pw), ref, atol=1e-6)
    assert torch.allclose(focal_bce(logits, y, gamma=0.0), F.binary_cross_entropy_with_logits(logits, y), atol=1e-6)
    # a positive at logit 0: p_t = 0.5 from the UNWEIGHTED bce, so loss = 0.25 * 29 * ln 2
    val = focal_bce(torch.zeros(1), torch.ones(1), gamma=2.0, pos_weight=pw).item()
    assert abs(val - 0.25 * 29 * math.log(2)) < 1e-5, val
    # a negative at logit 0: same focal weight, no pos_weight
    val = focal_bce(torch.zeros(1), torch.zeros(1), gamma=2.0, pos_weight=pw).item()
    assert abs(val - 0.25 * math.log(2)) < 1e-5, val
    print("focal_bce ok")


def test_init_heads_and_groups():
    from eeg_gpt.models.gpt import EEGGPT, SeizureModel, init_heads
    from eeg_gpt.training import recipe
    bb = EEGGPT(64, 4, 32, 2, 4, 64, 64, 0.0, (1,))
    m = SeizureModel(bb, 4, 2, context_blocks=2, dropout=0.1)
    init_heads(m, 0.2, [30, 10])
    lin = [x for x in m.binary_head.modules() if isinstance(x, torch.nn.Linear)][-1]
    assert float(lin.weight.abs().sum()) == 0.0 and abs(float(lin.bias) - math.log(0.2 / 0.8)) < 1e-6
    tl = [x for x in m.type_head.modules() if isinstance(x, torch.nn.Linear)][-1]
    assert abs(float(tl.bias[0]) - math.log(0.75)) < 1e-6 and abs(float(tl.bias[1]) - math.log(0.25)) < 1e-6

    class A:  # minimal args
        head_lr, agg_lr, lr, llrd = 1e-3, 3e-4, 1e-5, 0.5
    groups = recipe.param_groups(m, A)
    names = {g["name"]: g["lr"] for g in groups}
    assert names["heads"] == 1e-3 and names["agg"] == 3e-4
    assert abs(names["trunk.block1"] - 1e-5) < 1e-12 and abs(names["trunk.block0"] - 0.5e-5) < 1e-12
    assert abs(names["trunk.emb"] - 0.25e-5) < 1e-12
    n_params = sum(p.numel() for g in groups for p in g["params"])
    assert n_params == sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    # frozen trunk => no trunk groups
    m.encoder.freeze()
    assert not any(g["name"].startswith("trunk") for g in recipe.param_groups(m, A))
    recipe.unfreeze_trunk(m)
    assert all(p.requires_grad for p in m.encoder.backbone.parameters())
    assert any(g["name"].startswith("trunk") for g in recipe.param_groups(m, A))
    print("init_heads / param groups ok")


def test_thresholds_and_stats():
    from eeg_gpt.training import recipe
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]); y = np.array([0, 0, 0, 0, 0, 1, 1, 1.0])
    thr = recipe.thresholds_from(s, y, spec=0.9)
    assert thr["spec90"] is not None and thr["spec90"] > 0.6 and thr["ba"] is not None
    at = recipe.apply_threshold(s, y, thr["spec90"])
    assert at["sens"] == 1.0 and at["spec"] == 1.0
    st = recipe.score_stats(s, y)
    assert set(st) >= {"std", "frac_pos", "mean_logit_pos", "mean_logit_neg", "separation"} and st["separation"] > 0
    assert recipe.smoothed([float("nan"), 0.5, 0.7], 3)[1] == 0.6
    assert recipe.thresholds_from(s, np.zeros_like(y))["spec90"] is None
    print("thresholds / score stats ok")


def _patient_split(csv, ck, pid, window_s, context_blocks):
    """train = first session, val = second, eval = third (session-disjoint), as row indices of the patient frame."""
    from eeg_gpt.data.datasets import LabeledWindows, session_number
    full = LabeledWindows(csv, ck, split=None, patient=pid, window_s=window_s, context_blocks=context_blocks)
    sess = sorted(full.df["session_key"].unique().tolist(), key=session_number)
    assert len(sess) >= 3
    idx = lambda keys: full.df.index[full.df["session_key"].isin(keys)].tolist()  # noqa: E731
    return full, idx(sess[:1]), idx(sess[1:2]), idx(sess[2:3])


def test_fold_training(ck, csv, root):
    from eeg_gpt.data.datasets import LabeledWindows, patient_id
    from eeg_gpt.training import finetune, patient_sweep
    from eeg_gpt.training.device import get_device
    ds = LabeledWindows(csv, ck, split=None, context_blocks=1)
    pid = patient_id(ds.sessions[0])
    args = finetune.build_parser().parse_args([
        "--master-csv", csv, "--checkpoint-dir", ck, "--out", os.path.join(root, "ft"), "--patient", pid,
        "--epochs", "2", "--freeze-epochs", "1", "--ema", "0.9", "--select", "inner", "--steps-per-epoch", "3",
        "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--window-s", "1.0", "--context-blocks", "2",
        "--d", "32", "--layers", "1", "--heads", "4", "--ffn", "64", "--preictal-frac", "0.25", "--baseline", "--no-amp"])
    device = get_device("cpu"); args.amp = False
    full, tr_i, va_i, ev_i = _patient_split(csv, ck, pid, args.window_s, args.context_blocks)
    kw = dict(window_s=args.window_s, context_blocks=args.context_blocks, type_grouping=args.type_grouping)
    tr = LabeledWindows(csv, ck, split=None, patient=pid, row_indices=tr_i, **kw)
    va = LabeledWindows(csv, ck, split=None, patient=pid, row_indices=va_i, **kw)
    ev = LabeledWindows(csv, ck, split=None, patient=pid, row_indices=ev_i, **kw)
    model, _ = patient_sweep.build_pretrained_seizure_model(None, device, tr.n_channels, len(tr.type_classes), args)
    out_dir = os.path.join(root, "fold1")
    res = patient_sweep.run_one_fold_training(1, tr, ev, model, args, device, out_dir, val_ds=va)
    need = {"auroc", "auroc_oracle", "auroc_last", "selected_epoch", "selection", "thresholds", "history", "train_auroc",
            "score_stats", "baseline_auroc", "at_threshold", "sens_spec90_innerthr", "sens_at_spec", "smooth", "per_session",
            "type_macro_f1_ictal", "best_epoch"}
    missing = need - set(res)
    assert not missing, missing
    assert res["selection"] == "inner" and res["selected_epoch"] in (1, 2) and len(res["history"]) == 2
    assert res["history"][0]["phase"] == "frozen" and res["history"][1]["phase"] == "full"
    assert "_arrays" not in res
    if not math.isnan(res["auroc"]) and not math.isnan(res["auroc_oracle"]):
        assert res["auroc_oracle"] >= res["auroc"] - 1e-12
    assert res["thresholds"]["spec90"] is None or 0.0 <= res["thresholds"]["spec90"] <= 1.0
    assert os.path.exists(os.path.join(out_dir, "best.pt"))
    ck_ = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu")
    assert ck_["epoch"] == res["selected_epoch"] and ck_["selection"] == "inner"
    import json
    json.dumps({k: v for k, v in res.items() if k != "per_session"}, default=float)   # fold record must serialise
    # without a val set --select inner falls back to 'none' (last epoch)
    res2 = patient_sweep.run_one_fold_training(2, tr, ev, model, args, device, os.path.join(root, "fold2"))
    assert res2["selection"] == "none" and res2["selected_epoch"] == args.epochs
    # 'oracle' reproduces the max-over-epochs selection
    args.select = "oracle"
    res3 = patient_sweep.run_one_fold_training(3, tr, ev, model, args, device, os.path.join(root, "fold3"), val_ds=va)
    assert res3["selection"] == "oracle" and abs(res3["auroc"] - res3["auroc_oracle"]) < 1e-12
    print("fold training (LP-FT, EMA, inner selection) ok")


def test_type_head_auto_off(root):
    from eeg_gpt.data.datasets import LabeledWindows, patient_id
    from eeg_gpt.training import finetune, patient_sweep, recipe
    from eeg_gpt.training.device import get_device
    one = os.path.join(root, "one"); os.makedirs(one, exist_ok=True)
    ck, csv = make_corpus(one, n_patients=1, sessions_per_patient=3, seed=1)    # a single seizure type
    ds = LabeledWindows(csv, ck, split=None, context_blocks=1)
    pid = patient_id(ds.sessions[0])
    assert not recipe.type_head_enabled("auto", ds.df) and recipe.type_head_enabled("on", ds.df) and not recipe.type_head_enabled("off", ds.df)
    args = finetune.build_parser().parse_args([
        "--master-csv", csv, "--checkpoint-dir", ck, "--out", os.path.join(root, "ft1"), "--patient", pid,
        "--epochs", "1", "--freeze-epochs", "0", "--ema", "0", "--select", "none", "--steps-per-epoch", "2",
        "--batch-size", "4", "--accum", "1", "--num-workers", "0", "--window-s", "1.0", "--context-blocks", "1",
        "--d", "32", "--layers", "1", "--heads", "4", "--ffn", "64", "--no-amp"])
    device = get_device("cpu"); args.amp = False
    full, tr_i, va_i, ev_i = _patient_split(csv, ck, pid, args.window_s, args.context_blocks)
    kw = dict(window_s=args.window_s, context_blocks=args.context_blocks, type_grouping=args.type_grouping)
    tr = LabeledWindows(csv, ck, split=None, patient=pid, row_indices=tr_i + va_i, **kw)
    ev = LabeledWindows(csv, ck, split=None, patient=pid, row_indices=ev_i, **kw)
    model, _ = patient_sweep.build_pretrained_seizure_model(None, device, tr.n_channels, len(tr.type_classes), args)
    res = patient_sweep.run_one_fold_training(1, tr, ev, model, args, device, os.path.join(root, "fold_one"))
    assert res["type_head"] is False and res["type_macro_f1_ictal"] is None and res["batch_composition"]["reserved_ictal"] == 0
    print("type head auto-off ok")


def main():
    root = tempfile.mkdtemp(prefix="eeggpt_recipe_")
    try:
        test_sampler_composition(); test_focal_bce(); test_init_heads_and_groups(); test_thresholds_and_stats()
        ck, csv = make_corpus(root, n_patients=2, sessions_per_patient=3)
        test_fold_training(ck, csv, root)
        test_type_head_auto_off(root)
        print("RECIPE TESTS PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
