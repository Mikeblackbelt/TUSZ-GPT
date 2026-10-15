"""Stage 1 + Stage 2 in one go, with 5-fold cross-validation instead of the TUSZ train/dev/eval splits.

    python -m eeg_gpt.training.pipeline --master-csv ../master_full.csv --checkpoint-dir ../checkpoints --out runs/pipe \
        --pretrain-args "--steps 20000 --batch-size 64" --finetune-args "--epochs 10 --context-blocks 4"

1. Profile  : every patient of the master CSV is ranked for patient-specific modelling (sessions, preictal /
              background amount, number of seizures and the variety of seizure types). The table is printed and
              written to candidates.md; the user picks the patients (--patients, --top-k or the prompt).
2. Pretrain : the chosen patients are held out completely. The remaining patients are cut into --cv-folds
              patient-disjoint folds; fold k pretrains on the other folds and validates on fold k.
3. Finetune : each chosen patient gets its own --cv-folds cross-validation, started from the pretraining
              checkpoints (--pretrain-select paired: fine-tune fold k starts from pretraining fold k).
4. Summary  : SUMMARY.md (everything), pretrain/summary.md and finetune/<run>/summary.md.

Running it again on the same --out reuses whatever is finished: pretraining folds are skipped. Pretraining only
ever sees patients that are not fine-tuned patient-specifically, so a later run on the same --out may only name
patients that were held out when that pretraining was planned; anyone else needs a new --out.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import sys
import time

import numpy as np
import torch

from eeg_gpt.data.datasets import SessionStore, patient_id
from eeg_gpt.training import finetune, pretrain
from eeg_gpt.training.patient_sweep import format_candidates_table, profile_all_patients, run_patient_cv

log = logging.getLogger("pipeline")


def build_parser():
    p = argparse.ArgumentParser(description="pretrain -> patient-specific fine-tune, both under k-fold cross-validation")
    p.add_argument("--master-csv", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--stage", default="proc")
    p.add_argument("--out", required=True)
    p.add_argument("--only", choices=["all", "profile", "pretrain", "finetune"], default="all",
                   help="profile: rank patients and stop; pretrain: stop after Stage 1; finetune: reuse the finished Stage 1 folds")
    p.add_argument("--patients", default=None, help="comma list of patient ids (or ranks of the candidates table) for patient-specific training")
    p.add_argument("--top-k", type=int, default=0, help="take the k best ranked candidates instead of asking")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--pretrain-select", choices=["paired", "best"], default="paired",
                   help="Stage 1 checkpoint for a held-out patient: fine-tune fold k from pretraining fold k, or all folds from the best one")
    p.add_argument("--run-name", default=None, help="name of this fine-tuning run under <out>/finetune (default: timestamp)")
    p.add_argument("--min-preictal", type=int, default=5)
    p.add_argument("--min-sessions", type=int, default=2)
    p.add_argument("--min-background", type=int, default=20)
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--pretrain-args", default="", help="extra arguments for eeg_gpt.training.pretrain, as one quoted string (a single flag needs --pretrain-args=--compile)")
    p.add_argument("--finetune-args", default="", help="extra arguments for eeg_gpt.training.finetune, as one quoted string")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--split", default=None, help="filter master CSV to a specific split (e.g. 'train' or 'trn')")
    p.add_argument("--seed", type=int, default=0)
    return p


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _pm(mean, std, nd=3):
    return f"{_fmt(mean, nd)} ± {_fmt(std, nd)}"


def _common(args):
    extra = ["--seed", str(args.seed)]
    if args.device:
        extra += ["--device", args.device]
    if args.num_workers is not None:
        extra += ["--num-workers", str(args.num_workers)]
    return extra


# ----------------------------------------------------------------------------------------------- 1. profile

def candidates_markdown(profiles):
    lines = ["# Patient-specific candidates", "",
             "Ranked by the suitability score (preictal amount, sessions, background, seizure count and type variety, class balance).", "",
             "| rank | patient | ideal | score | sessions | with seizure | seizures | seizure types | preictal | background | ictal | hours | not ideal because |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(profiles, start=1):
        types = ", ".join(f"{t} ×{n}" for t, n in p["seizure_types"].items()) or "-"
        lines.append(f"| {i} | {p['patient_id']} | {'yes' if p['is_ideal_candidate'] else 'no'} | {p['suitability_score']:.2f} | {p['n_sessions']} | "
                     f"{p['sessions_with_seizure']} | {p['n_seizures']} | {types} | {p['n_preictal']} | {p['n_background']} | {p['n_ictal']} | "
                     f"{p['duration_hours']} | {'; '.join(p['reasons_not_ideal']) or '-'} |")
    return "\n".join(lines) + "\n"


def choose_patients(args, profiles):
    """--patients / --top-k, else ask on the terminal. Ranks refer to the printed table."""
    ids = [p["patient_id"] for p in profiles]
    ideal = [p["patient_id"] for p in profiles if p["is_ideal_candidate"]]
    if args.patients is None and args.top_k > 0:
        return (ideal or ids)[: args.top_k]
    raw = args.patients
    if raw is None:
        if not sys.stdin.isatty():
            raise SystemExit("no patients chosen: pass --patients id1,id2 (ids or table ranks) or --top-k N; see candidates.md")
        raw = input(f"Patients for patient-specific training (comma-separated ids or ranks; empty = the {len(ideal)} ideal candidates): ").strip()
        if not raw:
            if not ideal:
                raise SystemExit("no ideal candidates; name the patients explicitly")
            return ideal
    chosen = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        # an exact id wins over a rank: numeric TUSZ ids (00000254) are digits too
        pid = tok if tok in ids else ids[int(tok) - 1] if tok.isdigit() and 1 <= int(tok) <= len(ids) else tok
        if pid not in ids:
            raise SystemExit(f"unknown patient '{tok}' (not in the master CSV)")
        if pid not in chosen:
            chosen.append(pid)
    if not chosen:
        raise SystemExit("no patients chosen")
    return chosen


# ---------------------------------------------------------------------------------------------- 2. pretrain

def plan_pretrain_folds(sessions, held_out, n_folds, seed):
    """Patient-disjoint folds over every patient that is not held out for patient-specific training."""
    all_pats = {patient_id(s) for s in sessions}
    missing = sorted(set(held_out) - all_pats)
    if missing:   # holding out an id no recording has would leak that patient into pretraining unnoticed
        raise SystemExit(f"held-out patients {missing} match no recording in the checkpoint dir; "
                         f"ids there look like {sorted(all_pats)[:5]}")
    pats = sorted(all_pats - set(held_out))
    if len(pats) < n_folds:
        raise SystemExit(f"{len(pats)} patients left for pretraining, need >= {n_folds} for {n_folds}-fold CV")
    rng = np.random.default_rng(seed); rng.shuffle(pats)
    return [{"fold": k + 1, "val_patients": sorted(pats[k::n_folds])} for k in range(n_folds)]


def run_pretrain_cv(args, held_out):
    pre_dir = os.path.join(args.out, "pretrain"); os.makedirs(pre_dir, exist_ok=True)
    plan_path = os.path.join(pre_dir, "folds.json")
    sessions = SessionStore(args.checkpoint_dir, args.stage).list_sessions()
    if os.path.exists(plan_path):
        plan = _read_json(plan_path)
        log.info(f"reusing the pretraining folds of {plan_path} (held out: {plan['held_out']})")
    else:
        plan = {"held_out": sorted(held_out), "seed": args.seed, "folds": plan_pretrain_folds(sessions, held_out, args.cv_folds, args.seed)}
        _write_json(plan_path, plan)
    excluded = set(plan["held_out"])
    for f in plan["folds"]:
        fold_dir = os.path.join(pre_dir, f"fold_{f['fold']}")
        done = os.path.join(fold_dir, "fold.json")
        if os.path.exists(done) and os.path.exists(os.path.join(fold_dir, "best.pt")):
            log.info(f"pretraining fold {f['fold']}: already done, skipping")
            continue
        val_p = set(f["val_patients"])
        val_s = [s for s in sessions if patient_id(s) in val_p]
        train_s = [s for s in sessions if patient_id(s) not in val_p and patient_id(s) not in excluded]
        pa = pretrain.build_parser().parse_args(["--checkpoint-dir", args.checkpoint_dir, "--stage", args.stage, "--out", fold_dir,
                                                 *_common(args), *shlex.split(args.pretrain_args)])
        if os.path.exists(os.path.join(fold_dir, "last.pt")) and not pa.resume:
            pa.resume = os.path.join(fold_dir, "last.pt")
            log.info(f"pretraining fold {f['fold']}: resuming from {pa.resume}")
        log.info(f"=== pretraining fold {f['fold']}/{len(plan['folds'])}: {len(train_s)} train sessions, "
                 f"{len(val_s)} val sessions of {len(val_p)} held-out patients ===")
        pretrain.run_pretrain(pa, train_s, val_s)
        ck = torch.load(os.path.join(fold_dir, "best.pt"), map_location="cpu")
        _write_json(done, {"fold": f["fold"], "checkpoint": os.path.join(fold_dir, "best.pt"), "best_step": ck.get("step"), "val": ck.get("val", {}),
                           "train_sessions": len(train_s), "val_sessions": len(val_s),
                           "train_patients": len({patient_id(s) for s in train_s}), "val_patients": len(val_p)})
    return load_pretrain_state(args.out)


def load_pretrain_state(out):
    """The fold plan with the result of every finished fold merged in; None if pretraining never started."""
    pre_dir = os.path.join(out, "pretrain")
    plan_path = os.path.join(pre_dir, "folds.json")
    if not os.path.exists(plan_path):
        return None
    plan = _read_json(plan_path)
    for f in plan["folds"]:
        done = os.path.join(pre_dir, f"fold_{f['fold']}", "fold.json")
        f["result"] = _read_json(done) if os.path.exists(done) else None
        if f["result"]:   # valid from any working directory, and after the run directory was moved
            f["result"]["checkpoint"] = os.path.join(pre_dir, f"fold_{f['fold']}", "best.pt")
    return plan


def pretrain_markdown(plan):
    lines = ["# Pretraining cross-validation", "",
             f"Held out for patient-specific training (never pretrained on): {', '.join(plan['held_out']) or '-'}", "",
             "Floors: what copying the previous sample (copy) and a least-squares AR(8) predictor (ar8) score on the same held-out "
             "batches; a model near the floor has learned the local autocorrelation and little else. probe = AUROC of a linear "
             "probe on the frozen trunk's pooled features for preictal vs background windows of the held-out patients; band-power = "
             "the same probe on hand-crafted features of the same windows (the trunk is useful when probe > band-power).", "",
             "| fold | train patients | train sessions | val patients | val sessions | best step | objective (model) | floors copy / ar8 | ppl@1 | probe AUROC | band-power AUROC | val loss |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ces, probes, bps = [], [], []
    for f in plan["folds"]:
        r = f.get("result")
        if not r:
            lines.append(f"| {f['fold']} | | | {len(f['val_patients'])} | | not finished | | | | | | |")
            continue
        v = r.get("val", {}) or {}
        if v.get("mse_patch") is not None:
            model_s = f"mse_patch {_fmt(v.get('mse_patch'), 4)}"
            floors = f"{_fmt(v.get('mse_copy_patch'), 4)} / {_fmt(v.get('mse_ar8'), 4)}"
            ppl = "n/a"
        else:
            ce = v.get("ce@1", float("nan"))
            ce = float("nan") if ce is None else float(ce); ces.append(ce)
            model_s = f"ce@1 {_fmt(ce, 4)}"
            floors = f"{_fmt(v.get('ce_copy'), 4)} / {_fmt(v.get('ce_ar8'), 4)}"
            ppl = _fmt(math.exp(min(20.0, ce)), 1) if not math.isnan(ce) else "n/a"
        pr, bp = v.get("probe_auroc"), v.get("bandpower_auroc")
        if isinstance(pr, (int, float)) and not math.isnan(pr):
            probes.append(float(pr))
        if isinstance(bp, (int, float)) and not math.isnan(bp):
            bps.append(float(bp))
        lines.append(f"| {f['fold']} | {r.get('train_patients', '')} | {r.get('train_sessions', '')} | {r.get('val_patients', '')} | "
                     f"{r.get('val_sessions', '')} | {r.get('best_step', '')} | {model_s} | {floors} | {ppl} | {_fmt(pr)} | {_fmt(bp)} | "
                     f"{_fmt(v.get('loss'), 4)} |")
    ces = [c for c in ces if not math.isnan(c)]
    if ces:
        lines += ["", f"Held-out ce@1 over {len(ces)} folds: **{_pm(float(np.mean(ces)), float(np.std(ces)), 4)}**"]
    if probes:
        lines += ["", f"Probe AUROC over {len(probes)} folds: **{_pm(float(np.mean(probes)), float(np.std(probes)))}**"
                      + (f" vs band-power **{_pm(float(np.mean(bps)), float(np.std(bps)))}**" if bps else "")]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- 3. finetune

def checkpoints_for(pid, plan, select):
    """(checkpoints for the CV folds, note for the summary)."""
    done = [f for f in plan["folds"] if f.get("result")]
    if not done:
        raise SystemExit("no finished pretraining fold; run the pretraining stage first")
    if pid not in plan["held_out"]:
        raise SystemExit(f"patient {pid} is in the pretraining corpus of this --out; use a new --out to hold it out")
    if select == "best":
        best = min(done, key=lambda f: f["result"]["val"].get("ce@1", float("inf")))
        return [best["result"]["checkpoint"]], f"pretraining fold {best['fold']} (best held-out ce@1); patient never seen in pretraining"
    return [f["result"]["checkpoint"] for f in done], f"paired with pretraining folds {[f['fold'] for f in done]}; patient never seen in pretraining"


def run_finetune_cv(args, patients, plan, profiles):
    run_name = args.run_name or time.strftime("ft_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out, "finetune", run_name); os.makedirs(run_dir, exist_ok=True)
    prof = {p["patient_id"]: p for p in profiles}
    records = []
    for pid in patients:
        pdir = os.path.join(run_dir, f"patient_{pid}")
        ckpts, note = checkpoints_for(pid, plan, args.pretrain_select)
        rec = {"patient_id": pid, "init": note, "profile": {k: prof[pid][k] for k in
               ("n_sessions", "sessions_with_seizure", "n_seizures", "seizure_types", "n_preictal", "n_background", "n_ictal", "duration_hours")}}
        summary_path = os.path.join(pdir, "cv_summary.json")
        if os.path.exists(summary_path):
            log.info(f"patient {pid}: already done in run {run_name}, skipping")
            rec["cv"] = _read_json(summary_path)
        else:
            fa = finetune.build_parser().parse_args(["--master-csv", args.master_csv, "--checkpoint-dir", args.checkpoint_dir, "--stage", args.stage,
                                                     "--out", pdir, "--patient", pid, "--patient-base", "pretrained", "--window-s", str(args.window_s),
                                                     *_common(args), *shlex.split(args.finetune_args), "--cv-folds", str(args.cv_folds)])
            torch.manual_seed(fa.seed); np.random.seed(fa.seed)
            try:
                rec["cv"] = run_patient_cv(fa, pid, base_mode="pretrained", out_dir=pdir, pretrained_per_fold=ckpts)
            except ValueError as e:   # not enough rows / classes for the requested folds
                log.error(f"patient {pid}: skipped ({e})")
                rec["error"] = str(e)
        records.append(rec)
        _write_json(os.path.join(run_dir, "run.json"), {"run": run_name, "finetune_args": args.finetune_args, "cv_folds": args.cv_folds,
                                                        "pretrain_select": args.pretrain_select, "patients": records})
    _write_text(os.path.join(run_dir, "summary.md"), "# Fine-tuning run " + run_name + "\n\n" + finetune_markdown(_read_json(os.path.join(run_dir, "run.json")), True))
    return run_dir


def _num(v):
    """float or NaN for anything a fold record may hold (None, missing, nested)."""
    return float(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else float("nan")


def _cv_mean(cv, key, fold_key=None, sub=None):
    """`cv[key]` when the summary has it, else the mean over folds of `fold_key` (or `fold[sub][fold_key]`), finite values only."""
    v = _num(cv.get(key))
    if not math.isnan(v):
        return v
    vals = []
    for f in cv.get("folds", []):
        src = f.get(sub) if sub else f
        if isinstance(src, dict):
            vals.append(_num(src.get(fold_key or key)))
    vals = [x for x in vals if not math.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _fold_sens90(f):
    v = _num(f.get("sens_spec90"))
    return v if not math.isnan(v) else _num((f.get("sens_at_spec") or {}).get("spec90"))


def _strategy(cv):
    strat = cv.get("strategy_per_fold") or [f.get("strategy") for f in cv.get("folds", [])]
    strat = sorted({str(x) for x in strat if x})
    return ", ".join(strat) if strat else "n/a"


def _valid_folds(cv):
    n = cv.get("n_folds_valid")
    folds = cv.get("folds", [])
    if n is None:
        n = sum(1 for f in folds if not math.isnan(_num(f.get("auroc"))))
    return f"{n}/{len(folds)}" if folds else "n/a"


HOW_TO_READ = ("How to read: **AUROC (selected)** is the eval fold's window AUROC at the epoch chosen on the inner validation split "
               "(or the last epoch when there was none) - the honest number. **oracle** is the best epoch chosen on the eval fold "
               "itself (what the old pipeline reported; optimistic by the epoch-selection noise). **baseline** is a logistic "
               "regression on hand-crafted features (band powers + log amplitude) of the same windows: the floor the model must "
               "beat; `eeg_gpt.stats --metric baseline_delta` tests the difference. **smooth15** is the AUROC after a causal "
               "15-window moving average of the scores. **sens@spec90** is the sensitivity at 90 % specificity with the "
               "threshold taken from the validation split when there was one. Event sens / FA/h are seizure-level: the fraction "
               "of seizures with an alarm in their preictal span and false alarms per hour; compare sens with the chance level "
               "printed in the fold table. Chance for a window AUROC is 0.5; with ~200 preictal windows per patient, +-0.05 is noise.")


def finetune_markdown(run, per_fold=False):
    lines = [f"Stage 2 arguments: `{run.get('finetune_args') or '(defaults)'}` | {run.get('cv_folds', '?')}-fold CV per patient | "
             f"Stage 1 checkpoints: {run.get('pretrain_select', '?')}", "",
             "| patient | seizures | seizure types | preictal / background rows | AUROC (selected) | oracle | baseline | smooth15 | "
             "sens@spec90 | strategy | valid folds | initialised from |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    aurocs, bases, oracles = [], [], []
    for r in run.get("patients", []):
        pr = r.get("profile", {})
        head = (f"| {r.get('patient_id', '?')} | {pr.get('n_seizures', '')} | "
                f"{', '.join(f'{t} x{n}' for t, n in (pr.get('seizure_types') or {}).items()) or '-'} | "
                f"{pr.get('n_preictal', '')} / {pr.get('n_background', '')} | ")
        if "cv" not in r:
            lines.append(head + f"failed: {r.get('error', '?')} | | | | | | | {r.get('init', '')} |")
            continue
        cv = r["cv"]
        a, o, bl = _num(cv.get("mean_auroc")), _cv_mean(cv, "mean_auroc_oracle", "auroc_oracle"), _cv_mean(cv, "mean_baseline_auroc", "baseline_auroc")
        if not math.isnan(a):
            aurocs.append(a)
        if not math.isnan(bl):
            bases.append(bl)
        if not math.isnan(o):
            oracles.append(o)
        s90 = _cv_mean(cv, "mean_sens_spec90", "sens_spec90")
        if math.isnan(s90):
            s90 = _num(cv.get("mean_sens_at_spec90"))
        lines.append(head + f"{_pm(a, _num(cv.get('std_auroc')))} | {_fmt(o)} | {_fmt(bl)} | "
                            f"{_fmt(_cv_mean(cv, 'mean_auroc_smooth15', 'auroc_smooth15'))} | {_fmt(s90)} | {_strategy(cv)} | "
                            f"{_valid_folds(cv)} | {r.get('init', '')} |")
    if aurocs:
        line = f"Mean AUROC (selected) over {len(aurocs)} patients: **{_pm(float(np.mean(aurocs)), float(np.std(aurocs)))}** (+- = spread between patients)"
        if oracles:
            line += f" | oracle {_fmt(float(np.mean(oracles)))}"
        if bases:
            line += f" | baseline {_fmt(float(np.mean(bases)))} | model - baseline {_fmt(float(np.mean(aurocs)) - float(np.mean(bases)))}"
        lines += ["", line]
    if per_fold:
        for r in run.get("patients", []):
            if "cv" not in r:
                continue
            lines += ["", f"## Patient {r.get('patient_id', '?')}", "",
                      "| fold | train / val / eval rows | eval preictal | strategy | AUROC | oracle | last | baseline | smooth15 | balanced acc. | "
                      "sens@spec90 | events sens / FA/h (chance) | type macro-F1 | epoch | Stage 1 checkpoint |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for f in r["cv"].get("folds", []):
                ev = f.get("event") if isinstance(f.get("event"), dict) else {}
                ev_s = (f"{_fmt(_num(ev.get('sens_events')), 2)} / {_fmt(_num(ev.get('fa_per_h')), 2)} ({_fmt(_num(ev.get('chance_sens')), 2)})"
                        if ev else "n/a")
                lines.append(f"| {f.get('fold', '')} | {f.get('train_rows', '')} / {f.get('val_rows', '-')} / {f.get('eval_rows', '')} | "
                             f"{f.get('eval_pos', '')} | {f.get('strategy', 'n/a')} | {_fmt(_num(f.get('auroc')))} | {_fmt(_num(f.get('auroc_oracle')))} | "
                             f"{_fmt(_num(f.get('auroc_last')))} | {_fmt(_num(f.get('baseline_auroc')))} | {_fmt(_num(f.get('auroc_smooth15')))} | "
                             f"{_fmt(_num(f.get('sweep_ba')))} | {_fmt(_fold_sens90(f))} | {ev_s} | {_fmt(_num(f.get('type_macro_f1')))} | "
                             f"{f.get('selected_epoch', f.get('best_epoch', ''))} | {f.get('pretrained', '')} |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- 4. summary

def write_summary(out):
    """SUMMARY.md: the pretraining CV and every fine-tuning run found under <out>/finetune."""
    parts = ["# EEG-GPT pipeline summary", "", f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}. Candidate ranking: [candidates.md](candidates.md).", "",
             HOW_TO_READ, ""]
    plan = load_pretrain_state(out)
    if plan:
        md = pretrain_markdown(plan)
        _write_text(os.path.join(out, "pretrain", "summary.md"), md)
        parts += ["#" + md, ""]
    ft_root = os.path.join(out, "finetune")
    runs = sorted(d for d in os.listdir(ft_root) if os.path.exists(os.path.join(ft_root, d, "run.json"))) if os.path.isdir(ft_root) else []
    if runs:
        parts += ["## Patient-specific fine-tuning", ""]
    for name in runs:
        parts += [f"### Run {name}", "", f"Per-fold results: [finetune/{name}/summary.md](finetune/{name}/summary.md)", "",
                  finetune_markdown(_read_json(os.path.join(ft_root, name, "run.json"))), ""]
    path = os.path.join(out, "SUMMARY.md")
    _write_text(path, "\n".join(parts))
    return path


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "pipeline.log"))])
    profiles = profile_all_patients(args.master_csv, window_s=args.window_s, cv_folds=args.cv_folds, min_preictal=args.min_preictal,
                                    min_sessions=args.min_sessions, min_background=args.min_background, split=args.split)
    _write_json(os.path.join(args.out, "candidates.json"), profiles)
    _write_text(os.path.join(args.out, "candidates.md"), candidates_markdown(profiles))
    print("\n" + format_candidates_table(profiles) + "\n")
    if args.only == "profile":
        return
    patients = choose_patients(args, profiles)
    log.info(f"patient-specific patients: {patients}")
    plan = load_pretrain_state(args.out)
    if args.only == "finetune" and plan is None:
        raise SystemExit(f"--only finetune: no pretraining found under {args.out}")
    seen = [p for p in patients if plan is not None and p not in plan["held_out"]]
    if seen:   # their recordings are in the pretraining corpus: fine-tuning them would not be patient-disjoint
        raise SystemExit(f"{seen} were not held out when the pretraining in {args.out} was planned (held out: {plan['held_out']}); "
                         f"use a new --out to pretrain without them")
    if args.only != "finetune":
        plan = run_pretrain_cv(args, patients)
    if args.only != "pretrain":
        run_finetune_cv(args, patients, plan, profiles)
    log.info(f"summary written to {write_summary(args.out)}")


def _main_logged(argv=None):
    try:
        main(argv)
    except SystemExit:
        raise
    except BaseException:   # tracebacks go to stderr only; keep them in pipeline.log too
        log.exception("pipeline crashed")
        raise


if __name__ == "__main__":
    _main_logged()
