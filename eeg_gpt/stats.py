"""Statistical significance and effect size of finished patient-specific CV runs.

    python -m eeg_gpt.stats --run ../runs/full_v3/finetune/ft_20261014_0312            # vs chance
    python -m eeg_gpt.stats --run ../runs/full_v3/finetune/ft_A --compare ../runs/x/finetune/ft_B   # A vs B
    python -m eeg_gpt.stats --run ../runs/full_v3 --metric sweep_ba --out stats.md
    python -m eeg_gpt.stats --run ../runs/full_v3/finetune/ft_A --metric baseline_delta   # model minus feature baseline

Reads every cv_summary.json under --run (one per patient, written by patient_sweep.run_patient_cv).

Against chance (no --compare), for each patient over its CV folds and for the cohort over the patient means:
  mean, 95% t-interval, bootstrap 95% CI, one-sample t-test and Wilcoxon signed-rank vs the chance level,
  Cohen's d = (mean - chance) / sd. Per-patient p-values are Holm-corrected across patients.
Paired comparison (--compare): folds matched on (patient, fold), then patient means matched on patient:
  paired t-test, Wilcoxon signed-rank, Cohen's dz = mean(diff) / sd(diff), rank-biserial r, bootstrap CI of the
  mean difference. A paired fold comparison is only meaningful when both runs used the same --seed / --cv-folds.

Caveats printed with the report: CV folds of one patient share training data, so fold-level tests are optimistic;
the cohort test over patients (n = number of patients) is the one to quote, and with 5 patients it has little power.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Optional

import numpy as np
from scipy import stats as st

# chance level per metric (None = descriptive only). baseline_delta = auroc - baseline_auroc of the same fold, so its
# chance level is 0 and a positive cohort effect means the model beats the hand-crafted-feature floor; auroc_oracle is the
# old eval-fold-selected number (optimistic); sens_spec90 uses the validation-split threshold when the run had one;
# event_sens / fa_per_h are seizure-level (compare event_sens with the fold's chance_sens, not with a fixed level).
CHANCE = {"auroc": 0.5, "auroc_oracle": 0.5, "auroc_last": 0.5, "auroc_smooth15": 0.5, "auroc_norm": 0.5, "baseline_auroc": 0.5,
          "baseline_delta": 0.0, "sweep_ba": 0.5, "spec80": 0.2, "spec90": 0.1, "sens_spec90": 0.1, "type_macro_f1": None,
          "event_sens": None, "fa_per_h": None, "train_auroc": 0.5}
METRICS = list(CHANCE)


def _fold_value(f: dict, metric: str) -> float:
    """One number of a fold record (NaN when the record does not have it)."""
    if metric in ("spec80", "spec90"):
        v = (f.get("sens_at_spec") or {}).get(metric)
    elif metric == "baseline_delta":
        a, b = f.get("auroc"), f.get("baseline_auroc")
        v = None if a is None or b is None else float(a) - float(b)
    elif metric == "sens_spec90":
        v = f.get("sens_spec90")
        if v is None:
            v = (f.get("sens_at_spec") or {}).get("spec90")
    elif metric == "event_sens":
        v = (f.get("event") or {}).get("sens_events")
    elif metric == "fa_per_h":
        v = (f.get("event") or {}).get("fa_per_h")
    else:
        v = f.get(metric)
    try:
        return float("nan") if v is None else float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_run(path: str, metric: str) -> dict[str, list[tuple[int, float]]]:
    """{patient: [(fold, value), ...]} from every cv_summary.json under `path` (NaN folds dropped)."""
    files = [path] if path.endswith(".json") else glob.glob(os.path.join(path, "**", "cv_summary.json"), recursive=True)
    if not files:
        raise SystemExit(f"no cv_summary.json under {path}")
    out: dict[str, list[tuple[int, float]]] = {}
    for fp in sorted(files):
        with open(fp) as fh:
            s = json.load(fh)
        vals = [(int(f["fold"]), _fold_value(f, metric)) for f in s.get("folds", [])]
        vals = [(k, v) for k, v in vals if not math.isnan(v)]
        if vals:
            out.setdefault(str(s.get("patient_id", os.path.basename(os.path.dirname(fp)))), []).extend(vals)
    return out


def bootstrap_ci(x, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def describe(x, chance: Optional[float]) -> dict:
    """Location, spread and tests of `x` against `chance` (a paired difference passes chance=0)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    r = {"n": n, "mean": float(x.mean()) if n else float("nan"), "sd": float(x.std(ddof=1)) if n > 1 else float("nan")}
    r["ci_t"] = tuple(float(v) for v in st.t.interval(0.95, n - 1, loc=r["mean"], scale=r["sd"] / math.sqrt(n))) if n > 1 and r["sd"] > 0 else (float("nan"),) * 2
    r["ci_boot"] = bootstrap_ci(x)
    r["p_t"] = r["p_wilcoxon"] = r["d"] = r["rank_biserial"] = float("nan")
    if chance is None or n < 2:
        return r
    diff = x - chance
    if r["sd"] > 0:
        r["p_t"] = float(st.ttest_1samp(x, chance).pvalue)
        r["d"] = float(diff.mean() / r["sd"])
    nz = diff[diff != 0]
    if len(nz):
        r["p_wilcoxon"] = float(st.wilcoxon(nz).pvalue)
        ranks = st.rankdata(np.abs(nz))
        r["rank_biserial"] = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return r


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[i]))
        adj[i] = running
    return adj


def magnitude(d: float) -> str:
    if math.isnan(d):
        return "n/a"
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def _f(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _p(v):
    return "n/a" if math.isnan(v) else ("<0.001" if v < 0.001 else f"{v:.3f}")


def _ci(c):
    return f"[{_f(c[0])}, {_f(c[1])}]"


def vs_chance(run: dict, metric: str, chance: Optional[float]) -> tuple[str, dict]:
    pats = sorted(run)
    per = {p: describe([v for _, v in run[p]], chance) for p in pats}
    adj = holm([per[p]["p_t"] for p in pats])
    lines = [f"## {metric} vs chance ({_f(chance, 2) if chance is not None else 'no chance level: descriptive only'})", "",
             "| patient | folds | mean ± sd | 95% t-CI | bootstrap CI | t-test p | Holm p | Wilcoxon p | Cohen's d | effect |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p, a in zip(pats, adj):
        r = per[p]; r["p_t_holm"] = a
        lines.append(f"| {p} | {r['n']} | {_f(r['mean'])} ± {_f(r['sd'])} | {_ci(r['ci_t'])} | {_ci(r['ci_boot'])} | {_p(r['p_t'])} | {_p(a)} | "
                     f"{_p(r['p_wilcoxon'])} | {_f(r['d'], 2)} | {magnitude(r['d'])} |")
    cohort = describe([per[p]["mean"] for p in pats], chance)
    lines += ["", f"**Cohort (patient means, n = {cohort['n']})**: {_f(cohort['mean'])} ± {_f(cohort['sd'])}, 95% t-CI {_ci(cohort['ci_t'])}, "
                  f"bootstrap {_ci(cohort['ci_boot'])} | t-test p = {_p(cohort['p_t'])}, Wilcoxon p = {_p(cohort['p_wilcoxon'])} | "
                  f"Cohen's d = {_f(cohort['d'], 2)} ({magnitude(cohort['d'])}), rank-biserial r = {_f(cohort['rank_biserial'], 2)}"]
    return "\n".join(lines), {"patients": per, "cohort": cohort}


def paired(a: dict, b: dict, metric: str, name_a: str, name_b: str) -> tuple[str, dict]:
    fa = {(p, k): v for p, vs in a.items() for k, v in vs}
    fb = {(p, k): v for p, vs in b.items() for k, v in vs}
    keys = sorted(set(fa) & set(fb))
    pats = sorted({p for p, _ in keys})
    if not keys:
        raise SystemExit("the two runs share no (patient, fold) pair")
    fold = describe([fa[k] - fb[k] for k in keys], 0.0)
    pm = describe([np.mean([fa[k] for k in keys if k[0] == p]) - np.mean([fb[k] for k in keys if k[0] == p]) for p in pats], 0.0)
    wins = sum(fa[k] > fb[k] for k in keys)
    lines = [f"## {metric}: {name_a} − {name_b} (paired)", "",
             "| level | n | mean diff ± sd | 95% t-CI | bootstrap CI | paired t p | Wilcoxon p | Cohen's dz | effect | rank-biserial r |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for lvl, r in (("folds", fold), ("patients", pm)):
        lines.append(f"| {lvl} | {r['n']} | {_f(r['mean'])} ± {_f(r['sd'])} | {_ci(r['ci_t'])} | {_ci(r['ci_boot'])} | {_p(r['p_t'])} | "
                     f"{_p(r['p_wilcoxon'])} | {_f(r['d'], 2)} | {magnitude(r['d'])} | {_f(r['rank_biserial'], 2)} |")
    lines += ["", f"{name_a} better on {wins}/{len(keys)} folds. Only in one run: "
                  f"{len(set(fa) ^ set(fb))} (patient, fold) pairs (left out)."]
    return "\n".join(lines), {"folds": fold, "patients": pm, "wins": wins, "pairs": len(keys)}


CAVEATS = ("Notes: folds of one patient share training data, so fold-level p-values are optimistic; quote the cohort / "
           "patient-level test. With few patients (n ≤ 5) a Wilcoxon test cannot reach p < 0.05 two-sided "
           "(minimum p = 0.0625 at n = 5) and CIs are wide. Cohen's d bands: 0.2 small, 0.5 medium, 0.8 large.")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run dir (searched for cv_summary.json) or one cv_summary.json")
    p.add_argument("--compare", default=None, help="second run for a paired comparison (run − compare)")
    p.add_argument("--metric", choices=METRICS, default="auroc")
    p.add_argument("--chance", type=float, default=None, help="override the chance level of --metric")
    p.add_argument("--out", default=None, help="also write the report (.md) and numbers (.json next to it)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    chance = args.chance if args.chance is not None else CHANCE[args.metric]
    run = load_run(args.run, args.metric)
    parts = [f"# Statistics: {args.run}", "", f"{len(run)} patients, {sum(len(v) for v in run.values())} folds with a {args.metric}.", ""]
    text, res = vs_chance(run, args.metric, chance)
    parts.append(text)
    result = {"metric": args.metric, "chance": chance, "vs_chance": res}
    if args.compare:
        other = load_run(args.compare, args.metric)
        text, res = paired(run, other, args.metric, os.path.basename(os.path.normpath(args.run)), os.path.basename(os.path.normpath(args.compare)))
        parts += ["", text]
        result["paired"] = res
    parts += ["", CAVEATS]
    report = "\n".join(parts) + "\n"
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
            json.dump(result, fh, indent=1, default=float)
    return result


if __name__ == "__main__":
    main()
