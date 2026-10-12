"""Look at the patients in a master CSV and the best candidates for patient-specific training.

    python -m eeg_gpt.patients --master-csv ../master_full.csv              # top 10 candidates (table)
    python -m eeg_gpt.patients --master-csv ../master_full.csv --all        # every patient
    python -m eeg_gpt.patients --master-csv ../master_full.csv --json       # the hashmap itself
    python -m eeg_gpt.patients --master-csv ../master_full.csv --patient aaaaaajy

From Python:

    from eeg_gpt.patients import patient_map, top_candidates
    patients = patient_map("../master_full.csv")     # {patient_id: profile}, best candidate first
    patients["aaaaaajy"]["seizure_types"]            # {"fnsz": 3, "gnsz": 1}
    top = top_candidates("../master_full.csv", k=5)  # the 5 best, same shape

Each profile is what the pipeline's ranking uses (patient_sweep.profile_all_patients):
rank, suitability_score, is_ideal_candidate, reasons_not_ideal, n_sessions, sessions,
n_preictal / n_background / n_ictal, n_seizures, seizure_types, duration_hours, ...
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Optional


def patient_map(master_csv: str, **criteria) -> dict[str, dict[str, Any]]:
    """{patient_id: profile} for every patient, ordered best candidate first (dicts keep insertion order).

    `criteria` are passed to profile_all_patients: window_s, cv_folds, min_preictal, min_sessions, min_background.
    """
    from eeg_gpt.training.patient_sweep import profile_all_patients  # pulls in torch: keep the import lazy

    profiles = profile_all_patients(master_csv, **criteria)
    return {p["patient_id"]: {"rank": i, **p} for i, p in enumerate(profiles, start=1)}


def top_candidates(master_csv: Optional[str] = None, k: int = 10, ideal_only: bool = False,
                   patients: Optional[dict] = None, **criteria) -> dict[str, dict[str, Any]]:
    """The `k` best candidates as {patient_id: profile}; k <= 0 means no cap.

    `ideal_only` drops patients that miss the criteria instead of letting them fill the list.
    Pass `patients` (a patient_map result) to avoid reading the CSV again.
    """
    if patients is None:
        patients = patient_map(master_csv, **criteria)
    out = {pid: p for pid, p in patients.items() if p["is_ideal_candidate"] or not ideal_only}
    return dict(list(out.items())[:k]) if k > 0 else out


def format_patients(patients: dict[str, dict[str, Any]]) -> str:
    """Table of a patient_map / top_candidates result; ranks are the ones from the full ranking."""
    head = (f"{'Rank':<5} {'Patient':<10} {'Score':<7} {'Ideal':<6} {'Sess':<5} {'Seiz':<5} {'Preictal':<9} "
            f"{'Backgr.':<8} {'Ictal':<6} {'Hours':<7} Seizure types / why not ideal")
    lines = [head, "-" * len(head)]
    for pid, p in patients.items():
        types = ", ".join(f"{t} x{n}" for t, n in p["seizure_types"].items()) or "-"
        why = "" if p["is_ideal_candidate"] else "  [" + "; ".join(p["reasons_not_ideal"]) + "]"
        lines.append(f"{p['rank']:<5} {pid:<10} {p['suitability_score']:<7.2f} {'yes' if p['is_ideal_candidate'] else 'no':<6} "
                     f"{p['n_sessions']:<5} {p['n_seizures']:<5} {p['n_preictal']:<9} {p['n_background']:<8} {p['n_ictal']:<6} "
                     f"{p['duration_hours']:<7} {types}{why}")
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master-csv", required=True)
    p.add_argument("--top-k", type=int, default=10, help="how many candidates to show (default 10)")
    p.add_argument("--all", action="store_true", help="show every patient instead of the top k")
    p.add_argument("--ideal-only", action="store_true", help="only patients that meet the criteria")
    p.add_argument("--patient", default=None, help="show one patient's full profile")
    p.add_argument("--json", action="store_true", help="print the {patient_id: profile} map as JSON instead of a table")
    p.add_argument("--out", default=None, help="also write the JSON map to this file")
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--min-preictal", type=int, default=5)
    p.add_argument("--min-sessions", type=int, default=2)
    p.add_argument("--min-background", type=int, default=20)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    patients = patient_map(args.master_csv, window_s=args.window_s, cv_folds=args.cv_folds, min_preictal=args.min_preictal,
                           min_sessions=args.min_sessions, min_background=args.min_background)
    if args.patient:
        if args.patient not in patients:
            raise SystemExit(f"patient {args.patient!r} not in {args.master_csv} ({len(patients)} patients)")
        shown = {args.patient: patients[args.patient]}
    else:
        shown = top_candidates(k=0 if args.all else args.top_k, ideal_only=args.ideal_only, patients=patients)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(shown, f, indent=2)
    if args.json or args.patient:
        print(json.dumps(shown, indent=2))
    else:
        print(format_patients(shown))
        n_ideal = sum(p["is_ideal_candidate"] for p in patients.values())
        print(f"\n{len(shown)} of {len(patients)} patients shown | {n_ideal} meet the ideal-candidate criteria")
    return shown


if __name__ == "__main__":
    main()
