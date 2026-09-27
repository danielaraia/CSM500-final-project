"""Relaxed hours rescreen of the wide cohort. Writes p6-screening-cohort-relaxed.json."""

import contextlib
import json
import sys

from evaluation.alarms import HOURS_TO_RESOLVE_FP_MAX

import tools.screen_p6 as screen_p6
OUT_NAME = "p6-screening-cohort-relaxed.json"
RELAXED_MIN_TEST_INTERICTAL_HOURS = 3.0
# already tried once: packed at SOP 15, 10, 5 on the pod, qc_fail each time
RETRY_EXCLUDED = {"sub-043": "phase 1 pack qc_fail at SOP 15, 10, 5"}


@contextlib.contextmanager
def relaxed_gates():
    # screen_sop and choose_cohort read these screen_p6 module globals at call
    # time; patch them for the rescreen and restore them afterwards
    old_hours = screen_p6.MIN_TEST_INTERICTAL_HOURS
    old_excluded = screen_p6.EXCLUDED_PATIENTS
    screen_p6.MIN_TEST_INTERICTAL_HOURS = RELAXED_MIN_TEST_INTERICTAL_HOURS
    screen_p6.EXCLUDED_PATIENTS = tuple(old_excluded) + tuple(RETRY_EXCLUDED)
    try:
        yield
    finally:
        screen_p6.MIN_TEST_INTERICTAL_HOURS = old_hours
        screen_p6.EXCLUDED_PATIENTS = old_excluded


def fa_resolution(test_interictal_hours):
    if test_interictal_hours >= HOURS_TO_RESOLVE_FP_MAX:
        return "full"
    return "limited"


def screen_all_runs(meta_root=screen_p6.META_ROOT):
    rows = []
    n_event_files = 0
    for path in screen_p6.event_paths(meta_root):
        match = screen_p6.RUN_RE.search(path)
        if match is None:
            continue
        n_event_files += 1
        patient_id, run_id = match.group(1), match.group(2)
        row = screen_p6.screen_run(meta_root, patient_id, run_id, path)
        rows.append(row)
        if row.get("skip") is None:
            print(json.dumps({"screened": patient_id, "run": run_id}), flush=True)
    return rows, n_event_files


def main():
    with relaxed_gates():
        rows, n_event_files = screen_all_runs()
        cohort, all_candidates = screen_p6.choose_cohort(rows)
    for entry in cohort:
        entry["fa_resolution"] = fa_resolution(entry["test_interictal_hours"])
    summary = {
        "command": "python screen_p6_relaxed.py",
        "screen_sop_minutes": [s // 60 for s in screen_p6.SCREEN_SOP_SECONDS],
        "min_usable_seizures": screen_p6.MIN_USABLE_SEIZURES,
        "min_test_interictal_hours": RELAXED_MIN_TEST_INTERICTAL_HOURS,
        "strict_min_test_interictal_hours": HOURS_TO_RESOLVE_FP_MAX,
        "ram_gib_skip": screen_p6.RAM_GIB_SKIP,
        "cohort_cap": screen_p6.COHORT_CAP,
        "excluded_patients": list(screen_p6.EXCLUDED_PATIENTS) + list(RETRY_EXCLUDED),
        "retry_excluded": RETRY_EXCLUDED,
        "n_event_files": n_event_files,
        "n_runs_with_seizures": len([r for r in rows if r.get("skip") is None]),
        "n_patients_with_a_passing_run": len(all_candidates),
        "n_cohort": len(cohort),
        "cohort": cohort,
        "runs": rows,
    }
    with open(OUT_NAME, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    print(f"WROTE {OUT_NAME} n_cohort {len(cohort)} of {len(all_candidates)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
