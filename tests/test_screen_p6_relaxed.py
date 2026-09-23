import screen_p6
from alarms import HOURS_TO_RESOLVE_FP_MAX
from screen_p6 import hop1_ends, screen_sop

from screen_p6_relaxed import (
    RELAXED_MIN_TEST_INTERICTAL_HOURS,
    RETRY_EXCLUDED,
    fa_resolution,
    relaxed_gates,
)


def test_fa_resolution_labels():
    assert fa_resolution(HOURS_TO_RESOLVE_FP_MAX) == "full"
    assert fa_resolution(HOURS_TO_RESOLVE_FP_MAX + 1.0) == "full"
    assert fa_resolution(HOURS_TO_RESOLVE_FP_MAX - 0.01) == "limited"
    assert fa_resolution(RELAXED_MIN_TEST_INTERICTAL_HOURS) == "limited"


def test_relaxed_gates_patch_and_restore():
    old_hours = screen_p6.MIN_TEST_INTERICTAL_HOURS
    old_excluded = screen_p6.EXCLUDED_PATIENTS
    with relaxed_gates():
        assert screen_p6.MIN_TEST_INTERICTAL_HOURS == RELAXED_MIN_TEST_INTERICTAL_HOURS
        assert RELAXED_MIN_TEST_INTERICTAL_HOURS < HOURS_TO_RESOLVE_FP_MAX
        for patient_id in old_excluded:
            assert patient_id in screen_p6.EXCLUDED_PATIENTS
        for patient_id in RETRY_EXCLUDED:
            assert patient_id in screen_p6.EXCLUDED_PATIENTS
    assert screen_p6.MIN_TEST_INTERICTAL_HOURS == old_hours
    assert screen_p6.EXCLUDED_PATIENTS == old_excluded


def _passing_seizures():
    return [
        (5_000.0, 5_060.0),
        (20_000.0, 20_060.0),
        (35_000.0, 35_060.0),
        (40_000.0, 40_060.0),
    ]


def test_relaxed_gate_admits_limited_hours_run():
    # 4.225 clean test interictal hours: under the strict 6.67 gate, over the
    # relaxed 3 hour gate, so this run is admitted with limited FA resolution
    seizures = _passing_seizures()
    duration = 55_000
    ends = hop1_ends([True] * duration)
    strict_row = screen_sop(seizures, duration, ends, 15 * 60)
    assert strict_row["clock_pass"] is False
    assert "test interictal hours" in strict_row["skip"]
    assert RELAXED_MIN_TEST_INTERICTAL_HOURS <= strict_row["test_interictal_hours"]
    with relaxed_gates():
        row = screen_sop(seizures, duration, ends, 15 * 60)
    assert row["skip"] is None, row
    assert row["clock_pass"] is True
    assert row["test_interictal_hours"] == strict_row["test_interictal_hours"]
    assert fa_resolution(row["test_interictal_hours"]) == "limited"


def test_relaxed_gate_still_enforces_the_other_gates():
    # under 2 usable seizures is not negotiable, relaxed hours or not
    seizures = [(5_000.0, 5_060.0)]
    duration = 70_000
    ends = hop1_ends([True] * duration)
    with relaxed_gates():
        row = screen_sop(seizures, duration, ends, 15 * 60)
    assert row["clock_pass"] is False
    assert row["skip"] == "under 2 usable, we got 1"
    # and under 3 clean hours still fails the relaxed hours gate
    seizures = _passing_seizures()
    duration = 50_000  # 2.836 clean test interictal hours
    ends = hop1_ends([True] * duration)
    with relaxed_gates():
        row = screen_sop(seizures, duration, ends, 15 * 60)
    assert row["clock_pass"] is False
    assert "test interictal hours" in row["skip"]


def _run_row(patient_id, run_id, passes, n_usable, hours, ram):
    per_sop = {}
    for sop in ("15", "10", "5"):
        per_sop[sop] = {
            "clock_pass": sop in passes,
            "n_usable": n_usable,
            "test_interictal_hours": hours,
            "val_interictal_hours": hours,
            "n_train_seizures": 2,
            "n_test_seizures": 1,
            "n_test_leads_evaluable": 1,
        }
    return {
        "patient_id": patient_id,
        "run_id": run_id,
        "skip": None,
        "ram_gib": ram,
        "needs_pod": ram >= 14.0,
        "duration_seconds": 50_000,
        "per_sop": per_sop,
    }


def test_choose_cohort_under_relaxed_gates_excludes_retried_sub_043():
    rows = [
        _run_row("sub-043", "08", {"15", "10", "5"}, 20, 12.0, 5.0),
        _run_row("sub-022", "04", {"15", "10", "5"}, 3, 5.5, 21.0),
    ]
    with relaxed_gates():
        cohort, all_candidates = screen_p6.choose_cohort(rows)
    ids = [c["patient_id"] for c in all_candidates]
    assert "sub-043" not in ids
    assert ids == ["sub-022"]
    assert cohort == all_candidates
