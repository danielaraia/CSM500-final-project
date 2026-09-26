import json
import os
import random
import tempfile

from graphs.pearson import MOVEMENT_BUFFER_SECONDS
from data.spans import second_hits_a_span
from data.windows import WINDOW_SECONDS, window_end_times

from screen_p6 import (
    COHORT_CAP,
    SCREEN_SOP_SECONDS,
    choose_cohort,
    graph_ok_seconds,
    hop1_ends,
    screen_sop,
    sidecar_skip,
)


def test_graph_ok_matches_pack_drop_span_semantics():
    random.seed(7)
    duration = 400
    spans = [(50.5, 62.2), (100.0, 101.0), (250.7, 253.1), (398.2, 410.0)]
    graph_ok = graph_ok_seconds(duration, spans)
    assert len(graph_ok) == duration
    for second in range(duration):
        quality = not second_hits_a_span(second, spans)
        if second + 1 < MOVEMENT_BUFFER_SECONDS:
            assert graph_ok[second] is False
            continue
        buffer_ok = all(
            not second_hits_a_span(back, spans)
            for back in range(second - MOVEMENT_BUFFER_SECONDS + 1, second + 1)
        )
        assert graph_ok[second] == (quality and buffer_ok)


def test_hop1_ends_matches_window_end_times():
    random.seed(11)
    graph_ok = [random.random() > 0.05 for _ in range(500)]
    assert hop1_ends(graph_ok) == window_end_times(graph_ok, WINDOW_SECONDS, 1)


def _passing_seizures():
    return [
        (5_000.0, 5_060.0),
        (20_000.0, 20_060.0),
        (35_000.0, 35_060.0),
        (40_000.0, 40_060.0),
    ]


def test_screen_sop_passes_at_all_three_sops():
    seizures = _passing_seizures()
    duration = 70_000
    graph_ok = [True] * duration
    ends = hop1_ends(graph_ok)
    for sop_seconds in SCREEN_SOP_SECONDS:
        row = screen_sop(seizures, duration, ends, sop_seconds)
        assert row["skip"] is None, row
        assert row["clock_pass"] is True
        assert row["n_usable"] == 4
        assert row["n_train_seizures"] == 2
        assert row["n_test_seizures"] == 2
        assert row["n_positive_train"] > 0
        assert row["n_positive_monitor"] > 0
        assert row["n_negative_monitor"] > 0
        assert row["test_interictal_hours"] >= 6.67
        assert row["n_test_leads_evaluable"] == 2


def test_screen_sop_skips_under_two_usable():
    seizures = [(5_000.0, 5_060.0)]
    duration = 70_000
    ends = hop1_ends([True] * duration)
    row = screen_sop(seizures, duration, ends, 15 * 60)
    assert row["clock_pass"] is False
    assert row["skip"] == "under 2 usable, we got 1"
    assert row["n_usable"] == 1


def test_screen_sop_skips_when_test_hours_too_low():
    seizures = _passing_seizures()
    duration = 45_000  # test slice ends right after the last seizure
    ends = hop1_ends([True] * duration)
    row = screen_sop(seizures, duration, ends, 15 * 60)
    assert row["clock_pass"] is False
    assert "test interictal hours" in row["skip"]


def test_screen_sop_skips_when_gaps_break_evaluability():
    seizures = _passing_seizures()
    duration = 70_000
    graph_ok = [True] * duration
    # holes covering both test hit intervals kill the SOP 15 lookback (120 of 150)
    for second in range(33_000, 34_522):
        graph_ok[second] = False
    for second in range(38_700, 39_522):
        graph_ok[second] = False
    ends = hop1_ends(graph_ok)
    row = screen_sop(seizures, duration, ends, 15 * 60)
    assert row["clock_pass"] is False
    assert "not catch evaluable" in row["skip"]
    assert row["n_test_leads_evaluable"] == 0


def test_screen_sop_skips_when_monitor_misses_the_sop_class():
    # 3 usable seizures but the last train quarter lands on pure interictal,
    # the same honest failure sub-004 hit at SOP 15
    seizures = [(1_300.0, 1_360.0), (1_500.0, 1_560.0), (60_000.0, 60_060.0)]
    duration = 90_000
    ends = hop1_ends([True] * duration)
    row = screen_sop(seizures, duration, ends, 15 * 60)
    assert row["clock_pass"] is False
    assert row["skip"].startswith("monitor one class pos=0")


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


def test_choose_cohort_excludes_handled_patients_and_ranks():
    rows = [
        _run_row("sub-051", "01", {"15", "10", "5"}, 30, 12.0, 5.0),  # excluded
        _run_row("sub-200", "01", {"15", "10", "5"}, 6, 9.0, 5.0),
        _run_row("sub-201", "01", {"15", "10", "5"}, 9, 8.0, 5.0),
        _run_row("sub-202", "01", {"5"}, 12, 20.0, 5.0),
        _run_row("sub-203", "01", set(), 3, 1.0, 5.0),  # no pass anywhere
    ]
    cohort, all_candidates = choose_cohort(rows)
    ids = [c["patient_id"] for c in all_candidates]
    assert "sub-051" not in ids
    assert "sub-203" not in ids
    # seizure count first: sub-202 (12) before sub-201 (9) before sub-200 (6)
    assert ids == ["sub-202", "sub-201", "sub-200"]
    assert cohort == all_candidates[:COHORT_CAP]
    by_id = {c["patient_id"]: c for c in all_candidates}
    assert by_id["sub-202"]["longest_pass_sop_minutes"] == 5
    assert by_id["sub-201"]["longest_pass_sop_minutes"] == 15
    assert by_id["sub-201"]["passes"] == {"15": True, "10": True, "5": True}


def test_choose_cohort_picks_best_run_and_caps():
    rows = [
        _run_row("sub-300", "01", {"5"}, 4, 8.0, 5.0),
        _run_row("sub-300", "02", {"15", "10", "5"}, 5, 7.0, 6.0),
    ]
    for index in range(30):
        rows.append(_run_row(f"sub-4{index:02d}", "01", {"15"}, 3, 7.0, 5.0))
    cohort, all_candidates = choose_cohort(rows)
    assert len(all_candidates) == 31
    assert len(cohort) == COHORT_CAP
    top = [c for c in all_candidates if c["patient_id"] == "sub-300"][0]
    assert top["run_id"] == "02"  # the run passing at the longest sop wins
    assert top["n_usable_at_longest_sop"] == 5


def test_sidecar_skip_rules():
    with tempfile.TemporaryDirectory() as root:
        base = os.path.join(root, "sub-900", "ses-01")
        stem = "sub-900_ses-01_task-szMonitoring_run-01"
        assert sidecar_skip(root, "sub-900", "01") == "missing eeg json"
        for key in ("eeg", "ecg", "emg", "mov"):
            os.makedirs(os.path.join(base, key))
        for key, blob in (
            ("eeg", {"EEGChannelCount": 2}),
            ("ecg", {}),
            ("emg", {}),
            ("mov", {"MOVChannelCount": 12}),
        ):
            with open(os.path.join(base, key, f"{stem}_{key}.json"), "w") as handle:
                json.dump(blob, handle)
        assert sidecar_skip(root, "sub-900", "01") is None
        with open(os.path.join(base, "mov", f"{stem}_mov.json"), "w") as handle:
            json.dump({"MOVChannelCount": 6}, handle)
        assert sidecar_skip(root, "sub-900", "01") == "MOVChannelCount 6"
        with open(os.path.join(base, "mov", f"{stem}_mov.json"), "w") as handle:
            json.dump({"MOVChannelCount": 12}, handle)
        with open(os.path.join(base, "eeg", f"{stem}_eeg.json"), "w") as handle:
            json.dump({"EEGChannelCount": 1}, handle)
        assert sidecar_skip(root, "sub-900", "01") == "EEGChannelCount 1"
