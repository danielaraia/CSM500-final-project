from report_more_people import _overlap_flag, _unresolvable, events_relpath


def test_zero_scorable_hours_are_unresolvable():
    assert _unresolvable(0.0) is True
    assert _unresolvable(None) is True
    assert _unresolvable(1.5) is False


def test_overlap_flag_stays_none_when_a_class_is_missing():
    row = {
        "n_sop": 3,
        "n_interictal": 0,
        "frac_sop_at_or_under_interictal_max": None,
    }
    assert _overlap_flag(row) is None


def test_events_relpath_uses_meta_and_packed_run():
    path = events_relpath("sub-015", "04")
    assert path.endswith(
        "sub-015/ses-01/eeg/sub-015_ses-01_task-szMonitoring_run-04_events.tsv"
    )
    assert path.startswith("data/seizeit2-meta/")
