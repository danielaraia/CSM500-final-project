from report_073_039 import _overlap_flag, _unresolvable


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
