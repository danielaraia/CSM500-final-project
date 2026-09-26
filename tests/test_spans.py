import pytest

from data.spans import gap_spans_from_record_onsets, second_hits_a_span, spans_overlap


def test_overlap_and_just_touching():
    assert spans_overlap(0, 10, 5, 15) is True
    # they only touch at 10, so not overlapping
    assert spans_overlap(0, 10, 10, 20) is False


def test_second_hits_impd_or_gap():
    # fake impd from 4.2 to 6.0
    spans = [(4.2, 6.0)]
    assert second_hits_a_span(4, spans) is True
    assert second_hits_a_span(5, spans) is True
    assert second_hits_a_span(6, spans) is False
    assert second_hits_a_span(3, spans) is False


def test_record_jump_is_a_gap():
    # three 1 sec records, the third starts at 4, so clock 2..4 is missing
    spans = gap_spans_from_record_onsets([0.0, 1.0, 4.0], 1.0)
    assert spans == [(2.0, 4.0)]
    assert gap_spans_from_record_onsets([0.0, 1.0, 2.0], 1.0) == []


def test_first_record_after_zero_is_a_gap():
    spans = gap_spans_from_record_onsets([2.0, 3.0], 1.0)
    assert spans == [(0.0, 2.0)]


def test_overlapping_records_fail():
    with pytest.raises(ValueError, match="overlap"):
        gap_spans_from_record_onsets([0.0, 0.5], 1.0)
