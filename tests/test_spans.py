import pytest

from spans import second_hits_a_span, spans_overlap


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
