from alarm_leads import (
    blank_lead_flags,
    hit_interval,
    lead_event_counts,
    mark_caught_leads,
    mark_evaluable_at_second,
)
from labels import SOP_SECONDS, SPH_SECONDS


def test_blank_lead_flags_start_false():
    caught, evaluable = blank_lead_flags(2)
    assert caught == [False, False]
    assert evaluable == [False, False]


def test_mark_evaluable_only_inside_hit_interval():
    onset = 10_000.0
    start, end = hit_interval(onset)
    evaluable = [False]
    mark_evaluable_at_second(evaluable, [onset], start, SPH_SECONDS, SOP_SECONDS)
    assert evaluable == [True]
    later = [False]
    mark_evaluable_at_second(later, [onset], end, SPH_SECONDS, SOP_SECONDS)
    assert later == [False]


def test_mark_caught_leads_only_inside_hit_interval():
    onset = 10_000.0
    start, end = hit_interval(onset)
    caught = [False]
    mark_caught_leads(caught, [onset], start, SPH_SECONDS, SOP_SECONDS)
    assert caught == [True]
    later = [False]
    mark_caught_leads(later, [onset], end, SPH_SECONDS, SOP_SECONDS)
    assert later == [False]


def test_lead_counts_split_unevaluable_from_miss():
    empty = lead_event_counts([False], [False])
    assert empty["n_seizures_unevaluable"] == 1
    assert empty["n_seizures_missed"] == 0
    assert empty["n_seizures_caught"] == 0
    assert empty["event_sensitivity"] is None
    missed = lead_event_counts([False], [True])
    assert missed["n_seizures_evaluable"] == 1
    assert missed["n_seizures_missed"] == 1
    assert missed["event_sensitivity"] == 0.0
    caught = lead_event_counts([True], [True])
    assert caught["n_seizures_caught"] == 1
    assert caught["n_seizures_missed"] == 0
    assert caught["event_sensitivity"] == 1.0
