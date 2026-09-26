from alarm_leads import hit_interval
from catch_coverage import catch_coverage_from_ends
from data.labels import SOP_SECONDS, SPH_SECONDS


def test_dense_sop_lookback_is_evaluable():
    onset = 10_000.0
    start, end = hit_interval(onset, SPH_SECONDS, SOP_SECONDS)
    ends = []
    t = int(start)
    while t < int(end):
        ends.append(t)
        t += 1
    out = catch_coverage_from_ends(ends, [onset], int(onset) + 10)
    assert out["catch_evaluable"] is True
    assert out["n_test_leads_evaluable"] == 1
    assert out["n_test_leads_unevaluable"] == 0
    assert out["max_lookback_scores_in_hit"] == 300
    assert out["alarm_vote_need"] == 240


def test_gappy_lookback_under_240_is_unevaluable():
    onset = 10_000.0
    start, _end = hit_interval(onset, SPH_SECONDS, SOP_SECONDS)
    ends = []
    t = int(start)
    while t < int(start) + 145:
        ends.append(t)
        t += 1
    out = catch_coverage_from_ends(ends, [onset], int(onset) + 10)
    assert out["catch_evaluable"] is False
    assert out["n_test_leads_evaluable"] == 0
    assert out["n_test_leads_unevaluable"] == 1
    assert out["max_lookback_scores_in_hit"] == 145


def test_exactly_240_scored_seconds_is_evaluable():
    onset = 10_000.0
    start, _end = hit_interval(onset, SPH_SECONDS, SOP_SECONDS)
    ends = []
    t = int(start)
    while t < int(start) + 240:
        ends.append(t)
        t += 1
    out = catch_coverage_from_ends(ends, [onset], int(onset) + 10)
    assert out["catch_evaluable"] is True
    assert out["n_test_leads_evaluable"] == 1
    assert out["max_lookback_scores_in_hit"] == 240
