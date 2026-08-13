import math
import os
import tempfile

import pytest

from alarms import (
    FP_MAX_PER_HOUR,
    alarm_hits_a_seizure,
    choose_operating_threshold,
    hit_interval,
    load_window_scores,
    lookback_seconds,
    merge_lead_onsets,
    pick_threshold,
    pick_window_threshold,
    raise_alarms,
    save_window_scores,
    score_at_threshold,
    score_map,
    silent_seconds,
    study_criterion_met,
    vote_need,
)
from labels import SOP_SECONDS, SPH_SECONDS


def test_sop30_vote_is_240_of_300():
    assert lookback_seconds() == 300
    assert vote_need(300) == 240
    assert silent_seconds() == SPH_SECONDS + SOP_SECONDS


def test_240_of_300_fires_and_239_does_not():
    high = {}
    t = 0
    while t < 240:
        high[t] = 1.0
        t += 1
    while t < 300:
        high[t] = 0.0
        t += 1
    alarms = raise_alarms(high, 0.5, 300, lookback=300, need=240, silent=2100)
    assert alarms == [239]

    almost = dict(high)
    almost[0] = 0.0
    assert raise_alarms(almost, 0.5, 300, lookback=300, need=240, silent=2100) == []


def test_unscored_second_counts_against_the_vote():
    scores = {}
    t = 0
    while t < 4:
        scores[t] = 1.0
        t += 1
    # lookback at t=3 is seconds -1 to 3: 4 scored above and one missing, so it fires
    assert raise_alarms(scores, 0.5, 8, lookback=5, need=4, silent=20) == [3]
    # drop one scored second, 3 of 5 is not enough
    del scores[2]
    assert raise_alarms(scores, 0.5, 8, lookback=5, need=4, silent=20) == []


def test_silent_period_blocks_a_second_alarm():
    scores = {}
    t = 0
    while t < 40:
        scores[t] = 1.0
        t += 1
    alarms = raise_alarms(scores, 0.5, 40, lookback=5, need=4, silent=10)
    assert alarms[0] == 3
    assert alarms[1] == 13
    assert 4 not in alarms


def test_hit_is_sop_clock_not_sph():
    onset = 10_000.0
    start, end = hit_interval(onset)
    assert start == onset - SPH_SECONDS - SOP_SECONDS
    assert end == onset - SPH_SECONDS
    assert alarm_hits_a_seizure(start, [onset])
    assert alarm_hits_a_seizure(end - 1, [onset])
    assert not alarm_hits_a_seizure(end, [onset])
    assert not alarm_hits_a_seizure(start - 1, [onset])


def test_close_onsets_merge_into_one_lead():
    leads, n_merged = merge_lead_onsets([100.0, 200.0, 5000.0], 2100)
    assert leads == [100.0, 5000.0]
    assert n_merged == 1


def test_false_alarm_rate_uses_scorable_interictal_hours():
    # one seizure far away so 0..7999 is interictal
    seizures = [(20_000.0, 20_100.0)]
    scores = {}
    t = 0
    while t < 300:
        scores[t] = 1.0
        t += 1
    scored = score_at_threshold(
        scores,
        0.5,
        seizures,
        lead_onsets=[],
        duration_seconds=8000,
        part_start=0,
        part_end=8000,
        lookback=5,
        need=4,
        silent=1000,
        sph_seconds=2,
        sop_seconds=8,
    )
    assert scored["n_false_alarms"] >= 1
    assert scored["raw_interictal_hours"] == pytest.approx(8000 / 3600.0)
    assert scored["scorable_interictal_hours"] > 0
    assert scored["false_alarms_per_hour"] == pytest.approx(
        scored["n_false_alarms"] / scored["scorable_interictal_hours"]
    )
    assert scored["false_alarms_per_day"] == pytest.approx(
        scored["false_alarms_per_hour"] * 24.0
    )


def test_gappy_lookback_is_not_scorable():
    seizures = [(20_000.0, 20_100.0)]
    scores = {100: 1.0, 101: 1.0}
    scored = score_at_threshold(
        scores,
        0.5,
        seizures,
        lead_onsets=[],
        duration_seconds=200,
        part_start=0,
        part_end=200,
        lookback=5,
        need=4,
        silent=20,
        sph_seconds=2,
        sop_seconds=8,
    )
    assert scored["n_false_alarms"] == 0
    assert scored["scorable_interictal_hours"] == 0
    assert scored["false_alarms_per_hour"] is None
    assert scored["raw_interictal_hours"] == pytest.approx(200 / 3600.0)


def test_caught_and_missed_seizures():
    onset = 10_000.0
    seizures = [(onset, onset + 40.0)]
    start, _end = hit_interval(onset)
    scores = {}
    t = int(start)
    last = t + 10
    while t <= last:
        scores[t] = 1.0
        t += 1
    caught = score_at_threshold(
        scores,
        0.5,
        seizures,
        lead_onsets=[onset],
        duration_seconds=12_000,
        part_start=0,
        part_end=12_000,
        lookback=5,
        need=4,
        silent=2100,
    )
    assert caught["n_seizures_caught"] == 1
    assert caught["n_seizures_missed"] == 0
    assert caught["event_sensitivity"] == pytest.approx(1.0)

    missed = score_at_threshold(
        {},
        0.5,
        seizures,
        lead_onsets=[onset],
        duration_seconds=12_000,
        part_start=0,
        part_end=12_000,
        lookback=5,
        need=4,
        silent=2100,
    )
    assert missed["n_seizures_caught"] == 0
    assert missed["n_seizures_missed"] == 1
    assert missed["event_sensitivity"] == pytest.approx(0.0)


def test_empty_val_hours_freeze_highest_threshold():
    seizures = [(100.0, 140.0)]
    scores = {10: 0.9, 11: 0.8}
    # val window sits inside sph, so no interictal seconds
    threshold, info = pick_threshold(
        scores,
        seizures,
        duration_seconds=200,
        part_start=0,
        part_end=50,
        lookback=5,
        need=4,
        silent=20,
        sph_seconds=2,
        sop_seconds=8,
    )
    assert threshold is math.inf
    assert info["val_cap_met"] is False
    assert "hours are 0" in info["threshold_rule"]
    assert info["threshold_set"] == "val"


def test_empty_val_uses_monitor_not_test():
    seizures = [(80_000.0, 80_040.0)]
    duration = 8 * 3600
    scores = {}
    t = 200
    while t < 700:
        scores[t] = 2.0
        t += 1
    while t < 1200:
        scores[t] = 0.2
        t += 1
    # test scores that would pick a different operating point if we cheated
    t = 20_000
    while t < 20_400:
        scores[t] = 9.0
        t += 1
    threshold, info = choose_operating_threshold(
        scores,
        seizures,
        duration_seconds=duration,
        val_start=0,
        val_end=50,
        fallback_start=200,
        fallback_end=4000,
        fallback_interictal_logits=None,
        fallback_stride_seconds=10,
        lookback=5,
        need=4,
        silent=2100,
        sph_seconds=SPH_SECONDS,
        sop_seconds=SOP_SECONDS,
        fp_max_per_hour=FP_MAX_PER_HOUR,
    )
    assert info["threshold_set"] == "monitor"
    assert threshold < math.inf
    assert threshold != 9.0
    assert info["val_scorable_interictal_hours"] > 0
    at_chosen = score_at_threshold(
        scores,
        threshold,
        seizures,
        [],
        duration,
        200,
        4000,
        lookback=5,
        need=4,
        silent=2100,
    )
    assert at_chosen["false_alarms_per_hour"] <= FP_MAX_PER_HOUR


def test_empty_val_without_monitor_fails():
    seizures = [(100.0, 140.0)]
    scores = {10: 0.9}
    with pytest.raises(ValueError, match="no monitor fallback"):
        choose_operating_threshold(
            scores,
            seizures,
            duration_seconds=200,
            val_start=0,
            val_end=50,
            lookback=5,
            need=4,
            silent=20,
            sph_seconds=2,
            sop_seconds=8,
        )


def test_empty_val_hop10_monitor_uses_window_rate():
    seizures = [(80_000.0, 80_040.0)]
    duration = 8 * 3600
    scores = {}
    # hop 10 monitor: too sparse for the 240 of 300 clock vote
    t = 100
    while t < 400:
        scores[t] = 1.5
        t += 10
    t = 400
    while t < 700:
        scores[t] = -0.2
        t += 10
    # test scores must not set the threshold
    scores[20_000] = 50.0
    interictal = []
    t = 100
    while t < 700:
        interictal.append(scores[t])
        t += 10
    threshold, info = choose_operating_threshold(
        scores,
        seizures,
        duration_seconds=duration,
        val_start=0,
        val_end=50,
        fallback_start=100,
        fallback_end=700,
        fallback_interictal_logits=interictal,
        fallback_stride_seconds=10,
        lookback=300,
        need=240,
        silent=2100,
        sph_seconds=SPH_SECONDS,
        sop_seconds=SOP_SECONDS,
        fp_max_per_hour=FP_MAX_PER_HOUR,
    )
    assert info["threshold_set"] == "monitor windows"
    assert threshold == pytest.approx(1.5)
    assert 50.0 not in (threshold,)
    n_false = 0
    index = 0
    while index < len(interictal):
        if interictal[index] > threshold:
            n_false += 1
        index += 1
    assert n_false == 0


def test_window_threshold_keeps_zero_false_when_hours_are_short():
    logits = [0.4, 0.1, -0.2, 0.4]
    threshold, info = pick_window_threshold(logits, stride_seconds=10)
    assert threshold == pytest.approx(0.4)
    assert info["threshold_set"] == "monitor windows"
    assert info["val_false_alarms_per_hour"] == pytest.approx(0.0)


def test_val_with_hours_ignores_monitor_and_test():
    seizures = [(80_000.0, 80_040.0)]
    duration = 8 * 3600
    scores = {}
    t = 0
    while t < 400:
        scores[t] = 2.0
        t += 1
    while t < 800:
        scores[t] = 1.0
        t += 1
    scores[20_000] = 99.0
    threshold, info = choose_operating_threshold(
        scores,
        seizures,
        duration_seconds=duration,
        val_start=0,
        val_end=duration,
        fallback_start=20_000,
        fallback_end=20_100,
        fallback_interictal_logits=[99.0],
        fallback_stride_seconds=10,
        lookback=5,
        need=4,
        silent=2100,
        sph_seconds=SPH_SECONDS,
        sop_seconds=SOP_SECONDS,
        fp_max_per_hour=FP_MAX_PER_HOUR,
    )
    assert info["threshold_set"] == "val"
    assert threshold < math.inf
    assert threshold != 99.0


def test_threshold_is_the_lowest_that_stays_under_the_cap():
    seizures = [(80_000.0, 80_040.0)]
    duration = 8 * 3600
    scores = {}
    t = 0
    while t < 400:
        scores[t] = 2.0
        t += 1
    while t < 800:
        scores[t] = 1.0
        t += 1
    threshold, info = pick_threshold(
        scores,
        seizures,
        duration_seconds=duration,
        part_start=0,
        part_end=duration,
        lookback=5,
        need=4,
        silent=2100,
        sph_seconds=SPH_SECONDS,
        sop_seconds=SOP_SECONDS,
        fp_max_per_hour=FP_MAX_PER_HOUR,
    )
    assert info["val_cap_met"] is True
    assert threshold < math.inf
    at_chosen = score_at_threshold(
        scores,
        threshold,
        seizures,
        [],
        duration,
        0,
        duration,
        lookback=5,
        need=4,
        silent=2100,
    )
    assert at_chosen["false_alarms_per_hour"] <= FP_MAX_PER_HOUR
    lower = threshold - 0.5
    at_lower = score_at_threshold(
        scores,
        lower,
        seizures,
        [],
        duration,
        0,
        duration,
        lookback=5,
        need=4,
        silent=2100,
    )
    if at_lower["n_false_alarms"] > at_chosen["n_false_alarms"]:
        assert at_lower["false_alarms_per_hour"] > FP_MAX_PER_HOUR or at_lower[
            "false_alarms_per_hour"
        ] > at_chosen["false_alarms_per_hour"]


def test_study_criterion_needs_half_the_seizures_and_the_cap():
    assert study_criterion_met(0.5, 0.15, 1, True) is True
    assert study_criterion_met(0.4, 0.0, 0, True) is False
    assert study_criterion_met(1.0, 0.2, 2, True) is False
    assert study_criterion_met(1.0, 0.4, 0, False) is True
    assert study_criterion_met(1.0, 0.4, 1, False) is False


def test_save_and_load_window_scores():
    folder = tempfile.mkdtemp()
    try:
        path = os.path.join(folder, "val_scores.npz")
        save_window_scores(path, [10, 11], [0.2, 1.5])
        ends, logits = load_window_scores(path)
        assert list(ends) == [10, 11]
        assert list(logits) == pytest.approx([0.2, 1.5])
        mapped = score_map(ends, logits)
        assert mapped[10] == pytest.approx(0.2)
        assert mapped[11] == pytest.approx(1.5)
    finally:
        os.remove(path)
        os.rmdir(folder)


def test_folder_alarm_metrics_fail_without_scores():
    from metrics import write_metrics
    from score_alarms import alarm_metrics_from_folder

    folder = tempfile.mkdtemp()
    try:
        write_metrics(
            folder,
            {
                "patient_id": "sub-004",
                "t1": 1.0,
                "t2": 2.0,
                "n_test_seizures": 1,
            },
        )
        with pytest.raises(ValueError, match="window scores are missing"):
            alarm_metrics_from_folder(folder, "missing_events.tsv")
    finally:
        os.remove(os.path.join(folder, "metrics.json"))
        os.rmdir(folder)
