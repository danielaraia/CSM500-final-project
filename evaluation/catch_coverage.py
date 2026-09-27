import os

import numpy as np

from evaluation.alarm_leads import (
    blank_lead_flags,
    hit_interval,
    lead_event_counts,
    mark_evaluable_at_second,
)
from evaluation.alarms import lookback_seconds, test_leads_from_seizures, vote_need
from data.events import load_event_rows, seizure_spans
from data.labels import SOP_SECONDS, SPH_SECONDS
from evaluation.score_alarms import recording_duration_seconds


def catch_coverage_from_ends(ends, lead_onsets, duration_seconds, sop_seconds=SOP_SECONDS):
    lookback = lookback_seconds(sop_seconds)
    need = vote_need(lookback)
    if duration_seconds < 1:
        raise ValueError(
            f"duration has to be at least 1 sec, we got {duration_seconds}"
        )
    score_at = {}
    index = 0
    while index < len(ends):
        score_at[int(ends[index])] = True
        index += 1
    caught, evaluable = blank_lead_flags(len(lead_onsets))
    n_scores_in_lookback = 0
    max_in_hit = 0
    n_scorable_hit = 0
    t = 0
    while t < duration_seconds:
        leave = t - lookback
        if leave >= 0 and leave in score_at:
            n_scores_in_lookback -= 1
        if t in score_at:
            n_scores_in_lookback += 1
        in_hit = _second_in_any_hit(lead_onsets, t, sop_seconds)
        if in_hit and n_scores_in_lookback > max_in_hit:
            max_in_hit = n_scores_in_lookback
        if n_scores_in_lookback >= need:
            mark_evaluable_at_second(
                evaluable, lead_onsets, t, SPH_SECONDS, sop_seconds
            )
            if in_hit:
                n_scorable_hit += 1
        t += 1
    counts = lead_event_counts(caught, evaluable)
    return {
        "catch_evaluable": bool(counts["n_seizures_evaluable"] > 0),
        "n_test_leads": int(counts["n_seizures"]),
        "n_test_leads_evaluable": int(counts["n_seizures_evaluable"]),
        "n_test_leads_unevaluable": int(counts["n_seizures_unevaluable"]),
        "max_lookback_scores_in_hit": int(max_in_hit),
        "n_scorable_hit_seconds": int(n_scorable_hit),
        "alarm_lookback_seconds": int(lookback),
        "alarm_vote_need": int(need),
    }


def catch_coverage_from_folder(folder, events_path, sop_seconds=SOP_SECONDS):
    val_ends = _load_ends(folder, "val.npz")
    test_ends = _load_ends(folder, "test.npz")
    scored = []
    _extend_ints(scored, val_ends)
    _extend_ints(scored, test_ends)
    rows = load_event_rows(events_path)
    duration_seconds = recording_duration_seconds(rows)
    seizures = seizure_spans(rows)
    _t1, _t2, test_leads, _n_merged, _n_test = test_leads_from_seizures(
        seizures, sop_seconds
    )
    return catch_coverage_from_ends(scored, test_leads, duration_seconds, sop_seconds)


def _load_ends(folder, name):
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise ValueError(f"split pack is missing: {path}")
    loaded = np.load(path)
    if "ends" not in loaded.files:
        return None
    return loaded["ends"]


def _extend_ints(out, ends):
    if ends is None:
        return
    index = 0
    while index < len(ends):
        out.append(int(ends[index]))
        index += 1


def _second_in_any_hit(lead_onsets, t, sop_seconds=SOP_SECONDS):
    index = 0
    while index < len(lead_onsets):
        start, end = hit_interval(lead_onsets[index], SPH_SECONDS, sop_seconds)
        if start <= t < end:
            return True
        index += 1
    return False
