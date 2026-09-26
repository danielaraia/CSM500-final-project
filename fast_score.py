import math

import numpy as np

from alarm_leads import lead_event_counts
from alarms import (
    HOURS_TO_RESOLVE_FP_MAX,
    lookback_seconds,
    silent_seconds,
    vote_need,
)
from data.labels import INTERICTAL_GAP_SECONDS, SPH_SECONDS

SECONDS_PER_HOUR = 3600.0


def prepare_scorer(score_at, seizures, duration_seconds, sop_seconds):
    # everything score_at_threshold derives per second, up front in numpy.
    # the pure python scan took minutes per threshold; the report needs hundreds of probes
    if SPH_SECONDS + sop_seconds > INTERICTAL_GAP_SECONDS:
        raise ValueError("fast scoring needs the interictal gap to cover sph + sop")
    if duration_seconds < 1:
        raise ValueError(f"duration has to be at least 1 sec, we got {duration_seconds}")
    lookback = lookback_seconds(sop_seconds)
    need = vote_need(lookback)
    silent = silent_seconds(SPH_SECONDS, sop_seconds)
    seconds_list = sorted(score_at)
    seconds = np.array(seconds_list, dtype=np.int64)
    values = np.array([score_at[s] for s in seconds_list], dtype=np.float64)
    interictal = np.ones(duration_seconds, dtype=np.int64)
    for onset, offset in seizures:
        start = int(math.ceil(onset - INTERICTAL_GAP_SECONDS))
        end = int(math.ceil(offset + INTERICTAL_GAP_SECONDS))
        if start < 0:
            start = 0
        if end > duration_seconds:
            end = duration_seconds
        if start < end:
            interictal[start:end] = 0
    cumsum = np.zeros(duration_seconds + 1, dtype=np.int64)
    np.cumsum(interictal, out=cumsum[1:])
    return {
        "seconds": seconds,
        "values": values,
        "interictal": interictal,
        "interictal_cumsum": cumsum,
        "duration_seconds": duration_seconds,
        "lookback": lookback,
        "need": need,
        "silent": silent,
        "sop_seconds": sop_seconds,
    }


def _count_in_window(sorted_seconds, ts, lookback):
    # number of scored seconds in (t - lookback, t] for each t
    return np.searchsorted(sorted_seconds, ts, "right") - np.searchsorted(
        sorted_seconds, ts - lookback, "right"
    )


def fast_alarms(scorer, threshold):
    # same alarm times as alarms.raise_alarms
    seconds = scorer["seconds"]
    lookback = scorer["lookback"]
    need = scorer["need"]
    silent = scorer["silent"]
    above_seconds = seconds[scorer["values"] > threshold]  # n_above only grows at these
    alarms = []
    silent_until = 0
    index = 0
    n_above_seconds = len(above_seconds)
    while index < n_above_seconds:
        candidate = int(above_seconds[index])
        if candidate < silent_until:
            index += 1
            continue
        if silent_until < candidate:
            # a silent period just ended, alarm right here if the votes are still there
            n_at_wake = _count_in_window(
                above_seconds, np.array([silent_until]), lookback
            )[0]
            if n_at_wake >= need:
                alarms.append(silent_until)
                silent_until = silent_until + silent
                continue
        n_here = _count_in_window(above_seconds, np.array([candidate]), lookback)[0]
        if n_here >= need:
            alarms.append(candidate)
            silent_until = candidate + silent
        index += 1
    return alarms


def _hit_intervals(lead_onsets, sop_seconds):
    intervals = []
    for lead in lead_onsets:
        intervals.append((lead - SPH_SECONDS - sop_seconds, lead - SPH_SECONDS))
    return intervals


def vote_supremum(scorer, part_start, part_end):
    # The highest threshold where an interictal second in the part can still vote: for each
    # second the need-th biggest score in its window, then the biggest of those. at or above it, zero false alarms
    seconds = scorer["seconds"]
    values = scorer["values"]
    lookback = scorer["lookback"]
    need = scorer["need"]
    interictal = scorer["interictal"]
    if len(seconds) == 0:
        return None
    part_first = max(int(math.ceil(part_start)), 0)
    part_last = min(int(math.ceil(part_end)), scorer["duration_seconds"])
    best = None
    second = part_first
    while second < part_last:
        if interictal[second]:
            hi = int(np.searchsorted(seconds, second, "right"))
            lo = int(np.searchsorted(seconds, second - lookback, "right"))
            count = hi - lo
            if count >= need:
                window_values = values[lo:hi]
                v = float(np.partition(window_values, count - need)[count - need])
                if best is None or v > best:
                    best = v
        second += 1
    return best


def score_part_fast(scorer, threshold, part_start, part_end, lead_onsets):
    # same return fields as alarms.score_at_threshold, plus the alarm list
    if part_end < part_start:
        raise ValueError(f"part end is before part start, we got {part_start} {part_end}")
    duration = scorer["duration_seconds"]
    lookback = scorer["lookback"]
    need = scorer["need"]
    sop_seconds = scorer["sop_seconds"]
    seconds = scorer["seconds"]
    interictal = scorer["interictal"]
    cumsum = scorer["interictal_cumsum"]
    alarms = fast_alarms(scorer, threshold)
    part_first = max(int(math.ceil(part_start)), 0)
    part_last = min(int(math.ceil(part_end)), duration)
    if part_last > part_first:
        part_seconds = np.arange(part_first, part_last, dtype=np.int64)
        part_interictal = interictal[part_first:part_last].astype(bool)
        n_scores = _count_in_window(seconds, part_seconds, lookback)
        raw_seconds = int(part_interictal.sum())
        scorable_seconds = int((part_interictal & (n_scores >= need)).sum())
    else:
        raw_seconds = 0
        scorable_seconds = 0
    intervals = _hit_intervals(lead_onsets, sop_seconds)
    caught = [False] * len(lead_onsets)
    evaluable = [False] * len(lead_onsets)
    lead_index = 0
    while lead_index < len(lead_onsets):
        start, end = intervals[lead_index]
        first = max(int(math.ceil(start)), 0)
        last = min(int(math.ceil(end)), duration)
        if last > first:
            ts = np.arange(first, last, dtype=np.int64)
            if (_count_in_window(seconds, ts, lookback) >= need).any():
                evaluable[lead_index] = True
        lead_index += 1
    n_false = 0
    n_hit_alarms = 0
    warning_seconds = 0
    for alarm_time in alarms:
        is_hit = False
        lead_index = 0
        while lead_index < len(intervals):
            start, end = intervals[lead_index]
            if start <= alarm_time < end:
                is_hit = True
                caught[lead_index] = True
            lead_index += 1
        if is_hit:
            n_hit_alarms += 1
        elif part_start <= alarm_time < part_end and interictal[alarm_time]:
            n_false += 1
            warn_end = alarm_time + sop_seconds
            if warn_end > part_end:
                warn_end = part_end
            warn_last = min(int(math.ceil(warn_end)), duration)
            if warn_last > alarm_time:
                warning_seconds += int(cumsum[warn_last] - cumsum[alarm_time])
    leads = lead_event_counts(caught, evaluable)
    hours = scorable_seconds / SECONDS_PER_HOUR
    if hours <= 0:
        fa_per_hour = None
    else:
        fa_per_hour = n_false / hours
    fa_per_day = None
    if fa_per_hour is not None:
        fa_per_day = fa_per_hour * 24.0
    return {
        "n_alarms": len(alarms),
        "alarms": alarms,
        "n_hit_alarms": n_hit_alarms,
        "n_false_alarms": n_false,
        "n_seizures": leads["n_seizures"],
        "n_seizures_evaluable": leads["n_seizures_evaluable"],
        "n_seizures_unevaluable": leads["n_seizures_unevaluable"],
        "n_seizures_caught": leads["n_seizures_caught"],
        "n_seizures_missed": leads["n_seizures_missed"],
        "event_sensitivity": leads["event_sensitivity"],
        "raw_interictal_hours": raw_seconds / SECONDS_PER_HOUR,
        "scorable_interictal_hours": hours,
        "interictal_hours_dropped": (raw_seconds - scorable_seconds) / SECONDS_PER_HOUR,
        "false_alarms_per_hour": fa_per_hour,
        "false_alarms_per_day": fa_per_day,
        "time_in_warning_interictal_hours": warning_seconds / SECONDS_PER_HOUR,
        "resolves_0_15_per_hour": hours >= HOURS_TO_RESOLVE_FP_MAX,
    }
