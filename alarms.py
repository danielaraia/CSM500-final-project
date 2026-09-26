import math
import os

import numpy as np

from alarm_leads import (
    blank_lead_flags,
    hit_interval,
    lead_event_counts,
    mark_caught_leads,
    mark_evaluable_at_second,
)
from data.labels import (
    LABEL_INTERICTAL,
    SOP_SECONDS,
    SPH_SECONDS,
    label_second,
    seizures_with_full_sop,
    split_cut_times,
)

VOTE_FRACTION = 0.8
LOOKBACK_DIVISOR = 6
FP_MAX_PER_HOUR = 0.15
SECONDS_PER_HOUR = 3600.0
HOURS_TO_RESOLVE_FP_MAX = 1.0 / FP_MAX_PER_HOUR


def lookback_seconds(sop_seconds=SOP_SECONDS):
    if sop_seconds < LOOKBACK_DIVISOR:
        raise ValueError(f"sop is shorter than the lookback divisor, we got {sop_seconds}")
    if sop_seconds % LOOKBACK_DIVISOR != 0:
        raise ValueError(f"sop has to split into {LOOKBACK_DIVISOR} equal seconds, we got {sop_seconds}")
    return sop_seconds // LOOKBACK_DIVISOR


def vote_need(lookback):
    if lookback < 1:
        raise ValueError(f"lookback has to be at least 1 sec, we got {lookback}")
    return int(lookback * VOTE_FRACTION)


def silent_seconds(sph_seconds=SPH_SECONDS, sop_seconds=SOP_SECONDS):
    return sph_seconds + sop_seconds


def merge_lead_onsets(onsets, min_gap):
    if min_gap < 0:
        raise ValueError(f"merge gap cant be negative, we got {min_gap}")
    ordered = sorted(onsets)
    if len(ordered) == 0:
        return [], 0
    kept = [ordered[0]]
    n_merged = 0
    index = 1
    while index < len(ordered):
        if ordered[index] - kept[-1] < min_gap:
            n_merged += 1
        else:
            kept.append(ordered[index])
        index += 1
    return kept, n_merged


def score_map(ends, logits):
    if len(ends) != len(logits):
        raise ValueError(
            "ends and logits need the same number of windows, "
            f"we got {len(ends)} and {len(logits)}"
        )
    out = {}
    index = 0
    while index < len(ends):
        second = int(ends[index])
        if second in out:
            raise ValueError(f"two scores for the same second {second}")
        out[second] = float(logits[index])
        index += 1
    return out


def save_window_scores(path, ends, logits):
    ends = np.asarray(ends)
    logits = np.asarray(logits)
    if ends.shape[0] != logits.shape[0]:
        raise ValueError(
            "ends and logits need the same number of windows, "
            f"we got {ends.shape[0]} and {logits.shape[0]}"
        )
    np.savez_compressed(path, ends=ends, logits=logits)


def load_window_scores(path):
    if not os.path.isfile(path):
        raise ValueError(f"window scores are missing: {path}")
    loaded = np.load(path)
    if "ends" not in loaded.files or "logits" not in loaded.files:
        raise ValueError(f"window scores need ends and logits, we got {loaded.files}")
    return loaded["ends"], loaded["logits"]


def raise_alarms(
    score_at,
    threshold,
    duration_seconds,
    lookback=None,
    need=None,
    silent=None,
):
    if lookback is None:
        lookback = lookback_seconds()
    if need is None:
        need = vote_need(lookback)
    if silent is None:
        silent = silent_seconds()
    if duration_seconds < 1:
        raise ValueError(f"duration has to be at least 1 sec, we got {duration_seconds}")
    if lookback < 1:
        raise ValueError(f"lookback has to be at least 1 sec, we got {lookback}")
    if need < 1:
        raise ValueError(f"vote need has to be at least 1, we got {need}")
    if silent < 1:
        raise ValueError(f"silent period has to be at least 1 sec, we got {silent}")
    alarms = []
    silent_until = 0
    n_above = 0
    t = 0
    while t < duration_seconds:
        leave = t - lookback
        if leave >= 0:
            old = score_at.get(leave)
            if old is not None and old > threshold:
                n_above -= 1
        new = score_at.get(t)
        if new is not None and new > threshold:
            n_above += 1
        if t >= silent_until and n_above >= need:
            alarms.append(t)
            silent_until = t + silent
        t += 1
    return alarms


def alarm_hits_a_seizure(alarm_time, lead_onsets, sph_seconds=SPH_SECONDS, sop_seconds=SOP_SECONDS):
    index = 0
    while index < len(lead_onsets):
        start, end = hit_interval(lead_onsets[index], sph_seconds, sop_seconds)
        if start <= alarm_time < end:
            return True
        index += 1
    return False


def logit_to_probability(logit):
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    ez = math.exp(logit)
    return ez / (1.0 + ez)


def score_at_threshold(
    score_at,
    threshold,
    seizures,
    lead_onsets,
    duration_seconds,
    part_start,
    part_end,
    lookback=None,
    need=None,
    silent=None,
    sph_seconds=SPH_SECONDS,
    sop_seconds=SOP_SECONDS,
):
    if lookback is None:
        lookback = lookback_seconds(sop_seconds)
    if need is None:
        need = vote_need(lookback)
    if silent is None:
        silent = silent_seconds(sph_seconds, sop_seconds)
    if part_end < part_start:
        raise ValueError(f"part end is before part start, we got {part_start} {part_end}")
    alarms = raise_alarms(
        score_at, threshold, duration_seconds, lookback, need, silent
    )
    raw_seconds = 0
    scorable_seconds = 0
    warning_seconds = 0
    n_false = 0
    n_hit_alarms = 0
    caught, evaluable = blank_lead_flags(len(lead_onsets))
    n_scores_in_lookback = 0
    t = 0
    while t < duration_seconds:
        leave = t - lookback
        if leave >= 0 and leave in score_at:
            n_scores_in_lookback -= 1
        if t in score_at:
            n_scores_in_lookback += 1
        if n_scores_in_lookback >= need:
            mark_evaluable_at_second(
                evaluable, lead_onsets, t, sph_seconds, sop_seconds
            )
        in_part = part_start <= t < part_end
        if in_part and label_second(t, seizures, sop_seconds) == LABEL_INTERICTAL:
            raw_seconds += 1
            if n_scores_in_lookback >= need:
                scorable_seconds += 1
        t += 1
    alarm_index = 0
    while alarm_index < len(alarms):
        alarm_time = alarms[alarm_index]
        is_hit = alarm_hits_a_seizure(
            alarm_time, lead_onsets, sph_seconds, sop_seconds
        )
        if is_hit:
            n_hit_alarms += 1
            mark_caught_leads(
                caught, lead_onsets, alarm_time, sph_seconds, sop_seconds
            )
        elif part_start <= alarm_time < part_end:
            if label_second(alarm_time, seizures, sop_seconds) == LABEL_INTERICTAL:
                n_false += 1
                warn_t = alarm_time
                warn_end = alarm_time + sop_seconds
                if warn_end > part_end:
                    warn_end = part_end
                while warn_t < warn_end:
                    if label_second(warn_t, seizures, sop_seconds) == LABEL_INTERICTAL:
                        warning_seconds += 1
                    warn_t += 1
        alarm_index += 1
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


def pick_threshold(
    score_at,
    seizures,
    duration_seconds,
    part_start,
    part_end,
    lookback=None,
    need=None,
    silent=None,
    sph_seconds=SPH_SECONDS,
    sop_seconds=SOP_SECONDS,
    fp_max_per_hour=FP_MAX_PER_HOUR,
):
    probe = score_at_threshold(
        score_at,
        math.inf,
        seizures,
        [],
        duration_seconds,
        part_start,
        part_end,
        lookback=lookback,
        need=need,
        silent=silent,
        sph_seconds=sph_seconds,
        sop_seconds=sop_seconds,
    )
    if probe["scorable_interictal_hours"] <= 0:
        return math.inf, {
            "val_cap_met": False,
            "threshold_rule": "highest, val scorable interictal hours are 0",
            "val_false_alarms_per_hour": None,
            "val_scorable_interictal_hours": probe["scorable_interictal_hours"],
            "threshold_set": "val",
        }
    unique = []
    seen = {}
    for second, value in score_at.items():
        if part_start <= second < part_end and value not in seen:
            unique.append(value)
            seen[value] = True
    unique = sorted(unique, reverse=True)
    chosen = math.inf
    chosen_fpr = 0.0
    index = 0
    while index < len(unique):
        threshold = unique[index]
        scored = score_at_threshold(
            score_at,
            threshold,
            seizures,
            [],
            duration_seconds,
            part_start,
            part_end,
            lookback=lookback,
            need=need,
            silent=silent,
            sph_seconds=sph_seconds,
            sop_seconds=sop_seconds,
        )
        fpr = scored["false_alarms_per_hour"]
        if fpr is None:
            raise ValueError("val fpr is missing even though scorable hours were > 0")
        if fpr <= fp_max_per_hour:
            chosen = threshold
            chosen_fpr = fpr
        else:
            break
        index += 1
    if math.isinf(chosen):
        rule = "highest, no scored window sits above any finite threshold"
        cap_met = True
    else:
        rule = "lowest threshold with val false alarms per hour at most 0.15"
        cap_met = True
    return chosen, {
        "val_cap_met": cap_met,
        "threshold_rule": rule,
        "val_false_alarms_per_hour": chosen_fpr,
        "val_scorable_interictal_hours": probe["scorable_interictal_hours"],
        "threshold_set": "val",
    }


def choose_operating_threshold(
    score_at,
    seizures,
    duration_seconds,
    val_start,
    val_end,
    fallback_start=None,
    fallback_end=None,
    lookback=None,
    need=None,
    silent=None,
    sph_seconds=SPH_SECONDS,
    sop_seconds=SOP_SECONDS,
    fp_max_per_hour=FP_MAX_PER_HOUR,
):
    threshold, info = pick_threshold(
        score_at,
        seizures,
        duration_seconds,
        val_start,
        val_end,
        lookback=lookback,
        need=need,
        silent=silent,
        sph_seconds=sph_seconds,
        sop_seconds=sop_seconds,
        fp_max_per_hour=fp_max_per_hour,
    )
    if info["val_scorable_interictal_hours"] > 0:
        return threshold, info
    if fallback_start is None or fallback_end is None:
        raise ValueError(
            "val scorable interictal hours are 0 and there is no monitor fallback"
        )
    threshold, mon_info = pick_threshold(
        score_at,
        seizures,
        duration_seconds,
        fallback_start,
        fallback_end,
        lookback=lookback,
        need=need,
        silent=silent,
        sph_seconds=sph_seconds,
        sop_seconds=sop_seconds,
        fp_max_per_hour=fp_max_per_hour,
    )
    rule = mon_info["threshold_rule"]
    if rule == "lowest threshold with val false alarms per hour at most 0.15":
        rule = "lowest threshold with monitor false alarms per hour at most 0.15"
    elif rule == "highest, val scorable interictal hours are 0":
        rule = "highest, monitor scorable interictal hours are 0"
    elif rule == "highest, no scored window sits above any finite threshold":
        rule = "highest, no monitor scored window sits above any finite threshold"
    return threshold, {
        "val_cap_met": mon_info["val_cap_met"],
        "threshold_rule": rule,
        "val_false_alarms_per_hour": mon_info["val_false_alarms_per_hour"],
        "val_scorable_interictal_hours": mon_info["val_scorable_interictal_hours"],
        "threshold_set": "monitor",
    }


def test_leads_from_seizures(seizures, sop_seconds=SOP_SECONDS):
    usable = seizures_with_full_sop(seizures, sop_seconds)
    t1, t2, n_train, n_test = split_cut_times(usable, sop_seconds)
    onsets = []
    index = 0
    while index < len(usable):
        onsets.append(usable[index][0])
        index += 1
    leads, n_merged = merge_lead_onsets(onsets, silent_seconds(SPH_SECONDS, sop_seconds))
    test_leads = []
    lead_index = 0
    while lead_index < len(leads):
        if leads[lead_index] >= t2:
            test_leads.append(leads[lead_index])
        lead_index += 1
    return t1, t2, test_leads, n_merged, n_test


def study_criterion_met(sensitivity, fa_per_hour, n_false, resolves):
    if sensitivity is None:
        return False
    if sensitivity < 0.5:
        return False
    if not resolves:
        return n_false == 0
    if fa_per_hour is None:
        return False
    return fa_per_hour <= FP_MAX_PER_HOUR
