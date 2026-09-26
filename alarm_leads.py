from data.labels import SOP_SECONDS, SPH_SECONDS


def hit_interval(onset, sph_seconds=SPH_SECONDS, sop_seconds=SOP_SECONDS):
    return onset - sph_seconds - sop_seconds, onset - sph_seconds


def blank_lead_flags(n_leads):
    caught = []
    evaluable = []
    index = 0
    while index < n_leads:
        caught.append(False)
        evaluable.append(False)
        index += 1
    return caught, evaluable


def mark_evaluable_at_second(evaluable, lead_onsets, t, sph_seconds, sop_seconds):
    index = 0
    while index < len(lead_onsets):
        start, end = hit_interval(lead_onsets[index], sph_seconds, sop_seconds)
        if start <= t < end:
            evaluable[index] = True
        index += 1


def mark_caught_leads(caught, lead_onsets, alarm_time, sph_seconds, sop_seconds):
    index = 0
    while index < len(lead_onsets):
        start, end = hit_interval(lead_onsets[index], sph_seconds, sop_seconds)
        if start <= alarm_time < end:
            caught[index] = True
        index += 1


def lead_event_counts(caught, evaluable):
    n_caught = 0
    n_evaluable = 0
    index = 0
    while index < len(caught):
        if evaluable[index]:
            n_evaluable += 1
            if caught[index]:
                n_caught += 1
        index += 1
    n_leads = len(caught)
    n_unevaluable = n_leads - n_evaluable
    n_missed = n_evaluable - n_caught
    if n_evaluable == 0:
        sensitivity = None
    else:
        sensitivity = n_caught / float(n_evaluable)
    return {
        "n_seizures": n_leads,
        "n_seizures_evaluable": n_evaluable,
        "n_seizures_unevaluable": n_unevaluable,
        "n_seizures_caught": n_caught,
        "n_seizures_missed": n_missed,
        "event_sensitivity": sensitivity,
    }
