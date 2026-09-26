import math

SPH_SECONDS = 5 * 60
SOP_SECONDS = 30 * 60
INTERICTAL_GAP_SECONDS = 60 * 60
VALIDATION_INTERICTAL_SECONDS = int(6.7 * 3600)
MONITOR_FRACTION = 0.25

LABEL_SOP = "sop"
LABEL_INTERICTAL = "interictal"
LABEL_UNUSED = "unused"


def label_second(time_seconds, seizures, sop_seconds=SOP_SECONDS):
    if time_seconds < 0:
        raise ValueError("time cant be negative")
    _check_seizures(seizures)

    for onset, offset in seizures:
        if onset <= time_seconds < offset:
            return LABEL_UNUSED

    for onset, offset in seizures:
        # sph is the 5 min right before onset, we dont score those
        sph_start = onset - SPH_SECONDS
        if sph_start <= time_seconds < onset:
            return LABEL_UNUSED

    for onset, offset in seizures:
        sop_start = onset - SPH_SECONDS - sop_seconds
        sop_end = onset - SPH_SECONDS
        if sop_start <= time_seconds < sop_end:
            return LABEL_SOP

    for onset, offset in seizures:
        too_close_before = onset - INTERICTAL_GAP_SECONDS
        if too_close_before <= time_seconds < onset:
            return LABEL_UNUSED
        if offset <= time_seconds < offset + INTERICTAL_GAP_SECONDS:
            return LABEL_UNUSED

    return LABEL_INTERICTAL


def n_test_seizures(n_usable):
    if n_usable < 2:
        raise ValueError(
            f"need at least two usable seizures, we got {n_usable}"
        )
    n_test = (n_usable + 2) // 3
    n_max_test = n_usable - 1
    if n_test > n_max_test:
        n_test = n_max_test
    return n_test


def seizures_with_full_sop(seizures, sop_seconds=SOP_SECONDS):
    kept = []
    index = 0
    while index < len(seizures):
        onset, offset = seizures[index]
        if onset >= SPH_SECONDS + sop_seconds:
            kept.append((onset, offset))
        index += 1
    return kept


def split_cut_times(seizures, sop_seconds=SOP_SECONDS):
    n_usable = len(seizures)
    n_test = n_test_seizures(n_usable)
    n_train_seizures = n_usable - n_test
    last_train_seizure = seizures[n_train_seizures - 1]
    first_test_seizure = seizures[n_train_seizures]
    last_train_offset = last_train_seizure[1]
    first_test_onset = first_test_seizure[0]
    if first_test_onset < last_train_offset:
        raise ValueError("the test one starts before teh last training seizure even ended")
    t2 = (last_train_offset + first_test_onset) / 2.0
    t1 = _grow_validation_start(seizures, t2, last_train_offset, n_train_seizures, sop_seconds)
    return t1, t2, n_train_seizures, n_test


def monitoring_indices(train_labels):
    n_windows = len(train_labels)
    n_monitor = n_windows // 4
    if n_monitor == 0:
        raise ValueError(
            f"monitoring slice is empty: {n_windows} labelled train windows, "
            "need more so the last 25 percent isnt nothing"
        )
    monitor_start = n_windows - n_monitor
    monitor_labels = train_labels[monitor_start:]
    n_positive = 0
    n_negative = 0
    for label in monitor_labels:
        if label == LABEL_SOP:
            n_positive += 1
        elif label == LABEL_INTERICTAL:
            n_negative += 1
        else:
            raise ValueError(f"monitoring slice has unused label {label}, that shouldnt be in here")
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "monitoring slice is missing a class, "
            f"we got n_positive={n_positive} n_negative={n_negative}"
        )
    return monitor_start, n_positive, n_negative


def _grow_validation_start(
    seizures, t2, last_train_offset, n_train_seizures, sop_seconds=SOP_SECONDS
):
    required_train_interictal = n_train_seizures * sop_seconds
    prefix = _interictal_prefix(seizures, t2)
    t1 = t2
    earliest = last_train_offset
    while t1 > earliest:
        next_t1 = t1 - 1
        train_interictal = _prefix_count(prefix, 0, next_t1)
        if train_interictal < required_train_interictal:
            break
        t1 = next_t1
        validation_interictal = _prefix_count(prefix, t1, t2)
        if validation_interictal >= VALIDATION_INTERICTAL_SECONDS:
            break
    return t1


def _interictal_prefix(seizures, end_time):
    last_second = math.ceil(end_time) - 1
    if last_second < 0:
        return [0]
    prefix = [0]
    count = 0
    time_seconds = 0
    while time_seconds <= last_second:
        if label_second(time_seconds, seizures) == LABEL_INTERICTAL:
            count += 1
        prefix.append(count)
        time_seconds += 1
    return prefix


def _prefix_count(prefix, start_time, end_time):
    first = math.ceil(start_time)
    last = math.ceil(end_time) - 1
    if last < first:
        return 0
    if last + 1 >= len(prefix):
        raise ValueError("prefix is too short for that time range")
    return prefix[last + 1] - prefix[first]


def _check_seizures(seizures):
    if len(seizures) == 0:
        raise ValueError("no seizures in the list")
    previous_offset = None
    for onset, offset in seizures:
        if offset <= onset:
            raise ValueError(f"offset {offset} is not after onset {onset}")
        if previous_offset is not None and onset < previous_offset:
            raise ValueError("seizures overlap or are in the wrong order")
        previous_offset = offset
