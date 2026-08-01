import numpy as np

# rms of the whole 1 sec, in microvolts. 13 and 150 are from the plan
EEG_RMS_MIN = 13.0
EEG_RMS_MAX = 150.0
FLAT_RUN_FRACTION = 0.5


def root_mean_square(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cant take rms of nothing")
    return float(np.sqrt(np.mean(values * values)))


def is_flat(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.size < 2:
        raise ValueError("need at least 2 samples to check flat")
    if np.var(values) == 0.0:
        return True
    longest_run = _longest_identical_run(values)
    return longest_run / float(values.size) > FLAT_RUN_FRACTION


def samples_on_rail(samples, min_value, max_value):
    if min_value >= max_value:
        raise ValueError("rail min has to be below rail max")
    values = np.asarray(samples, dtype=np.float64)
    for value in values:
        if value <= min_value or value >= max_value:
            return True
    return False


def second_is_bad(
    eeg_a,
    eeg_b,
    ecg,
    emg,
    neck_acc,
    neck_gyro,
    chest_acc,
    chest_gyro,
    rail_min=None,
    rail_max=None,
):
    electrical = [eeg_a, eeg_b, ecg, emg]
    movement = [neck_acc, neck_gyro, chest_acc, chest_gyro]
    for series in electrical:
        if is_flat(series):
            return True
        if rail_min is not None and rail_max is not None:
            if samples_on_rail(series, rail_min, rail_max):
                return True
    # both ears must fail 13 to 150. one quiet ear is still a live second
    if eeg_rms_is_out_of_band(eeg_a) and eeg_rms_is_out_of_band(eeg_b):
        return True
    for series in movement:
        if is_flat(series):
            return True
    return False


def eeg_rms_is_out_of_band(samples):
    rms = root_mean_square(samples)
    return rms < EEG_RMS_MIN or rms > EEG_RMS_MAX


def _longest_identical_run(values):
    longest = 1
    current = 1
    index = 1
    while index < values.size:
        if values[index] == values[index - 1]:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 1
        index += 1
    return longest
