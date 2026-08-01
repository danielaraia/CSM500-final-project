import numpy as np

# rms of the whole 1 sec, in microvolts. 13 and 150 are from the plan
EEG_RMS_MIN = 13.0
EEG_RMS_MAX = 150.0
FLAT_RUN_FRACTION = 0.5
N_ELECTRICAL = 4


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
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        raise ValueError("rail bounds have to be finite")
    values = np.asarray(samples, dtype=np.float64)
    for value in values:
        if value <= min_value or value >= max_value:
            return True
    return False


def need_four_rails(electrical_rails):
    if electrical_rails is None:
        raise ValueError("need converter rails from the edf header")
    if len(electrical_rails) != N_ELECTRICAL:
        raise ValueError(
            f"need {N_ELECTRICAL} electrical rails, we got {len(electrical_rails)}"
        )
    index = 0
    while index < N_ELECTRICAL:
        pair = electrical_rails[index]
        if len(pair) != 2:
            raise ValueError(
                f"each rail needs min and max, we got {pair}"
            )
        rail_min, rail_max = pair
        if rail_min >= rail_max:
            raise ValueError("rail min has to be below rail max")
        if not np.isfinite(rail_min) or not np.isfinite(rail_max):
            raise ValueError("rail bounds have to be finite")
        index += 1


def electrical_rail_reason(electrical_raw, electrical_rails):
    need_four_rails(electrical_rails)
    values = np.asarray(electrical_raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != N_ELECTRICAL:
        raise ValueError(
            f"raw electrical should be ({N_ELECTRICAL}, samples), we got {values.shape}"
        )
    index = 0
    while index < N_ELECTRICAL:
        rail_min, rail_max = electrical_rails[index]
        if samples_on_rail(values[index], rail_min, rail_max):
            return "electrical_rail"
        index += 1
    return None


def second_is_bad(
    eeg_a,
    eeg_b,
    ecg,
    emg,
    neck_acc,
    neck_gyro,
    chest_acc,
    chest_gyro,
):
    return (
        second_drop_reason(
            eeg_a,
            eeg_b,
            ecg,
            emg,
            neck_acc,
            neck_gyro,
            chest_acc,
            chest_gyro,
        )
        is not None
    )


def second_drop_reason(
    eeg_a,
    eeg_b,
    ecg,
    emg,
    neck_acc,
    neck_gyro,
    chest_acc,
    chest_gyro,
):
    electrical = [eeg_a, eeg_b, ecg, emg]
    movement = [neck_acc, neck_gyro, chest_acc, chest_gyro]
    for series in electrical:
        if is_flat(series):
            return "electrical_flat"
    # both ears must fail 13 to 150. one quiet ear is still a live second
    if eeg_rms_is_out_of_band(eeg_a) and eeg_rms_is_out_of_band(eeg_b):
        return "both_ears_rms"
    for series in movement:
        if is_flat(series):
            return "movement_flat"
    return None


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
