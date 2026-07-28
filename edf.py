import os

import mne
import numpy as np

from channels import pick_run_channels
from filters import EEG_SAMPLE_RATE, MOVEMENT_SAMPLE_RATE

# mne get_data is volts. quality.py rms is microvolts
VOLTS_TO_UV = 1e6


def load_edf(path):
    if not os.path.isfile(path):
        raise ValueError(f"edf file is missing: {path}")
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    names = list(raw.ch_names)
    if len(names) < 1:
        raise ValueError(f"edf has no channels: {path}")
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()
    return names, sfreq, data


def load_run(eeg_path, ecg_path, emg_path, mov_path):
    eeg_names, eeg_rate, eeg_data = load_edf(eeg_path)
    ecg_names, ecg_rate, ecg_data = load_edf(ecg_path)
    emg_names, emg_rate, emg_data = load_edf(emg_path)
    mov_names, mov_rate, mov_data = load_edf(mov_path)
    _need_rate("eeg", eeg_rate, EEG_SAMPLE_RATE)
    _need_rate("ecg", ecg_rate, EEG_SAMPLE_RATE)
    _need_rate("emg", emg_rate, EEG_SAMPLE_RATE)
    _need_rate("movement", mov_rate, MOVEMENT_SAMPLE_RATE)
    picked = pick_run_channels(eeg_names, ecg_names, emg_names, mov_names)
    electrical = np.stack(
        [
            _row(eeg_names, eeg_data, picked["eeg_a"]) * VOLTS_TO_UV,
            _row(eeg_names, eeg_data, picked["eeg_b"]) * VOLTS_TO_UV,
            _row(ecg_names, ecg_data, picked["ecg"]) * VOLTS_TO_UV,
            _row(emg_names, emg_data, picked["emg"]) * VOLTS_TO_UV,
        ]
    )
    movement_labels = (
        picked["neck_acc"]
        + picked["neck_gyr"]
        + picked["chest_acc"]
        + picked["chest_gyr"]
    )
    movement = np.stack(
        [_row(mov_names, mov_data, label) for label in movement_labels]
    )
    n_elec = _n_seconds(electrical.shape[1], EEG_SAMPLE_RATE, "electrical")
    n_mov = _n_seconds(movement.shape[1], MOVEMENT_SAMPLE_RATE, "movement")
    if n_elec != n_mov:
        raise ValueError(
            f"electrical and movement need the same number of seconds, "
            f"we got {n_elec} and {n_mov}"
        )
    return electrical, movement, picked


def _need_rate(what, got, want):
    if got != float(want):
        raise ValueError(f"{what} should be {want} Hz, we got {got}")


def _row(names, data, label):
    if label not in names:
        raise ValueError(f"channel {label} is missing, we got {names}")
    return data[names.index(label)]


def _n_seconds(n_samples, rate, what):
    if n_samples % rate != 0:
        raise ValueError(
            f"{what} length has to split into 1 sec chunks, we got {n_samples}"
        )
    return n_samples // rate
