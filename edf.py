import os

import mne


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
