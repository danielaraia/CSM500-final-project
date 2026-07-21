# names come from the edf header, we dont guess a label
EEG_LABEL_ORDER = ["BTEleft SD", "BTEright SD", "CROSStop SD"]
ECG_LABEL = "ECG SD"
EMG_LABEL = "EMG SD"
NECK_ACC_LABELS = ["EEG SD ACC X", "EEG SD ACC Y", "EEG SD ACC Z"]
NECK_GYR_LABELS = ["EEG SD GYR A", "EEG SD GYR B", "EEG SD GYR C"]
CHEST_ACC_LABELS = ["ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z"]
CHEST_GYR_LABELS = ["ECGEMG SD GYR A", "ECGEMG SD GYR B", "ECGEMG SD GYR C"]


def pick_eeg_pair(channel_names):
    present = []
    for label in EEG_LABEL_ORDER:
        if label in channel_names:
            present.append(label)
    if len(present) < 2:
        raise ValueError(
            "need two behind-the-ear eeg channels, we got "
            + ", ".join(channel_names)
        )
    return present[0], present[1]


def require_label(channel_names, label, what):
    if label not in channel_names:
        raise ValueError(f"missing {what} label {label}, we got {channel_names}")
    return label


def require_group(channel_names, labels, what):
    missing = []
    for label in labels:
        if label not in channel_names:
            missing.append(label)
    if len(missing) > 0:
        raise ValueError(f"missing {what}: {missing}")
    return list(labels)


def pick_run_channels(eeg_names, ecg_names, emg_names, mov_names):
    eeg_a, eeg_b = pick_eeg_pair(eeg_names)
    return {
        "eeg_a": eeg_a,
        "eeg_b": eeg_b,
        "ecg": require_label(ecg_names, ECG_LABEL, "ecg"),
        "emg": require_label(emg_names, EMG_LABEL, "emg"),
        "neck_acc": require_group(mov_names, NECK_ACC_LABELS, "neck acc"),
        "neck_gyr": require_group(mov_names, NECK_GYR_LABELS, "neck gyr"),
        "chest_acc": require_group(mov_names, CHEST_ACC_LABELS, "chest acc"),
        "chest_gyr": require_group(mov_names, CHEST_GYR_LABELS, "chest gyr"),
    }
