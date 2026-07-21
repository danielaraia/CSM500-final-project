import pytest

from channels import pick_eeg_pair, pick_run_channels


def test_picks_left_and_right_when_all_three_eeg_are_there():
    a, b = pick_eeg_pair(["CROSStop SD", "BTEright SD", "BTEleft SD"])
    assert a == "BTEleft SD"
    assert b == "BTEright SD"


def test_one_eeg_channel_fails():
    with pytest.raises(ValueError, match="two behind-the-ear"):
        pick_eeg_pair(["BTEleft SD"])


def test_full_run_has_all_eight_node_groups():
    eeg = ["BTEleft SD", "CROSStop SD"]
    mov = [
        "EEG SD ACC X", "EEG SD ACC Y", "EEG SD ACC Z",
        "EEG SD GYR A", "EEG SD GYR B", "EEG SD GYR C",
        "ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z",
        "ECGEMG SD GYR A", "ECGEMG SD GYR B", "ECGEMG SD GYR C",
    ]
    picked = pick_run_channels(eeg, ["ECG SD"], ["EMG SD"], mov)
    assert picked["eeg_a"] == "BTEleft SD"
    assert picked["eeg_b"] == "CROSStop SD"
    assert picked["ecg"] == "ECG SD"
    assert len(picked["neck_gyr"]) == 3


def test_missing_neck_gyro_fails():
    eeg = ["BTEleft SD", "BTEright SD"]
    mov = [
        "EEG SD ACC X", "EEG SD ACC Y", "EEG SD ACC Z",
        "ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z",
        "ECGEMG SD GYR A", "ECGEMG SD GYR B", "ECGEMG SD GYR C",
    ]
    with pytest.raises(ValueError, match="neck gyr"):
        pick_run_channels(eeg, ["ECG SD"], ["EMG SD"], mov)
