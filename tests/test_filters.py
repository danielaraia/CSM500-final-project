import numpy as np

from data.filters import EEG_SAMPLE_RATE, NOTCH_HZ, filter_ecg, filter_eeg, filter_emg, filter_movement_axis


def test_eeg_keeps_ten_hz_and_cuts_dc():
    n_samples = EEG_SAMPLE_RATE * 8
    time = np.arange(n_samples) / EEG_SAMPLE_RATE
    # 10 Hz should stay, teh big DC offset should go
    ten_hz = np.sin(2 * np.pi * 10.0 * time)
    dc_plus_ten = ten_hz + 50.0
    filtered = filter_eeg(dc_plus_ten)
    assert np.abs(np.mean(filtered)) < 1.0
    assert _amplitude_near(filtered, time, 10.0) > 0.5


def test_eeg_notch_cuts_fifty_hz():
    n_samples = EEG_SAMPLE_RATE * 8
    time = np.arange(n_samples) / EEG_SAMPLE_RATE
    fifty_hz = np.sin(2 * np.pi * NOTCH_HZ * time)
    filtered = filter_eeg(fifty_hz)
    assert _amplitude_near(filtered, time, NOTCH_HZ) < 0.1


def test_ecg_cuts_sixty_hz():
    n_samples = EEG_SAMPLE_RATE * 8
    time = np.arange(n_samples) / EEG_SAMPLE_RATE
    sixty_hz = np.sin(2 * np.pi * 60.0 * time)
    filtered = filter_ecg(sixty_hz)
    assert _amplitude_near(filtered, time, 60.0) < 0.1


def test_emg_cuts_low_frequency():
    n_samples = EEG_SAMPLE_RATE * 8
    time = np.arange(n_samples) / EEG_SAMPLE_RATE
    five_hz = np.sin(2 * np.pi * 5.0 * time)
    filtered = filter_emg(five_hz)
    assert _amplitude_near(filtered, time, 5.0) < 0.1


def test_movement_highpass_cuts_stillness():
    n_samples = 25 * 8
    time = np.arange(n_samples) / 25.0
    slow = np.sin(2 * np.pi * 0.2 * time) + 3.0
    filtered = filter_movement_axis(slow)
    assert np.abs(np.mean(filtered)) < 0.2


def _amplitude_near(samples, time, freq_hz):
    cosine = np.cos(2 * np.pi * freq_hz * time)
    sine = np.sin(2 * np.pi * freq_hz * time)
    real = 2.0 * np.mean(samples * cosine)
    imag = 2.0 * np.mean(samples * sine)
    return float(np.sqrt(real * real + imag * imag))
