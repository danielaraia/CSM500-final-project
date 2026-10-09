from scipy import signal

EEG_SAMPLE_RATE = 256
MOVEMENT_SAMPLE_RATE = 25
NOTCH_HZ = 50
NOTCH_Q = 30
BUTTER_ORDER = 4
EMG_HIGHPASS_ORDER = 2


def filter_eeg(samples):
    band = _butter_bandpass(samples, 0.5, 100.0, EEG_SAMPLE_RATE, BUTTER_ORDER)
    return _notch_50(band, EEG_SAMPLE_RATE)


def filter_ecg(samples):
    band = _butter_bandpass(samples, 0.5, 40.0, EEG_SAMPLE_RATE, BUTTER_ORDER)
    return _notch_50(band, EEG_SAMPLE_RATE)


def filter_emg(samples):
    # emg highpass is order 2, everything else is 4
    highpassed = _butter_highpass(samples, 20.0, EEG_SAMPLE_RATE, EMG_HIGHPASS_ORDER)
    notched = _notch_50(highpassed, EEG_SAMPLE_RATE)
    return _butter_lowpass(notched, 100.0, EEG_SAMPLE_RATE, BUTTER_ORDER)


def filter_movement_axis(samples):
    return _butter_highpass(samples, 2.0, MOVEMENT_SAMPLE_RATE, BUTTER_ORDER)


def _butter_bandpass(samples, low_hz, high_hz, sample_rate, order):
    sos = signal.butter(
        order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    return signal.sosfilt(sos, samples)


def _butter_highpass(samples, cutoff_hz, sample_rate, order):
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, samples)


def _butter_lowpass(samples, cutoff_hz, sample_rate, order):
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, samples)


def _notch_50(samples, sample_rate):
    b, a = signal.iirnotch(NOTCH_HZ, NOTCH_Q, fs=sample_rate)
    sos = signal.tf2sos(b, a)
    return signal.sosfilt(sos, samples)
