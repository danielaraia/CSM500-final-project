import numpy as np
import pytest

from features import LOG_EPSILON, N_FFT_BINS, SAMPLES_1S, log_amplitude_fft


def test_ten_hz_sine_peaks_at_the_bin_we_expect():
    time = np.arange(SAMPLES_1S) / float(SAMPLES_1S)
    # 10 Hz for 1 sec at 256 Hz. after dropping dc, bin 10 is index 9
    samples = np.sin(2 * np.pi * 10.0 * time)
    features = log_amplitude_fft(samples)
    assert features.shape == (N_FFT_BINS,)
    peak_index = int(np.argmax(features))
    assert peak_index == 9


def test_log_fft_is_128_long_and_finite():
    samples = np.ones(SAMPLES_1S)
    features = log_amplitude_fft(samples)
    assert features.shape == (N_FFT_BINS,)
    assert np.all(np.isfinite(features))


def test_wrong_length_fails():
    with pytest.raises(ValueError, match="we got"):
        log_amplitude_fft(np.ones(10))


def test_zeros_go_to_log_epsilon_not_inf():
    samples = np.zeros(SAMPLES_1S)
    features = log_amplitude_fft(samples, epsilon=LOG_EPSILON)
    assert np.allclose(features, np.log(LOG_EPSILON))
