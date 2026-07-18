import numpy as np

SAMPLES_1S = 256
N_FFT_BINS = 128
LOG_EPSILON = 1e-8


def log_amplitude_fft(samples, epsilon=LOG_EPSILON):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("need a 1d second of signal")
    if values.shape[0] != SAMPLES_1S:
        raise ValueError(f"need {SAMPLES_1S} samples for 1 sec, we got {values.shape[0]}")
    if epsilon <= 0:
        raise ValueError("epsilon has to be > 0 or the log can explode")

    spectrum = np.fft.rfft(values)
    amplitude = np.abs(spectrum)
    # rfft gives 129 numbers. bin 0 is dc, log of that is a disaster, so we drop it
    kept = amplitude[1 : N_FFT_BINS + 1]
    if kept.shape[0] != N_FFT_BINS:
        raise ValueError(f"wanted {N_FFT_BINS} bins, we got {kept.shape[0]}")

    features = np.log(kept + epsilon)
    if not np.all(np.isfinite(features)):
        raise ValueError("fft features have inf or nan, something went wrong")
    return features
