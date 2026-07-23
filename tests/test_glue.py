import numpy as np
import pytest

from features import N_FFT_BINS
from glue import SECONDS_NEEDED, build_one_window
from pearson import N_NODES
from windows import WINDOW_SECONDS


def test_fake_run_makes_one_sixty_step_window():
    features, graphs = _fake_window()
    assert features.shape == (WINDOW_SECONDS, N_NODES, N_FFT_BINS)
    assert graphs.shape == (WINDOW_SECONDS, N_NODES, N_NODES)
    # 10 Hz sine, dc dropped, so bin 10 is index 9
    peak_index = int(np.argmax(features[30, 0]))
    assert peak_index == 9
    np.testing.assert_allclose(graphs[30], graphs[30].T)
    assert graphs[30, 0, 1] == pytest.approx(1.0)
    node = 0
    while node < N_NODES:
        assert graphs[30, node, node] == pytest.approx(1.0)
        node += 1


def test_too_short_fails():
    electrical = np.ones((4, 256))
    movement = np.ones((12, 25))
    with pytest.raises(ValueError, match="we got 1"):
        build_one_window(electrical, movement)


def test_second_count_mismatch_fails():
    electrical = np.ones((4, 256 * 4))
    movement = np.ones((12, 25 * 3))
    with pytest.raises(ValueError, match="we got 4 and 3"):
        build_one_window(electrical, movement)


def _fake_window():
    n_sec = SECONDS_NEEDED
    time = np.arange(n_sec * 256) / 256.0
    eeg = 80.0 * np.sin(2 * np.pi * 10.0 * time)
    emg = 80.0 * np.sin(2 * np.pi * 40.0 * time)
    electrical = np.stack([eeg, eeg, eeg, emg])
    rng = np.random.default_rng(123)
    movement = rng.normal(size=(12, n_sec * 25))
    return build_one_window(electrical, movement)
