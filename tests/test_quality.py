import numpy as np
import pytest

from quality import EEG_RMS_MAX, EEG_RMS_MIN, electrical_rail_reason, is_flat, root_mean_square, second_drop_reason, second_is_bad


def _ok_wave(scale):
    time = np.arange(256) / 256.0
    return scale * np.sin(2 * np.pi * 10.0 * time)


def test_rms_of_known_values():
    # 3, 4, 12 -> rms = sqrt((9+16+144)/3) = sqrt(56.333...) 
    samples = np.array([3.0, 4.0, 12.0])
    expected = np.sqrt((9.0 + 16.0 + 144.0) / 3.0)
    assert root_mean_square(samples) == pytest.approx(expected)


def test_flat_is_true_when_all_the_same():
    assert is_flat(np.ones(256)) is True
    assert is_flat(_ok_wave(50.0)) is False


def test_one_quiet_ear_keeps_the_second():
    quiet = _ok_wave(1.0)
    loud_ok = _ok_wave(80.0)
    movement = _ok_wave(1.0)
    assert root_mean_square(quiet) < EEG_RMS_MIN
    assert EEG_RMS_MIN < root_mean_square(loud_ok) < EEG_RMS_MAX
    assert second_is_bad(quiet, loud_ok, loud_ok, loud_ok, movement, movement, movement, movement) is False
    assert second_is_bad(loud_ok, quiet, loud_ok, loud_ok, movement, movement, movement, movement) is False


def test_both_ears_quiet_drops_the_second():
    quiet = _ok_wave(1.0)
    loud_ok = _ok_wave(80.0)
    movement = _ok_wave(1.0)
    assert second_is_bad(quiet, quiet, loud_ok, loud_ok, movement, movement, movement, movement) is True
    assert second_is_bad(loud_ok, loud_ok, loud_ok, loud_ok, movement, movement, movement, movement) is False


def test_drop_reason_names_the_fail():
    quiet = _ok_wave(1.0)
    loud_ok = _ok_wave(80.0)
    movement = _ok_wave(1.0)
    assert second_drop_reason(quiet, quiet, loud_ok, loud_ok, movement, movement, movement, movement) == "both_ears_rms"
    assert second_drop_reason(loud_ok, loud_ok, loud_ok, loud_ok, movement, movement, movement, movement) is None
    flat = np.ones(256)
    assert second_drop_reason(flat, loud_ok, loud_ok, loud_ok, movement, movement, movement, movement) == "electrical_flat"


def test_rail_hit_drops_the_second():
    loud_ok = _ok_wave(80.0)
    on_rail = loud_ok.copy()
    on_rail[10] = 1000.0
    raw = np.stack([on_rail, loud_ok, loud_ok, loud_ok])
    rails = [(-200.0, 200.0), (-200.0, 200.0), (-200.0, 200.0), (-200.0, 200.0)]
    assert electrical_rail_reason(raw, rails) == "electrical_rail"
    clean = np.stack([loud_ok, loud_ok, loud_ok, loud_ok])
    assert electrical_rail_reason(clean, rails) is None


def test_missing_rails_fail():
    loud_ok = _ok_wave(80.0)
    raw = np.stack([loud_ok, loud_ok, loud_ok, loud_ok])
    with pytest.raises(ValueError, match="need converter rails"):
        electrical_rail_reason(raw, None)
