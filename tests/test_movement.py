import numpy as np
import pytest

from movement import axis_magnitude, upsample_movement_to_256


def test_magnitude_is_the_3_4_5_triangle():
    x = np.array([3.0, 3.0])
    y = np.array([4.0, 4.0])
    z = np.array([0.0, 0.0])
    mag = axis_magnitude(x, y, z)
    np.testing.assert_allclose(mag, [5.0, 5.0])


def test_upsample_gives_256_samples():
    samples = np.sin(2 * np.pi * np.arange(25) / 25.0)
    up = upsample_movement_to_256(samples)
    assert up.shape == (256,)


def test_mismatched_axes_fail():
    with pytest.raises(ValueError, match="we got"):
        axis_magnitude(np.ones(3), np.ones(3), np.ones(2))
