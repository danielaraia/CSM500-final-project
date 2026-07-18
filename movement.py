import numpy as np
from scipy import signal

MOVEMENT_RATE = 25
ELECTRICAL_RATE = 256
SAMPLES_MOVEMENT_1S = 25
SAMPLES_ELECTRICAL_1S = 256


def axis_magnitude(x, y, z):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"x y z need the same lenght, we got {x.shape}, {y.shape}, {z.shape}"
        )
    # per sample, sqrt(x*x + y*y + z*z), not divided by 3
    return np.sqrt(x * x + y * y + z * z)


def upsample_movement_to_256(samples_25hz):
    values = np.asarray(samples_25hz, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("movement series needs to be 1d")
    if values.shape[0] != SAMPLES_MOVEMENT_1S:
        raise ValueError(
            f"1 sec of movement should be {SAMPLES_MOVEMENT_1S} samples, we got {values.shape[0]}"
        )
    return signal.resample(values, SAMPLES_ELECTRICAL_1S)
