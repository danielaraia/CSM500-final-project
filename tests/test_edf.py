import os
import tempfile

import numpy as np
import pytest

from edf import load_edf


def test_missing_file_fails():
    with pytest.raises(ValueError, match="edf file is missing"):
        load_edf("/tmp/no-such-file-please.edf")


def test_reads_header_name_rate_and_samples():
    time = np.arange(256) / 256.0
    samples = 50.0 * np.sin(2 * np.pi * 10.0 * time)
    path = _write_edf("BTEleft SD", 256, samples)
    try:
        names, sfreq, data = load_edf(path)
        assert names == ["BTEleft SD"]
        assert sfreq == 256.0
        assert data.shape == (1, 256)
        # mne returns volts, we wrote microvolts
        assert np.max(np.abs(data[0] * 1e6 - samples)) < 1.0
    finally:
        os.remove(path)


def _pad(text, width):
    raw = text.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"edf field too long: {text}")
    return raw + b" " * (width - len(raw))


def _write_edf(label, sfreq, samples):
    samples = np.asarray(samples, dtype=np.float64)
    n_samples = int(samples.size)
    if n_samples != sfreq:
        raise ValueError("this tiny writer only does one 1-second record")
    phys_max = 1000.0
    phys_min = -1000.0
    dig_max = 32767
    dig_min = -32767
    scaled = (samples - phys_min) / (phys_max - phys_min)
    digital = np.round(scaled * (dig_max - dig_min) + dig_min).astype(np.int16)

    n_header = 256 + 256
    header = b"".join(
        [
            _pad("0", 8),
            _pad("tiny", 80),
            _pad("tiny", 80),
            _pad("01.01.70", 8),
            _pad("00.00.00", 8),
            _pad(str(n_header), 8),
            _pad("", 44),
            _pad("1", 8),
            _pad("1", 8),
            _pad("1", 4),
        ]
    )
    channel = b"".join(
        [
            _pad(label, 16),
            _pad("", 80),
            _pad("uV", 8),
            _pad(str(phys_min), 8),
            _pad(str(phys_max), 8),
            _pad(str(dig_min), 8),
            _pad(str(dig_max), 8),
            _pad("", 80),
            _pad(str(n_samples), 8),
            _pad("", 32),
        ]
    )
    handle = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    try:
        handle.write(header)
        handle.write(channel)
        handle.write(digital.tobytes())
    finally:
        handle.close()
    return handle.name
