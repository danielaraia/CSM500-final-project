import os
import tempfile

import numpy as np
import pytest

from channels import (
    CHEST_ACC_LABELS,
    CHEST_GYR_LABELS,
    NECK_ACC_LABELS,
    NECK_GYR_LABELS,
)
from edf import edf_gap_spans, edf_record_onsets, load_edf, load_run


def test_missing_file_fails():
    with pytest.raises(ValueError, match="edf file is missing"):
        load_edf("/tmp/no-such-file-please.edf")


def test_reads_header_name_rate_and_samples():
    time = np.arange(256) / 256.0
    samples = 50.0 * np.sin(2 * np.pi * 10.0 * time)
    path = _write_edf([("BTEleft SD", samples)], 256)
    try:
        names, sfreq, data = load_edf(path)
        assert names == ["BTEleft SD"]
        assert sfreq == 256.0
        assert data.shape == (1, 256)
        # mne returns volts, we wrote microvolts
        assert np.max(np.abs(data[0] * 1e6 - samples)) < 1.0
    finally:
        os.remove(path)


def test_load_run_stacks_four_files():
    paths = _write_tiny_run()
    try:
        electrical, movement, picked = load_run(
            paths["eeg"], paths["ecg"], paths["emg"], paths["mov"]
        )
        assert electrical.shape == (4, 256)
        assert movement.shape == (12, 25)
        assert picked["eeg_a"] == "BTEleft SD"
        assert picked["eeg_b"] == "CROSStop SD"
        assert np.max(np.abs(electrical[0] - _wave(256, 50.0))) < 1.0
    finally:
        _remove_paths(paths)


def test_wrong_eeg_rate_fails():
    time = np.arange(128) / 128.0
    samples = 50.0 * np.sin(2 * np.pi * 10.0 * time)
    bad = _write_edf([("BTEleft SD", samples), ("CROSStop SD", samples)], 128)
    paths = _write_tiny_run()
    try:
        with pytest.raises(ValueError, match="eeg should be 256 Hz"):
            load_run(bad, paths["ecg"], paths["emg"], paths["mov"])
    finally:
        os.remove(bad)
        _remove_paths(paths)


def test_plain_edf_has_no_tal_channel():
    time = np.arange(256) / 256.0
    samples = 50.0 * np.sin(2 * np.pi * 10.0 * time)
    path = _write_edf([("BTEleft SD", samples)], 256)
    try:
        with pytest.raises(ValueError, match="EDF Annotations"):
            edf_gap_spans(path)
    finally:
        os.remove(path)


def test_continuous_edf_plus_has_no_gap():
    path = _write_edf_plus([0.0, 1.0, 2.0], reserved="EDF+C")
    try:
        onsets, duration = edf_record_onsets(path)
        assert onsets == [0.0, 1.0, 2.0]
        assert duration == 1.0
        assert edf_gap_spans(path) == []
    finally:
        os.remove(path)


def test_edf_plus_jump_is_a_gap():
    path = _write_edf_plus([0.0, 1.0, 4.0], reserved="EDF+D")
    try:
        onsets, duration = edf_record_onsets(path)
        assert onsets == [0.0, 1.0, 4.0]
        assert duration == 1.0
        assert edf_gap_spans(path) == [(2.0, 4.0)]
    finally:
        os.remove(path)


def _wave(n, scale):
    time = np.arange(n) / float(n)
    return scale * np.sin(2 * np.pi * 10.0 * time)


def _write_tiny_run():
    eeg_wave = _wave(256, 50.0)
    mov_wave = _wave(25, 1.0)
    mov_channels = []
    for label in NECK_ACC_LABELS + NECK_GYR_LABELS + CHEST_ACC_LABELS + CHEST_GYR_LABELS:
        mov_channels.append((label, mov_wave))
    paths = {
        "eeg": _write_edf(
            [("BTEleft SD", eeg_wave), ("CROSStop SD", eeg_wave)], 256
        ),
        "ecg": _write_edf([("ECG SD", eeg_wave)], 256),
        "emg": _write_edf([("EMG SD", eeg_wave)], 256),
        "mov": _write_edf(mov_channels, 25),
    }
    return paths


def _remove_paths(paths):
    for path in paths.values():
        os.remove(path)


def _pad(text, width):
    raw = text.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"edf field too long: {text}")
    return raw + b" " * (width - len(raw))


def _write_edf(channels, sfreq):
    n_ch = len(channels)
    n_samples = int(np.asarray(channels[0][1]).size)
    if n_samples != sfreq:
        raise ValueError("this tiny writer only does one 1-second record")
    phys_max = 1000.0
    phys_min = -1000.0
    dig_max = 32767
    dig_min = -32767
    n_header = 256 + 256 * n_ch
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
            _pad(str(n_ch), 4),
        ]
    )
    labels = b""
    transducers = b""
    units = b""
    phys_mins = b""
    phys_maxs = b""
    dig_mins = b""
    dig_maxs = b""
    prefilters = b""
    nsamps = b""
    reserved = b""
    bodies = []
    for label, samples in channels:
        samples = np.asarray(samples, dtype=np.float64)
        scaled = (samples - phys_min) / (phys_max - phys_min)
        digital = np.round(scaled * (dig_max - dig_min) + dig_min).astype(np.int16)
        labels += _pad(label, 16)
        transducers += _pad("", 80)
        units += _pad("uV", 8)
        phys_mins += _pad(str(phys_min), 8)
        phys_maxs += _pad(str(phys_max), 8)
        dig_mins += _pad(str(dig_min), 8)
        dig_maxs += _pad(str(dig_max), 8)
        prefilters += _pad("", 80)
        nsamps += _pad(str(n_samples), 8)
        reserved += _pad("", 32)
        bodies.append(digital.tobytes())
    handle = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    try:
        handle.write(header)
        handle.write(labels)
        handle.write(transducers)
        handle.write(units)
        handle.write(phys_mins)
        handle.write(phys_maxs)
        handle.write(dig_mins)
        handle.write(dig_maxs)
        handle.write(prefilters)
        handle.write(nsamps)
        handle.write(reserved)
        for body in bodies:
            handle.write(body)
    finally:
        handle.close()
    return handle.name


def _tal_bytes(onset, n_samples):
    if onset == int(onset):
        text = f"+{int(onset)}\x14\x14\x00"
    else:
        text = f"+{onset}\x14\x14\x00"
    raw = text.encode("ascii")
    nbytes = n_samples * 2
    if len(raw) > nbytes:
        raise ValueError("tal too long for the annotation channel")
    return raw + b"\x00" * (nbytes - len(raw))


def _write_edf_plus(record_onsets, reserved="EDF+C"):
    n_records = len(record_onsets)
    annot_nsamp = 40
    dummy_nsamp = 8
    n_ch = 2
    n_header = 256 + 256 * n_ch
    header = b"".join(
        [
            _pad("0", 8),
            _pad("tiny", 80),
            _pad("tiny", 80),
            _pad("01.01.70", 8),
            _pad("00.00.00", 8),
            _pad(str(n_header), 8),
            _pad(reserved, 44),
            _pad(str(n_records), 8),
            _pad("1", 8),
            _pad(str(n_ch), 4),
        ]
    )
    labels = _pad("EDF Annotations", 16) + _pad("BTEleft SD", 16)
    transducers = _pad("", 80) + _pad("", 80)
    units = _pad("", 8) + _pad("uV", 8)
    phys_mins = _pad("-32768", 8) + _pad("-1000", 8)
    phys_maxs = _pad("32767", 8) + _pad("1000", 8)
    dig_mins = _pad("-32768", 8) + _pad("-32767", 8)
    dig_maxs = _pad("32767", 8) + _pad("32767", 8)
    prefilters = _pad("", 80) + _pad("", 80)
    nsamps = _pad(str(annot_nsamp), 8) + _pad(str(dummy_nsamp), 8)
    ch_reserved = _pad("", 32) + _pad("", 32)
    handle = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    try:
        handle.write(header)
        handle.write(labels)
        handle.write(transducers)
        handle.write(units)
        handle.write(phys_mins)
        handle.write(phys_maxs)
        handle.write(dig_mins)
        handle.write(dig_maxs)
        handle.write(prefilters)
        handle.write(nsamps)
        handle.write(ch_reserved)
        dummy = np.zeros(dummy_nsamp, dtype=np.int16)
        for onset in record_onsets:
            handle.write(_tal_bytes(onset, annot_nsamp))
            handle.write(dummy.tobytes())
    finally:
        handle.close()
    return handle.name
