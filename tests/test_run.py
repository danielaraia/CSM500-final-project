import os
import shutil
import tempfile

import numpy as np
import pytest

from core.channels import (
    CHEST_ACC_LABELS,
    CHEST_GYR_LABELS,
    ECG_LABEL,
    EMG_LABEL,
    NECK_ACC_LABELS,
    NECK_GYR_LABELS,
)
from data.labels import LABEL_INTERICTAL, LABEL_SOP
from pack import load_split
from run import (
    labelled_windows_from_files,
    write_scaled_splits,
    write_scaled_splits_from_seizures,
    write_split_run,
)


def test_files_and_events_keep_the_sop_window():
    paths = _write_tiny_labelled_run(n_seconds=80)
    try:
        features, graphs, ends, labels, drop_counts = labelled_windows_from_files(
            paths["eeg"],
            paths["ecg"],
            paths["emg"],
            paths["mov"],
            paths["events"],
        )
        assert ends == [69]
        assert labels == [LABEL_SOP]
        assert features.shape[0] == 1
        assert graphs.shape[0] == 1
        assert drop_counts["n_steps_dropped_short_buffer"] >= 1
    finally:
        _remove_paths(paths)


def test_files_keep_train_hop_ten_before_t1():
    paths = _write_tiny_labelled_run(n_seconds=80)
    try:
        features, graphs, ends, labels, drop_counts = labelled_windows_from_files(
            paths["eeg"],
            paths["ecg"],
            paths["emg"],
            paths["mov"],
            paths["events"],
            stride_seconds=1,
            t1=70,
            train_stride_seconds=10,
        )
        assert 69 in ends
        assert 70 in ends
        assert 71 in ends
        assert 68 not in ends
        assert 75 not in ends
        assert labels[0] == LABEL_SOP
        assert drop_counts["n_steps_dropped_short_buffer"] >= 1
    finally:
        _remove_paths(paths)


def test_impd_on_the_sop_window_leaves_nothing():
    paths = _write_tiny_labelled_run(n_seconds=80, impd=(15.0, 1.0))
    try:
        with pytest.raises(ValueError, match="didnt keep any labelled windows"):
            labelled_windows_from_files(
                paths["eeg"],
                paths["ecg"],
                paths["emg"],
                paths["mov"],
                paths["events"],
            )
    finally:
        _remove_paths(paths)


def test_silent_gap_fails():
    paths = _write_tiny_labelled_run(n_seconds=3, eeg_onsets=[0.0, 1.0, 4.0])
    try:
        with pytest.raises(ValueError, match="silent gaps"):
            labelled_windows_from_files(
                paths["eeg"],
                paths["ecg"],
                paths["emg"],
                paths["mov"],
                paths["events"],
            )
    finally:
        _remove_paths(paths)


def test_write_scaled_splits_zscore_from_train_only():
    n_windows = 20
    features = np.zeros((n_windows, 1, 8, 1))
    graphs = np.zeros((n_windows, 1, 8, 8))
    ends = list(range(n_windows))
    labels = (
        [LABEL_INTERICTAL] * 10
        + [LABEL_SOP] * 2
        + [LABEL_SOP] * 2
        + [LABEL_INTERICTAL] * 2
        + [LABEL_SOP] * 2
        + [LABEL_SOP, LABEL_INTERICTAL]
    )
    # first 12 windows are gradient train: 0, 2, 0, 2 ... so mean 1 std 1
    index = 0
    while index < 12:
        features[index, 0, :, 0] = 0.0 if index % 2 == 0 else 2.0
        index += 1
    features[12:18, 0, :, 0] = 100.0
    features[18:20, 0, :, 0] = 3.0
    folder = tempfile.mkdtemp()
    try:
        counts = write_scaled_splits(
            folder, features, graphs, ends, labels, t1=16, t2=18
        )
        assert counts["n_train_windows"] == 12
        assert counts["n_monitor_windows"] == 4
        assert counts["n_validation_windows"] == 2
        assert counts["n_test_windows"] == 2
        train_f, _g, train_y, _train_ends = load_split(os.path.join(folder, "train.npz"))
        mon_f, _mg, mon_y, _mon_ends = load_split(os.path.join(folder, "monitor.npz"))
        _val_f, _vg, val_y, _val_ends = load_split(os.path.join(folder, "val.npz"))
        test_f, _tg, test_y, _test_ends = load_split(os.path.join(folder, "test.npz"))
        packed = np.load(os.path.join(folder, "test.npz"))
        np.testing.assert_array_equal(packed["ends"], np.array([18, 19]))
        assert train_f[0, 0, 0, 0] == pytest.approx(-1.0)
        assert train_f[1, 0, 0, 0] == pytest.approx(1.0)
        # 100 used train stats, not its own
        assert mon_f[0, 0, 0, 0] == pytest.approx(99.0)
        assert test_f[0, 0, 0, 0] == pytest.approx(2.0)
        np.testing.assert_array_equal(train_y[:10], np.zeros(10))
        np.testing.assert_array_equal(train_y[10:], np.ones(2))
        np.testing.assert_array_equal(mon_y, np.array([1, 1, 0, 0]))
        np.testing.assert_array_equal(val_y, np.array([1, 1]))
        np.testing.assert_array_equal(test_y, np.array([1, 0]))
        assert counts["n_positive_train"] == 2
        assert counts["n_negative_train"] == 10
        assert counts["n_positive_monitor"] == 2
        assert counts["n_negative_monitor"] == 2
        assert counts["n_positive_test"] == 1
        assert counts["n_negative_test"] == 1
    finally:
        shutil.rmtree(folder)


def test_write_scaled_splits_packs_hop_one_monitor():
    ends = [59, 69, 79, 89, 90, 91, 92, 100, 101, 110, 111]
    n_windows = len(ends)
    features = np.zeros((n_windows, 1, 8, 1))
    graphs = np.zeros((n_windows, 1, 8, 8))
    index = 0
    while index < 3:
        features[index, 0, :, 0] = 0.0 if index % 2 == 0 else 2.0
        index += 1
    features[3:, 0, :, 0] = 100.0
    labels = (
        [LABEL_SOP, LABEL_INTERICTAL, LABEL_SOP]
        + [LABEL_SOP, LABEL_SOP, LABEL_INTERICTAL, LABEL_INTERICTAL]
        + [LABEL_SOP, LABEL_SOP]
        + [LABEL_SOP, LABEL_INTERICTAL]
    )
    folder = tempfile.mkdtemp()
    try:
        counts = write_scaled_splits(
            folder, features, graphs, ends, labels, t1=100, t2=110
        )
        assert counts["n_train_windows"] == 3
        assert counts["n_monitor_windows"] == 4
        assert counts["monitor_stride_seconds"] == 1
        _tf, _g, _y, train_ends = load_split(os.path.join(folder, "train.npz"))
        _mf, _mg, _my, mon_ends = load_split(os.path.join(folder, "monitor.npz"))
        packed = np.load(os.path.join(folder, "monitor.npz"))
        np.testing.assert_array_equal(packed["ends"], np.array([89, 90, 91, 92]))
        np.testing.assert_array_equal(train_ends, np.array([59, 69, 79]))
        assert 90 in list(mon_ends)
        assert 91 in list(mon_ends)
    finally:
        shutil.rmtree(folder)


def test_write_scaled_splits_fails_when_train_has_no_sop():
    n_windows = 20
    features = np.zeros((n_windows, 1, 8, 1))
    graphs = np.zeros((n_windows, 1, 8, 8))
    ends = list(range(n_windows))
    labels = (
        [LABEL_INTERICTAL] * 12
        + [LABEL_SOP] * 2
        + [LABEL_INTERICTAL] * 2
        + [LABEL_SOP] * 2
        + [LABEL_SOP, LABEL_INTERICTAL]
    )
    folder = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="gradient train has no sop windows"):
            write_scaled_splits(
                folder, features, graphs, ends, labels, t1=16, t2=18
            )
        assert not os.path.exists(os.path.join(folder, "train.npz"))
    finally:
        shutil.rmtree(folder)


def test_write_scaled_splits_uses_seizure_cut_times():
    seizures = [
        (10_000, 10_100),
        (30_000, 30_100),
        (50_000, 50_100),
    ]
    t1 = 30_100.0
    t2 = 40_050.0
    n_windows = 20
    features = np.zeros((n_windows, 1, 8, 1))
    graphs = np.zeros((n_windows, 1, 8, 8))
    ends = list(range(16)) + [t1, t1 + 1.0, t2, t2 + 1.0]
    index = 0
    while index < 12:
        features[index, 0, :, 0] = 0.0 if index % 2 == 0 else 2.0
        index += 1
    labels = (
        [LABEL_INTERICTAL] * 10
        + [LABEL_SOP] * 2
        + [LABEL_SOP] * 2
        + [LABEL_INTERICTAL] * 2
        + [LABEL_SOP] * 2
        + [LABEL_SOP, LABEL_INTERICTAL]
    )
    folder = tempfile.mkdtemp()
    try:
        counts = write_scaled_splits_from_seizures(
            folder, features, graphs, ends, labels, seizures
        )
        assert counts["t1"] == pytest.approx(t1)
        assert counts["t2"] == pytest.approx(t2)
        assert counts["n_train_seizures"] == 2
        assert counts["n_test_seizures"] == 1
        assert counts["n_train_windows"] == 12
        assert counts["n_monitor_windows"] == 4
        assert counts["n_validation_windows"] == 2
        assert counts["n_test_windows"] == 2
        _val_f, _vg, val_y, _val_ends = load_split(os.path.join(folder, "val.npz"))
        _test_f, _tg, test_y, _test_ends = load_split(os.path.join(folder, "test.npz"))
        np.testing.assert_array_equal(val_y, np.array([1, 1]))
        np.testing.assert_array_equal(test_y, np.array([1, 0]))
    finally:
        shutil.rmtree(folder)


def test_one_seizure_cannot_make_split_files():
    folder = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="at least two usable seizures"):
            write_scaled_splits_from_seizures(
                folder,
                np.zeros((4, 1, 8, 1)),
                np.zeros((4, 1, 8, 8)),
                [1, 2, 3, 4],
                [LABEL_SOP, LABEL_SOP, LABEL_INTERICTAL, LABEL_INTERICTAL],
                [(10_000, 10_100)],
            )
    finally:
        shutil.rmtree(folder)


def test_write_split_run_fails_with_one_seizure():
    paths = _write_tiny_labelled_run(n_seconds=80)
    root = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="at least two usable seizures"):
            write_split_run(
                root,
                "sub-004",
                paths["eeg"],
                paths["ecg"],
                paths["emg"],
                paths["mov"],
                paths["events"],
            )
    finally:
        _remove_paths(paths)
        shutil.rmtree(root)


def test_missing_events_file_fails():
    paths = _write_tiny_labelled_run(n_seconds=3)
    try:
        with pytest.raises(ValueError, match="events file is missing"):
            labelled_windows_from_files(
                paths["eeg"],
                paths["ecg"],
                paths["emg"],
                paths["mov"],
                "/tmp/no-such-events-please.tsv",
            )
    finally:
        _remove_paths(paths)


def _write_tiny_labelled_run(n_seconds, impd=None, eeg_onsets=None):
    eeg_wave = _wave(n_seconds * 256, 50.0)
    mov_wave = _noise(n_seconds * 25)
    mov_channels = []
    for label in NECK_ACC_LABELS + NECK_GYR_LABELS + CHEST_ACC_LABELS + CHEST_GYR_LABELS:
        mov_channels.append((label, mov_wave))
    paths = {
        "eeg": _write_edf_plus_signals(
            [("BTEleft SD", eeg_wave), ("CROSStop SD", eeg_wave)],
            256,
            eeg_onsets,
        ),
        "ecg": _write_edf_plus_signals([(ECG_LABEL, eeg_wave)], 256, None),
        "emg": _write_edf_plus_signals([(EMG_LABEL, eeg_wave)], 256, None),
        "mov": _write_edf_plus_signals(mov_channels, 25, None),
        "events": _write_events(impd),
    }
    return paths


def _write_events(impd):
    lines = ["onset\tduration\teventType\n", "375.0\t25.0\tsz\n"]
    if impd is not None:
        start, duration = impd
        lines.append(f"{start}\t{duration}\timpd\n")
    handle = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
    handle.write("".join(lines))
    handle.close()
    return handle.name


def _wave(n, scale):
    time = np.arange(n) / 256.0
    return scale * np.sin(2 * np.pi * 10.0 * time)


def _noise(n):
    rng = np.random.default_rng(123)
    return rng.normal(size=n)


def _remove_paths(paths):
    for path in paths.values():
        os.remove(path)


def _pad(text, width):
    raw = text.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"edf field too long: {text}")
    return raw + b" " * (width - len(raw))


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


def _write_edf_plus_signals(channels, sfreq, record_onsets):
    n_samples = int(np.asarray(channels[0][1]).size)
    if n_samples % sfreq != 0:
        raise ValueError("samples have to split into 1 sec records")
    n_records = n_samples // sfreq
    if record_onsets is None:
        record_onsets = []
        index = 0
        while index < n_records:
            record_onsets.append(float(index))
            index += 1
    if len(record_onsets) != n_records:
        raise ValueError("need one onset per data record")
    annot_nsamp = 40
    n_sig = len(channels)
    n_ch = n_sig + 1
    n_header = 256 + 256 * n_ch
    header = b"".join(
        [
            _pad("0", 8),
            _pad("tiny", 80),
            _pad("tiny", 80),
            _pad("01.01.70", 8),
            _pad("00.00.00", 8),
            _pad(str(n_header), 8),
            _pad("EDF+C", 44),
            _pad(str(n_records), 8),
            _pad("1", 8),
            _pad(str(n_ch), 4),
        ]
    )
    labels = _pad("EDF Annotations", 16)
    transducers = _pad("", 80)
    units = _pad("", 8)
    phys_mins = _pad("-32768", 8)
    phys_maxs = _pad("32767", 8)
    dig_mins = _pad("-32768", 8)
    dig_maxs = _pad("32767", 8)
    prefilters = _pad("", 80)
    nsamps = _pad(str(annot_nsamp), 8)
    ch_reserved = _pad("", 32)
    digital_rows = []
    phys_min = -1000.0
    phys_max = 1000.0
    dig_min = -32767
    dig_max = 32767
    for label, samples in channels:
        samples = np.asarray(samples, dtype=np.float64)
        scaled = (samples - phys_min) / (phys_max - phys_min)
        digital = np.round(scaled * (dig_max - dig_min) + dig_min).astype(np.int16)
        digital_rows.append(digital)
        labels += _pad(label, 16)
        transducers += _pad("", 80)
        units += _pad("uV", 8)
        phys_mins += _pad(str(phys_min), 8)
        phys_maxs += _pad(str(phys_max), 8)
        dig_mins += _pad(str(dig_min), 8)
        dig_maxs += _pad(str(dig_max), 8)
        prefilters += _pad("", 80)
        nsamps += _pad(str(sfreq), 8)
        ch_reserved += _pad("", 32)
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
        record_index = 0
        while record_index < n_records:
            handle.write(_tal_bytes(record_onsets[record_index], annot_nsamp))
            ch_index = 0
            while ch_index < n_sig:
                start = record_index * sfreq
                handle.write(digital_rows[ch_index][start : start + sfreq].tobytes())
                ch_index += 1
            record_index += 1
    finally:
        handle.close()
    return handle.name
