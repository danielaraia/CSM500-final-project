import os

import mne
import numpy as np

from channels import pick_run_channels
from filters import EEG_SAMPLE_RATE, MOVEMENT_SAMPLE_RATE
from spans import gap_spans_from_record_onsets

# mne get_data is volts. quality.py rms is microvolts
VOLTS_TO_UV = 1e6


ANNOTATION_LABEL = "EDF Annotations"


def load_edf(path):
    if not os.path.isfile(path):
        raise ValueError(f"edf file is missing: {path}")
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    names = list(raw.ch_names)
    if len(names) < 1:
        raise ValueError(f"edf has no channels: {path}")
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()
    return names, sfreq, data


def load_run(eeg_path, ecg_path, emg_path, mov_path):
    eeg_names, eeg_rate, eeg_data = load_edf(eeg_path)
    ecg_names, ecg_rate, ecg_data = load_edf(ecg_path)
    emg_names, emg_rate, emg_data = load_edf(emg_path)
    mov_names, mov_rate, mov_data = load_edf(mov_path)
    _need_rate("eeg", eeg_rate, EEG_SAMPLE_RATE)
    _need_rate("ecg", ecg_rate, EEG_SAMPLE_RATE)
    _need_rate("emg", emg_rate, EEG_SAMPLE_RATE)
    _need_rate("movement", mov_rate, MOVEMENT_SAMPLE_RATE)
    picked = pick_run_channels(eeg_names, ecg_names, emg_names, mov_names)
    electrical_rails = electrical_rails_uv(eeg_path, ecg_path, emg_path, picked)
    electrical = np.stack(
        [
            _row(eeg_names, eeg_data, picked["eeg_a"]) * VOLTS_TO_UV,
            _row(eeg_names, eeg_data, picked["eeg_b"]) * VOLTS_TO_UV,
            _row(ecg_names, ecg_data, picked["ecg"]) * VOLTS_TO_UV,
            _row(emg_names, emg_data, picked["emg"]) * VOLTS_TO_UV,
        ]
    )
    movement_labels = (
        picked["neck_acc"]
        + picked["neck_gyr"]
        + picked["chest_acc"]
        + picked["chest_gyr"]
    )
    movement = np.stack(
        [_row(mov_names, mov_data, label) for label in movement_labels]
    )
    n_elec = _n_seconds(electrical.shape[1], EEG_SAMPLE_RATE, "electrical")
    n_mov = _n_seconds(movement.shape[1], MOVEMENT_SAMPLE_RATE, "movement")
    if n_elec != n_mov:
        raise ValueError(
            f"electrical and movement need the same number of seconds, "
            f"we got {n_elec} and {n_mov}"
        )
    return electrical, movement, picked, electrical_rails


def _need_rate(what, got, want):
    if got != float(want):
        raise ValueError(f"{what} should be {want} Hz, we got {got}")


def _row(names, data, label):
    if label not in names:
        raise ValueError(f"channel {label} is missing, we got {names}")
    return data[names.index(label)]


def _n_seconds(n_samples, rate, what):
    if n_samples % rate != 0:
        raise ValueError(
            f"{what} length has to split into 1 sec chunks, we got {n_samples}"
        )
    return n_samples // rate


def edf_gap_spans(path):
    onsets, record_duration = edf_record_onsets(path)
    return gap_spans_from_record_onsets(onsets, record_duration)


def edf_record_onsets(path):
    if not os.path.isfile(path):
        raise ValueError(f"edf file is missing: {path}")
    with open(path, "rb") as handle:
        preamble = handle.read(256)
        if len(preamble) < 256:
            raise ValueError(f"edf header is truncated: {path}")
        n_header = _ascii_int(preamble[184:192], "header bytes")
        n_records = _ascii_int(preamble[236:244], "n records")
        record_duration = _ascii_float(preamble[244:252], "record duration")
        n_ch = _ascii_int(preamble[252:256], "n channels")
        rest = handle.read(n_header - 256)
        if len(rest) != n_header - 256:
            raise ValueError(f"edf header is truncated: {path}")
        labels, _units, _phys_mins, _phys_maxs, nsamp = _channel_header(rest, n_ch)
        annot_i = _annotation_channel(labels)
        bytes_before = 0
        index = 0
        while index < annot_i:
            bytes_before += nsamp[index] * 2
            index += 1
        annot_bytes = nsamp[annot_i] * 2
        bytes_per_record = 0
        index = 0
        while index < n_ch:
            bytes_per_record += nsamp[index] * 2
            index += 1
        if n_records < 1:
            raise ValueError(f"edf has no data records: {path}")
        onsets = []
        record_index = 0
        while record_index < n_records:
            handle.seek(n_header + record_index * bytes_per_record + bytes_before)
            raw = handle.read(annot_bytes)
            if len(raw) != annot_bytes:
                raise ValueError(
                    f"edf annotations were cut short on record {record_index}"
                )
            onsets.append(_first_tal_onset(raw, record_index))
            record_index += 1
    return onsets, record_duration


def electrical_rails_uv(eeg_path, ecg_path, emg_path, picked):
    eeg_ranges = edf_physical_ranges(eeg_path)
    ecg_ranges = edf_physical_ranges(ecg_path)
    emg_ranges = edf_physical_ranges(emg_path)
    return [
        _range_to_uv(eeg_ranges, picked["eeg_a"]),
        _range_to_uv(eeg_ranges, picked["eeg_b"]),
        _range_to_uv(ecg_ranges, picked["ecg"]),
        _range_to_uv(emg_ranges, picked["emg"]),
    ]


def edf_physical_ranges(path):
    if not os.path.isfile(path):
        raise ValueError(f"edf file is missing: {path}")
    with open(path, "rb") as handle:
        preamble = handle.read(256)
        if len(preamble) < 256:
            raise ValueError(f"edf header is truncated: {path}")
        n_header = _ascii_int(preamble[184:192], "header bytes")
        n_ch = _ascii_int(preamble[252:256], "n channels")
        rest = handle.read(n_header - 256)
        if len(rest) != n_header - 256:
            raise ValueError(f"edf header is truncated: {path}")
        labels, units, phys_mins, phys_maxs, _nsamp = _channel_header(rest, n_ch)
    ranges = {}
    index = 0
    while index < len(labels):
        label = labels[index]
        if label != ANNOTATION_LABEL:
            pmin = phys_mins[index]
            pmax = phys_maxs[index]
            if not np.isfinite(pmin) or not np.isfinite(pmax):
                raise ValueError(
                    f"physical range for {label} is not finite, we got {pmin} {pmax}"
                )
            if pmin >= pmax:
                raise ValueError(
                    f"physical min has to be below max for {label}, "
                    f"we got {pmin} {pmax}"
                )
            ranges[label] = (pmin, pmax, units[index])
        index += 1
    if len(ranges) < 1:
        raise ValueError(f"edf has no signal channels with a physical range: {path}")
    return ranges


def _range_to_uv(ranges, label):
    if label not in ranges:
        raise ValueError(
            f"no physical range for {label}, we got {list(ranges)}"
        )
    pmin, pmax, dim = ranges[label]
    scale = _unit_to_uv_scale(dim)
    return (pmin * scale, pmax * scale)


def _unit_to_uv_scale(dim):
    if dim == "uV":
        return 1.0
    if dim == "mV":
        return 1000.0
    if dim == "V":
        return 1_000_000.0
    raise ValueError(
        f"electrical physical unit should be uV, mV, or V, we got {dim!r}"
    )


def _channel_header(rest, n_ch):
    labels = _ascii_fields(rest, 0, n_ch, 16)
    offset = n_ch * 16
    offset += n_ch * 80
    units = _ascii_fields(rest, offset, n_ch, 8)
    offset += n_ch * 8
    phys_mins = _float_fields(rest, offset, n_ch, "physical min")
    offset += n_ch * 8
    phys_maxs = _float_fields(rest, offset, n_ch, "physical max")
    offset += n_ch * 8
    offset += n_ch * 8
    offset += n_ch * 8
    offset += n_ch * 80
    nsamp = []
    index = 0
    while index < n_ch:
        start = offset + index * 8
        nsamp.append(_ascii_int(rest[start : start + 8], "samples per record"))
        index += 1
    return labels, units, phys_mins, phys_maxs, nsamp


def _ascii_fields(rest, offset, n_ch, width):
    fields = []
    index = 0
    while index < n_ch:
        start = offset + index * width
        fields.append(rest[start : start + width].decode("ascii").strip())
        index += 1
    return fields


def _float_fields(rest, offset, n_ch, what):
    fields = []
    index = 0
    while index < n_ch:
        start = offset + index * 8
        fields.append(_ascii_float(rest[start : start + 8], what))
        index += 1
    return fields


def _annotation_channel(labels):
    index = 0
    while index < len(labels):
        if labels[index] == ANNOTATION_LABEL:
            return index
        index += 1
    raise ValueError(
        "edf has no EDF Annotations channel, we got " + ", ".join(labels)
    )


def _first_tal_onset(raw, record_index):
    index = 0
    while index < len(raw) and raw[index] == 0:
        index += 1
    if index >= len(raw) or raw[index] not in (43, 45):
        raise ValueError(
            f"data record {record_index} has no time keeping tal"
        )
    sign = 1.0
    if raw[index] == 45:
        sign = -1.0
    index += 1
    start = index
    while index < len(raw) and raw[index] in b"0123456789.":
        index += 1
    if index == start or index >= len(raw) or raw[index] != 20:
        raise ValueError(
            f"data record {record_index} has no time keeping tal"
        )
    return sign * float(raw[start:index].decode("ascii"))


def _ascii_int(field, what):
    text = field.decode("ascii").strip()
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{what} should be an int, we got {text!r}")


def _ascii_float(field, what):
    text = field.decode("ascii").strip()
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{what} should be a number, we got {text!r}")
