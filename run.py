import os

from config import start_run
from edf import edf_gap_spans, load_run
from events import impd_spans, load_event_rows, seizure_spans
from glue import build_windows, cut_monitoring_slice, keep_labelled_windows, split_labelled_windows
from labels import split_cut_times
from normalize import apply_node_zscore, node_mean_and_std
from pack import save_named_labels
from windows import TRAIN_STRIDE_SECONDS


def labelled_windows_from_files(
    eeg_path,
    ecg_path,
    emg_path,
    mov_path,
    events_path,
    stride_seconds=TRAIN_STRIDE_SECONDS,
):
    electrical, movement, _picked = load_run(
        eeg_path, ecg_path, emg_path, mov_path
    )
    gaps = []
    for path in (eeg_path, ecg_path, emg_path, mov_path):
        gaps.extend(edf_gap_spans(path))
    if len(gaps) > 0:
        raise ValueError(
            "edf has silent gaps, clock and samples wont line up, "
            f"we got {gaps}"
        )
    if not os.path.isfile(events_path):
        raise ValueError(f"events file is missing: {events_path}")
    rows = load_event_rows(events_path)
    features, graphs, ends = build_windows(
        electrical,
        movement,
        stride_seconds=stride_seconds,
        drop_spans=impd_spans(rows),
    )
    return keep_labelled_windows(
        features, graphs, ends, seizure_spans(rows)
    )


def write_labelled_run(
    root,
    patient_id,
    eeg_path,
    ecg_path,
    emg_path,
    mov_path,
    events_path,
    extra=None,
):
    features, graphs, _ends, labels = labelled_windows_from_files(
        eeg_path, ecg_path, emg_path, mov_path, events_path
    )
    folder = start_run(root, patient_id, extra)
    save_named_labels(os.path.join(folder, "windows.npz"), features, graphs, labels)
    return folder


def write_scaled_splits_from_seizures(folder, features, graphs, ends, labels, seizures):
    t1, t2, n_train_seizures, n_test = split_cut_times(seizures)
    counts = write_scaled_splits(folder, features, graphs, ends, labels, t1, t2)
    counts["t1"] = t1
    counts["t2"] = t2
    counts["n_train_seizures"] = int(n_train_seizures)
    counts["n_test_seizures"] = int(n_test)
    return counts


def write_scaled_splits(folder, features, graphs, ends, labels, t1, t2):
    if not os.path.isdir(folder):
        raise ValueError(f"run folder is missing: {folder}")
    train_path = os.path.join(folder, "train.npz")
    if os.path.exists(train_path):
        raise ValueError(f"split files already exist, we wont write over them: {train_path}")
    train, val, test = split_labelled_windows(features, graphs, ends, labels, t1, t2)
    train, monitor, _monitor_start, _n_pos, _n_neg = cut_monitoring_slice(
        train[0], train[1], train[2], train[3]
    )
    # zscore from gradient train only, then the same numbers on monitor, val, and test
    means, stds = node_mean_and_std(train[0])
    train_f = apply_node_zscore(train[0], means, stds)
    monitor_f = apply_node_zscore(monitor[0], means, stds)
    val_f = apply_node_zscore(val[0], means, stds)
    test_f = apply_node_zscore(test[0], means, stds)
    save_named_labels(train_path, train_f, train[1], train[3])
    save_named_labels(os.path.join(folder, "monitor.npz"), monitor_f, monitor[1], monitor[3])
    save_named_labels(os.path.join(folder, "val.npz"), val_f, val[1], val[3])
    save_named_labels(os.path.join(folder, "test.npz"), test_f, test[1], test[3])
    return {
        "n_train_windows": int(train_f.shape[0]),
        "n_monitor_windows": int(monitor_f.shape[0]),
        "n_validation_windows": int(val_f.shape[0]),
        "n_test_windows": int(test_f.shape[0]),
        "means": means,
        "stds": stds,
    }
