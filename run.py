import os
import time

import torch

from config import LOCKED, start_run
from edf import edf_gap_spans, load_run
from events import impd_spans, load_event_rows, seizure_spans
from glue import build_windows, cut_monitoring_slice, keep_labelled_windows, split_labelled_windows
from labels import seizures_with_full_sop, split_cut_times
from metrics import write_metrics
from normalize import apply_node_zscore, node_mean_and_std
from pack import load_split, save_named_labels
from pos_weight import weight_from_class_numbers
from train_loop import score_windows, train_and_write_metrics, train_until_stop
from windows import EVAL_STRIDE_SECONDS, TRAIN_STRIDE_SECONDS


def labelled_windows_from_files(
    eeg_path,
    ecg_path,
    emg_path,
    mov_path,
    events_path,
    stride_seconds=TRAIN_STRIDE_SECONDS,
    t1=None,
    train_stride_seconds=None,
):
    electrical, movement, _picked, electrical_rails = load_run(
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
    seizures = seizure_spans(rows)
    features, graphs, ends, drop_counts = build_windows(
        electrical,
        movement,
        stride_seconds=stride_seconds,
        drop_spans=impd_spans(rows),
        electrical_rails=electrical_rails,
        seizures=seizures,
        t1=t1,
        train_stride_seconds=train_stride_seconds,
    )
    kept = keep_labelled_windows(
        features, graphs, ends, seizures
    )
    return kept[0], kept[1], kept[2], kept[3], drop_counts


def write_split_run(
    root,
    patient_id,
    eeg_path,
    ecg_path,
    emg_path,
    mov_path,
    events_path,
    extra=None,
    leaf="none",
):
    rows = load_event_rows(events_path)
    seizures = seizures_with_full_sop(seizure_spans(rows))
    t1 = split_cut_times(seizures)[0]
    features, graphs, ends, labels, drop_counts = labelled_windows_from_files(
        eeg_path,
        ecg_path,
        emg_path,
        mov_path,
        events_path,
        stride_seconds=EVAL_STRIDE_SECONDS,
        t1=t1,
        train_stride_seconds=TRAIN_STRIDE_SECONDS,
    )
    folder = start_run(root, patient_id, extra, leaf=leaf)
    counts = write_scaled_splits_from_seizures(
        folder, features, graphs, ends, labels, seizures
    )
    for key in drop_counts:
        counts[key] = drop_counts[key]
    counts["train_stride_seconds"] = TRAIN_STRIDE_SECONDS
    counts["eval_stride_seconds"] = EVAL_STRIDE_SECONDS
    counts["scored_at_train_stride"] = False
    return folder, counts


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
    save_named_labels(train_path, train_f, train[1], train[3], ends=train[2])
    save_named_labels(
        os.path.join(folder, "monitor.npz"),
        monitor_f,
        monitor[1],
        monitor[3],
        ends=monitor[2],
    )
    save_named_labels(
        os.path.join(folder, "val.npz"), val_f, val[1], val[3], ends=val[2]
    )
    save_named_labels(
        os.path.join(folder, "test.npz"), test_f, test[1], test[3], ends=test[2]
    )
    return {
        "n_train_windows": int(train_f.shape[0]),
        "n_monitor_windows": int(monitor_f.shape[0]),
        "n_validation_windows": int(val_f.shape[0]),
        "n_test_windows": int(test_f.shape[0]),
        "means": means,
        "stds": stds,
    }


def train_packed_run(
    folder,
    model,
    optimizer,
    extra=None,
    max_epochs=None,
    patience=None,
    train_batch_size=None,
    device=None,
):
    train_f, train_g, train_y, _train_ends = _load_split_file(folder, "train.npz")
    monitor_f, monitor_g, monitor_y, _monitor_ends = _load_split_file(
        folder, "monitor.npz"
    )
    packed_extra = {
        "n_validation_windows": int(_load_split_file(folder, "val.npz")[2].shape[0]),
        "n_test_windows": int(_load_split_file(folder, "test.npz")[2].shape[0]),
    }
    if extra is not None:
        for key in extra:
            packed_extra[key] = extra[key]
    torch.manual_seed(LOCKED["seed"])
    return train_and_write_metrics(
        folder,
        model,
        optimizer,
        train_f,
        train_g,
        train_y,
        monitor_f,
        monitor_g,
        monitor_y,
        extra=packed_extra,
        max_epochs=max_epochs,
        patience=patience,
        train_batch_size=train_batch_size,
        device=device,
    )


def timed_patient_run(
    root,
    patient_id,
    eeg_path,
    ecg_path,
    emg_path,
    mov_path,
    events_path,
    model,
    optimizer,
    extra=None,
    device=None,
    max_epochs=None,
    patience=None,
    train_batch_size=None,
    compile_seconds=None,
):
    clock_start = time.perf_counter()
    rows = load_event_rows(events_path)
    seizures = seizures_with_full_sop(seizure_spans(rows))
    t1 = split_cut_times(seizures)[0]
    features, graphs, ends, labels, drop_counts = labelled_windows_from_files(
        eeg_path,
        ecg_path,
        emg_path,
        mov_path,
        events_path,
        stride_seconds=EVAL_STRIDE_SECONDS,
        t1=t1,
        train_stride_seconds=TRAIN_STRIDE_SECONDS,
    )
    folder = start_run(root, patient_id)
    split_counts = write_scaled_splits_from_seizures(
        folder, features, graphs, ends, labels, seizures
    )
    preprocess_seconds = time.perf_counter() - clock_start
    packed_extra = dict(drop_counts)
    packed_extra["n_validation_windows"] = int(split_counts["n_validation_windows"])
    packed_extra["n_test_windows"] = int(split_counts["n_test_windows"])
    packed_extra["t1"] = split_counts["t1"]
    packed_extra["t2"] = split_counts["t2"]
    packed_extra["n_train_seizures"] = int(split_counts["n_train_seizures"])
    packed_extra["n_test_seizures"] = int(split_counts["n_test_seizures"])
    packed_extra["scored_at_train_stride"] = False
    packed_extra["train_stride_seconds"] = TRAIN_STRIDE_SECONDS
    packed_extra["eval_stride_seconds"] = EVAL_STRIDE_SECONDS
    packed_extra["patient_id"] = patient_id
    if extra is not None:
        for key in extra:
            packed_extra[key] = extra[key]
    return train_score_and_write_metrics(
        folder,
        model,
        optimizer,
        packed_extra,
        clock_start,
        preprocess_seconds,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        train_batch_size=train_batch_size,
        compile_seconds=compile_seconds,
    )


def train_score_and_write_metrics(
    folder,
    model,
    optimizer,
    extra,
    clock_start,
    preprocess_seconds,
    device=None,
    max_epochs=None,
    patience=None,
    train_batch_size=None,
    compile_seconds=None,
):
    train_f, train_g, train_y, _train_ends = _load_split_file(folder, "train.npz")
    monitor_f, monitor_g, monitor_y, _monitor_ends = _load_split_file(
        folder, "monitor.npz"
    )
    val_f, val_g, _val_y, _val_ends = _load_split_file(folder, "val.npz")
    test_f, test_g, _test_y, _test_ends = _load_split_file(folder, "test.npz")
    torch.manual_seed(LOCKED["seed"])
    pos_weight, n_negative, n_positive = weight_from_class_numbers(train_y)
    _mon_w, n_negative_monitor, n_positive_monitor = weight_from_class_numbers(monitor_y)
    train_start = time.perf_counter()
    best_epoch, best_loss = train_until_stop(
        model,
        optimizer,
        train_f,
        train_g,
        train_y,
        monitor_f,
        monitor_g,
        monitor_y,
        pos_weight,
        max_epochs=max_epochs,
        patience=patience,
        train_batch_size=train_batch_size,
        checkpoint_dir=os.path.join(folder, "checkpoints"),
        device=device,
    )
    train_seconds = time.perf_counter() - train_start
    score_start = time.perf_counter()
    n_val_scored = score_windows(model, val_f, val_g, device=device)
    n_test_scored = score_windows(model, test_f, test_g, device=device)
    score_seconds = time.perf_counter() - score_start
    total_seconds = time.perf_counter() - clock_start
    n_params = 0
    for parameter in model.parameters():
        n_params += parameter.numel()
    metrics = {
        "best_epoch": int(best_epoch),
        "best_monitor_bce": float(best_loss),
        "pos_weight": float(pos_weight),
        "n_positive_train": int(n_positive),
        "n_negative_train": int(n_negative),
        "n_positive_monitor": int(n_positive_monitor),
        "n_negative_monitor": int(n_negative_monitor),
        "n_train_windows": int(len(train_y)),
        "n_monitor_windows": int(len(monitor_y)),
        "parameter_count": int(n_params),
        "negative_control": "none",
        "early_stopping_metric": LOCKED["early_stopping_metric"],
        "preprocess_seconds": float(preprocess_seconds),
        "train_seconds": float(train_seconds),
        "score_seconds": float(score_seconds),
        "end_to_end_wall_clock": float(total_seconds),
        "patient_wall_clock_seconds": float(total_seconds),
        "n_seconds_scored": int(n_val_scored + n_test_scored),
        "n_validation_windows_scored": int(n_val_scored),
        "n_test_windows_scored": int(n_test_scored),
        "patient_id": extra.get("patient_id") if extra else None,
        "graph_strategy": LOCKED["graph_strategy"],
        "sop_minutes": LOCKED["sop_minutes"],
        "sph_minutes": LOCKED["sph_minutes"],
    }
    if metrics["patient_id"] is None:
        metrics.pop("patient_id")
    if compile_seconds is not None:
        metrics["compile_seconds"] = float(compile_seconds)
    if extra is not None:
        for key in extra:
            metrics[key] = extra[key]
    if "peak_memory_allocated_bytes" in metrics:
        metrics["peak_memory_bytes"] = metrics["peak_memory_allocated_bytes"]
        metrics["peak_memory_allocated_gib"] = float(
            metrics["peak_memory_allocated_bytes"]
        ) / float(1024 ** 3)
        if "gpu_memory_bytes" in metrics:
            metrics["concurrent_train_processes_arithmetic_formula"] = (
                "gpu_memory_bytes // peak_memory_bytes"
            )
    if "batch_128_one_step_fits" in metrics:
        metrics["batch_128_fits_24gb"] = metrics["batch_128_one_step_fits"]
    write_metrics(folder, metrics)
    return folder, metrics


def _load_split_file(folder, name):
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise ValueError(f"split file is missing: {path}")
    return load_split(path)
