import os

from data.edf import edf_gap_spans, load_run
from data.events import impd_spans, load_event_rows, seizure_spans
from glue import build_windows, keep_labelled_windows
from data.labels import LABEL_INTERICTAL, LABEL_SOP
from metrics import read_metrics, replace_metrics_keys
from data.normalize import apply_node_zscore, node_mean_and_std
from packs.pack import load_split, save_named_labels
from data.windows import EVAL_STRIDE_SECONDS, TRAIN_STRIDE_SECONDS, end_is_on_stride


def monitor_clock_from_train_hop10(ends):
    hop10_i = []
    index = 0
    while index < len(ends):
        if end_is_on_stride(int(ends[index]), TRAIN_STRIDE_SECONDS):
            hop10_i.append(index)
        index += 1
    if len(hop10_i) == 0 or len(hop10_i) == len(ends):
        return None
    n_hop10 = len(hop10_i)
    n_monitor = n_hop10 // 4
    if n_monitor == 0:
        return None
    return ends[hop10_i[n_hop10 - n_monitor]]


def cut_monitoring_slice_at_clock(features, adjacency, ends, labels, monitor_clock):
    train_i, monitor_i = _indices_at_clock(ends, monitor_clock, end_limit=None)
    if len(train_i) == 0:
        raise ValueError("didnt leave any train windows after the monitoring cut")
    if len(monitor_i) == 0:
        raise ValueError("monitoring slice is empty after the clock cut")
    train_ends, train_labels = _take_ends_labels(ends, labels, train_i)
    monitor_ends, monitor_labels = _take_ends_labels(ends, labels, monitor_i)
    train = (features[train_i], adjacency[train_i], train_ends, train_labels)
    monitor = (features[monitor_i], adjacency[monitor_i], monitor_ends, monitor_labels)
    n_positive, n_negative = _count_named_monitor_classes(monitor_labels)
    return train, monitor, monitor_clock, n_positive, n_negative


def rewrite_monitor_eval(folder, eeg_path, ecg_path, emg_path, mov_path, events_path):
    if not os.path.isdir(folder):
        raise ValueError(f"run folder is missing: {folder}")
    train_path = os.path.join(folder, "train.npz")
    monitor_path = os.path.join(folder, "monitor.npz")
    if not os.path.isfile(train_path):
        raise ValueError(f"train pack is missing: {train_path}")
    if not os.path.isfile(monitor_path):
        raise ValueError(f"monitor pack is missing: {monitor_path}")
    t1 = float(read_metrics(folder)["t1"])
    train_f, _g, _y, train_ends = load_split(train_path)
    _mf, _mg, _my, old_monitor_ends = load_split(monitor_path)
    if train_ends is None:
        raise ValueError("train pack has no stored ends")
    if old_monitor_ends is None or len(old_monitor_ends) < 1:
        raise ValueError("monitor pack has no stored ends")
    monitor_clock = int(min(old_monitor_ends))
    features, graphs, ends, labels = _labelled_windows(
        eeg_path, ecg_path, emg_path, mov_path, events_path, t1
    )
    train_i, monitor_i = _indices_at_clock(ends, monitor_clock, end_limit=t1)
    if len(train_i) == 0:
        raise ValueError("didnt rebuild any train windows for the z score")
    if len(monitor_i) == 0:
        raise ValueError("didnt rebuild any hop 1 monitor windows")
    if _int_ends(ends, train_i) != _int_ends(train_ends, None):
        raise ValueError(
            "rebuilt train ends do not match the stored train pack, "
            f"we got {_int_ends(ends, train_i)[:5]} vs {_int_ends(train_ends, None)[:5]}"
        )
    means, stds = node_mean_and_std(features[train_i])
    train_scaled = apply_node_zscore(features[train_i], means, stds)
    if train_scaled.shape != train_f.shape:
        raise ValueError(
            "rebuilt train shape does not match the stored train pack, "
            f"we got {train_scaled.shape} vs {train_f.shape}"
        )
    monitor_ends, monitor_labels = _take_ends_labels(ends, labels, monitor_i)
    save_named_labels(
        monitor_path,
        apply_node_zscore(features[monitor_i], means, stds),
        graphs[monitor_i],
        monitor_labels,
        ends=monitor_ends,
    )
    scores_path = os.path.join(folder, "monitor_scores.npz")
    if os.path.isfile(scores_path):
        os.remove(scores_path)
    n_positive, n_negative = _count_named_monitor_classes(monitor_labels)
    extra = {
        "n_monitor_windows": int(len(monitor_ends)),
        "n_positive_monitor": int(n_positive),
        "n_negative_monitor": int(n_negative),
        "n_monitor_ends": int(len(monitor_ends)),
        "monitor_stride_seconds": EVAL_STRIDE_SECONDS,
        "monitor_clock": int(monitor_clock),
        "n_monitor_windows_with_scores": 0,
        "monitor_logits_loaded": False,
    }
    replace_metrics_keys(folder, extra)
    return extra


def _labelled_windows(eeg_path, ecg_path, emg_path, mov_path, events_path, t1):
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
    features, graphs, ends, _drop_counts = build_windows(
        electrical,
        movement,
        stride_seconds=EVAL_STRIDE_SECONDS,
        drop_spans=impd_spans(rows),
        electrical_rails=electrical_rails,
        seizures=seizures,
        t1=t1,
        train_stride_seconds=TRAIN_STRIDE_SECONDS,
    )
    kept = keep_labelled_windows(features, graphs, ends, seizures)
    return kept[0], kept[1], kept[2], kept[3]


def _indices_at_clock(ends, monitor_clock, end_limit):
    train_i = []
    monitor_i = []
    index = 0
    while index < len(ends):
        end_time = ends[index]
        index += 1
        if end_limit is not None and end_time >= end_limit:
            continue
        if end_time < monitor_clock:
            train_i.append(index - 1)
        else:
            monitor_i.append(index - 1)
    return train_i, monitor_i


def _int_ends(ends, indices):
    out = []
    if indices is None:
        pos = 0
        while pos < len(ends):
            out.append(int(ends[pos]))
            pos += 1
        return out
    pos = 0
    while pos < len(indices):
        out.append(int(ends[indices[pos]]))
        pos += 1
    return out


def _take_ends_labels(ends, labels, indices):
    kept_ends = []
    kept_labels = []
    pos = 0
    while pos < len(indices):
        i = indices[pos]
        kept_ends.append(ends[i])
        kept_labels.append(labels[i])
        pos += 1
    return kept_ends, kept_labels


def _count_named_monitor_classes(labels):
    n_positive = 0
    n_negative = 0
    index = 0
    while index < len(labels):
        lab = labels[index]
        if lab == LABEL_SOP:
            n_positive += 1
        elif lab == LABEL_INTERICTAL:
            n_negative += 1
        else:
            raise ValueError(
                f"monitoring slice has unused label {lab}, that shouldnt be in here"
            )
        index += 1
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "monitoring slice is missing a class, "
            f"we got n_positive={n_positive} n_negative={n_negative}"
        )
    return n_positive, n_negative
