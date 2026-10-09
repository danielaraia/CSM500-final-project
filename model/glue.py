import numpy as np

from data.features import log_amplitude_fft
from data.filters import filter_ecg, filter_eeg, filter_emg, filter_movement_axis
from data.labels import LABEL_UNUSED, SOP_SECONDS, label_second, monitoring_indices
from data.movement import axis_magnitude, upsample_movement_to_256
from graphs.pearson import (
    ELECTRICAL_SAMPLES_1S,
    MOVEMENT_BUFFER_SECONDS,
    MOVEMENT_SAMPLES_4S,
    N_ELECTRICAL_NODES,
    N_MOVEMENT_NODES,
    N_NODES,
    build_pearson_graph,
    count_short_buffer_drops,
    steps_with_full_movement_buffer,
)
from data.quality import electrical_rail_reason, need_four_rails, second_drop_reason
from data.spans import second_hits_a_span
from data.windows import (
    EVAL_STRIDE_SECONDS,
    TRAIN_STRIDE_SECONDS,
    WINDOW_SECONDS,
    end_is_on_stride,
    window_end_times,
)

N_MOVEMENT_AXES = 12
AXES_PER_DEVICE = 3
SAMPLES_MOVEMENT_1S = 25
# mixed pearson pairs need 4 sec at 25 Hz, so graphs 0,1,2 dont exist
SECONDS_BEFORE_FIRST_GRAPH = MOVEMENT_BUFFER_SECONDS - 1
SECONDS_NEEDED = SECONDS_BEFORE_FIRST_GRAPH + WINDOW_SECONDS


def build_one_window(electrical, movement_axes, electrical_rails):
    features, graphs, _ends, _drop_counts = build_windows(
        electrical,
        movement_axes,
        stride_seconds=EVAL_STRIDE_SECONDS,
        electrical_rails=electrical_rails,
    )
    if features.shape[0] < 1:
        raise ValueError("didnt get a window, too many bad seconds")
    return features[0], graphs[0]


def build_windows(
    electrical,
    movement_axes,
    stride_seconds=TRAIN_STRIDE_SECONDS,
    drop_spans=None,
    electrical_rails=None,
    seizures=None,
    t1=None,
    train_stride_seconds=None,
    graph_builder=None,
    sop_seconds=SOP_SECONDS,
):
    if (t1 is None) != (train_stride_seconds is None):
        raise ValueError("t1 and train stride have to be set together, or neither")
    if graph_builder is None:
        graph_builder = build_pearson_graph
    need_four_rails(electrical_rails)
    electrical = np.asarray(electrical, dtype=np.float64)
    movement_axes = np.asarray(movement_axes, dtype=np.float64)
    _n_seconds(electrical, movement_axes)

    electrical_raw_seconds = _cut_seconds(electrical, ELECTRICAL_SAMPLES_1S)
    electrical = _filter_electrical(electrical)
    movement = _movement_magnitudes(movement_axes)
    electrical_seconds = _cut_seconds(electrical, ELECTRICAL_SAMPLES_1S)
    movement_seconds = _cut_seconds(movement, SAMPLES_MOVEMENT_1S)

    quality_ok, drop_counts = _quality_ok_and_drop_counts(
        electrical_seconds,
        movement_seconds,
        drop_spans,
        electrical_raw_seconds,
        electrical_rails,
    )
    graph_ok = _graph_ok_seconds(quality_ok)
    ends = window_end_times(graph_ok, WINDOW_SECONDS, stride_seconds)
    ends = _keep_pack_ends(ends, seizures, t1, train_stride_seconds, sop_seconds)

    n_windows = len(ends)
    features = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, 128), dtype=np.float64)
    adjacency = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, N_NODES), dtype=np.float64)
    feat_by_second = {}
    graph_by_second = {}
    window_index = 0
    while window_index < n_windows:
        end_time = ends[window_index]
        start_time = end_time - WINDOW_SECONDS + 1
        step = 0
        while step < WINDOW_SECONDS:
            index = start_time + step
            if index not in feat_by_second:
                if not graph_ok[index]:
                    raise ValueError(
                        f"window end {end_time} wants sec {index} "
                        f"bjt that second isnt graph ok"
                    )
                feat_by_second[index] = _features_for_second(
                    electrical_seconds[index],
                    movement_seconds[index],
                )
                graph_by_second[index] = _graph_for_second(
                    electrical_seconds, movement_seconds, index, graph_builder
                )
            features[window_index, step] = feat_by_second[index]
            adjacency[window_index, step] = graph_by_second[index]
            step += 1
        window_index += 1
    return features, adjacency, ends, drop_counts


def keep_labelled_windows(features, adjacency, ends, seizures, sop_seconds=SOP_SECONDS):
    features = np.asarray(features)
    adjacency = np.asarray(adjacency)
    if features.shape[0] != adjacency.shape[0] or features.shape[0] != len(ends):
        raise ValueError(
            "features, graphs and ends need the same count, "
            f"we got {features.shape[0]}, {adjacency.shape[0]}, {len(ends)}"
        )
    kept_features = []
    kept_graphs = []
    kept_ends = []
    kept_labels = []
    index = 0
    while index < len(ends):
        # label sits on the last second of the window, not the whole 60
        lab = label_second(ends[index], seizures, sop_seconds)
        if lab != LABEL_UNUSED:
            kept_features.append(features[index])
            kept_graphs.append(adjacency[index])
            kept_ends.append(ends[index])
            kept_labels.append(lab)
        index += 1
    if len(kept_labels) == 0:
        raise ValueError("didnt keep any labelled windows")
    return (
        np.stack(kept_features, axis=0),
        np.stack(kept_graphs, axis=0),
        kept_ends,
        kept_labels,
    )


def split_labelled_windows(features, adjacency, ends, labels, t1, t2):
    features = np.asarray(features)
    adjacency = np.asarray(adjacency)
    if t2 < t1:
        raise ValueError(f"t2 is befor t1, we got t1={t1} t2={t2}")
    if features.shape[0] != adjacency.shape[0] or features.shape[0] != len(ends):
        raise ValueError(
            "features, graphs and ends need the same count, "
            f"we got {features.shape[0]}, {adjacency.shape[0]}, {len(ends)}"
        )
    if len(labels) != len(ends):
        raise ValueError(
            f"labels and ends need the same count, we got {len(labels)} and {len(ends)}"
        )
    train_i = []
    val_i = []
    test_i = []
    index = 0
    while index < len(ends):
        end_time = ends[index]
        start_time = end_time - features.shape[1] + 1
        if end_time < t1:
            train_i.append(index)
        elif start_time >= t1 and end_time < t2:
            val_i.append(index)
        elif start_time >= t2:
            test_i.append(index)
        index += 1
    if len(train_i) == 0:
        raise ValueError("no train windows")
    if len(test_i) == 0:
        raise ValueError("no test windows")
    train = _gather_windows(train_i, features, adjacency, ends, labels)
    val = _gather_windows(val_i, features, adjacency, ends, labels)
    test = _gather_windows(test_i, features, adjacency, ends, labels)
    return train, val, test


def cut_monitoring_slice(features, adjacency, ends, labels):
    features = np.asarray(features)
    adjacency = np.asarray(adjacency)
    monitor_start, n_positive, n_negative = monitoring_indices(labels)
    n_windows = len(labels)
    train_i = []
    monitor_i = []
    monitor_clock = ends[monitor_start]
    train_end_limit = monitor_clock - features.shape[1] + 1
    index = 0
    while index < n_windows:
        if ends[index] < train_end_limit:
            train_i.append(index)
        elif index >= monitor_start:
            monitor_i.append(index)
        index += 1
    if len(train_i) == 0:
        raise ValueError("didnt leave any train windows after the monitoring cut")
    train = _gather_windows(train_i, features, adjacency, ends, labels)
    monitor = _gather_windows(monitor_i, features, adjacency, ends, labels)
    return train, monitor, monitor_start, n_positive, n_negative


def _keep_pack_ends(ends, seizures, t1, train_stride_seconds, sop_seconds=SOP_SECONDS):
    labelled = []
    index = 0
    while index < len(ends):
        end_time = ends[index]
        index += 1
        if seizures is not None:
            if label_second(end_time, seizures, sop_seconds) == LABEL_UNUSED:
                continue
        labelled.append(end_time)
    monitor_clock = None
    if t1 is not None and seizures is not None:
        hop10_train = []
        pos = 0
        while pos < len(labelled):
            end_time = labelled[pos]
            pos += 1
            if end_time < t1 and end_is_on_stride(end_time, train_stride_seconds):
                hop10_train.append(end_time)
        n_hop10 = len(hop10_train)
        n_monitor = n_hop10 // 4
        if n_monitor > 0:
            monitor_clock = hop10_train[n_hop10 - n_monitor]
    kept = []
    index = 0
    while index < len(labelled):
        end_time = labelled[index]
        index += 1
        if t1 is not None and end_time < t1:
            if monitor_clock is None:
                if not end_is_on_stride(end_time, train_stride_seconds):
                    continue
            elif end_time < monitor_clock:
                if not end_is_on_stride(end_time, train_stride_seconds):
                    continue
        kept.append(end_time)
    return kept


def _gather_windows(indices, features, adjacency, ends, labels):
    if len(indices) == 0:
        return features[:0], adjacency[:0], [], []
    kept_features = []
    kept_graphs = []
    kept_ends = []
    kept_labels = []
    pos = 0
    while pos < len(indices):
        i = indices[pos]
        kept_features.append(features[i])
        kept_graphs.append(adjacency[i])
        kept_ends.append(ends[i])
        kept_labels.append(labels[i])
        pos += 1
    return (
        np.stack(kept_features, axis=0),
        np.stack(kept_graphs, axis=0),
        kept_ends,
        kept_labels,
    )


def _n_seconds(electrical, movement_axes):
    if electrical.ndim != 2 or electrical.shape[0] != N_ELECTRICAL_NODES:
        raise ValueError(
            f"electrical should be ({N_ELECTRICAL_NODES}, samples), we got {electrical.shape}"
        )
    if movement_axes.ndim != 2 or movement_axes.shape[0] != N_MOVEMENT_AXES:
        raise ValueError(
            f"movement should be ({N_MOVEMENT_AXES}, samples), we got {movement_axes.shape}"
        )
    if electrical.shape[1] % ELECTRICAL_SAMPLES_1S != 0:
        raise ValueError(
            f"electrical length has to split into 1 sec chunks, we got {electrical.shape[1]}"
        )
    if movement_axes.shape[1] % SAMPLES_MOVEMENT_1S != 0:
        raise ValueError(
            f"movement length has to split into 1 sec chunks, we got {movement_axes.shape[1]}"
        )
    n_electrical = electrical.shape[1] // ELECTRICAL_SAMPLES_1S
    n_movement = movement_axes.shape[1] // SAMPLES_MOVEMENT_1S
    if n_electrical != n_movement:
        raise ValueError(
            f"electrical and movement need the same number of seconds, "
            f"we got {n_electrical} and {n_movement}"
        )
    return n_electrical


def _quality_ok_and_drop_counts(
    electrical_seconds,
    movement_seconds,
    drop_spans,
    electrical_raw_seconds,
    electrical_rails,
):
    need_four_rails(electrical_rails)
    n_seconds = len(electrical_seconds)
    if electrical_raw_seconds.shape[0] != n_seconds:
        raise ValueError(
            "raw and filtered electrical need the same second count, "
            f"we got {electrical_raw_seconds.shape[0]} and {n_seconds}"
        )
    quality_ok = []
    n_electrical_flat = 0
    n_electrical_rail = 0
    n_both_ears_rms = 0
    n_movement_flat = 0
    n_drop_span = 0
    second_index = 0
    while second_index < n_seconds:
        rail_reason = electrical_rail_reason(
            electrical_raw_seconds[second_index],
            electrical_rails,
        )
        if rail_reason is not None:
            quality_ok.append(False)
            n_electrical_rail += 1
        else:
            reason = second_drop_reason(
                electrical_seconds[second_index, 0],
                electrical_seconds[second_index, 1],
                electrical_seconds[second_index, 2],
                electrical_seconds[second_index, 3],
                movement_seconds[second_index, 0],
                movement_seconds[second_index, 1],
                movement_seconds[second_index, 2],
                movement_seconds[second_index, 3],
            )
            if reason is None:
                quality_ok.append(True)
            elif reason == "electrical_flat":
                quality_ok.append(False)
                n_electrical_flat += 1
            elif reason == "both_ears_rms":
                quality_ok.append(False)
                n_both_ears_rms += 1
            elif reason == "movement_flat":
                quality_ok.append(False)
                n_movement_flat += 1
            else:
                raise ValueError(f"unknown quality reason {reason}")
        second_index += 1
    second_index = 0
    while second_index < n_seconds:
        if quality_ok[second_index] and second_hits_a_span(
            second_index, drop_spans or []
        ):
            n_drop_span += 1
        second_index += 1
    quality_ok = _apply_drop_spans(quality_ok, drop_spans)
    n_qc = (
        n_electrical_flat
        + n_electrical_rail
        + n_both_ears_rms
        + n_movement_flat
        + n_drop_span
    )
    drop_counts = {
        "n_steps_dropped_qc": n_qc,
        "n_steps_dropped_short_buffer": count_short_buffer_drops(quality_ok),
        "n_steps_qc_electrical_flat": n_electrical_flat,
        "n_steps_qc_electrical_rail": n_electrical_rail,
        "n_steps_qc_both_ears_rms": n_both_ears_rms,
        "n_steps_qc_movement_flat": n_movement_flat,
        "n_steps_qc_drop_span": n_drop_span,
    }
    return quality_ok, drop_counts


def _apply_drop_spans(quality_ok, drop_spans):
    if drop_spans is None:
        drop_spans = []
    ok = []
    second_index = 0
    while second_index < len(quality_ok):
        hit = second_hits_a_span(second_index, drop_spans)
        ok.append(bool(quality_ok[second_index] and not hit))
        second_index += 1
    return ok


def _graph_ok_seconds(quality_ok):
    has_buffer = steps_with_full_movement_buffer(quality_ok)
    graph_ok = []
    second_index = 0
    while second_index < len(quality_ok):
        graph_ok.append(bool(quality_ok[second_index] and has_buffer[second_index]))
        second_index += 1
    return graph_ok


def _filter_electrical(electrical):
    out = np.zeros_like(electrical)
    out[0] = filter_eeg(electrical[0])
    out[1] = filter_eeg(electrical[1])
    out[2] = filter_ecg(electrical[2])
    out[3] = filter_emg(electrical[3])
    return out


def _movement_magnitudes(movement_axes):
    magnitudes = np.zeros(
        (N_MOVEMENT_NODES, movement_axes.shape[1]), dtype=np.float64
    )
    device = 0
    while device < N_MOVEMENT_NODES:
        start = device * AXES_PER_DEVICE
        x = filter_movement_axis(movement_axes[start])
        y = filter_movement_axis(movement_axes[start + 1])
        z = filter_movement_axis(movement_axes[start + 2])
        magnitudes[device] = axis_magnitude(x, y, z)
        device += 1
    return magnitudes


def _cut_seconds(run, samples_per_second):
    n_seconds = run.shape[1] // samples_per_second
    n_nodes = run.shape[0]
    seconds = np.zeros((n_seconds, n_nodes, samples_per_second), dtype=np.float64)
    second_index = 0
    while second_index < n_seconds:
        start = second_index * samples_per_second
        seconds[second_index] = run[:, start : start + samples_per_second]
        second_index += 1
    return seconds


def _features_for_second(electrical_1s, movement_1s):
    feats = np.zeros((N_NODES, 128), dtype=np.float64)
    node = 0
    while node < N_ELECTRICAL_NODES:
        feats[node] = log_amplitude_fft(electrical_1s[node])
        node += 1
    node = 0
    while node < N_MOVEMENT_NODES:
        up = upsample_movement_to_256(movement_1s[node])
        feats[N_ELECTRICAL_NODES + node] = log_amplitude_fft(up)
        node += 1
    return feats


def _graph_for_second(electrical_seconds, movement_seconds, second_index, graph_builder):
    electrical_1s = electrical_seconds[second_index]
    electrical_chunks = []
    movement_chunks = []
    offset = 0
    while offset < MOVEMENT_BUFFER_SECONDS:
        index = second_index - SECONDS_BEFORE_FIRST_GRAPH + offset
        electrical_chunks.append(electrical_seconds[index])
        movement_chunks.append(movement_seconds[index])
        offset += 1
    electrical_4s = np.concatenate(electrical_chunks, axis=1)
    movement_4s = np.concatenate(movement_chunks, axis=1)
    if movement_4s.shape[1] != MOVEMENT_SAMPLES_4S:
        raise ValueError(
            f"4 sec movement should be {MOVEMENT_SAMPLES_4S} samples, we got {movement_4s.shape[1]}"
        )
    return graph_builder(electrical_1s, electrical_4s, movement_4s)
